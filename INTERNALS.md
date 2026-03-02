# Self-Correction Agent : 내부 동작 추적

> 각 함수가 어떤 데이터를 받고 어떻게 처리하는지 추적한다.
> 전체 아키텍처/파일 구조는 `DOCS.md` 참조.

---

## 0. 진입점

```
python3 -m self_correction_agent --query "..."
  └─ orchestrator.run_agent(query, model_name, db_path)
```

`--model` 미지정 → `mode="mock"` / 지정 시 `create_pydantic_agent()` 호출 후 `mode="pydantic-ai"`

> **EXAONE 특이사항**: EXAONE 3.5는 Ollama에서 function calling을 지원하지 않는다.
> `_LOCAL_NO_TOOL_MODELS = {"exaone"}` 에 등록되어 tools 없는 텍스트 파싱 경로로 실행된다.

### 상태 머신 구조 — orchestrator.py

흐름 제어는 pydantic-ai Graph API나 LangGraph를 사용하지 않는 **직접 구현 while 루프**다.

```python
state = AgentState(query=query)          # 전체 상태를 담는 단일 객체

while state.phase not in (Phase.DONE, Phase.FAILED):
    if   state.phase == Phase.PLANNING:
        plan_search_queries(...)
        state.phase = Phase.SEARCHING    # 전이: 코드로 직접 지정

    elif state.phase == Phase.SEARCHING:
        execute_search(...)
        state.phase = Phase.DRAFTING

    elif state.phase == Phase.DRAFTING:
        state.draft = worker_generate_draft(...)
        state.phase = Phase.CRITIQUING

    elif state.phase == Phase.CRITIQUING:
        verdict = critic_evaluate(...)
        if verdict.passed:     state.phase = Phase.DONE
        elif state.can_retry:  state.phase = Phase.PLANNING   # Self-Healing
        else:                  state.phase = Phase.FAILED
```

- **pydantic**: `AgentState`, `CriticVerdict` 등 데이터 모델 정의만 담당
- **pydantic-ai Agent**: LLM 호출 래퍼 (`agent.run_sync()`). 흐름 제어 불관여
- 노드 함수(planner/searcher/worker/critic)는 일반 Python 함수 — 특별한 등록 불필요

---

## 1. 초기화

### LanceDB 초기화 — `LocalRAG.initialize(KNOWLEDGE_BASE)`

```
입력: 10건의 dict {"id", "topic", "text", "source"}

처리:
  if TABLE_NAME in db.table_names():
      open existing table
      sample = table.to_arrow().slice(0, 1)
      if len(sample[0]["vector"]) != 384:   # 구 BoW(48차원) 감지
          db.drop_table() → seed(seed_docs)  # 자동 마이그레이션
      else:
          _rebuild_bm25()  # BM25 인덱스 재구성 후 반환
  else:
      seed(KNOWLEDGE_BASE) → 신규 생성

출력: 현재 인덱싱된 청크 수
```

> `initialize()`는 테이블이 있으면 재생성하지 않는다 — 웹 UI에서 추가한 문서가 재시작 후에도 유지됨.
> 강제 재초기화가 필요하면 `/db/seed` 엔드포인트 또는 `LocalRAG.seed()` 직접 호출.

### Sentence Transformers 임베딩 — `text_to_vector(text)`

```
입력: "GPT-4o (omni) was released in May 2024..."

처리:
  1. 모듈 레벨 싱글톤 _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') 로드
     (한/영 다국어, ~470MB, 첫 호출 시 초기화)
  2. model.encode(text, normalize_embeddings=True)
     → 384차원 float32 numpy 배열 (L2 정규화 포함)
  3. .tolist() → Python float 리스트 반환

출력: [0.037, -0.009, 0.028, ...] (384차원 float 리스트)
```

> BoW(48차원) 대비: 어휘 겹침 없이도 의미 유사도 계산 가능, 한국어 정확도 대폭 향상.

### 문서 청킹 — `chunker.chunk_text(text)`

```
입력: 긴 문서 문자열

처리:
  if len(text) <= 500: return [text]
  슬라이딩 윈도우:
    start=0, end=min(start+500, len)
    chunks.append(text[start:end])
    start = end - 50   ← 50자 오버랩
  마지막 청크 < 125자이면 이전 청크에 흡수

출력: 청크 리스트 (각 ≤500자, 50자 오버랩)
```

각 청크는 개별 `text_to_vector()` 호출 → 별도 384차원 벡터 → LanceDB에 같은 `parent_id`로 저장.

---

## 2. PLANNING — `nodes/planner.py:plan_search_queries()`

```
입력: state, base_query, model_name (optional)

── 재시도 분기 (state.retry_count > 0) ─────────────────────────────
  기존 쿼리 유지 + Critic 누락 키워드만 추가:
    queries += [f"{kw} latest" for kw in critique.missing_keywords]
  → 기존 쿼리 4개 + 누락 키워드 쿼리 N개
  → SEARCHING 전이

── 첫 시도: LLM 확장 (model_name 있을 때) ──────────────────────────
  expand_query_with_llm(base_query, model_name)  [nodes/llm_helpers.py]

    tool-capable 모델 (GPT 등):
      → Agent(result_type=ExpandedQueries)로 구조화 JSON 출력
    no-tool 모델 (EXAONE 등, _LOCAL_NO_TOOL_MODELS에 등록):
      → Agent()로 줄바꿈 텍스트 출력 → _parse_lines()로 파싱
    실패 시 [] 반환 → heuristic fallback

  generate_eval_criteria(base_query, model_name)  [nodes/llm_helpers.py]
    tool-capable 모델: result_type=EvalCriteria (JSON)
    no-tool 모델 (EXAONE 등): 줄바꿈 텍스트 → _parse_lines() → 최대 10개 추출
    → state.eval_criteria에 저장 → CRITIQUING에서 사용
    → 실패 시 → REQUIRED_KEYWORDS fallback

── 첫 시도: Heuristic fallback (LLM 없거나 실패) ───────────────────
  _TOPIC_ALIASES 딕셔너리로 쿼리 키워드 감지
    예) "openai" 포함 → 3개 보조 쿼리
        매칭 없음     → 범용 AI 쿼리 3개
  queries = [base_query] + supplements  (총 4개)

출력: 4~N개 쿼리 리스트 + state.search_queries_used 업데이트
     + state.eval_criteria (LLM 모드 첫 시도 시)
상태 전이: → SEARCHING
```

---

## 3. SEARCHING — `nodes/searcher.py:execute_search()`

```
입력: rag, queries (4~N개), state, top_k=2, alpha=0.7, distance_threshold=1.5

처리:
  seen = set()
  for sq in queries:
      hits = rag.hybrid_search(sq, top_k=2, alpha=alpha, distance_threshold=distance_threshold)
        └─ 벡터 검색: text_to_vector(sq) → 384차원 → LanceDB.search() → 거리 기반 후보
        └─ BM25 검색: BM25Okapi.get_scores(sq.lower().split()) → 키워드 점수
        └─ 결합: alpha * norm_vec + (1-alpha) * norm_bm25
        └─ parent_id 중복 제거 (최고 combined 청크만 유지)
      for h in hits:
          key = h["text"][:80]   ← 쿼리 간 중복 방지 키
          if key not in seen: seen.add(key); all_results.append(h)

  state.search_results = all_results

alpha/threshold 값은 orchestrator가 settings.json에서 로드하여 전달.

출력: 1차 시도 약 5건, 재시도 약 7~10건 (중복 제거 후)
상태 전이: 결과 있음→DRAFTING, 없음→FAILED
```

---

## 4. DRAFTING — `nodes/worker.py:worker_generate_draft()`

### Mock 모드 — `_mock_generate(state)`

```
입력: state.search_results (5건), state.query

처리:
  topics = {토픽명: [결과들]} 으로 그룹핑
  parts = ["# {user_query}", "## 요약", "This report covers N areas..."]
  for topic, results in topics:
      parts += ["## {topic}", result["text"], "_출처: {source}_"]
  parts += ["## 결론", "..."]
  return "\n\n".join(parts)
```

### Pydantic AI 모드 — `_build_worker_prompt(state)`

```
프롬프트 구성:
  사용자 요청: {state.query}

  지식베이스:
  - [{topic}] {text}
  ...

  [재시도 시] PREVIOUS ATTEMPT FEEDBACK:
  Missing topics: Sora, o1
  You MUST address these gaps.

  작성 규칙: 한국어로, ## 헤더, 출처: ... 형식, 사실 기반

tools 지원 모델 (GPT 등):
  agent.run_sync(prompt, deps=rag)  # LanceDB search tool 사용 가능

no-tool 모델 (EXAONE 등):
  agent.run_sync(prompt)  # 검색 결과는 프롬프트에 직접 포함 (tools 없음)

return result.output  # pydantic-ai >= 1.0
```

---

## 5. CRITIQUING — `nodes/critic.py:critic_evaluate()`

```
입력: draft (리포트 문자열), eval_criteria (optional)

키워드 목록 결정:
  eval_criteria 있음 (LLM 모드): LLM이 생성한 expected_topics 사용
  eval_criteria 없음 (Mock 모드): REQUIRED_KEYWORDS fallback
    = ["GPT-4o", "Sora", "API", "o1",
       ("멀티모달","multimodal"), ("파트너십","partnership"),
       ("엔터프라이즈","enterprise","기업용"), ("안전성","안전","safety")]

Check 1 — 키워드 커버리지 (×0.4):
  튜플 항목은 OR — 하나라도 있으면 통과
  kw_ratio = found / len(kws)
  score_1 = kw_ratio * 0.4

Check 2 — 콘텐츠 길이 (×0.2):
  score_2 = min(len(draft) / 500, 1.0) * 0.2

Check 3 — 문서 구조 (×0.2):
  score_3 = min(draft.count("## ") / 4, 1.0) * 0.2

Check 4 — 출처 인용 (×0.2):
  score_4 = min(draft.lower().count("source:") / 3, 1.0) * 0.2

Hard Gate: kw_ratio < 0.85 → 총점과 무관하게 passed=False
passed = (total >= 0.7) AND (kw_ratio >= 0.85)

출력: CriticVerdict { passed, score, feedback, missing_keywords, suggestions }
상태 전이: passed→DONE / can_retry→PLANNING / else→FAILED
```

---

## 6. Self-Healing 분기 — `orchestrator.py`

```python
if verdict.passed:
    state.final_result = state.draft
    state.phase = Phase.DONE

elif state.can_retry:               # retry_count < max_retries(3)
    state.retry_count += 1
    state.phase = Phase.PLANNING    # critique.missing_keywords 보존된 채 재진입

else:
    state.final_result = state.draft  # Best-Effort 반환
    state.phase = Phase.FAILED
```

---

## 7. LocalRAG CRUD — `infra/vectordb.py`

| 메서드 | 기능 |
|--------|------|
| `initialize(seed_docs)` | 테이블 없을 때만 생성+시드. 있으면 차원 확인 후 마이그레이션 또는 재사용 |
| `seed(documents)` | 강제 재초기화 (DROP → 청킹+임베딩 → CREATE). 관리자 전용 |
| `hybrid_search(query, top_k, alpha, distance_threshold)` | 벡터+BM25 하이브리드 검색. `parent_id` 중복 제거 후 `top_k` 반환 |
| `search(query, top_k)` | `hybrid_search` 기본값(alpha=0.7) 래퍼 |
| `add_document(doc)` | 청킹+임베딩 후 추가. `parent_id` 반환. `_rebuild_bm25()` 자동 호출 |
| `delete_document(doc_id)` | `parent_id` 기준으로 해당 문서의 모든 청크 삭제 |
| `list_documents(offset, limit)` | `chunk_index==0` 청크만 반환 (`parent_id` 중복 제거). `id` 컬럼은 `parent_id` 값 |
| `_rebuild_bm25()` | 전체 청크 텍스트로 BM25Okapi 인덱스 재구성 |

---

## 8. 전체 실행 타임라인 (1차 실패 → 2차 성공 예시)

```
시간  함수                              phase 전이            핵심 데이터
─────┬──────────────────────────────────┬─────────────────────┬──────────────────────
 T0  │ run_agent()                      │                     │ mode=pydantic-ai
 T1  │ load_settings(db_path)           │                     │ alpha=0.7, top_k=2
 T2  │ LocalRAG.initialize(10건)        │                     │ ST 384차원, BM25 재구성
 T3  │ AgentState(query=...)            │ → PLANNING          │ 빈 상태 초기화
─────┼── 1차 시도 ───────────────────────┼─────────────────────┼──────────────────────
 T4  │ plan_search_queries()            │ PLANNING→SEARCHING  │ LLM 확장 4개 쿼리
     │   expand_query_with_llm()        │                     │ eval_criteria 생성 → 저장
     │   generate_eval_criteria()       │                     │
 T5  │ execute_search()                 │ SEARCHING→DRAFTING  │ hybrid_search × 4쿼리
     │   hybrid_search(alpha=0.7)       │                     │ search_results=[5건]
 T6  │ worker_generate_draft()          │ DRAFTING→CRITIQUING │ draft 생성
 T7  │ critic_evaluate(eval_criteria=…) │ CRITIQUING→PLANNING │ score=0.85, passed=F
     │                                  │                     │ missing=[Sora,o1]
     │                                  │                     │ retry_count: 0→1
─────┼── 2차 시도 (Self-Healing) ────────┼─────────────────────┼──────────────────────
 T8  │ plan_search_queries()            │ PLANNING→SEARCHING  │ 기존 4 + 누락 2 = 6쿼리
 T9  │ execute_search()                 │ SEARCHING→DRAFTING  │ search_results=[7건]
 T10 │ worker_generate_draft()          │ DRAFTING→CRITIQUING │ Sora, o1 포함 draft
 T11 │ critic_evaluate(eval_criteria=…) │ CRITIQUING→DONE     │ score=1.00, passed=T
─────┼──────────────────────────────────┼─────────────────────┼──────────────────────
 T12 │ return state.final_result        │ DONE                │ 최종 리포트 반환
```

---

## 9. Ollama 연결 구조

### Docker ↔ Ollama 통신 경로

```
Docker Container
  Agent("ollama:exaone3.5:7.8b")
    └─ pydantic-ai 1.x: ollama: prefix → OpenAIChatModel 사용
    └─ OLLAMA_BASE_URL = http://host.docker.internal:11434/v1
    └─ POST http://host.docker.internal:11434/v1/chat/completions
         ↕ host.docker.internal (macOS: 자동 해석, Linux: extra_hosts 필요)
Ollama (호스트 macOS, brew services)
  :11434/v1/chat/completions  ← OpenAI 호환 엔드포인트
  exaone3.5:7.8b 모델 응답
```

### 환경변수 역할

| 변수 | 값 | 설명 |
|------|----|------|
| `AGENT_MODEL` | `ollama:exaone3.5:7.8b` | pydantic-ai에 전달되는 모델 문자열 |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` | pydantic-ai `ollama:` prefix가 사용하는 base URL. `/v1` 필수 |
| `OPENAI_BASE_URL` | `http://host.docker.internal:11434/v1` | `openai:` prefix 또는 OpenAI SDK 직접 사용 시 |
| `OPENAI_API_KEY` | `dummy-key-for-local` | OpenAI SDK가 키를 요구하나 로컬은 임의 값 허용 |

### pydantic-ai 1.x `ollama:` prefix 처리 메커니즘

```python
# pydantic-ai가 "ollama:exaone3.5:7.8b" 문자열을 받으면:
# 1. OpenAIChatModel로 라우팅 (OpenAI 호환 클라이언트)
# 2. base_url = os.environ["OLLAMA_BASE_URL"]
# 3. 최종 URL = f"{base_url}/chat/completions"
#    → "http://host.docker.internal:11434/v1/chat/completions" ✓
#    (/v1 없으면 → "/chat/completions" → 404)
```

### EXAONE tool calling 미지원

```python
# agent.py / llm_helpers.py
_LOCAL_NO_TOOL_MODELS = {"exaone"}  # 부분 문자열 매칭

# "ollama:exaone3.5:7.8b".lower()에 "exaone" 포함 → use_tools = False
# → result_type=ExpandedQueries 사용 안 함 (structured output 불가)
# → 줄바꿈 텍스트 출력 → _parse_lines() 파싱
```

---

## 10. 함수 호출 그래프

```
run_agent()
  ├─ load_settings(db_path)             # settings.json → alpha/threshold/top_k
  ├─ create_pydantic_agent()            # model 있을 때만
  ├─ LocalRAG.initialize()
  │    └─ (없을 때만) seed()
  │         ├─ chunk_text(doc["text"])  # 500자/50자 슬라이딩 윈도우
  │         ├─ text_to_vector(chunk) × 청크마다   # ST 384차원
  │         └─ _rebuild_bm25()         # BM25Okapi 인덱스 구성
  └─ while loop:
       ├─ [PLANNING]   plan_search_queries(state, query, model_name)
       │    ├─ (첫 시도 + LLM) expand_query_with_llm()    → ExpandedQueries
       │    │    └─ 실패 시 heuristic fallback
       │    ├─ (첫 시도 + LLM) generate_eval_criteria()   → EvalCriteria
       │    │    └─ state.eval_criteria 저장
       │    └─ (재시도) missing_keywords → 추가 쿼리
       │
       ├─ [SEARCHING]  execute_search(rag, queries, state, alpha, threshold)
       │    └─ rag.hybrid_search(query, top_k, alpha, distance_threshold) × N쿼리
       │         ├─ text_to_vector(query) → LanceDB.search() → vec_sim 점수
       │         ├─ BM25Okapi.get_scores() → bm25_sim 점수
       │         ├─ combined = alpha * vec_sim + (1-alpha) * bm25_sim
       │         └─ parent_id 중복 제거 → top_k 반환
       │
       ├─ [DRAFTING]   worker_generate_draft(state, mode, ...)
       │    ├─ mock: _mock_generate() → 템플릿 마크다운
       │    └─ pydantic-ai: agent.run_sync(prompt, deps=rag)
       │
       └─ [CRITIQUING] critic_evaluate(draft, eval_criteria)
            ├─ eval_criteria 있으면 동적 기준 / 없으면 REQUIRED_KEYWORDS
            ├─ 키워드 OR 그룹 체크 (×0.4)
            ├─ 길이 체크 (×0.2)
            ├─ ## 헤더 체크 (×0.2)
            ├─ 출처 인용 체크 (×0.2)
            ├─ Hard Gate: kw_ratio < 85% → 무조건 RETRY
            └─ CriticVerdict → DONE / PLANNING(retry) / FAILED
```
