# Self-Correction Agent : 내부 동작 기술 문서

> 쿼리 입력부터 최종 리포트 출력까지, 각 함수가 어떤 데이터를 받고 어떻게 처리하는지 추적한다.

---

## 0. 진입점 — `__main__.py` → `orchestrator.run_agent()`

```
python3 -m self_correction_agent --query "OpenAI 최신 동향 리포트를 작성해주세요"
```

### `__main__.py:main()`

| 단계 | 동작 |
|------|------|
| 1 | `argparse`로 `--model`, `--query`, `--db-path` 파싱 |
| 2 | `--model` 미지정 시 환경변수 `AGENT_MODEL` 확인, 둘 다 없으면 `None` (mock 모드) |
| 3 | `orchestrator.run_agent(query, model_name, db_path)` 호출 |
| 4 | 반환된 최종 리포트 문자열을 `print()` |

---

## 1. 초기화 — `orchestrator.run_agent()` (L26-61)

### 1-1. 모드 결정

```python
mode = "mock"
pydantic_agent = None
if model_name:
    pydantic_agent = create_pydantic_agent(model_name)  # agent.py
    if pydantic_agent:
        mode = "pydantic-ai"
```

- `--model`이 없으면 → `mode="mock"` (LLM 호출 없이 템플릿 생성)
- `--model ollama:llama3.2` 등 지정 시 → `agent.py`가 Pydantic AI Agent 객체 생성

### 1-2. LanceDB 초기화 + 시드 데이터 인덱싱

```python
rag = LocalRAG(db_path)        # infra/vectordb.py
n = rag.seed(KNOWLEDGE_BASE)   # knowledge/openai_trends.py의 10건
```

**`LocalRAG.__init__(db_path)`**
- `lancedb.connect(db_path)` → `/tmp/lancedb_self_correction_agent/` 에 로컬 DB 연결
- `_table = None` (아직 테이블 참조 없음)

**`LocalRAG.seed(documents)` — 핵심 인덱싱 과정:**

```
입력: KNOWLEDGE_BASE (dict 10건)
      [{"id":"1", "topic":"GPT-4o", "text":"GPT-4o (omni) was...", "source":"..."},
       {"id":"2", "topic":"Sora",   "text":"Sora is OpenAI's...", "source":"..."}, ...]

처리:
  for doc in documents:                           # 10회 반복
      vec = text_to_vector(doc["text"])            # → 60차원 float 벡터
      records.append({
          "id", "text", "topic", "source",
          "vector": vec                            # 임베딩 벡터 추가
      })

  if "knowledge_base" 테이블이 이미 존재:
      DROP TABLE                                   # 기존 데이터 삭제
  CREATE TABLE "knowledge_base" with records       # 새로 생성

출력: 10 (인덱싱된 문서 수)
```

### 1-3. `text_to_vector()` — Bag-of-Words 임베딩 (infra/embedding.py)

60개 단어로 구성된 고정 어휘(`_VOCAB`)를 기준으로 벡터를 생성한다.

```
입력: "GPT-4o (omni) was released in May 2024 as OpenAI's most advanced
       multimodal model..."

처리:
  1. text.lower() → 소문자 변환
  2. _VOCAB (60개 단어) 각각에 대해 출현 횟수 카운트
     예: "model" → 2회, "multimodal" → 1회, "gpt" → 1회, "sora" → 0회 ...
  3. 60차원 벡터 생성: [0, 0, ..., 1, 2, ..., 0]
  4. L2 정규화: vec / ||vec||₂
     → 벡터 크기를 1로 맞춰서 코사인 유사도 검색 가능

출력: [0.0, 0.0, 0.123, 0.456, ...] (60차원 float 리스트)
```

### 1-4. AgentState 초기화

```python
state = AgentState(query=query)
```

```
AgentState {
    phase:              Phase.PLANNING     ← 시작 단계
    query:              "OpenAI 최신 동향 리포트를 작성해주세요"
    search_results:     []
    search_queries_used: []
    draft:              ""
    critique:           None
    retry_count:        0
    max_retries:        3
    history:            []
    final_result:       None
}
```

---

## 2. 메인 루프 — 상태 머신 (orchestrator.py L68-136)

```python
while state.phase not in (Phase.DONE, Phase.FAILED):
    if state.phase == Phase.PLANNING:   ...
    elif state.phase == Phase.SEARCHING: ...
    elif state.phase == Phase.DRAFTING:  ...
    elif state.phase == Phase.CRITIQUING: ...
```

`phase` 값에 따라 해당 노드 함수를 호출하고, 완료되면 다음 phase로 전이한다.

---

## 3. PLANNING 단계 — `nodes/planner.py:plan_search_queries()`

```
입력:
  - state: AgentState (phase=PLANNING)
  - base_query: "OpenAI 최신 동향 리포트를 작성해주세요"

처리:
  1. 기본 쿼리 4개 하드코딩 생성:
     queries = [
         "OpenAI 최신 동향 리포트를 작성해주세요",   ← 원본 쿼리
         "OpenAI GPT model multimodal",              ← 모델 관련
         "OpenAI API developer platform",            ← API 관련
         "OpenAI safety alignment",                  ← 안전성 관련
     ]

  2. [재시도 시만] Critic이 남긴 missing_keywords가 있으면 타깃 쿼리 추가:
     예: missing_keywords = ["Sora", "o1"] 일 때
     queries += ["OpenAI Sora latest", "OpenAI o1 latest"]

  3. state.search_queries_used = queries   ← 상태에 저장

출력: ["OpenAI 최신 동향...", "OpenAI GPT model...", ...] (4~6개 쿼리 리스트)

상태 전이: phase → SEARCHING
```

---

## 4. SEARCHING 단계 — `nodes/searcher.py:execute_search()`

```
입력:
  - rag: LocalRAG (LanceDB에 10건 인덱싱된 상태)
  - queries: ["OpenAI 최신 동향...", "OpenAI GPT model...", ...] (4개)
  - state: AgentState
  - top_k: 2 (쿼리당 최대 2건 반환)

처리:
  seen = set()         ← 중복 방지용

  for sq in queries:   ← 4개 쿼리 순회
      hits = rag.search(sq, top_k=2)
      ├── text_to_vector(sq) → 60차원 쿼리 벡터
      ├── LanceDB.search(query_vec).limit(2) → L2 거리 기반 유사 문서 2건
      └── 반환: [{"text", "topic", "source", "distance"}, ...]

      for h in hits:
          key = h["text"][:80]      ← 텍스트 앞 80자를 중복 키로 사용
          if key not in seen:
              seen.add(key)
              all_results.append(h)  ← 중복 아닌 것만 추가

  state.search_results = all_results

출력 예시 (1차 시도):
  쿼리 4개 × top_k 2 = 최대 8건 → 중복 제거 후 약 5건
  [
    {"text": "GPT-4o (omni) was...",        "topic": "GPT-4o",              "distance": 0.12},
    {"text": "The OpenAI API platform...",  "topic": "API Platform",        "distance": 0.15},
    {"text": "OpenAI invested heavily...",  "topic": "Safety & Alignment",  "distance": 0.18},
    {"text": "ChatGPT Enterprise and...",   "topic": "ChatGPT Enterprise",  "distance": 0.21},
    {"text": "OpenAI's multimodal...",      "topic": "Multimodal",          "distance": 0.23},
  ]

상태 전이:
  - 결과 있음 → phase → DRAFTING
  - 결과 없음 → phase → FAILED
```

### LanceDB 검색 내부 동작 (`vectordb.py:search()`)

```
1. text_to_vector(query) → 쿼리를 60차원 BoW 벡터로 변환
2. _table.search(query_vec) → LanceDB가 내부적으로 L2 거리 계산
   - 저장된 10개 문서 벡터와 쿼리 벡터 간 거리 산출
   - 임베딩이 L2 정규화되어 있으므로 L2 거리 ≈ 코사인 유사도 역순
3. .limit(top_k) → 거리 가장 가까운 top_k개 반환
4. distance 값을 소수점 4자리로 반올림하여 반환
```

---

## 5. DRAFTING 단계 — `nodes/worker.py:worker_generate_draft()`

### 5-1. Mock 모드 (기본) — `_mock_generate(state)`

```
입력: state.search_results (5건의 검색 결과)

처리:
  1. 토픽별 그룹핑:
     topics = {
         "GPT-4o":             [결과1],
         "API Platform":       [결과2],
         "Safety & Alignment": [결과3],
         "ChatGPT Enterprise": [결과4],
         "Multimodal":         [결과5],
     }

  2. 마크다운 리포트 조립:
     parts = []
     parts += "# OpenAI Latest Trends Report"
     parts += "_Generated: 2026-02-09 15:30_"
     parts += "## Executive Summary"
     parts += "This report covers 5 key areas..."

     for topic, results in topics:
         parts += "## {topic}"          ← 토픽별 섹션 헤더
         parts += result["text"]        ← 검색 결과 원문 그대로
         parts += "_Source: {source}_"  ← 출처 인용

     parts += "## Conclusion"
     parts += "OpenAI continues advancing..."

  3. "\n\n".join(parts) → 하나의 문자열로 합치기

출력 예시:
  "# OpenAI Latest Trends Report
   _Generated: 2026-02-09 15:30_

   ## Executive Summary
   This report covers 5 key areas...

   ## GPT-4o
   GPT-4o (omni) was released in May 2024...
   _Source: OpenAI Blog 2024_

   ## API Platform
   The OpenAI API platform introduced...
   _Source: OpenAI Developer Docs 2024_

   ... (중략) ...

   ## Conclusion
   OpenAI continues advancing AI capabilities..."
```

### 5-2. Pydantic AI 모드 (`--model` 지정 시)

```
입력: state.search_results + state.critique (재시도 시 피드백 포함)

처리:
  1. _build_worker_prompt(state) → 검색 결과를 포함한 프롬프트 생성
     "Write a comprehensive 'OpenAI Latest Trends Report' using this knowledge:
      - [GPT-4o] GPT-4o (omni) was released...
      - [API Platform] The OpenAI API platform...
      [재시도 시] PREVIOUS ATTEMPT FEEDBACK: Missing topics: Sora, o1
      Requirements: ## headers, Source citations, Cover ALL topics"

  2. pydantic_agent.run_sync(prompt, deps=rag) → 실제 LLM 호출
     - LLM이 search_knowledge_base 도구를 자율적으로 사용 가능
     - 도구 내부: rag.search(query, top_k=5) → LanceDB 추가 검색

  3. result.data → LLM이 생성한 리포트 문자열 반환

state.draft = 생성된 리포트 문자열
상태 전이: phase → CRITIQUING
```

---

## 6. CRITIQUING 단계 — `nodes/critic.py:critic_evaluate()`

```
입력: state.draft (리포트 초안 문자열)

처리: 4가지 기준으로 점수 산출 (LLM 호출 없음, 순수 문자열 분석)
```

### Check 1: 키워드 커버리지 (가중치 40%)

```
필수 키워드 8개:
  ["GPT-4o", "Sora", "API", "safety", "multimodal", "o1", "enterprise", "partnership"]

draft.lower()에서 각 키워드 존재 여부 확인:
  "gpt-4o"      → ✅ found
  "sora"        → ❌ missing   ← 1차 시도에서 검색 안 됨
  "api"         → ✅ found
  "safety"      → ✅ found
  "multimodal"  → ✅ found
  "o1"          → ❌ missing   ← 1차 시도에서 검색 안 됨
  "enterprise"  → ✅ found
  "partnership" → ❌ missing   ← 1차 시도에서 검색 안 됨

kw_ratio = 5/8 = 0.625
score_1 = 0.625 × 0.4 = 0.25

missing_keywords = ["Sora", "o1", "partnership"]
```

### Check 2: 콘텐츠 길이 (가중치 20%)

```
MIN_LENGTH = 500

len(draft) = 1200 (예시)
length_score = min(1200 / 500, 1.0) = 1.0
score_2 = 1.0 × 0.2 = 0.20
```

### Check 3: 문서 구조 (가중치 20%)

```
필요 섹션: 최소 4개

draft.count("## ") = 7  (Executive Summary, GPT-4o, API Platform, Safety, ...)
struct_score = min(7 / 4, 1.0) = 1.0
score_3 = 1.0 × 0.2 = 0.20
```

### Check 4: 출처 인용 (가중치 20%)

```
필요 인용: 최소 3개

draft.lower().count("source:") = 5
cite_score = min(5 / 3, 1.0) = 1.0
score_4 = 1.0 × 0.2 = 0.20
```

### 최종 판정

```
total = 0.25 + 0.20 + 0.20 + 0.20 = 0.85
threshold = 0.7 → total >= threshold ✅

BUT keyword_gate:
  kw_ratio (0.625) >= MIN_KEYWORD_RATIO (0.85) → ❌ HARD GATE FAILED

passed = (total >= 0.7) AND (keyword_gate) = True AND False = False
```

```
출력: CriticVerdict {
    passed: False,
    score: 0.85,
    feedback: "Score 0.85/0.7 — RETRY NEEDED\n
               Keywords: 5/8 (need 85%+)\n
               HARD GATE FAILED: keyword coverage 62% < 85%\n
               Missing: Sora, o1, partnership",
    missing_keywords: ["Sora", "o1", "partnership"],
    suggestions: ["Search for and include content about: Sora, o1, partnership"]
}
```

---

## 7. Self-Healing 분기 — `orchestrator.py` (L120-136)

```python
if verdict.passed:
    → phase = DONE            # 통과 → 종료

elif state.can_retry:         # retry_count(0) < max_retries(3) → True
    state.retry_count += 1    # 0 → 1
    state.phase = PLANNING    # ← 다시 PLANNING으로 되돌아감

else:
    → phase = FAILED          # 재시도 한도 초과 → 실패
```

**핵심: `state.critique.missing_keywords = ["Sora", "o1", "partnership"]`가 보존된 채로 다시 PLANNING에 진입한다.**

---

## 8. 2차 시도 — Self-Healing Loop

### PLANNING (재시도)

```
plan_search_queries() 내부:

  queries = [기본 4개]  (동일)

  state.critique.missing_keywords = ["Sora", "o1", "partnership"]  ← 1차 Critic 결과
  for kw in missing_keywords:
      queries.append(f"OpenAI {kw} latest")

  최종 queries = [
      "OpenAI 최신 동향 리포트를 작성해주세요",
      "OpenAI GPT model multimodal",
      "OpenAI API developer platform",
      "OpenAI safety alignment",
      "OpenAI Sora latest",           ← NEW: 누락 키워드 타깃 쿼리
      "OpenAI o1 latest",             ← NEW
      "OpenAI partnership latest",    ← NEW
  ]
```

### SEARCHING (재시도)

```
7개 쿼리 × top_k=2 = 최대 14건 → 중복 제거 후 약 8건
  기존 5건 + "Sora" 문서 + "o1 Reasoning Model" 문서 + "Partnerships" 문서
```

### DRAFTING (재시도)

```
8건 검색 결과 → 토픽 8개 섹션 포함 리포트 생성
  이전에 없던 Sora, o1, Partnerships 섹션이 추가됨
```

### CRITIQUING (재시도)

```
키워드 체크:
  "gpt-4o" ✅  "sora" ✅  "api" ✅  "safety" ✅
  "multimodal" ✅  "o1" ✅  "enterprise" ✅  "partnership" ✅

  kw_ratio = 8/8 = 1.0
  keyword_gate: 1.0 >= 0.85 → ✅

total = (1.0 × 0.4) + (1.0 × 0.2) + (1.0 × 0.2) + (1.0 × 0.2) = 1.0
passed = (1.0 >= 0.7) AND (keyword_gate) = True

→ CriticVerdict { passed: True, score: 1.0 }
```

---

## 9. 완료 — DONE

```python
state.final_result = state.draft   # 최종 승인된 리포트 저장
state.phase = Phase.DONE           # 루프 탈출 조건 충족
```

`while` 루프 종료 후:
```python
return state.final_result or "Agent failed to produce a result."
```

`__main__.py`에서 이 문자열을 `print()` → 사용자에게 최종 리포트 출력.

---

## 전체 실행 타임라인 (1차 실패 → 2차 성공)

```
시간  함수                              phase 전이           핵심 데이터 변화
─────┬───────────────────────────────────┬────────────────────┬──────────────────────
 T0  │ run_agent()                       │                    │ mode=mock
 T1  │ LocalRAG.seed(10건)               │                    │ LanceDB 테이블 생성, 10벡터
 T2  │ AgentState(query=...)             │ → PLANNING         │ 빈 상태 초기화
─────┼── 1차 시도 ────────────────────────┼────────────────────┼──────────────────────
 T3  │ plan_search_queries()             │ PLANNING→SEARCHING │ queries=[4개]
 T4  │ execute_search()                  │ SEARCHING→DRAFTING │ search_results=[5건]
 T5  │ _mock_generate()                  │ DRAFTING→CRITIQUING│ draft="# OpenAI..."
 T6  │ critic_evaluate()                 │ CRITIQUING→PLANNING│ score=0.85, passed=F
     │                                   │                    │ missing=[Sora,o1,partnership]
     │                                   │                    │ retry_count: 0→1
─────┼── 2차 시도 (Self-Healing) ─────────┼────────────────────┼──────────────────────
 T7  │ plan_search_queries()             │ PLANNING→SEARCHING │ queries=[4+3=7개]
 T8  │ execute_search()                  │ SEARCHING→DRAFTING │ search_results=[8건]
 T9  │ _mock_generate()                  │ DRAFTING→CRITIQUING│ draft에 Sora,o1 포함
 T10 │ critic_evaluate()                 │ CRITIQUING→DONE    │ score=1.00, passed=T
─────┼─────────────────────────────────────┼────────────────────┼──────────────────────
 T11 │ return state.final_result         │ DONE               │ 최종 리포트 반환
```

---

## 함수 호출 그래프

```
__main__.main()
  └─ orchestrator.run_agent(query, model_name, db_path)
       ├─ agent.create_pydantic_agent(model_name)     # model 있을 때만
       ├─ LocalRAG(db_path)
       │    └─ lancedb.connect()
       ├─ LocalRAG.seed(KNOWLEDGE_BASE)
       │    ├─ text_to_vector(doc["text"]) × 10       # infra/embedding.py
       │    │    ├─ text.lower()
       │    │    ├─ _VOCAB 60단어 카운트 → numpy 벡터
       │    │    └─ L2 정규화
       │    ├─ drop_table() (기존 있으면)
       │    └─ create_table(records)
       ├─ AgentState(query=query)
       │
       └─ while loop:
            ├─ [PLANNING]  planner.plan_search_queries(state, query)
            │    ├─ 기본 쿼리 4개 생성
            │    └─ (재시도) critique.missing_keywords → 추가 쿼리
            │
            ├─ [SEARCHING] searcher.execute_search(rag, queries, state)
            │    ├─ for query in queries:
            │    │    └─ rag.search(query, top_k=2)
            │    │         ├─ text_to_vector(query)
            │    │         └─ LanceDB.search().limit(2)
            │    └─ 중복 제거 (text[:80] 기준)
            │
            ├─ [DRAFTING]  worker.worker_generate_draft(state, mode, ...)
            │    ├─ mock: _mock_generate(state)
            │    │    ├─ 토픽별 그룹핑
            │    │    ├─ 마크다운 섹션 조립
            │    │    └─ "\n\n".join()
            │    └─ pydantic-ai: agent.run_sync(prompt, deps=rag)
            │         └─ LLM이 search_knowledge_base 도구 자율 호출
            │
            └─ [CRITIQUING] critic.critic_evaluate(draft)
                 ├─ 키워드 커버리지  (×0.4) — 8개 필수 키워드
                 ├─ 콘텐츠 길이     (×0.2) — 500자 기준
                 ├─ 문서 구조       (×0.2) — ## 헤더 4개 기준
                 ├─ 출처 인용       (×0.2) — "source:" 3개 기준
                 ├─ Hard Gate: kw_ratio < 85% → 무조건 RETRY
                 └─ 반환: CriticVerdict { passed, score, missing_keywords }
                      ├─ passed=True  → DONE (final_result 저장)
                      ├─ passed=False + can_retry → PLANNING (Self-Healing)
                      └─ passed=False + !can_retry → FAILED (best-effort 반환)
```
