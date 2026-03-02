# Self-Correction Agent 학습 가이드

> 실행 확인 완료 (2026-03-02) — EXAONE + Docker 모드 정상 동작
> v3: sentence-transformers 384차원 + BM25 하이브리드 검색 + 문서 청킹 + 검색 설정 UI
> LLM: Ollama EXAONE 3.5 7.8B (호스트 macOS 직접 실행) ← Docker 컨테이너가 `host.docker.internal:11434/v1`로 접근
> EXAONE은 tool calling 미지원 → `_LOCAL_NO_TOOL_MODELS`에 등록, 텍스트 파싱 경로로 동작

---

## 1. 실행 흐름 이해하기

실행하면 아래 순서로 로그가 출력됩니다.

```
[PLAN]   → 쿼리 변형 몇 개 생성했는지 확인 (LLM 모드: eval_criteria도 생성)
[SEARCH] → 각 쿼리가 몇 개 문서를 찾았는지 확인
[DRAFT]  → 리포트 몇 글자, 몇 섹션으로 작성됐는지 확인
[CRITIC] → Score가 얼마인지, 통과/실패 여부, 누락 키워드 확인
           → Failed이면 자동으로 [PLAN]부터 다시 시작 (Self-Healing!)
[DONE]   → 최종 승인
```

> 실행 명령어는 `DOCS.md` 6절, Ollama/EXAONE 설정은 `DOCS.md` 6절 참조.

---

## 2. 실험 목록

### 실험 1: Planner — Mock vs LLM 모드 비교

**Mock 모드 (heuristic)**
```bash
python3 -m self_correction_agent --query "Google Gemini 최신 동향"
python3 -m self_correction_agent --query "너는 어떤 모델이야?"
```

**LLM 모드 (Docker — EXAONE 기본값)**
```bash
docker compose up  # AGENT_MODEL=ollama:exaone3.5:7.8b 기본값
```

**LLM 모드 (CLI 직접 실행)**
```bash
python3 -m self_correction_agent --query "Google Gemini 최신 동향" --model ollama:exaone3.5:7.8b
```

**확인할 것:**
- Mock 모드: `nodes/planner.py`의 `_TOPIC_ALIASES`로 보조 쿼리 선택. "Gemini" 키 없으면 범용 AI 쿼리 3개 사용
- EXAONE(no-tool): `expand_query_with_llm()`이 줄바꿈 텍스트 출력 → `_parse_lines()`로 파싱 (result_type 미사용)
- GPT 등(tool-capable): `result_type=ExpandedQueries`로 구조화 JSON 출력
- LLM 응답 실패 시 heuristic fallback이 자동으로 작동하는가?

---

### 실험 2: Hybrid Critic 직접 뜯어보기

```python
# Python REPL에서 직접 호출
from self_correction_agent.nodes.critic import critic_evaluate

# Mock 모드 (eval_criteria 없음 → REQUIRED_KEYWORDS fallback)
draft = "## GPT-4o\ntest\n## Sora\ntest Source: blog"
result = critic_evaluate(draft)
print(result.score, result.passed, result.missing_keywords)

# LLM 모드 시뮬레이션 (eval_criteria 직접 주입)
custom_criteria = ["Gemini", "Google AI", "multimodal", "Bard"]
result2 = critic_evaluate(draft, eval_criteria=custom_criteria)
print(result2.score, result2.passed, result2.missing_keywords)
```

**확인할 것:**
- `eval_criteria` 주입 시 REQUIRED_KEYWORDS 대신 해당 기준으로 채점되는가?
- 키워드 커버리지 85% 미만이면 총점 무관하게 RETRY — Hard Gate 위치: `nodes/critic.py` 93번째 줄 근처
- REQUIRED_KEYWORDS의 튜플 항목은 OR 조건 — `("멀티모달", "multimodal")` 중 하나만 있어도 통과?

---

### 실험 3: Self-Healing 한계 테스트 (FAILED 상태 만들기)

`domain/constants.py`에서 REQUIRED_KEYWORDS에 존재하지 않는 키워드 추가:

```python
REQUIRED_KEYWORDS = [..., "NonExistentTopic123"]
```

**확인할 것:**
- 최대 3회 재시도 후 어떤 메시지가 나오는가?
- FAILED 상태에서도 마지막 초안을 반환하는가? (Best-Effort 원칙)

---

### 실험 4: 웹 UI에서 지식베이스 CRUD

브라우저 `http://localhost:8000` → 좌측 사이드바 **지식베이스** 클릭

**문서 목록 탭**
1. **검색**: 검색창에 "Sora" 입력 → 실시간 필터링
2. **삭제**: 행의 `삭제` 버튼 → 즉시 제거 + 목록 새로고침

**추가 / 관리 탭**
3. **문서 직접 추가**: Topic / Source / 내용 입력 → `추가` → 문서 목록 탭으로 이동
4. **파일 업로드**: `.txt` 파일 선택 → 자동 청킹+임베딩 저장 → 목록 탭으로 이동
5. **초기화**: `초기화 실행` → 모든 문서 삭제 후 기본 10건 복원
6. **검색 전략 설정**: alpha 슬라이더, 거리 임계값, top-k 조정 → `설정 저장`

**연동 확인**
7. 사이드바 **채팅** 탭 → 쿼리 입력 → 새로 추가한 문서가 리포트에 반영되는지 확인

---

### 실험 5: 하이브리드 검색 직접 확인

**웹 UI 검색 테스트**

`http://localhost:8000` → 지식베이스 → 문서 목록 탭 상단 검색창

```
검색어: "Sora video"
Alpha:  0.7 (기본)
```

- 결과 테이블의 `벡터`, `BM25`, `종합` 점수 확인
- Alpha=0.0 (순수 BM25): "GPT-4o" 검색 → GPT-4o 문서 최상위?
- Alpha=1.0 (순수 벡터): "AI 이미지 생성" 검색 → DALL-E 문서 상위?

**Python REPL에서 직접 호출**

```python
from self_correction_agent.infra.vectordb import LocalRAG

rag = LocalRAG()
rag.initialize()

# 하이브리드 (alpha=0.7)
results = rag.hybrid_search("OpenAI 이미지 생성", top_k=3)
for r in results:
    print(r["topic"], r["vector_score"], r["bm25_score"], r["combined_score"])

# 순수 BM25 (키워드 정밀)
results_bm25 = rag.hybrid_search("GPT-4o API", top_k=3, alpha=0.0)

# 순수 벡터 (의미 기반)
results_vec = rag.hybrid_search("AI 이미지 생성", top_k=3, alpha=1.0)
```

**확인할 것:**
- alpha에 따라 순위가 어떻게 달라지는가?
- 한국어 쿼리로 영어 문서가 검색되는가? (다국어 ST 임베딩 효과)
- `combined_score`가 `alpha * vector_score + (1-alpha) * bm25_score`와 일치하는가?

---

### 실험 6: 임베딩 직접 확인

```python
from self_correction_agent.infra.embedding import text_to_vector
import numpy as np

v1 = text_to_vector("OpenAI GPT-4o multimodal")
v2 = text_to_vector("Sora video generation")
v3 = text_to_vector("banana smoothie recipe")

cosine = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"AI 도메인 유사도: {cosine(v1, v2):.4f}")   # 높아야 정상
print(f"다른 도메인 유사도: {cosine(v1, v3):.4f}") # 낮아야 정상

# 벡터 차원 확인
print(f"차원: {len(v1)}")  # 384
```

**확인할 것:**
- 384차원 ST 임베딩은 어휘 겹침 없이도 의미 유사도를 계산함 (BoW 대비 개선)
- 한국어 "AI 이미지 생성" vs 영어 "AI image generation" — 유사도가 높은가?
- `normalize_embeddings=True`로 L2 정규화 → 코사인 유사도 = 내적

---

### 실험 7: 청킹 확인

```python
from self_correction_agent.infra.chunker import chunk_text

# 긴 문서 청킹
long_text = "A" * 1200
chunks = chunk_text(long_text)
print(f"청크 수: {len(chunks)}")   # 3개 예상 (500+50 오버랩)
print([len(c) for c in chunks])

# 짧은 문서 — 1개 반환
short = chunk_text("짧은 문서")
print(len(short))  # 1

# 웹 UI에서 긴 .txt 파일 업로드 후
# 문서 목록에 원본 1건만 표시되는지 확인 (청크 여러 개가 parent_id로 묶임)
```

**확인할 것:**
- 청크 간 오버랩이 50자인지 확인 (청크 끝 50자 = 다음 청크 시작 50자)
- `list_documents()`가 `chunk_index==0`만 반환하는지: `infra/vectordb.py`의 `list_documents()` 확인

---

### 실험 8: LanceDB 직접 조회

```python
import lancedb

db = lancedb.connect("/tmp/lancedb_self_correction_agent")
print(db.table_names())

table = db.open_table("knowledge_base")
rows = table.to_arrow().to_pylist()
for r in rows[:3]:
    print(r["topic"], r["parent_id"], r["chunk_index"], len(r["vector"]))
```

**확인할 것:**
- `parent_id`, `chunk_index` 컬럼이 있는가? (v3 스키마)
- `vector` 길이가 384인가? (48이면 자동 마이그레이션 안 된 것)
- 같은 문서에서 나온 청크들이 같은 `parent_id`를 공유하는가?
- `initialize()`는 기존 테이블을 DROP하지 않음 — 웹 UI에서 추가한 문서가 재시작 후에도 유지되는가?

---

## 3. 코드 읽기 순서 (추천)

```
1. domain/state.py          → Phase, AgentState 구조 (전체 상태 모델)
2. domain/constants.py      → REQUIRED_KEYWORDS, 임계값
3. orchestrator.py          → 메인 while 루프 — 상태 머신 흐름
4. nodes/llm_helpers.py     → LLM 쿼리 확장 + eval_criteria 생성 (ExpandedQueries, EvalCriteria)
5. nodes/planner.py         → LLM 확장 + heuristic fallback 로직
6. infra/chunker.py         → 문자 기반 슬라이딩 윈도우 청킹 (500자/50자)
7. infra/settings.py        → alpha/threshold/top_k 영속화 (settings.json)
8. infra/embedding.py       → sentence-transformers 384차원 임베딩
9. infra/vectordb.py        → LanceDB + BM25 하이브리드 검색 + CRUD
10. nodes/searcher.py       → hybrid_search 실행 (alpha/threshold 파라미터)
11. nodes/worker.py         → Mock 리포트 생성 + LLM 프롬프트 구성
12. nodes/critic.py         → 점수 계산 공식 (eval_criteria + OR 그룹 키워드 체크)
13. agent.py                → Pydantic AI 연동 (Ollama tools 모드)
```

---

## 4. 핵심 개념 체크리스트

### 아키텍처
- [ ] Planner / Worker / Critic 각각의 역할이 뭔가?
- [ ] AgentState가 왜 "Single Source of Truth"인가?
- [ ] 왜 domain 레이어에서 numpy/lancedb를 직접 import하면 안 되는가?
- [ ] 이 프로젝트의 상태 머신은 LangGraph와 구체적으로 어떻게 다른가? (`orchestrator.py` while 루프 확인)
- [ ] pydantic, pydantic-ai, orchestrator 각각이 담당하는 역할의 경계는?

### Self-Healing
- [ ] Critic의 피드백이 어떻게 다음 Planner에 전달되는가?
- [ ] 최대 재시도 횟수는 어디서 제어하는가?
- [ ] Hard Gate(키워드 85% 미만 무조건 RETRY)를 왜 별도로 두는가?

### RAG
- [ ] 왜 전체 지식베이스를 그대로 넣지 않고 벡터 검색으로 관련 문서만 꺼내는가?
- [ ] sentence-transformers가 BoW(Bag-of-Words) 대비 어떤 점이 개선되었는가?
- [ ] 하이브리드 검색에서 alpha=0.0과 alpha=1.0은 각각 어떤 상황에 유리한가?
- [ ] 문서 청킹이 없으면 긴 문서에서 어떤 문제가 생기는가? `parent_id`의 역할은?
- [ ] `_rebuild_bm25()`가 `add_document()`, `delete_document()`, `seed()` 후 항상 호출되는 이유는?

### Hybrid Critic
- [ ] LLM이 채점 기준을 생성하고 규칙 엔진이 채점하는 방식의 장점은?
- [ ] eval_criteria가 None일 때 REQUIRED_KEYWORDS로 fallback되는 코드는 어디인가?
- [ ] 소형 모델을 Full LLM Judge로 쓰면 왜 신뢰도가 낮은가?

### 확장성 & 배포
- [ ] Mock → Ollama EXAONE → OpenAI API로 바꾸려면 무엇만 바꾸면 되는가?
- [ ] `initialize()` vs `seed()` 차이는? 언제 각각 사용하는가?
- [ ] `result_type=ExpandedQueries`가 tools 미지원 모델에서 실패하는 이유는?
- [ ] `_LOCAL_NO_TOOL_MODELS`에 새 모델을 추가하면 어떤 경로로 실행되는가?
- [ ] settings.json을 삭제하면 어떤 기본값이 사용되는가? (`infra/settings.py` 확인)
- [ ] pydantic-ai 1.x에서 `ollama:model` prefix는 실제로 어떤 클라이언트를 사용하는가?
- [ ] `OLLAMA_BASE_URL`에 `/v1`이 없으면 왜 404가 발생하는가?
- [ ] Docker 컨테이너에서 호스트 프로세스(Ollama)에 어떻게 접근하는가? (`host.docker.internal`)

---

## 5. 심화 과제

1. **BM25 vs 벡터 검색 성능 비교** — alpha=0.0, 0.5, 1.0으로 동일 쿼리 검색 후 결과 순위 비교
2. **청킹 크기 실험** — `infra/chunker.py`의 `chunk_size`를 200자, 1000자로 바꾸면 검색 정확도가 어떻게 달라지는가?
3. **새 Tool 추가** — `agent.py`에 웹 검색 Tool 추가 (tools 지원 모델 전용)
4. **llm_helpers 프롬프트 튜닝** — `expand_query_with_llm()` 프롬프트 수정해서 쿼리 다양성 개선
5. **eval_criteria 품질 측정** — LLM이 생성한 기준과 사람이 작성한 기준 비교 실험
6. **다국어 지식베이스** — 한국어 문서 추가 후 한/영 혼합 검색 결과 확인 (ST 다국어 임베딩 효과)
7. **LangGraph로 마이그레이션** — `orchestrator.py`의 while 루프를 `StateGraph`로 교체. 병렬 검색 fan-out, 체크포인팅 추가
