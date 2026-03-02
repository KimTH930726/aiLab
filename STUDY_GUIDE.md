# Self-Correction Agent 학습 가이드

> 실행 확인 완료 (2026-03-02) — Mock 모드 정상 동작, 2회 시도 만에 Score 1.00 달성
> v2 업데이트: LLM 기반 쿼리 확장(no-tool 지원) + Hybrid Critic + 웹 UI DB CRUD + 좌측 사이드바 UI

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

> 실행 명령어는 `DOCS.md` 6절, Docker/BitNet 설정은 `BITNET_GUIDE.md` 참조.

---

## 2. 실험 목록

### 실험 1: Planner — Mock vs LLM 모드 비교

**Mock 모드 (heuristic)**
```bash
python3 -m self_correction_agent --query "Google Gemini 최신 동향"
python3 -m self_correction_agent --query "너는 어떤 모델이야?"
```

**LLM 모드 (동적 확장)**
```bash
python3 -m self_correction_agent --query "Google Gemini 최신 동향" --model openai:bitnet
```

**확인할 것:**
- Mock 모드: `nodes/planner.py`의 `_TOPIC_ALIASES`로 보조 쿼리 선택. "Gemini" 키 없으면 범용 AI 쿼리 3개 사용
- LLM 모드: `expand_query_with_llm()`이 쿼리 변형 3~4개 생성, `generate_eval_criteria()`가 `eval_criteria` 생성
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
4. **파일 업로드**: `.txt` 파일 선택 → 자동 임베딩 저장 → 목록 탭으로 이동
5. **초기화**: `초기화 실행` → 모든 문서 삭제 후 기본 10건 복원

**연동 확인**
6. 사이드바 **채팅** 탭 → 쿼리 입력 → 새로 추가한 문서가 리포트에 반영되는지 확인

> 코드로 직접 테스트하려면 `vectordb.py`의 `add_document()`, `delete_document()` 호출

---

### 실험 5: 임베딩 직접 확인

```python
from self_correction_agent.infra.embedding import text_to_vector
import numpy as np

v1 = text_to_vector("OpenAI GPT-4o multimodal")
v2 = text_to_vector("Sora video generation")
v3 = text_to_vector("banana smoothie recipe")

cosine = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(cosine(v1, v2))  # 높아야 정상 (같은 AI 도메인)
print(cosine(v1, v3))  # 낮아야 정상 (다른 도메인)
```

**확인할 것:**
- Bag-of-Words 방식이라 어휘가 겹쳐야 유사도가 높아짐
- 대소문자 처리는 어떻게 되는가? (`text.lower()` 확인)

---

### 실험 6: LanceDB 직접 조회

```python
import lancedb

db = lancedb.connect("/tmp/lancedb_self_correction_agent")
print(db.table_names())

table = db.open_table("knowledge_base")
rows = table.to_arrow().to_pylist()  # pandas 없이 조회
for r in rows:
    print(r["topic"], r["source"])
```

**확인할 것:**
- 어떤 컬럼으로 저장되는가? (id, text, topic, source, vector)
- v2에서 `initialize()`가 기존 테이블을 DROP하지 않음 — 웹 UI에서 추가한 문서가 재실행 후에도 유지되는가?
- 반면 `seed()`(또는 웹 UI `초기화` 버튼)는 DROP+재생성

---

## 3. 코드 읽기 순서 (추천)

```
1. domain/state.py          → Phase, AgentState 구조 (전체 상태 모델)
2. domain/constants.py      → REQUIRED_KEYWORDS, 임계값
3. orchestrator.py          → 메인 while 루프 — 상태 머신 흐름
4. nodes/llm_helpers.py     → LLM 쿼리 확장 + eval_criteria 생성 (ExpandedQueries, EvalCriteria)
5. nodes/planner.py         → LLM 확장 + heuristic fallback 로직
6. infra/embedding.py       → Bag-of-Words 벡터 변환
7. infra/vectordb.py        → LanceDB initialize/search/CRUD 래퍼
8. nodes/searcher.py        → LanceDB 실제 검색
9. nodes/worker.py          → Mock 리포트 생성 + LLM 프롬프트 구성
10. nodes/critic.py         → 점수 계산 공식 (eval_criteria + OR 그룹 키워드 체크)
11. agent.py                → Pydantic AI 연동 (no-tool 모드 포함)
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
- [ ] Bag-of-Words 임베딩의 한계는? Sentence Transformers와 차이는?
- [ ] top_k=2로 쿼리당 2개만 가져오는데, 왜 중복 제거가 필요한가?

### Hybrid Critic
- [ ] LLM이 채점 기준을 생성하고 규칙 엔진이 채점하는 방식의 장점은?
- [ ] 2B 소형 모델(BitNet)을 Full LLM Judge로 쓰면 왜 신뢰도가 낮은가?
- [ ] eval_criteria가 None일 때 REQUIRED_KEYWORDS로 fallback되는 코드는 어디인가?

### 확장성
- [ ] Mock → Ollama → OpenAI API로 바꾸려면 무엇만 바꾸면 되는가?
- [ ] BitNet이 Function Calling을 지원하지 않는데, 어떻게 해결했는가? (`llm_helpers.py`의 `_is_no_tool()` 확인)
- [ ] `initialize()` vs `seed()` 차이는? 언제 각각 사용하는가?
- [ ] `result_type=ExpandedQueries`가 no-tool 모델에서 실패하는 이유는?

---

## 5. 심화 과제

1. **임베딩을 Sentence Transformers로 교체** — `infra/embedding.py`만 수정. BoW 대비 유사도 품질 향상 확인
2. **새 Tool 추가** — `agent.py`에 웹 검색 Tool 추가 (tools 지원 모델 전용)
3. **llm_helpers 프롬프트 튜닝** — `expand_query_with_llm()` 프롬프트 수정해서 쿼리 다양성 개선
4. **eval_criteria 품질 측정** — LLM이 생성한 기준과 사람이 작성한 기준 비교 실험
5. **다국어 지식베이스** — 한국어 문서 추가 후 웹 UI 지식베이스 탭으로 삽입, 한/영 혼합 Critic 키워드 대응 확인
6. **LangGraph로 마이그레이션** — `orchestrator.py`의 while 루프를 `StateGraph`로 교체. 병렬 검색 fan-out, 체크포인팅 추가
