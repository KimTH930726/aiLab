# Self-Correction Agent

## Local-First AI : 자가 수정 에이전트

> GPU 클라우드 없이, 로컬 CPU만으로 동작하는 "끈질긴" AI 에이전트

---

## 1. 프로젝트 개요

단순히 한 번 답변을 생성하고 끝나는 것이 아니라,
**Critic(검증자)이 결과물을 평가하고, 부족하면 스스로 재검색-재작성을 반복**하여
목표 품질에 도달할 때까지 자율적으로 개선합니다.

| 특성 | 설명 |
|------|------|
| **Local-First** | 네트워크 없이 로컬 디스크 + CPU만으로 완전 동작 |
| **Self-Healing** | Critic이 부족한 부분을 진단하면 자동으로 재검색/재작성 (최대 3회) |
| **Lightweight** | Bag-of-Words 임베딩 + LanceDB로 외부 모델 서버 불필요 |
| **Swappable** | Mock 모드에서 Ollama/OpenAI/BitNet까지 한 줄 변경으로 전환 가능 |
| **Hybrid Critic** | LLM이 동적 채점 기준 생성 → 규칙 기반 엔진이 채점 (쿼리 적응형) |

---

## 2. 기술 스택

| 패키지 | 역할 |
|--------|------|
| `lancedb` | 로컬 벡터 DB (RAG 검색) |
| `numpy` | 임베딩 벡터 연산 |
| `pydantic` | 상태/결과 데이터 모델 (`AgentState`, `CriticVerdict` 등) |
| `pydantic-ai` | LLM 호출 전담 (`Agent.run_sync`) — 쿼리 확장·eval_criteria·Draft 생성에 사용. Graph API는 미사용 |
| `fastapi` + `sse-starlette` | 웹 UI 서버 (SSE 실시간 스트리밍 + CRUD REST API) |

> **pydantic vs pydantic-ai 역할 구분**
> - `pydantic`: 데이터 구조 정의만 담당 (BaseModel)
> - `pydantic-ai Agent`: LLM 호출 래퍼. 흐름 제어는 하지 않음
> - 흐름 제어(그래프/상태 전이)는 `orchestrator.py`의 **직접 구현 while 루프**가 담당

---

## 3. 아키텍처

### 상태 머신 흐름

```
  ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐
  │ PLANNING │────>│ SEARCHING│────>│ DRAFTING │────>│ CRITIQUING│
  └─────────┘     └──────────┘     └──────────┘     └───────────┘
       ^                                                   │
       │◄──────── (Self-Healing Retry: 최대 3회) ◄─────────┤
                                                           │
                                                    ┌──────┴──────┐
                                                 ┌──┴──┐       ┌──┴────┐
                                                 │ DONE│       │FAILED │
                                                 └─────┘       └───────┘
```

> **구현 방식**: LangGraph나 pydantic-ai Graph API를 사용하지 않는 **직접 구현 상태 머신**.
> `orchestrator.py`의 `while` 루프 + `if/elif` 분기가 전이 로직 전체를 담당하며,
> `state.phase = Phase.X` 한 줄로 상태를 전이한다.
>
> | | 이 프로젝트 | LangGraph |
> |--|------------|-----------|
> | 노드 | 일반 함수 | `@node` 등록 |
> | 전이 | `state.phase = X` 직접 지정 | `add_edge()` / `add_conditional_edges()` |
> | 실행 | `while` 루프 | 프레임워크 런타임 |
> | 병렬 실행 | 없음 | fan-out 지원 |

| Phase | 역할 |
|-------|------|
| **PLANNING** | LLM으로 쿼리 변형 3~4개 생성 + 동적 eval_criteria 생성. LLM 없으면 heuristic fallback. 재시도 시 Critic 누락 키워드 기반 추가 쿼리만 보충 |
| **SEARCHING** | LanceDB 벡터 검색 (쿼리당 top_k=2, 중복 제거) |
| **DRAFTING** | Mock 템플릿 또는 Pydantic AI Agent가 검색 결과 기반 리포트 작성 |
| **CRITIQUING** | eval_criteria(LLM 동적 기준) 또는 REQUIRED_KEYWORDS(fallback)로 키워드 커버리지(40%) + 길이(20%) + 구조(20%) + 출처(20%) 점수화 |

**Hybrid Critic**: LLM이 쿼리에 맞는 `expected_topics` 생성 → 규칙 기반 엔진이 그 기준으로 채점.
2B 소형 모델(BitNet)은 Full LLM Judge로 사용 시 신뢰도 55~70% → 하이브리드가 최적.

**Critic Hard Gate**: 키워드 커버리지 85% 미만이면 총점과 무관하게 무조건 RETRY

### Self-Healing 흐름 예시

```
[1차 시도] CRITIC: "Sora", "o1" 누락 → score 0.85 → RETRY
[2차 시도] PLANNER: "Sora latest", "o1 latest" 쿼리 추가
           CRITIC: 8/8 키워드 충족, score 1.00 → APPROVED
```

---

## 4. 파일 구조

```
aiLab/
├── web_server.py                   # FastAPI 웹 서버 (SSE + CRUD REST API)
├── static/index.html               # 단일 파일 웹 UI (좌측 사이드바 — 채팅 / 지식베이스)
├── Dockerfile / docker-compose.yml
├── requirements.txt
│
└── self_correction_agent/
    ├── orchestrator.py             # run_agent() 메인 루프 (상태 머신)
    ├── agent.py                    # Pydantic AI Agent 팩토리
    ├── domain/
    │   ├── state.py                # Phase, AgentState, CriticVerdict
    │   └── constants.py            # REQUIRED_KEYWORDS, 임계값
    ├── infra/
    │   ├── embedding.py            # Bag-of-Words 벡터 변환
    │   └── vectordb.py             # LocalRAG (LanceDB 래퍼 + CRUD)
    ├── nodes/
    │   ├── llm_helpers.py          # LLM 쿼리 확장 + eval_criteria 생성
    │   ├── planner.py              # 동적 검색 쿼리 생성 (LLM + heuristic)
    │   ├── searcher.py             # LanceDB 검색 실행
    │   ├── worker.py               # 리포트 초안 생성
    │   └── critic.py               # 품질 평가 엔진 (Hybrid: 동적 기준 + 규칙)
    └── knowledge/
        └── openai_trends.py        # 시드 데이터 10건
```

**계층별 의존 방향**: `orchestrator` → `nodes` → `domain` ← `infra` ← `knowledge`
domain 레이어는 numpy/lancedb를 직접 import하지 않습니다.

---

## 5. 웹 API 엔드포인트

| 메서드 | 경로 | 기능 |
|--------|------|------|
| GET | `/` | 웹 UI (index.html) |
| POST | `/run` | 에이전트 실행 시작 → `{session_id}` 반환 |
| GET | `/stream/{session_id}` | SSE 실시간 이벤트 스트림 |
| GET | `/db` | DB 상태 조회 (count) |
| GET | `/db/documents` | 문서 목록 (페이지네이션 + 텍스트 검색) |
| POST | `/db/documents` | 문서 추가 → `{id, topic, source}` 반환 |
| DELETE | `/db/documents/{id}` | 문서 삭제 |
| POST | `/db/upload` | `.txt` 파일 업로드 → 자동 임베딩 저장 |
| POST | `/db/seed` | 지식베이스 초기 데이터로 초기화 (전체 재생성) |

---

## 6. 실행 방법

### CLI (Mock 모드 — LLM 불필요)

```bash
pip install -r requirements.txt
python3 -m self_correction_agent
python3 -m self_correction_agent --query "원하는 질문" --model openai:gpt-4o-mini
```

### Docker + 웹 UI (권장)

```bash
# Mock 모드
docker compose up

# BitNet 로컬 LLM 모드 (BITNET_GUIDE.md 참조)
AGENT_MODEL=openai:bitnet docker compose up
```

브라우저: `http://localhost:8000`

---

## 7. 확장 포인트

| 목표 | 수정 파일 |
|------|-----------|
| 임베딩을 Sentence Transformers로 교체 | `infra/embedding.py` 전체 교체 |
| LLM 쿼리 확장 프롬프트 튜닝 | `nodes/llm_helpers.py` |
| 지식베이스 교체 | `knowledge/openai_trends.py` 또는 웹 UI 지식베이스 탭에서 직접 추가 |
| 새 LLM Tool 추가 | `agent.py`에 `@agent.tool` 데코레이터로 추가 |
| eval_criteria 기준 조정 | `nodes/llm_helpers.py`의 `generate_eval_criteria()` 프롬프트 수정 |
| 상태 머신 → LangGraph 마이그레이션 | `orchestrator.py` while 루프를 `StateGraph`로 교체 |

---

## 8. 설계 원칙

**Local-First**: Mock 모드에서는 인터넷 연결 전혀 불필요. LanceDB가 로컬 파일시스템에 벡터 인덱스 저장.

**Self-Healing**: Critic은 단순 pass/fail이 아닌 구체적 누락 항목을 명시 → 누락된 키워드가 다음 검색 쿼리에 직접 반영 → 최대 3회로 무한 루프 방지 → 재시도 초과 시에도 마지막 초안 반환(Best-Effort).

**Hybrid Critic**: 소형 LLM(2B)이 채점 기준 목록(`expected_topics`)을 생성하면 결정론적(deterministic) 규칙 엔진이 채점. LLM의 창의성 + 규칙 기반의 신뢰성을 결합.

---

## 참고

- [LanceDB](https://lancedb.github.io/lancedb/) — 로컬 임베디드 벡터 DB
- [Pydantic AI](https://ai.pydantic.dev/) — Python Agent 프레임워크
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — 1-bit LLM 아키텍처
- [bitnet.cpp](https://github.com/microsoft/BitNet) — Microsoft 1-bit LLM 추론 엔진
- `BITNET_GUIDE.md` — BitNet 설치/실행 전체 가이드
- `INTERNALS.md` — 함수별 데이터 흐름 상세 추적
