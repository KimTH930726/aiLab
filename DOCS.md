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
| **Hybrid RAG** | sentence-transformers(384차원) + BM25 하이브리드 검색, 500자 청킹 |
| **Swappable** | Mock 모드에서 Ollama(EXAONE)/OpenAI까지 한 줄 변경으로 전환 가능 |
| **Hybrid Critic** | LLM이 동적 채점 기준 생성 → 규칙 기반 엔진이 채점 (쿼리 적응형) |

---

## 2. 기술 스택

| 패키지 | 역할 |
|--------|------|
| `lancedb` | 로컬 벡터 DB (RAG 검색) |
| `sentence-transformers` | 다국어 임베딩 (`paraphrase-multilingual-MiniLM-L12-v2`, 384차원) |
| `rank-bm25` | BM25 키워드 검색 인덱스 (in-memory, `BM25Okapi`) |
| `numpy` | 벡터 연산 보조 |
| `pydantic` | 상태/결과 데이터 모델 (`AgentState`, `CriticVerdict` 등) |
| `pydantic-ai` | LLM 호출 전담 (`Agent.run_sync`) — 쿼리 확장·eval_criteria·Draft 생성에 사용 |
| `fastapi` + `sse-starlette` | 웹 UI 서버 (SSE 실시간 스트리밍 + CRUD REST API) |
| `ollama` | 로컬 LLM 런타임 (EXAONE 3.5 등 실행). Docker 외부 호스트에서 실행, `host.docker.internal:11434`로 접근 |

> **pydantic vs pydantic-ai 역할 구분**
> - `pydantic`: 데이터 구조 정의만 담당 (BaseModel)
> - `pydantic-ai Agent`: LLM 호출 래퍼. 흐름 제어는 하지 않음
> - 흐름 제어(그래프/상태 전이)는 `orchestrator.py`의 **직접 구현 while 루프**가 담당

---

## 3. 아키텍처

### 배포 구조 (Docker + Ollama)

```
┌─────────────────── macOS 호스트 ──────────────────────┐
│                                                       │
│  ┌─────────────────────────────────┐                 │
│  │  Docker Container               │                 │
│  │  FastAPI + LanceDB + BM25       │                 │
│  │  :8000                          │                 │
│  │                                 │   host.docker   │
│  │  OLLAMA_BASE_URL                │   .internal     │
│  │  = http://host.docker.internal  │◄───────────────►│  Ollama (brew)
│  │    :11434/v1                    │                 │  :11434
│  └─────────────────────────────────┘                 │  exaone3.5:7.8b
│                                                       │
│  브라우저 → http://localhost:8000                     │
└───────────────────────────────────────────────────────┘
```

**pydantic-ai 1.x Ollama 연결 방식**:
- `ollama:exaone3.5:7.8b` → `OpenAIChatModel` (OpenAI 호환 클라이언트로 처리)
- `OLLAMA_BASE_URL` 값을 base URL로 사용 → `/chat/completions` 경로 추가
- 따라서 `OLLAMA_BASE_URL`은 반드시 `/v1`을 포함해야 함: `http://host.docker.internal:11434/v1`

**EXAONE tool calling 미지원**:
- Ollama의 EXAONE 3.5는 function calling을 지원하지 않음
- `agent.py` / `llm_helpers.py`의 `_LOCAL_NO_TOOL_MODELS = {"exaone"}` 에 등록
- Worker는 tools 없이 검색 결과를 프롬프트에 직접 포함, Planner/Critic도 줄바꿈 텍스트 파싱 경로 사용

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

| Phase | 역할 |
|-------|------|
| **PLANNING** | LLM으로 쿼리 변형 3~4개 생성 + 동적 eval_criteria 생성. LLM 없으면 heuristic fallback. 재시도 시 Critic 누락 키워드 기반 추가 쿼리만 보충 |
| **SEARCHING** | 하이브리드 검색 (벡터 + BM25, alpha/threshold/top_k 설정 기반) |
| **DRAFTING** | Mock 템플릿 또는 Pydantic AI Agent가 검색 결과 기반 리포트 작성 |
| **CRITIQUING** | eval_criteria(LLM 동적 기준) 또는 REQUIRED_KEYWORDS(fallback)로 키워드 커버리지(40%) + 길이(20%) + 구조(20%) + 출처(20%) 점수화 |

### RAG 파이프라인

```
문서 추가
  └─ chunk_text(500자/50자 오버랩)
  └─ text_to_vector(384차원 ST 임베딩) × 청크마다
  └─ LanceDB 저장 (parent_id, chunk_index 포함)
  └─ BM25Okapi 재구성

검색
  └─ 벡터 검색: LanceDB.search() → 거리 기반 후보 선정
  └─ BM25 검색: BM25Okapi.get_scores() → 키워드 점수
  └─ 결합: alpha * norm_vector + (1-alpha) * norm_bm25
  └─ parent_id 중복 제거 → top_k 반환
```

**Hybrid Critic**: LLM이 쿼리에 맞는 `expected_topics` 생성 → 규칙 기반 엔진이 그 기준으로 채점.

**Critic Hard Gate**: 키워드 커버리지 85% 미만이면 총점과 무관하게 무조건 RETRY

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
    │   ├── embedding.py            # sentence-transformers 384차원 임베딩
    │   ├── vectordb.py             # LocalRAG (LanceDB + BM25 하이브리드 검색 + CRUD)
    │   ├── chunker.py              # 문자 기반 슬라이딩 윈도우 청킹 (500자/50자)
    │   └── settings.py             # 검색 설정 영속화 (alpha, threshold, top_k)
    ├── nodes/
    │   ├── llm_helpers.py          # LLM 쿼리 확장 + eval_criteria 생성
    │   ├── planner.py              # 동적 검색 쿼리 생성 (LLM + heuristic)
    │   ├── searcher.py             # 하이브리드 검색 실행
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
| POST | `/db/documents` | 문서 추가 (청킹+임베딩 자동) → `{id, topic, source}` 반환 |
| DELETE | `/db/documents/{id}` | 문서 삭제 (parent_id의 모든 청크 제거) |
| POST | `/db/upload` | `.txt` 파일 업로드 → 청킹+임베딩 저장 |
| POST | `/db/seed` | 지식베이스 초기 데이터로 초기화 (전체 재생성) |
| GET | `/db/settings` | 검색 설정 조회 `{alpha, distance_threshold, top_k}` |
| POST | `/db/settings` | 검색 설정 저장 (settings.json 영속화) |
| GET | `/db/search` | 하이브리드 검색 테스트 (vector/bm25/combined 점수 반환) |

---

## 6. 실행 방법

### 사전 준비 — Ollama + EXAONE

```bash
# Ollama 설치 (최초 1회)
brew install ollama

# Ollama 서비스 시작 (로그인 시 자동 실행)
brew services start ollama

# EXAONE 3.5 모델 다운로드 (~4.8GB, 최초 1회)
ollama pull exaone3.5:7.8b

# 동작 확인
ollama list          # exaone3.5:7.8b 표시 확인
curl http://localhost:11434  # "Ollama is running"
```

### Docker + 웹 UI (권장)

```bash
# EXAONE 기본 모드 (docker-compose.yml의 AGENT_MODEL 기본값 사용)
docker compose up

# Mock 모드 (Ollama 없이 동작 확인용)
AGENT_MODEL= docker compose up

# 다른 Ollama 모델 사용
AGENT_MODEL=ollama:llama3.1 docker compose up
```

브라우저: `http://localhost:8000`

> 웹 UI Model 입력창은 `ollama:exaone3.5:7.8b`가 기본값. 비우면 Mock 모드로 동작.

### CLI (로컬 직접 실행)

```bash
pip install -r requirements.txt

# Mock 모드
python3 -m self_correction_agent --query "원하는 질문"

# EXAONE 모드 (Ollama 실행 중이어야 함)
python3 -m self_correction_agent --query "원하는 질문" --model ollama:exaone3.5:7.8b

# OpenAI 모드
python3 -m self_correction_agent --query "원하는 질문" --model openai:gpt-4o-mini
```

> CLI 직접 실행 시 `OLLAMA_BASE_URL`을 환경변수로 주지 않으면 기본값 `http://localhost:11434/v1` 사용.

### Ollama 관리 명령어

```bash
ollama list                    # 설치된 모델 목록
ollama ps                      # 현재 메모리에 로드된 모델
ollama rm exaone3.5:7.8b       # 모델 삭제
brew services stop ollama      # 서비스 중지
brew services restart ollama   # 서비스 재시작
```

---

## 7. 검색 설정 (alpha / threshold / top_k)

웹 UI → 지식베이스 탭 → **추가/관리** → 검색 전략 설정

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `alpha` | 0.7 | 0=순수 BM25(키워드), 1=순수 벡터(의미) |
| `distance_threshold` | 1.5 | 벡터 거리 임계값 (낮을수록 엄격) |
| `top_k` | 2 | 쿼리당 반환 문서 수 |

설정은 `{DB_PATH}/settings.json`에 저장되며 Docker 볼륨과 함께 영속됩니다.

---

## 8. 확장 포인트

| 목표 | 수정 파일 |
|------|-----------|
| 청킹 크기/오버랩 조정 | `infra/chunker.py` chunk_size, overlap 파라미터 |
| 임베딩 모델 교체 | `infra/embedding.py` _MODEL_NAME 변경 |
| 검색 전략 변경 (RRF 등) | `infra/vectordb.py` hybrid_search() |
| LLM 쿼리 확장 프롬프트 튜닝 | `nodes/llm_helpers.py` |
| 지식베이스 교체 | `knowledge/openai_trends.py` 또는 웹 UI 지식베이스 탭에서 직접 추가 |
| 새 LLM Tool 추가 | `agent.py`에 `@agent.tool` 데코레이터로 추가 |
| eval_criteria 기준 조정 | `nodes/llm_helpers.py`의 `generate_eval_criteria()` 프롬프트 수정 |
| 다른 Ollama 모델로 교체 | `AGENT_MODEL=ollama:<모델명>` 환경변수만 변경 |
| tools 미지원 모델 추가 | `agent.py` + `llm_helpers.py`의 `_LOCAL_NO_TOOL_MODELS`에 모델명 부분문자열 추가 |
| 상태 머신 → LangGraph 마이그레이션 | `orchestrator.py` while 루프를 `StateGraph`로 교체 |

---

## 9. 설계 원칙

**Local-First**: Mock 모드에서는 인터넷 연결 전혀 불필요. LanceDB가 로컬 파일시스템에 벡터 인덱스 저장.

**Self-Healing**: Critic은 단순 pass/fail이 아닌 구체적 누락 항목을 명시 → 누락된 키워드가 다음 검색 쿼리에 직접 반영 → 최대 3회로 무한 루프 방지 → 재시도 초과 시에도 마지막 초안 반환(Best-Effort).

**Hybrid RAG**: sentence-transformers 384차원 임베딩으로 의미 기반 검색, BM25로 키워드 정밀 검색. alpha 파라미터로 두 전략의 비율을 조정. 문서 입력 시 자동 청킹으로 긴 문서도 정확하게 검색.

**Hybrid Critic**: LLM이 채점 기준 목록(`expected_topics`)을 생성하면 결정론적(deterministic) 규칙 엔진이 채점. LLM의 창의성 + 규칙 기반의 신뢰성을 결합.

---

## 참고

- [LanceDB](https://lancedb.github.io/lancedb/) — 로컬 임베디드 벡터 DB
- [sentence-transformers](https://www.sbert.net/) — 다국어 임베딩 모델
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — Python BM25 구현
- [Pydantic AI](https://ai.pydantic.dev/) — Python Agent 프레임워크
- [EXAONE 3.5](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct) — LG AI Research 한국어 특화 LLM
- [Ollama](https://ollama.com/) — 로컬 LLM 런타임
- `INTERNALS.md` — 함수별 데이터 흐름 상세 추적
