# Self-Correction Agent

## Local-First AI : 자가 수정 에이전트

> GPU 클라우드 없이, 로컬 CPU만으로 동작하는 "끈질긴" AI 에이전트

---

## 1. 프로젝트 개요

이 프로젝트는 **고가의 GPU 클라우드 의존성 없이** 사용자의 로컬 환경(CPU)에서
완벽하게 동작하는 Self-Correction(자가 수정) 에이전트입니다.

단순히 한 번 답변을 생성하고 끝나는 것이 아니라,
**Critic(검증자)이 결과물을 평가하고, 부족하면 스스로 재검색-재작성을 반복**하여
목표 품질에 도달할 때까지 자율적으로 개선합니다.

### 핵심 가치

| 특성 | 설명 |
|------|------|
| **Local-First** | 네트워크 없이 로컬 디스크 + CPU만으로 완전 동작 |
| **Self-Healing** | Critic이 부족한 부분을 진단하면 자동으로 재검색/재작성 (최대 3회) |
| **Lightweight** | Bag-of-Words 임베딩 + LanceDB로 외부 모델 서버 불필요 |
| **Swappable** | Mock 모드에서 Ollama/OpenAI까지 한 줄 변경으로 전환 가능 |

---

## 2. 기술 스택

```
+──────────────────────────────────────────────────+
│              Self-Correction Agent                │
+──────────────────────────────────────────────────+
│  Inference  │  BitNet b1.58 (Simulated / Mock)   │
│  Logic      │  Python 3.9+ / Pydantic BaseModel  │
│  Agent      │  Pydantic AI (Tool-augmented LLM)  │
│  Memory     │  LanceDB (Local Embedded Vector DB) │
│  Embedding  │  Bag-of-Words (CPU) / sentence-transformers (옵션) │
│  Pattern    │  Planner-Worker-Critic Loop         │
+──────────────────────────────────────────────────+
```

### 의존성

| 패키지 | 역할 | 버전 |
|--------|------|------|
| `lancedb` | 로컬 벡터 DB (RAG 검색) | >= 0.6.0 |
| `numpy` | 임베딩 벡터 연산 | >= 1.24.0 |
| `pydantic` | 상태 모델 (타입 안전) | >= 2.0.0 |
| `pydantic-ai` | LLM Agent 프레임워크 (선택) | >= 0.0.20 |

---

## 3. 아키텍처

### 3.1 전체 흐름 (State Machine)

```
  ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐
  │ PLANNING │────>│ SEARCHING│────>│ DRAFTING │────>│ CRITIQUING│
  └─────────┘     └──────────┘     └──────────┘     └───────────┘
       ^                                                   │
       │                                          ┌────────┴────────┐
       │                                          │                 │
       │                                    passed?            failed?
       │                                          │                 │
       │                                     ┌────┴──┐        ┌────┴──┐
       │                                     │ DONE  │        │can    │
       │                                     └───────┘        │retry? │
       │                                                      └───┬───┘
       │                                                 yes      │ no
       │◄─────────── (Self-Healing Retry) ◄───────────────┘       │
                                                             ┌────┴──┐
                                                             │FAILED │
                                                             └───────┘
```

**각 Phase의 역할:**

| Phase | 역할 | 핵심 동작 |
|-------|------|----------|
| **PLANNING** | 검색 전략 수립 | 기본 4개 쿼리 생성. 재시도 시 Critic 피드백 기반 타깃 쿼리 추가 |
| **SEARCHING** | LanceDB 벡터 검색 | 각 쿼리별 top_k=2 검색, 결과 중복 제거 |
| **DRAFTING** | 리포트 초안 작성 | Mock 템플릿 또는 Pydantic AI Agent가 검색 결과 기반 작성 |
| **CRITIQUING** | 품질 검증 | 키워드 커버리지, 길이, 구조, 출처 인용을 점수화 |
| **DONE** | 성공 완료 | Critic 통과, 최종 결과 반환 |
| **FAILED** | 실패 종료 | 최대 재시도 초과 또는 검색 결과 없음 |

### 3.2 핵심 컴포넌트

#### AgentState (상태 모델)

모든 노드를 관통하는 **단일 진실 공급원(Single Source of Truth)** 입니다.

```python
class AgentState(BaseModel):
    phase: Phase               # 현재 단계
    query: str                 # 사용자 질의
    search_results: List[...]  # LanceDB 검색 결과
    draft: str                 # Worker가 생성한 초안
    critique: CriticVerdict    # Critic 평가 결과
    retry_count: int           # 현재 재시도 횟수 (max 3)
    final_result: str          # 최종 승인된 결과
```

#### LocalRAG (LanceDB 래퍼)

```python
class LocalRAG:
    def seed(docs)    # 문서 벡터화 후 LanceDB에 인덱싱
    def search(query) # 쿼리 벡터화 -> 코사인 유사도 검색
```

- 저장 경로: `/tmp/lancedb_self_correction_agent` (변경 가능)
- 임베딩: 60차원 Bag-of-Words 벡터 (L2 정규화)
- 프로덕션 대안: `sentence-transformers`의 `all-MiniLM-L6-v2`

#### Worker (초안 생성)

| 모드 | 설명 |
|------|------|
| `mock` | LLM 없이 템플릿 기반 리포트 생성 (기본값) |
| `pydantic-ai` | 실제 LLM이 LanceDB 도구를 사용하여 자율 작성 |

#### Critic (품질 검증 엔진)

규칙 기반 점수 체계로 CPU 부담 없이 즉시 평가합니다.

```
총점 = (키워드 커버리지 x 0.4)
     + (콘텐츠 길이    x 0.2)
     + (문서 구조      x 0.2)
     + (출처 인용      x 0.2)
```

**Hard Gate**: 키워드 커버리지가 85% 미만이면 총점과 무관하게 **무조건 RETRY**

```
필수 키워드: GPT-4o, Sora, API, safety, multimodal, o1, enterprise, partnership
```

### 3.3 Self-Healing (자가 수정) 메커니즘

이 에이전트의 핵심 차별점은 **실패 시 자동 복구**입니다.

```
[1차 시도]
  PLAN: 기본 4개 쿼리
  SEARCH: 5개 문서 발견
  DRAFT: 5개 토픽 커버
  CRITIC: "Sora", "o1" 누락 -> RETRY 명령

[2차 시도 - Self-Healing]
  PLAN: 기본 4개 + "OpenAI Sora latest" + "OpenAI o1 latest" 추가
  SEARCH: 7개 문서 발견 (누락 토픽 포함)
  DRAFT: 7개 토픽 커버
  CRITIC: 8/8 키워드 충족, Score 1.00 -> APPROVED
```

**Critic의 피드백이 다음 Planner의 검색 전략에 직접 반영**되는 것이 핵심입니다.

---

## 4. 파일 구조 (DDD 패키지)

```
self_correction_agent/              # 패키지 루트
├── __init__.py                     # 공개 API: run_agent
├── __main__.py                     # python -m self_correction_agent 진입점
│
├── domain/                         # 순수 도메인 — pydantic 외 외부 의존성 없음
│   ├── __init__.py
│   ├── state.py                    # Phase, PHASE_ICONS, CriticVerdict, AgentState
│   └── constants.py                # REQUIRED_KEYWORDS, 임계값 상수
│
├── infra/                          # 인프라 계층 — 외부 라이브러리(numpy, lancedb) 접점
│   ├── __init__.py
│   ├── embedding.py                # _VOCAB, EMBED_DIM, text_to_vector()
│   └── vectordb.py                 # LocalRAG (LanceDB 래퍼)
│
├── nodes/                          # 비즈니스 로직 노드 — 도메인 모델만 참조
│   ├── __init__.py
│   ├── planner.py                  # plan_search_queries()
│   ├── searcher.py                 # execute_search()
│   ├── worker.py                   # worker_generate_draft(), _mock_generate()
│   └── critic.py                   # critic_evaluate()
│
├── knowledge/                      # 데이터 계층
│   ├── __init__.py
│   └── openai_trends.py            # KNOWLEDGE_BASE 시드 데이터 (10건)
│
├── agent.py                        # Pydantic AI Agent 팩토리 + LanceDB Tool
└── orchestrator.py                 # run_agent() 메인 루프 (상태 머신)
```

### 계층별 의존 방향

```
  __main__.py
      │
      ▼
  orchestrator.py ──────► nodes/*  (planner, searcher, worker, critic)
      │                      │
      ├──► agent.py          ├──► domain/*  (state, constants)
      │       │              │
      ▼       ▼              ▼
  infra/*  (vectordb, embedding)
      │
      ▼
  knowledge/*  (openai_trends)
```

- **domain/**: 순수 도메인. numpy, lancedb 직접 import 금지.
- **infra/**: 외부 라이브러리 접점. embedding.py를 sentence-transformers로 교체해도 nodes/에 영향 없음.
- **nodes/**: AgentState를 받아서 처리하는 순수 비즈니스 로직.
- **orchestrator**: infra + nodes를 조립하여 while 루프 구동.

---

## 5. 빌드 및 실행

### 5.1 환경 설정

```bash
# 의존성 설치
pip install lancedb numpy pydantic

# (선택) Pydantic AI — 실제 LLM 연동 시 필요
pip install pydantic-ai
```

### 5.2 실행 방법

#### Mock 모드 (기본 — LLM 불필요)

```bash
python3 -m self_correction_agent
```

외부 API 키나 모델 서버 없이 즉시 실행됩니다.
Bag-of-Words 임베딩 + 템플릿 기반 생성으로 전체 Self-Healing 루프를 시연합니다.

#### Ollama 로컬 LLM 모드

```bash
# 1. Ollama 설치 후 모델 다운로드
ollama pull llama3.2

# 2. 에이전트 실행
python3 -m self_correction_agent --model ollama:llama3.2
```

#### OpenAI API 모드

```bash
# 환경변수로 API 키 설정
export OPENAI_API_KEY="sk-..."

python3 -m self_correction_agent --model openai:gpt-4o-mini
```

#### BitNet b1.58 모드 (bitnet.cpp)

```bash
# 1. bitnet.cpp 서버 실행 (OpenAI 호환 엔드포인트)
./bitnet-server -m model.gguf --port 8080

# 2. OpenAI 호환 모드로 연결
python3 -m self_correction_agent --model openai:bitnet
```

#### Python 코드에서 호출

```python
from self_correction_agent import run_agent

report = run_agent(query="OpenAI 최신 동향 리포트를 작성해주세요")
print(report)
```

### 5.3 CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `None` (mock) | Pydantic AI 모델 식별자 |
| `--query` | `"OpenAI 최신 동향 리포트를 작성해주세요"` | 에이전트에게 전달할 질의 |
| `--db-path` | `/tmp/lancedb_self_correction_agent` | LanceDB 저장 경로 |

환경변수 `AGENT_MODEL`로도 모델을 지정할 수 있습니다:

```bash
AGENT_MODEL=ollama:mistral python3 -m self_correction_agent
```

---

## 6. 실행 결과 예시

```
============================================================
  Self-Correction Agent | Local-First AI
  LanceDB (RAG) + Pydantic AI + BitNet b1.58 (Sim)
============================================================
  Mode   : mock
  LanceDB: 10 documents indexed
  Query  : OpenAI 최신 동향 리포트를 작성해주세요
============================================================

──────────────────────────────────────────────────
  Phase: planning  |  Attempt: 1/4
──────────────────────────────────────────────────
  [21:48:47] [PLAN] Analyzing query, planning search strategy...
  [21:48:47] [PLAN] Planned 4 search queries

──────────────────────────────────────────────────
  Phase: searching  |  Attempt: 1/4
──────────────────────────────────────────────────
  [21:48:47] [SEARCH] Querying LanceDB knowledge base...
  [21:48:47] [SEARCH] Total unique results: 5

──────────────────────────────────────────────────
  Phase: critiquing  |  Attempt: 1/4
──────────────────────────────────────────────────
  [21:48:47] [CRITIC] Score: 0.90 | Passed: False    <-- 키워드 부족!
  [21:48:47] [CRITIC] RETRY #1 triggered
  [21:48:47] [CRITIC]   Missing: Sora, o1            <-- 누락 항목 진단

──────────────────────────────────────────────────
  Phase: planning  |  Attempt: 2/4                    <-- 자가 수정 시작
──────────────────────────────────────────────────
  [21:48:47] [PLAN] Targeted queries added for: Sora, o1

──────────────────────────────────────────────────
  Phase: critiquing  |  Attempt: 2/4
──────────────────────────────────────────────────
  [21:48:47] [CRITIC] Score: 1.00 | Passed: True     <-- 통과!
  [21:48:47] [DONE] Draft APPROVED by Critic.

============================================================
  Result  : DONE
  Retries : 1/3
  Score   : 1.00
============================================================
```

---

## 7. 아키텍처 확장 가이드

### 7.1 임베딩 교체 (Bag-of-Words -> Sentence Transformers)

`infra/embedding.py` 파일만 수정하면 됩니다. 다른 계층에 영향 없음:

```python
# self_correction_agent/infra/embedding.py 전체 교체

from sentence_transformers import SentenceTransformer

_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")  # 384차원, CPU 가능
EMBED_DIM = 384

def text_to_vector(text: str) -> list[float]:
    return _ENCODER.encode(text).tolist()
```

기존 `_VOCAB` 관련 코드는 자동으로 무시됩니다 (파일 전체 교체).

### 7.2 Critic을 LLM 기반으로 전환

```python
from pydantic_ai import Agent

critic_agent = Agent(
    "ollama:llama3.2",
    result_type=CriticVerdict,   # 구조화된 출력
    system_prompt="당신은 리포트 품질 검수 전문가입니다...",
)

# critic_evaluate() 대신 사용:
verdict = critic_agent.run_sync(f"다음 초안을 평가하세요:\n{draft}")
```

### 7.3 지식 베이스 확장

`KNOWLEDGE_BASE` 리스트에 문서를 추가하거나, 외부 파일에서 로드하도록 변경:

```python
import json

with open("knowledge.json", "r") as f:
    KNOWLEDGE_BASE = json.load(f)
```

### 7.4 Pydantic AI Tool 추가

Worker Agent에 새로운 도구를 추가하여 기능을 확장할 수 있습니다:

```python
@agent.tool
def web_search(ctx: RunContext[LocalRAG], query: str) -> str:
    """웹에서 실시간 정보를 검색합니다."""
    # 실제 웹 검색 로직
    ...

@agent.tool
def calculate(ctx: RunContext[LocalRAG], expression: str) -> str:
    """수학 계산을 수행합니다."""
    return str(eval(expression))  # 프로덕션에서는 안전한 파서 사용
```

---

## 8. 설계 원칙

### Local-First 원칙

1. **네트워크 제로 의존성**: Mock 모드에서는 인터넷 연결이 전혀 필요 없음
2. **디스크 기반 영속성**: LanceDB가 로컬 파일시스템에 벡터 인덱스 저장
3. **CPU 최적화**: Bag-of-Words 임베딩은 GPU 없이 수 밀리초 내 완료
4. **점진적 확장**: Mock -> Ollama -> Cloud API 순으로 단계적 업그레이드 가능

### Self-Healing 원칙

1. **구조적 피드백**: Critic은 단순 pass/fail이 아닌 구체적 누락 항목을 명시
2. **적응적 재검색**: 누락된 키워드가 다음 검색 쿼리에 직접 반영
3. **유한 루프**: 최대 3회 재시도로 무한 루프 방지
4. **Best-Effort 반환**: 재시도 초과 시에도 마지막 초안을 반환 (완전 실패 방지)

---

## 9. 라이선스 및 참고

### 참고 기술

- [LanceDB](https://lancedb.github.io/lancedb/) — 로컬 임베디드 벡터 데이터베이스
- [Pydantic AI](https://ai.pydantic.dev/) — Python Agent 프레임워크
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — 1-bit LLM 아키텍처
- [bitnet.cpp](https://github.com/microsoft/BitNet) — Microsoft의 1-bit LLM 추론 엔진
