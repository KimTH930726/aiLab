# BitNet + Self-Correction Agent 전체 가이드

## 시스템 구성 개요

```
[Browser]
    │ HTTP
    ▼
[Docker: FastAPI 웹앱 :8000]        ← aiLab/ 프로젝트 폴더에서 실행
    │ SSE (Server-Sent Events)
    │ threading + asyncio.Queue
    ▼
run_agent(on_event=callback)
    │ OpenAI 호환 API 호출
    │ host.docker.internal:8080
    ▼
[Host: llama-server(BitNet) :8080]  ← ~/BitNet/ 에서 별도 실행
```

---

## 설치 위치 및 파일 구조

| 경로 | 설명 |
|------|------|
| `~/aiLab/` | Self-Correction Agent 웹앱 (Docker 이미지 빌드 대상) |
| `~/BitNet/` | bitnet.cpp 소스 및 빌드 결과 |
| `~/aiLab/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf` | 1.1GB GGUF 모델 파일 |

> **중요**: 모델 파일은 `aiLab/models/` 디렉토리에 있지만,
> `llama-server`는 **호스트(Mac)** 에서 직접 실행됩니다 — Docker 내부가 아닙니다.

---

## 전체 시작 순서 (매 세션마다)

### 1단계: BitNet 서버 시작 (호스트 터미널)

```bash
# 터미널 1 — BitNet llama-server 실행
cd ~/BitNet

~/BitNet/build/bin/llama-server \
  -m ~/aiLab/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 2048 \
  -n 512 \
  --log-disable
```

서버가 뜨면 다음 메시지 확인:
```
llama server listening at http://0.0.0.0:8080
```

> **포트 8080이 이미 사용 중이면**: `lsof -i :8080 | grep LISTEN` 으로 확인 후 종료.

---

### 2단계: Docker 앱 실행 (별도 터미널)

```bash
# 터미널 2 — Self-Correction Agent 웹앱
cd ~/aiLab

# BitNet 모드로 실행 (기본값)
AGENT_MODEL=openai:bitnet docker compose up

# 또는 이미지를 새로 빌드해야 할 때
AGENT_MODEL=openai:bitnet docker compose up --build
```

---

### 3단계: 브라우저 접속

```
http://localhost:8000
```

- **Model 입력란**: 기본값 `openai:bitnet` (그대로 사용)
- **Chat 탭**: 질문 입력 → 에이전트 단계별 실시간 진행 확인
- **DB 탭**: LanceDB 지식베이스 문서 10개 확인

---

## BitNet 성능 (M2 MacBook 기준)

| 조건 | 예상 소요 시간 |
|------|---------------|
| 2B 모델, CPU only, Draft 1회 | 20~40초 |
| Self-Healing Retry 1회 발생 | 추가 20~40초 |
| 전체 (DONE까지) | 40~80초 |

**이 속도가 정상입니다.** BitNet b1.58은 1비트 양자화 LLM으로 GPU 없이 CPU만으로
동작하도록 설계되었습니다. GPU가 없는 환경에서 실용적으로 사용할 수 있는 가장 가벼운 LLM 중 하나입니다.

속도 개선 옵션:
```bash
# 스레드 수 늘리기 (물리 코어 수에 맞게 조정)
~/BitNet/build/bin/llama-server \
  -m ~/aiLab/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 2048 -n 512 \
  -t 8            # ← 스레드 수 (M2 Pro: 최대 10, M2: 최대 8)
```

---

## BitNet을 다른 프로젝트에서도 사용하는 방법

BitNet은 **OpenAI 호환 API 서버** (`llama-server`)로 동작합니다.
포트 8080에서 실행 중이면 어떤 프로젝트에서도 동일하게 사용 가능합니다.

### Python (pydantic-ai)

```python
from pydantic_ai import Agent

# BitNet llama-server가 8080에서 실행 중이어야 함
agent = Agent(
    "openai:bitnet",           # 모델명은 임의 — 실제 모델은 서버에 로드된 것
    system_prompt="당신은 AI 연구원입니다. 한국어로 답변하세요.",
)

result = agent.run_sync("GPT-4o의 주요 특징을 설명해주세요.")
print(result.output)
```

환경변수 설정 (pydantic-ai가 자동으로 사용):
```bash
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=dummy-key          # 로컬 서버는 키 검증 안 함
```

### Python (OpenAI SDK 직접)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy-key",
)

response = client.chat.completions.create(
    model="bitnet",   # 서버에서 로드된 모델 사용
    messages=[{"role": "user", "content": "안녕하세요!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

### Docker 컨테이너에서 사용 시

컨테이너 내부에서는 `localhost:8080` 대신 `host.docker.internal:8080` 사용:

```yaml
# docker-compose.yml
environment:
  - OPENAI_BASE_URL=http://host.docker.internal:8080/v1
  - OPENAI_API_KEY=dummy-key
extra_hosts:
  - "host.docker.internal:host-gateway"  # Linux 필수, Mac은 자동
```

---

## BitNet의 제약 사항

| 기능 | 지원 여부 |
|------|-----------|
| 텍스트 생성 | ✅ |
| 스트리밍 | ✅ |
| Function Calling (tools) | ❌ |
| JSON Mode | ⚠️ 불안정 |
| Embeddings | ❌ |

**Function Calling 불가** → pydantic-ai의 `@agent.tool` 기능을 사용하는 경우
`LOCAL_NO_TOOL_MODELS` 목록에 모델을 추가하거나, 검색 결과를 프롬프트에 임베드하는 방식을 사용하세요.
(이 프로젝트는 이미 자동으로 no-tool 모드를 사용합니다.)

---

## 문제 해결

### BitNet 서버가 시작되지 않을 때

```bash
# 바이너리 확인
ls ~/BitNet/build/bin/llama-server

# 직접 경로로 실행
/Users/$(whoami)/BitNet/build/bin/llama-server --help
```

### Docker가 BitNet에 연결되지 않을 때

```bash
# 호스트에서 BitNet 서버 확인
curl http://localhost:8080/v1/models

# Docker 내부에서 확인
docker exec -it ailab-agent-1 curl http://host.docker.internal:8080/v1/models
```

### Docker 컨테이너 재빌드

```bash
cd ~/aiLab
docker compose down
docker compose build --no-cache
AGENT_MODEL=openai:bitnet docker compose up
```

### LanceDB 초기화 (지식베이스 리셋)

```bash
# Docker volume 삭제 후 재시작
docker compose down -v
AGENT_MODEL=openai:bitnet docker compose up
```

---

## Mock 모드 (LLM 없이 테스트)

BitNet 없이 빠르게 UI/흐름을 테스트할 때:

```bash
# Model 입력란을 비워두거나
docker compose up

# 또는 명시적으로
AGENT_MODEL= docker compose up
```

Mock 모드는 LLM 호출 없이 템플릿 기반으로 즉시 결과를 반환합니다.

---

## 전체 설치 이력 (이 맥북에서 수행한 내용)

1. `huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf`로 모델 다운로드
2. `git clone https://github.com/microsoft/BitNet.git ~/BitNet`
3. ARM64 cmake 사용 (`/opt/homebrew/Cellar/cmake/4.2.3/bin/cmake`)
4. macOS SDK C++ 헤더 주입을 위한 컴파일러 래퍼 생성 후 `setup_env.py` 실행
5. `~/BitNet/build/bin/llama-server` 바이너리 생성 완료
6. Docker + FastAPI 웹앱 완성 (SSE 실시간 스트리밍)
