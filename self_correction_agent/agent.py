"""
Pydantic AI Agent 팩토리: LanceDB Tool이 주입된 LLM Agent 생성.

--model 옵션이 제공될 때만 활성화된다.
BitNet b1.58 via bitnet.cpp:
  1. bitnet.cpp 서버 실행 (OpenAI 호환 엔드포인트)
  2. --model openai:bitnet 으로 연결
"""
import json
import textwrap
from typing import Any

from .infra.vectordb import LocalRAG

# pydantic-ai 선택적 임포트 (모듈 레벨 — annotations 지연 평가 방지)
try:
    from pydantic_ai import Agent, RunContext
    _PYDANTIC_AI_AVAILABLE = True
except ImportError:
    _PYDANTIC_AI_AVAILABLE = False


def create_pydantic_agent(model_name: str) -> Any:
    """
    Pydantic AI Agent 생성.

    model_name 예시:
      - 'ollama:llama3.2'    -> 로컬 Ollama (tools 지원)
      - 'openai:gpt-4o-mini' -> OpenAI API (tools 지원)
      - 'openai:bitnet'      -> BitNet via bitnet.cpp (tools 미지원 → no-tool 모드)

    반환값: (agent, use_tools: bool) 튜플
    """
    if not _PYDANTIC_AI_AVAILABLE:
        print("  [WARN] pydantic-ai not installed. Falling back to mock mode.")
        return None, False

    # bitnet.cpp / 로컬 서버는 function calling(tools)을 지원하지 않음
    # → 검색 결과를 프롬프트에 임베드하는 no-tool 모드 사용
    LOCAL_NO_TOOL_MODELS = {"bitnet", "local"}
    model_id = model_name.split(":")[-1].lower()
    use_tools = model_id not in LOCAL_NO_TOOL_MODELS

    agent = Agent(
        model_name,
        system_prompt=textwrap.dedent("""\
            You are a research report writer specializing in AI industry analysis.
            IMPORTANT: Always respond in Korean (한국어). All output must be in Korean.
            Write reports with ## markdown headers and cite sources with
            "출처: ..." format. Be comprehensive and factual.
        """),
    )

    if use_tools:
        # tools API 지원 모델: Agent가 직접 LanceDB 검색 수행
        agent = Agent(
            model_name,
            deps_type=LocalRAG,
            system_prompt=textwrap.dedent("""\
                You are a research report writer specializing in AI industry analysis.
                IMPORTANT: Always respond in Korean (한국어). All output must be in Korean.
                Use the search_knowledge_base tool to find relevant information.
                Write reports with ## markdown headers and cite sources with
                "출처: ..." format. Be comprehensive and factual.
            """),
        )

        @agent.tool
        def search_knowledge_base(ctx: RunContext[LocalRAG], query: str) -> str:
            """Search the local LanceDB knowledge base for relevant information."""
            results = ctx.deps.search(query, top_k=5)
            return json.dumps(results, ensure_ascii=False, indent=2)

    return agent, use_tools
