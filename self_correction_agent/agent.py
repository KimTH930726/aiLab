"""
Pydantic AI Agent 팩토리: LanceDB Tool이 주입된 LLM Agent 생성.

--model 옵션이 제공될 때만 활성화된다.
BitNet b1.58 via bitnet.cpp:
  1. bitnet.cpp 서버 실행 (OpenAI 호환 엔드포인트)
  2. --model openai:bitnet 으로 연결
"""
from __future__ import annotations

import json
import textwrap
from typing import Any

from .infra.vectordb import LocalRAG


def create_pydantic_agent(model_name: str) -> Any:
    """
    Pydantic AI Agent + LanceDB 검색 Tool 생성.

    model_name 예시:
      - 'ollama:llama3.2'    -> 로컬 Ollama
      - 'openai:gpt-4o-mini' -> OpenAI API
    """
    try:
        from pydantic_ai import Agent, RunContext
    except ImportError:
        print("  [WARN] pydantic-ai not installed. Falling back to mock mode.")
        return None

    agent = Agent(
        model_name,
        deps_type=LocalRAG,
        system_prompt=textwrap.dedent("""\
            You are a research report writer specializing in AI industry analysis.
            Use the search_knowledge_base tool to find relevant information.
            Write reports with ## markdown headers and cite sources with
            "Source: ..." format. Be comprehensive and factual.
        """),
    )

    @agent.tool
    def search_knowledge_base(ctx: RunContext[LocalRAG], query: str) -> str:
        """Search the local LanceDB knowledge base for relevant information."""
        results = ctx.deps.search(query, top_k=5)
        return json.dumps(results, ensure_ascii=False, indent=2)

    return agent
