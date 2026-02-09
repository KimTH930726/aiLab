"""
Worker 노드: 검색 결과 기반 리포트 초안 생성.

mock 모드 (LLM 불필요)와 pydantic-ai 모드 (실제 LLM) 지원.
"""
from __future__ import annotations

import textwrap
from datetime import datetime
from typing import Any, Optional

from ..domain.state import AgentState
from ..infra.vectordb import LocalRAG


def worker_generate_draft(
    state: AgentState,
    mode: str = "mock",
    pydantic_agent: Any = None,
    rag: Optional[LocalRAG] = None,
) -> str:
    """
    Worker 노드 진입점.

    mode="mock"       : 템플릿 기반 (BitNet 시뮬레이션, LLM 불필요)
    mode="pydantic-ai": Pydantic AI Agent가 LanceDB Tool로 자율 작성
    """
    if mode == "pydantic-ai" and pydantic_agent is not None and rag is not None:
        prompt = _build_worker_prompt(state)
        result = pydantic_agent.run_sync(prompt, deps=rag)
        return result.data

    return _mock_generate(state)


def _build_worker_prompt(state: AgentState) -> str:
    """Pydantic AI Agent용 프롬프트 생성."""
    context = "\n".join(
        f"- [{r['topic']}] {r['text']}"
        for r in state.search_results
    )

    feedback = ""
    if state.critique and state.critique.feedback:
        feedback = (
            f"\n\nPREVIOUS ATTEMPT FEEDBACK (retry #{state.retry_count}):\n"
            f"{state.critique.feedback}\n"
            f"Missing topics: {', '.join(state.critique.missing_keywords)}\n"
            f"You MUST address these gaps in the new draft."
        )

    return textwrap.dedent(f"""\
        Write a comprehensive "OpenAI Latest Trends Report" using this knowledge:

        {context}
        {feedback}

        Requirements:
        - Use ## headers for each major topic
        - Cite sources with "Source: ..." format
        - Cover ALL topics present in the search results
        - Be comprehensive and factual
    """)


def _mock_generate(state: AgentState) -> str:
    """
    템플릿 기반 초안 생성 (로컬 BitNet 출력 시뮬레이션).

    LLM 호출 없이 CPU 문자열 연산만으로 동작.
    """
    topics: dict[str, list[dict]] = {}
    for r in state.search_results:
        topics.setdefault(r.get("topic", "General"), []).append(r)

    parts: list[str] = []

    parts.append("# OpenAI Latest Trends Report")
    parts.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

    parts.append("## Executive Summary")
    parts.append(
        f"This report covers {len(topics)} key areas of OpenAI's recent "
        f"developments, synthesized from {len(state.search_results)} sources."
    )

    for topic, results in topics.items():
        parts.append(f"## {topic}")
        for r in results:
            parts.append(r["text"])
            parts.append(f"_Source: {r['source']}_\n")

    parts.append("## Conclusion")
    parts.append(
        "OpenAI continues advancing AI capabilities across multimodal models, "
        "reasoning, safety research, and enterprise adoption while expanding "
        "its developer platform and strategic partnerships."
    )

    return "\n\n".join(parts)
