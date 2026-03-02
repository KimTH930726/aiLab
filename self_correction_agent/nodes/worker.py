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

    mode="mock"       : 템플릿 기반 (LLM 불필요)
    mode="pydantic-ai": Pydantic AI Agent가 LanceDB Tool로 자율 작성
    """
    if pydantic_agent is not None:
        prompt = _build_worker_prompt(state)
        if mode == "pydantic-ai" and rag is not None:
            # tools 지원 모델: Agent가 직접 LanceDB 검색 수행
            result = pydantic_agent.run_sync(prompt, deps=rag)
        else:
            # tools 미지원 모델: 검색 결과를 프롬프트에 임베드
            result = pydantic_agent.run_sync(prompt)
        # pydantic-ai 버전별 반환 속성 호환 (>=0.1.0: output, 구버전: data)
        return result.output if hasattr(result, "output") else result.data

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

    user_query = state.query or "AI 최신 동향 리포트를 작성해주세요"

    return textwrap.dedent(f"""\
        사용자 요청: {user_query}

        아래 지식베이스 내용을 활용하여 위 요청에 맞는 리포트를 작성하세요:

        {context}
        {feedback}

        작성 규칙:
        - 반드시 한국어(Korean)로 작성할 것
        - 각 주요 토픽에 ## 헤더를 사용할 것
        - 출처는 "출처: ..." 형식으로 표기할 것
        - 검색 결과의 모든 토픽을 포함할 것
        - 사실에 기반하여 상세하게 작성할 것
    """)


def _mock_generate(state: AgentState) -> str:
    """
    템플릿 기반 초안 생성. LLM 호출 없이 CPU 문자열 연산만으로 동작.
    """
    topics: dict[str, list[dict]] = {}
    for r in state.search_results:
        topics.setdefault(r.get("topic", "General"), []).append(r)

    parts: list[str] = []

    title = state.query or "AI 최신 동향 리포트"
    parts.append(f"# {title}")
    parts.append(f"_생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

    parts.append("## 요약")
    parts.append(
        f"이 리포트는 {len(topics)}개 핵심 분야를 다루며, "
        f"{len(state.search_results)}개 출처에서 종합하였습니다."
    )

    for topic, results in topics.items():
        parts.append(f"## {topic}")
        for r in results:
            parts.append(r["text"])
            parts.append(f"_출처: {r['source']}_\n")

    parts.append("## 결론")
    parts.append(
        "AI 분야는 멀티모달 모델, 추론 능력, 안전성 연구, 기업 도입 확산 등 "
        "다방면에서 빠르게 발전하고 있으며, 개발자 플랫폼과 전략적 파트너십도 "
        "지속적으로 확대되고 있습니다."
    )

    return "\n\n".join(parts)
