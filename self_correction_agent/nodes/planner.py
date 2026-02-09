"""
Planner 노드: Critic 피드백을 반영한 검색 전략 수립.
"""
from __future__ import annotations

from ..domain.state import AgentState


def plan_search_queries(state: AgentState, base_query: str) -> list[str]:
    """
    검색 쿼리 목록을 생성한다.

    - 최초 실행: 광범위한 기본 쿼리 4개
    - 재시도: 기본 쿼리 + Critic이 지적한 누락 키워드별 타깃 쿼리 추가
    """
    queries = [
        base_query,
        "OpenAI GPT model multimodal",
        "OpenAI API developer platform",
        "OpenAI safety alignment",
    ]

    if state.critique and state.critique.missing_keywords:
        for kw in state.critique.missing_keywords:
            queries.append(f"OpenAI {kw} latest")
        state.log(
            f"Targeted queries added for: "
            f"{', '.join(state.critique.missing_keywords)}"
        )

    state.search_queries_used = queries
    state.log(f"Planned {len(queries)} search queries")
    return queries
