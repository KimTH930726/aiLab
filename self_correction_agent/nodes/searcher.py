"""
Searcher 노드: LanceDB 벡터 검색 실행 및 결과 집계.
"""
from __future__ import annotations

from typing import Any

from ..domain.state import AgentState
from ..infra.vectordb import LocalRAG


def execute_search(
    rag: LocalRAG,
    queries: list[str],
    state: AgentState,
    top_k: int = 2,
) -> list[dict[str, Any]]:
    """
    여러 쿼리로 LanceDB를 검색하고 중복 제거된 결과를 반환한다.
    """
    all_results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for sq in queries:
        hits = rag.search(sq, top_k=top_k)
        for h in hits:
            key = h["text"][:80]
            if key not in seen:
                seen.add(key)
                all_results.append(h)
        state.log(f"  '{sq[:40]}' -> {len(hits)} hits")

    state.search_results = all_results
    state.log(f"Total unique results: {len(all_results)}")
    return all_results
