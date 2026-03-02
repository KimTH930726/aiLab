"""
Searcher 노드: 하이브리드 검색 실행 및 결과 집계.

v3: rag.hybrid_search() 사용 (alpha, distance_threshold 파라미터 추가).
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
    alpha: float = 0.7,
    distance_threshold: float = 1.5,
) -> list[dict[str, Any]]:
    """
    여러 쿼리로 하이브리드 검색을 실행하고 중복 제거된 결과를 반환한다.

    alpha: 0=순수 BM25, 1=순수 벡터
    distance_threshold: 벡터 거리 임계값 (낮을수록 엄격)
    """
    all_results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for sq in queries:
        hits = rag.hybrid_search(sq, top_k=top_k, alpha=alpha,
                                 distance_threshold=distance_threshold)
        for h in hits:
            key = h["text"][:80]
            if key not in seen:
                seen.add(key)
                all_results.append(h)
        state.log(f"  '{sq[:40]}' -> {len(hits)} hits")

    state.search_results = all_results
    state.log(f"Total unique results: {len(all_results)}")
    return all_results
