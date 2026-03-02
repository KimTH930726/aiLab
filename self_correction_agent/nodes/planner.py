"""
Planner 노드: 사용자 쿼리 기반 검색 전략 수립.

v2: LLM으로 쿼리 확장 + 동적 채점 기준 생성.
LLM 없으면 heuristic fallback.
"""
from __future__ import annotations

from typing import Optional

from ..domain.state import AgentState
from .llm_helpers import expand_query_with_llm, generate_eval_criteria

# 토픽 키워드 → 보조 쿼리 (LLM 없을 때 fallback)
_TOPIC_ALIASES: dict[str, list[str]] = {
    "openai":   ["OpenAI GPT multimodal", "OpenAI API platform", "OpenAI safety alignment"],
    "gpt":      ["GPT model capabilities", "GPT API usage", "GPT fine-tuning"],
    "llm":      ["large language model capabilities", "LLM API platform", "LLM safety research"],
    "ai":       ["AI model multimodal", "AI developer platform", "AI safety alignment"],
    "모델":      ["AI model latest", "language model capabilities", "model API platform"],
    "언어모델":  ["language model capabilities", "LLM API platform", "AI safety"],
    "sora":     ["OpenAI Sora video generation", "text-to-video AI", "diffusion transformer"],
    "dalle":    ["OpenAI DALL-E image generation", "multimodal AI", "image model"],
    "whisper":  ["OpenAI Whisper speech recognition", "audio AI model", "voice processing"],
    "chatgpt":  ["ChatGPT enterprise", "ChatGPT API", "ChatGPT features"],
    "exaone":   ["EXAONE 3.5 Korean LLM", "LG AI Research language model", "efficient Korean model"],
    "o1":       ["OpenAI o1 reasoning", "chain of thought model", "AI reasoning benchmark"],
}

_DEFAULT_SUPPLEMENTS = [
    "AI model latest capabilities",
    "AI API developer platform",
    "AI safety alignment research",
]


def _heuristic_supplements(query: str) -> list[str]:
    """쿼리 키워드 감지 → 보조 쿼리 반환 (LLM 없을 때 fallback)."""
    ql = query.lower()
    for kw, supplements in _TOPIC_ALIASES.items():
        if kw in ql:
            return supplements
    return _DEFAULT_SUPPLEMENTS


def plan_search_queries(
    state: AgentState,
    base_query: str,
    model_name: Optional[str] = None,
) -> list[str]:
    """
    검색 쿼리 목록을 생성한다.

    - 첫 시도 + LLM 있음: LLM으로 쿼리 확장 + eval_criteria 생성
    - 첫 시도 + LLM 없음: heuristic 보조 쿼리 3개
    - 재시도: 기존 쿼리 + Critic 누락 키워드 타깃 쿼리 추가
    """
    # ── 재시도: Critic 피드백 기반 보충 쿼리만 추가 ──────────────────
    if state.retry_count > 0 and state.critique and state.critique.missing_keywords:
        existing = list(state.search_queries_used)
        for kw in state.critique.missing_keywords:
            existing.append(f"{kw} latest")
        state.search_queries_used = existing
        state.log(
            f"Targeted queries added for: "
            f"{', '.join(state.critique.missing_keywords)}"
        )
        state.log(f"Total queries: {len(existing)}")
        return existing

    # ── 첫 시도: LLM 쿼리 확장 시도 ─────────────────────────────────
    if model_name:
        llm_queries = expand_query_with_llm(base_query, model_name)
        if llm_queries:
            queries = [base_query] + [q for q in llm_queries if q != base_query]
            state.log(f"LLM expanded to {len(queries)} queries")

            # 동적 채점 기준 생성 (첫 시도에만)
            criteria = generate_eval_criteria(base_query, model_name)
            if criteria:
                state.eval_criteria = criteria
                state.log(f"LLM generated {len(criteria)} eval criteria")
            else:
                state.log("Eval criteria generation failed — using fallback")

            state.search_queries_used = queries
            state.log(f"Planned {len(queries)} search queries")
            return queries

        state.log("LLM query expansion failed — using heuristic fallback")

    # ── Fallback: heuristic 보조 쿼리 ────────────────────────────────
    queries = [base_query] + _heuristic_supplements(base_query)
    state.search_queries_used = queries
    state.log(f"Planned {len(queries)} search queries")
    return queries
