"""
Critic 노드: 규칙 기반 품질 검증 엔진.

CPU 효율을 위해 LLM 호출 없이 동작한다.
LLM 기반 Critic으로 확장 시, Pydantic AI Agent에
CriticVerdict를 result_type으로 지정하면 된다.
"""
from __future__ import annotations

from typing import List, Optional

from ..domain.state import CriticVerdict
from ..domain.constants import (
    REQUIRED_KEYWORDS,
    PASS_THRESHOLD,
    MIN_LENGTH,
    MIN_KEYWORD_RATIO,
)


def critic_evaluate(
    draft: str,
    required_keywords: Optional[List[str]] = None,
    threshold: float = PASS_THRESHOLD,
) -> CriticVerdict:
    """
    규칙 기반 Critic: 초안 품질 평가.

    점수 가중치:
      - 키워드 커버리지 : 40%
      - 콘텐츠 길이     : 20%
      - 문서 구조 (##)  : 20%
      - 출처 인용       : 20%

    Hard gate: 키워드 커버리지 85% 미만이면 무조건 RETRY.
    """
    kws = required_keywords or REQUIRED_KEYWORDS
    draft_lower = draft.lower()
    scores: list[float] = []
    missing: list[str] = []
    suggestions: list[str] = []

    # ── Check 1: 키워드 커버리지 (40%) ──
    found = 0
    for kw in kws:
        if kw.lower() in draft_lower:
            found += 1
        else:
            missing.append(kw)

    kw_ratio = found / len(kws) if kws else 1.0
    scores.append(kw_ratio * 0.4)

    if missing:
        suggestions.append(f"Search for and include content about: {', '.join(missing)}")

    # ── Check 2: 콘텐츠 길이 (20%) ──
    length_score = min(len(draft) / MIN_LENGTH, 1.0)
    scores.append(length_score * 0.2)

    if len(draft) < MIN_LENGTH:
        suggestions.append(f"Expand draft ({len(draft)} chars, need {MIN_LENGTH}+)")

    # ── Check 3: 문서 구조 — 섹션 헤더 (20%) ──
    headers = draft.count("## ")
    struct_score = min(headers / 4, 1.0)
    scores.append(struct_score * 0.2)

    if headers < 4:
        suggestions.append(f"Add more sections ({headers} found, need 4+)")

    # ── Check 4: 출처 인용 (20%) ──
    cites = draft_lower.count("source:")
    cite_score = min(cites / 3, 1.0)
    scores.append(cite_score * 0.2)

    if cites < 3:
        suggestions.append(f"Add source citations ({cites} found, need 3+)")

    # ── 최종 판정 ──
    total = sum(scores)

    keyword_gate = kw_ratio >= MIN_KEYWORD_RATIO
    passed = total >= threshold and keyword_gate

    if passed:
        feedback = f"Score {total:.2f}/{threshold} — PASSED"
    else:
        lines = [f"Score {total:.2f}/{threshold} — RETRY NEEDED"]
        lines.append(f"Keywords: {found}/{len(kws)} (need {MIN_KEYWORD_RATIO:.0%}+)")
        if not keyword_gate:
            lines.append(
                f"HARD GATE FAILED: keyword coverage {kw_ratio:.0%} < {MIN_KEYWORD_RATIO:.0%}"
            )
        if missing:
            lines.append(f"Missing: {', '.join(missing)}")
        lines.extend(suggestions)
        feedback = "\n".join(lines)

    return CriticVerdict(
        passed=passed,
        score=round(total, 3),
        feedback=feedback,
        missing_keywords=missing,
        suggestions=suggestions,
    )
