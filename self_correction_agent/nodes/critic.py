"""
Critic 노드: 규칙 기반 품질 검증 엔진 (v2: 동적 eval_criteria 지원).

- eval_criteria 있음 (LLM 모드): 동적으로 생성된 토픽 목록으로 채점
- eval_criteria 없음 (mock 모드): REQUIRED_KEYWORDS fallback
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
    eval_criteria: Optional[List[str]] = None,
    threshold: float = PASS_THRESHOLD,
) -> CriticVerdict:
    """
    규칙 기반 Critic: 초안 품질 평가.

    점수 가중치:
      - 키워드 커버리지 : 40%  (eval_criteria 또는 REQUIRED_KEYWORDS)
      - 콘텐츠 길이     : 20%
      - 문서 구조 (##)  : 20%
      - 출처 인용       : 20%

    Hard gate: 키워드 커버리지 85% 미만이면 무조건 RETRY.
    """
    # eval_criteria(LLM 동적 기준) 또는 REQUIRED_KEYWORDS(fallback)
    kws: list = eval_criteria if eval_criteria else REQUIRED_KEYWORDS
    draft_lower = draft.lower()
    scores: list[float] = []
    missing: list[str] = []
    suggestions: list[str] = []

    # ── Check 1: 키워드 커버리지 (40%) ──
    # 각 항목은 str(단일) 또는 tuple(OR 그룹) — 하나라도 있으면 통과
    found = 0
    for kw in kws:
        if isinstance(kw, (list, tuple)):
            # OR 그룹: 하나라도 있으면 통과
            hit = any(variant.lower() in draft_lower for variant in kw)
            if hit:
                found += 1
            else:
                # 누락 표시는 첫 번째 변형(한국어 우선)으로 표기
                missing.append(kw[0])
        else:
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
