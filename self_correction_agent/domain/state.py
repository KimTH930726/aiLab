"""
도메인 핵심 모델: 에이전트 상태 머신.

순수 도메인 계층 — pydantic 외 외부 의존성 없음.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Phase(str, Enum):
    """에이전트 라이프사이클 단계."""
    PLANNING   = "planning"
    SEARCHING  = "searching"
    DRAFTING   = "drafting"
    CRITIQUING = "critiquing"
    DONE       = "done"
    FAILED     = "failed"


PHASE_ICONS = {
    Phase.PLANNING:   "[PLAN]",
    Phase.SEARCHING:  "[SEARCH]",
    Phase.DRAFTING:   "[DRAFT]",
    Phase.CRITIQUING: "[CRITIC]",
    Phase.DONE:       "[DONE]",
    Phase.FAILED:     "[FAIL]",
}


class CriticVerdict(BaseModel):
    """Critic 노드의 구조화된 평가 결과."""
    passed: bool = False
    score: float = 0.0
    feedback: str = ""
    missing_keywords: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """
    에이전트 실행 전체 라이프사이클을 추적하는 단일 진실 공급원(SSOT).

    모든 노드(Planner, Searcher, Worker, Critic)가 이 객체를 공유하며,
    phase 전이를 통해 상태 머신이 구동된다.
    """
    phase: Phase = Phase.PLANNING
    query: str = ""

    # RAG
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    search_queries_used: List[str] = Field(default_factory=list)

    # Worker
    draft: str = ""

    # Critic
    critique: Optional[CriticVerdict] = None

    # Self-healing
    retry_count: int = 0
    max_retries: int = 3
    history: List[str] = Field(default_factory=list)

    # Final
    final_result: Optional[str] = None

    @property
    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        icon = PHASE_ICONS.get(self.phase, "[???]")
        entry = f"[{ts}] {icon} {msg}"
        self.history.append(entry)
        print(f"  {entry}")
