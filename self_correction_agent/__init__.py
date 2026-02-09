"""
Self-Correction Agent: Local-First AI Architecture.

Stack  : LanceDB (RAG) + Pydantic AI (Agent) + BitNet b1.58 (Sim)
Pattern: Planner -> Worker -> Critic -> Self-Healing Loop
"""
from .orchestrator import run_agent

__all__ = ["run_agent"]
