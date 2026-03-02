"""
Self-Correction Agent: Local-First AI Architecture.

Stack  : LanceDB (RAG) + sentence-transformers + BM25 + Pydantic AI (Agent) + Ollama
Pattern: Planner -> Worker -> Critic -> Self-Healing Loop
"""
from .orchestrator import run_agent

__all__ = ["run_agent"]
