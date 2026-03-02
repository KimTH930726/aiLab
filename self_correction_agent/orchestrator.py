"""
오케스트레이터: Phase 기반 상태 머신 메인 루프.

  PLANNING -> SEARCHING -> DRAFTING -> CRITIQUING
      ^                                    |
      |______ (Self-Healing Retry) ________|
                                           |
                                 DONE  or  FAILED

모든 의존성(infra, nodes)을 여기서 조립한다.
"""
from __future__ import annotations

from typing import Callable, Optional

from .domain.state import Phase, AgentState
from .infra.vectordb import LocalRAG
from .infra.settings import load_settings
from .knowledge import KNOWLEDGE_BASE
from .nodes.planner import plan_search_queries
from .nodes.searcher import execute_search
from .nodes.worker import worker_generate_draft
from .nodes.critic import critic_evaluate
from .agent import create_pydantic_agent


def run_agent(
    query: str = "OpenAI 최신 동향 리포트를 작성해주세요",
    model_name: Optional[str] = None,
    db_path: str = "/tmp/lancedb_self_correction_agent",
    verbose: bool = True,
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
    """
    Self-Correction Agent 메인 진입점.
    Plan -> Search -> Draft -> Critique -> (Retry | Done)
    """

    # ── Setup ──
    if verbose:
        print("\n" + "=" * 60)
        print("  Self-Correction Agent | Local-First AI")
        print("  LanceDB (RAG) + Pydantic AI + Ollama EXAONE 3.5")
        print("=" * 60)

    mode = "mock"
    pydantic_agent = None
    if model_name:
        pydantic_agent, use_tools = create_pydantic_agent(model_name)
        if pydantic_agent:
            mode = "pydantic-ai" if use_tools else "pydantic-ai-notool"

    if verbose:
        print(f"  Mode   : {mode}" + (f" ({model_name})" if model_name else ""))

    # 검색 설정 로딩 (settings.json)
    settings = load_settings(db_path)
    alpha = float(settings.get("alpha", 0.7))
    distance_threshold = float(settings.get("distance_threshold", 1.5))
    top_k = int(settings.get("top_k", 2))

    rag = LocalRAG(db_path)
    n = rag.initialize(KNOWLEDGE_BASE)  # v3: 차원 불일치 시 자동 마이그레이션

    if verbose:
        print(f"  LanceDB: {n} chunks indexed")
        print(f"  Search : alpha={alpha}, threshold={distance_threshold}, top_k={top_k}")
        print(f"  Query  : {query}")
        print("=" * 60)

    state = AgentState(query=query)
    state.on_event = on_event  # 웹 UI 콜백 주입

    # ═══════════════════════════════════════════════════
    #  MAIN LOOP — Phase 기반 상태 머신
    # ═══════════════════════════════════════════════════

    while state.phase not in (Phase.DONE, Phase.FAILED):

        attempt = state.retry_count + 1
        max_attempts = state.max_retries + 1

        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  Phase: {state.phase.value}  |  Attempt: {attempt}/{max_attempts}")
            print(f"{'─' * 50}")

        if on_event:
            on_event({
                "type": "phase",
                "phase": state.phase.value,
                "attempt": attempt,
                "max_attempts": max_attempts,
            })

        # ── PLANNING ──
        if state.phase == Phase.PLANNING:
            state.log("Analyzing query, planning search strategy...")
            plan_search_queries(state, query, model_name=model_name)
            state.phase = Phase.SEARCHING

        # ── SEARCHING ──
        elif state.phase == Phase.SEARCHING:
            state.log("Querying LanceDB knowledge base (hybrid search)...")
            results = execute_search(
                rag, state.search_queries_used, state,
                top_k=top_k,
                alpha=alpha,
                distance_threshold=distance_threshold,
            )

            if not results:
                state.log("No results found. Cannot proceed.")
                state.phase = Phase.FAILED
            else:
                state.phase = Phase.DRAFTING

        # ── DRAFTING ──
        elif state.phase == Phase.DRAFTING:
            state.log("Worker generating report draft...")
            state.draft = worker_generate_draft(
                state=state,
                mode=mode,
                pydantic_agent=pydantic_agent,
                rag=rag,
            )
            char_count = len(state.draft)
            section_count = state.draft.count("## ")
            state.log(f"Draft: {char_count} chars, {section_count} sections")
            state.phase = Phase.CRITIQUING

        # ── CRITIQUING ──
        elif state.phase == Phase.CRITIQUING:
            state.log("Critic evaluating draft quality...")
            verdict = critic_evaluate(state.draft, eval_criteria=state.eval_criteria)
            state.critique = verdict
            state.log(f"Score: {verdict.score:.2f} | Passed: {verdict.passed}")

            if verdict.passed:
                state.final_result = state.draft
                state.phase = Phase.DONE
                state.log("Draft APPROVED by Critic.")

            elif state.can_retry:
                state.retry_count += 1
                state.log(
                    f"RETRY #{state.retry_count} triggered — "
                    f"{verdict.feedback.splitlines()[0]}"
                )
                if verdict.missing_keywords:
                    state.log(f"  Missing: {', '.join(verdict.missing_keywords)}")
                state.phase = Phase.PLANNING

            else:
                state.log(
                    f"Max retries ({state.max_retries}) exceeded. "
                    f"Returning best-effort draft."
                )
                state.final_result = state.draft
                state.phase = Phase.FAILED

    # ═══════════════════════════════════════════════════
    #  RESULT SUMMARY
    # ═══════════════════════════════════════════════════

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Result  : {state.phase.value.upper()}")
        print(f"  Retries : {state.retry_count}/{state.max_retries}")
        if state.critique:
            print(f"  Score   : {state.critique.score:.2f}")
        print(f"{'=' * 60}")

        print("\n  --- Execution Trace ---")
        for entry in state.history:
            print(f"  {entry}")
        print()

    final = state.final_result or "Agent failed to produce a result."

    if on_event:
        on_event({
            "type": "result",
            "status": state.phase.value,
            "markdown": final,
            "score": state.critique.score if state.critique else 0.0,
            "retries": state.retry_count,
        })

    return final
