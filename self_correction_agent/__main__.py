"""
CLI 진입점: python -m self_correction_agent

사용법:
    python -m self_correction_agent
    python -m self_correction_agent --model ollama:llama3.2
    AGENT_MODEL=openai:gpt-4o-mini python -m self_correction_agent
"""
from __future__ import annotations

import argparse
import os
import textwrap

from .orchestrator import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-Correction Agent: Local-First AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m self_correction_agent
              python -m self_correction_agent --model ollama:llama3.2
              AGENT_MODEL=openai:gpt-4o-mini python -m self_correction_agent
        """),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AGENT_MODEL"),
        help="Pydantic AI model (e.g. 'ollama:llama3.2', 'openai:gpt-4o-mini')",
    )
    parser.add_argument(
        "--query",
        default="OpenAI 최신 동향 리포트를 작성해주세요",
        help="User query for the agent",
    )
    parser.add_argument(
        "--db-path",
        default="/tmp/lancedb_self_correction_agent",
        help="Path for LanceDB storage",
    )

    args = parser.parse_args()

    report = run_agent(
        query=args.query,
        model_name=args.model,
        db_path=args.db_path,
    )

    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60 + "\n")
    print(report)


if __name__ == "__main__":
    main()
