"""Graph runner — interrupt/resume loop and CLI interaction."""

import json

from langgraph.types import Command

from src.agent.graph import build_graph
from src.agent.output import FinancialReport, print_report, save_report
from src.config.settings import Settings

DIVIDER = "=" * 64


def _human_interrupt(label: str, payload: dict) -> str:
    """Print interrupt payload and collect human feedback from stdin.

    Returns the typed response, or 'approved' if the user pressed Enter.
    """
    print(f"\n{DIVIDER}")
    print(f"  INTERRUPT — {label}")
    print(DIVIDER)
    print(json.dumps(payload, indent=2, default=str))
    print(DIVIDER)
    print("  Enter your feedback (or press Enter to approve as-is):")
    response = input("  > ").strip()
    return response or "approved"


async def run(ticker: str, doc_type: str, settings: Settings) -> FinancialReport | None:
    """Execute the full analyst graph with two human-in-the-loop interrupts.

    Interrupt #1 — human reviews the proposed analysis plan.
    Interrupt #2 — human reviews identified risk flags before finalising.

    Args:
        ticker:   Company ticker symbol (e.g. 'ACM').
        doc_type: Document type ('annual_report' | 'earnings_call').
        settings: Application settings.

    Returns:
        FinancialReport if successful, None otherwise.
    """
    thread_config = {"configurable": {"thread_id": f"{ticker}-{doc_type}"}}
    initial_input = {
        "messages": [],
        "document_id": None,
        "ticker": ticker,
        "doc_type": doc_type,
    }

    print(f"\n{DIVIDER}")
    print(f"  Financial Document Analyst")
    print(f"  Ticker: {ticker}  |  Document: {doc_type}")
    print(DIVIDER)

    async with build_graph(settings) as graph:
        # ── Run 1: start → interrupt #1 (plan review) ──────────────────────
        print("\n[agent] Planning analysis …")
        async for chunk in graph.astream(initial_input, thread_config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                human_plan_feedback = _human_interrupt(
                    "Review Analysis Plan", chunk["__interrupt__"][0].value
                )
                break

        # ── Run 2: resume → interrupt #2 (risk review) ─────────────────────
        print("\n[agent] Extracting data and computing ratios …")
        async for chunk in graph.astream(
            Command(resume=human_plan_feedback), thread_config, stream_mode="updates"
        ):
            if "__interrupt__" in chunk:
                human_risk_feedback = _human_interrupt(
                    "Review Risk Flags", chunk["__interrupt__"][0].value
                )
                break

        # ── Run 3: resume → final report ────────────────────────────────────
        print("\n[agent] Formatting final report …")
        final_state = None
        async for chunk in graph.astream(
            Command(resume=human_risk_feedback), thread_config, stream_mode="values"
        ):
            final_state = chunk

    report: FinancialReport | None = final_state.get("report") if final_state else None

    if report:
        print_report(report)
        path = save_report(report, settings.results_dir)
        print(f"\n[saved] {path}")
    else:
        print("\n[error] Agent did not return a structured report.")

    return report
