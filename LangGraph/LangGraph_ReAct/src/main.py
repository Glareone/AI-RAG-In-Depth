import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.agent import AMLReport, build_graph
from src.config import Settings
from src.telemetry import init_telemetry

DEFAULT_CASE_ID = os.getenv("AML_CASE_ID", "CASE-2024-001")
RESULTS_DIR = Path("results")


def _save_report(report: AMLReport) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"{report.case_id}_{timestamp}.json"
    output_path.write_text(report.model_dump_json(indent=2))
    return output_path


def run(case_id: str = DEFAULT_CASE_ID) -> AMLReport | None:
    settings = Settings()
    os.environ.setdefault("DATA_DIR", settings.data_dir)

    init_telemetry(settings)
    graph = build_graph(settings)

    print(f"\n{'='*60}")
    print(f"AML ReAct Agent — analysing case: {case_id}")
    print(f"{'='*60}\n")

    result = graph.invoke(
        {"messages": [HumanMessage(content=f"Analyse AML case {case_id}.")], "case_id": case_id},
        config={"configurable": {"thread_id": case_id}},
    )

    report: AMLReport | None = result.get("report")

    print("\n" + "="*60)
    if report:
        print(f"CASE:           {report.case_id}")
        print(f"RISK LEVEL:     {report.risk_level}")
        print(f"RECOMMENDATION: {report.recommendation}")
        print(f"RATIONALE:      {report.recommendation_rationale}")
        print("\nFACTS:")
        for i, fact in enumerate(report.facts, 1):
            print(f"  [{i}] {fact}")
        print(f"\nSUMMARY:\n{report.analyst_summary}")
        output_path = _save_report(report)
        print(f"\n[saved] {output_path}")
    else:
        print("Formatter did not return a structured report.")

    print("="*60)
    return report


if __name__ == "__main__":
    run()
