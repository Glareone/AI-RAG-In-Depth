"""A/B comparison of AML prompt versions using Arize Phoenix.

Workflow:
  1. Fetch the most recent baseline span for the case from Phoenix.
  2. Edit src/agent/prompts.yaml with your updated prompt (manual step).
  3. Run this script — it will:
       a) re-run the agent with the new prompt
       b) fetch the new (candidate) span from Phoenix
       c) upload a 2-row dataset (baseline + candidate) to Phoenix
       d) run a Phoenix Experiment on that dataset with precision/recall evaluators

Usage:
    uv run python ab_testing/compare_runs.py --case-id CASE-2024-001
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from phoenix.client import Client
from phoenix.client.resources.spans import SpanQuery

from src.config import Settings
from ab_testing.evaluators import build_evaluators, passthrough_task

LANGGRAPH_SPAN_NAME = "LangGraph"


# ---------------------------------------------------------------------------
# Phoenix span helpers
# ---------------------------------------------------------------------------

def _make_phoenix_client(settings: Settings) -> Client:
    return Client(base_url=settings.phoenix_endpoint.rstrip("/"))


def _fetch_span_for_case(
    client: Client,
    project_name: str,
    case_id: str,
    *,
    after_timestamp: datetime | None = None,
) -> "pd.Series | None":
    query = (
        SpanQuery()
        .where(f"name == '{LANGGRAPH_SPAN_NAME}'")
        .select("input.value", "output.value", "start_time")
    )
    df = client.spans.get_spans_dataframe(
        query=query,
        project_name=project_name,
        root_spans_only=True,
        start_time=after_timestamp,
        limit=1000,
    )
    if df is None or df.empty:
        return None

    df = df[df["input.value"].str.contains(case_id, na=False)]
    if df.empty:
        return None

    return df.sort_values("start_time", ascending=False).iloc[0]


def _extract_span_data(span: "pd.Series", case_id: str) -> dict:
    """Pull AMLReport fields and raw input out of a root LangGraph span row."""
    raw_output = span.get("output.value", "{}")
    raw_input  = span.get("input.value", "{}")

    try:
        output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
    except json.JSONDecodeError:
        output = {}

    report = output.get("report") or {}
    return {
        "case_id":          report.get("case_id") or case_id,
        "risk_level":       report.get("risk_level", ""),
        "recommendation":   report.get("recommendation", ""),
        "facts":            "\n".join(report.get("facts") or []),
        "analyst_summary":  report.get("analyst_summary", ""),
        "raw_input":        raw_input,
    }


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def _run_agent(case_id: str) -> None:
    from src.main import run
    run(case_id)


# ---------------------------------------------------------------------------
# Dataset + experiment
# ---------------------------------------------------------------------------

def _create_dataset_and_run_experiment(
    client: Client,
    case_id: str,
    baseline: dict,
    candidate: dict,
    settings: Settings,
) -> None:
    rows = [
        {"prompt_version": "baseline",  **baseline},
        {"prompt_version": "candidate", **candidate},
    ]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dataset_name = f"aml-ab-{case_id}-{timestamp}"

    print(f"\n[5/5] Creating dataset '{dataset_name}' and running experiment …")
    dataset = client.datasets.create_dataset(
        name=dataset_name,
        dataframe=pd.DataFrame(rows),
        input_keys=["case_id", "prompt_version", "raw_input"],
        output_keys=["facts", "risk_level", "recommendation", "analyst_summary"],
    )

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=passthrough_task,
        evaluators=build_evaluators(settings),
        experiment_name=f"ab-eval-{case_id}-{timestamp}",
        experiment_description=(
            f"A/B evaluation of prompt versions for {case_id}. "
            "Metrics: token F1, G-eval recall/precision (LLM judge), label exact-match."
        ),
        print_summary=True,
    )

    url = client.experiments.get_experiment_url(
        dataset_id=experiment["dataset_id"],
        experiment_id=experiment["experiment_id"],
    )
    print(f"\n[done] Open Phoenix → Experiments:\n  {url}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare(case_id: str) -> None:
    settings = Settings()
    os.environ.setdefault("DATA_DIR", settings.data_dir)
    client = _make_phoenix_client(settings)

    # Step 1: baseline span
    print(f"\n[1/5] Fetching baseline span for {case_id} …")
    baseline_span = _fetch_span_for_case(client, settings.phoenix_project_name, case_id)
    if baseline_span is None:
        print(
            "  ERROR: No baseline span found.\n"
            "  Run 'uv run python -m src.main' first to produce a baseline trace."
        )
        sys.exit(1)
    print(f"  start_time: {baseline_span.get('start_time')}")

    # Step 2: extract baseline
    print("\n[2/5] Extracting baseline data …")
    baseline_data = _extract_span_data(baseline_span, case_id)
    print(f"  risk_level={baseline_data['risk_level']}  "
          f"recommendation={baseline_data['recommendation']}  "
          f"facts={len(baseline_data['facts'].splitlines())}")

    # Step 3: re-run agent with updated prompt
    print("\n[3/5] Re-running agent with current prompts.yaml …")
    run_started_at = datetime.now(timezone.utc)
    _run_agent(case_id)

    print("  Waiting for Phoenix to ingest trace …")
    time.sleep(5)

    # Step 4: candidate span
    print("\n[4/5] Fetching candidate span …")
    candidate_span = _fetch_span_for_case(
        client, settings.phoenix_project_name, case_id, after_timestamp=run_started_at
    )
    if candidate_span is None:
        print("  ERROR: No new span found after re-run. Check Phoenix ingestion.")
        sys.exit(1)
    print(f"  start_time: {candidate_span.get('start_time')}")

    # Step 5: dataset + experiment
    candidate_data = _extract_span_data(candidate_span, case_id)
    _create_dataset_and_run_experiment(client, case_id, baseline_data, candidate_data, settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B test AML prompt versions")
    parser.add_argument(
        "--case-id",
        default=os.getenv("AML_CASE_ID", "CASE-2024-001"),
        help="AML case identifier (default: CASE-2024-001)",
    )
    args = parser.parse_args()
    compare(args.case_id)
