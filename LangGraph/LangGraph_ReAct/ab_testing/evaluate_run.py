"""Evaluate an existing A/B comparison dataset with precision / recall metrics.

Loads a dataset previously created by compare_runs.py (name prefix: aml-ab-<case_id>-),
then runs a Phoenix Experiment on it with four evaluators imported from evaluators.py:

  1. token_precision / token_recall / token_f1  — deterministic, bag-of-words overlap
  2. geval_recall    — LLM (Bedrock) judges % of reference facts semantically covered
  3. geval_precision — LLM (Bedrock) judges % of generated facts that are grounded
  4. risk_level_match / recommendation_match    — exact label comparison

Results appear in Phoenix → Experiments board linked to the comparison dataset,
so baseline and candidate rows are evaluated side-by-side.

Usage:
    # Auto-select the most recent aml-ab-CASE-2024-001-* dataset:
    uv run python ab_testing/evaluate_run.py --case-id CASE-2024-001

    # Target a specific dataset by name:
    uv run python ab_testing/evaluate_run.py --dataset-name aml-ab-CASE-2024-001-20260305T120714
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phoenix.client import Client

from src.config import Settings
from ab_testing.evaluators import build_evaluators, passthrough_task


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _find_latest_dataset(client: Client, case_id: str) -> str:
    """Return the name of the most recently created aml-ab-<case_id>-* dataset."""
    datasets = client.datasets.list()
    prefix = f"aml-ab-{case_id}-"
    matches = [d for d in datasets if d["name"].startswith(prefix)]
    if not matches:
        raise ValueError(
            f"No dataset found with prefix '{prefix}'.\n"
            "Run compare_runs.py first to create a baseline/candidate dataset."
        )
    return sorted(matches, key=lambda d: d["created_at"])[-1]["name"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(dataset_name: str, settings: Settings) -> None:
    client = Client(base_url=settings.phoenix_endpoint.rstrip("/"))

    print(f"\n[1/3] Loading dataset '{dataset_name}' …")
    dataset = client.datasets.get_dataset(dataset=dataset_name)
    print(f"  {len(dataset.examples)} examples found")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    print("[2/3] Running experiment (token metrics → G-eval → label match) …")
    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=passthrough_task,
        evaluators=build_evaluators(settings),
        experiment_name=f"eval-{dataset_name}-{timestamp}",
        experiment_description=(
            "Precision/recall evaluation against ground truth. "
            "Metrics: token F1, G-eval recall/precision (LLM judge), label exact-match."
        ),
        print_summary=True,
    )

    url = client.experiments.get_experiment_url(
        dataset_id=experiment["dataset_id"],
        experiment_id=experiment["experiment_id"],
    )
    print(f"\n[3/3] Done — open Phoenix → Experiments:\n  {url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an A/B comparison dataset against ground truth"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--case-id",
        default=os.getenv("AML_CASE_ID", "CASE-2024-001"),
        help="Auto-select the latest aml-ab-<case_id>-* dataset (default: CASE-2024-001)",
    )
    group.add_argument(
        "--dataset-name",
        help="Exact Phoenix dataset name to evaluate",
    )
    args = parser.parse_args()

    settings = Settings()
    client = Client(base_url=settings.phoenix_endpoint.rstrip("/"))

    dataset_name = args.dataset_name or _find_latest_dataset(client, args.case_id)
    print(f"Dataset: {dataset_name}")

    run_evaluation(dataset_name, settings)


if __name__ == "__main__":
    main()
