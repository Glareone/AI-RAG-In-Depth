"""FinancialReport — structured output model + persistence helpers."""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class Recommendation(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class FinancialReport(BaseModel):
    ticker: str
    period: str
    doc_type: str
    key_metrics: dict[str, float | str]
    risk_summary: list[str]
    investment_thesis: str
    recommendation: Recommendation


DIVIDER = "=" * 64


def save_report(report: FinancialReport, results_dir: str) -> Path:
    """Persist a FinancialReport as JSON. Returns the file path."""
    out_dir = Path(results_dir)
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{report.ticker}_{timestamp}.json"
    path.write_text(report.model_dump_json(indent=2))
    return path


def print_report(report: FinancialReport) -> None:
    """Pretty-print a FinancialReport to stdout."""
    print(f"\n{DIVIDER}")
    print(f"  FINANCIAL REPORT — {report.ticker}  ({report.period})")
    print(DIVIDER)
    print(f"  Document type  : {report.doc_type}")
    print(f"  Recommendation : {report.recommendation}")
    print(f"\n  Key Metrics:")
    for k, v in report.key_metrics.items():
        print(f"    {k}: {v}")
    print(f"\n  Risk Summary:")
    for risk in report.risk_summary:
        print(f"    • {risk}")
    print(f"\n  Investment Thesis:\n    {report.investment_thesis}")
    print(DIVIDER)
