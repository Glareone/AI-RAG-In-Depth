"""Entry point — Financial Document Analyst with Human-in-the-Loop.

Run:
    python -m src.main
    python -m src.main --ticker TFI
    python -m src.main --ticker ACM --doc-type earnings_call
"""

import argparse
import asyncio
import os

from src.agent.runner import run
from src.config.settings import Settings
from src.telemetry.setup import init_telemetry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Financial Document Analyst — Human-in-the-Loop")
    parser.add_argument("--ticker", default="ACM", help="Company ticker (e.g. ACM, TFI)")
    parser.add_argument(
        "--doc-type",
        default="annual_report",
        choices=["annual_report", "earnings_call"],
        help="Document type to analyse",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = Settings()
    os.environ.setdefault("DATA_DIR", settings.data_dir)
    init_telemetry(settings)
    asyncio.run(run(ticker=args.ticker, doc_type=args.doc_type, settings=settings))


if __name__ == "__main__":
    main()
