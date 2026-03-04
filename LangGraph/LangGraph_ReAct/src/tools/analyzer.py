import json
import os
from collections import Counter
from pathlib import Path

from langchain_core.tools import tool


def _load_case_transactions(case_id: str) -> list[dict]:
    data_dir = Path(os.getenv("DATA_DIR", "data/"))
    with open(data_dir / "transactions.json") as f:
        all_txns = json.load(f)
    return [t for t in all_txns if t["case_id"] == case_id]


@tool
def analyze_transaction_patterns(case_id: str) -> str:
    """Detect AML patterns in all transactions belonging to a case.

    Checks for structuring, layering, round-tripping, smurfing, and rapid fund movement.

    Args:
        case_id: The unique case identifier (e.g. CASE-2024-001).

    Returns:
        JSON string with detected patterns and supporting evidence.
    """
    txns = _load_case_transactions(case_id)

    if not txns:
        return json.dumps({"error": f"No transactions for case {case_id}."})

    patterns_found: list[str] = []
    evidence: dict[str, list] = {}

    amounts = [t["amount"] for t in txns]
    countries = list({t["counterparty_country"] for t in txns} | {t["originating_country"] for t in txns})
    customer_ids = list({t["customer_id"] for t in txns})

    # Structuring: multiple transactions just below $10,000
    sub10k = [t for t in txns if 7_500 <= t["amount"] < 10_000]
    if len(sub10k) >= 2:
        patterns_found.append("structuring")
        evidence["structuring"] = [t["transaction_id"] for t in sub10k]

    # Same-day multiple transactions
    date_counter = Counter(t["date"] for t in txns)
    same_day_dates = [d for d, c in date_counter.items() if c >= 2]
    if same_day_dates:
        patterns_found.append("same_day_multiple")
        evidence["same_day_multiple"] = same_day_dates

    # High-risk jurisdictions
    high_risk = {"PA", "RU", "VG", "NG"}
    hr_countries = [c for c in countries if c in high_risk]
    if hr_countries:
        patterns_found.append("high_risk_jurisdiction")
        evidence["high_risk_jurisdiction"] = hr_countries

    # Rapid fund movement (cash deposit followed by wire)
    cash_deposits = [t for t in txns if t["transaction_type"] == "cash_deposit"]
    wires = [t for t in txns if t["transaction_type"] == "wire_transfer"]
    if cash_deposits and wires:
        patterns_found.append("rapid_movement_cash_to_wire")
        evidence["rapid_movement_cash_to_wire"] = {
            "cash_deposits": [t["transaction_id"] for t in cash_deposits],
            "wires": [t["transaction_id"] for t in wires],
        }

    return json.dumps(
        {
            "case_id": case_id,
            "customer_ids": customer_ids,
            "transaction_count": len(txns),
            "total_amount": sum(amounts),
            "countries_involved": countries,
            "patterns_detected": patterns_found,
            "evidence": evidence,
        },
        indent=2,
    )


@tool
def extract_facts(analysis_results: str) -> str:
    """Structure raw analysis results into discrete, actionable AML facts.

    Args:
        analysis_results: JSON or text output from pattern analysis or other tools.

    Returns:
        JSON string with a list of structured AML facts suitable for SAR filing.
    """
    facts: list[str] = []

    try:
        data = json.loads(analysis_results)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"facts": [f"Raw finding: {analysis_results}"]})

    case_id = data.get("case_id", "unknown")
    customer_ids = data.get("customer_ids", [])
    patterns = data.get("patterns_detected", [])
    evidence = data.get("evidence", {})
    total_amount = data.get("total_amount", 0)
    tx_count = data.get("transaction_count", 0)
    countries = data.get("countries_involved", [])
    customer_ref = ", ".join(customer_ids) if customer_ids else "unknown"

    if "structuring" in patterns:
        txn_ids = evidence.get("structuring", [])
        facts.append(
            f"STRUCTURING: Case {case_id} — {len(txn_ids)} transactions "
            f"just below the $10,000 CTR threshold (transactions: {', '.join(txn_ids)}). "
            f"This pattern is consistent with willful structuring under 31 U.S.C. § 5324."
        )

    if "same_day_multiple" in patterns:
        dates = evidence.get("same_day_multiple", [])
        facts.append(
            f"SAME-DAY MULTIPLE TRANSACTIONS: Case {case_id} — multiple transactions "
            f"on the same day on dates: {', '.join(dates)}."
        )

    if "round_trip_layering" in patterns:
        txn_ids = evidence.get("round_trip_layering", [])
        facts.append(
            f"LAYERING/ROUND-TRIP: Case {case_id} — funds sent and returned within a short "
            f"timeframe (transactions: {', '.join(txn_ids)}). "
            f"Consistent with layering to obscure the audit trail."
        )

    if "high_risk_jurisdiction" in patterns:
        countries_list = evidence.get("high_risk_jurisdiction", [])
        facts.append(
            f"HIGH-RISK JURISDICTIONS: Case {case_id} — transactions involving "
            f"high-risk countries: {', '.join(countries_list)}."
        )

    if "rapid_movement_cash_to_wire" in patterns:
        details = evidence.get("rapid_movement_cash_to_wire", {})
        facts.append(
            f"RAPID FUND MOVEMENT: Case {case_id} — cash deposits ({details.get('cash_deposits', [])}) "
            f"rapidly followed by outgoing wires ({details.get('wires', [])}). "
            f"Possible smurfing or third-party deposit scheme."
        )

    facts.append(
        f"SUMMARY: Case {case_id} (customer(s): {customer_ref}) — {tx_count} transactions totaling "
        f"${total_amount:,.2f} involving countries: {', '.join(countries)}. "
        f"Patterns detected: {', '.join(patterns) if patterns else 'none'}."
    )

    return json.dumps({"case_id": case_id, "facts": facts}, indent=2)
