import json
from datetime import datetime, timedelta

from langchain_core.tools import tool

# Country risk tiers: higher score = higher risk
_COUNTRY_RISK: dict[str, float] = {
    "US": 0.1,
    "GB": 0.1,
    "DE": 0.1,
    "FR": 0.1,
    "CA": 0.1,
    "AU": 0.1,
    "CH": 0.5,
    "PA": 0.8,
    "MX": 0.6,
    "RU": 0.9,
    "CN": 0.5,
    "AE": 0.6,
    "NG": 0.8,
    "VG": 0.9,  # British Virgin Islands
}
_DEFAULT_COUNTRY_RISK = 0.5


@tool
def calculate_risk_score(amounts: list[float], countries: list[str]) -> str:
    """Calculate a composite AML risk score from transaction amounts and countries.

    Combines transaction volume risk with geographic risk.

    Args:
        amounts: List of transaction amounts in USD.
        countries: List of country codes (ISO 2-letter) involved in the transactions.

    Returns:
        JSON string with risk_score (0-100), volume_risk, geo_risk, and interpretation.
    """
    if not amounts:
        return json.dumps({"error": "No amounts provided."})

    total = sum(amounts)
    max_single = max(amounts)
    count = len(amounts)

    # Volume risk: scale 0-1 based on total and single-transaction size
    volume_risk = min(1.0, total / 500_000)
    if max_single >= 10_000:
        volume_risk = min(1.0, volume_risk + 0.2)
    # Penalise structuring: many transactions just below $10k
    sub10k = [a for a in amounts if 7_500 <= a < 10_000]
    if len(sub10k) >= 2:
        volume_risk = min(1.0, volume_risk + 0.3)

    # Geo risk: average country risk scores
    country_scores = [_COUNTRY_RISK.get(c.upper(), _DEFAULT_COUNTRY_RISK) for c in countries]
    geo_risk = sum(country_scores) / len(country_scores) if country_scores else 0.0

    composite = (volume_risk * 0.5 + geo_risk * 0.5) * 100

    if composite >= 75:
        interpretation = "CRITICAL"
    elif composite >= 50:
        interpretation = "HIGH"
    elif composite >= 25:
        interpretation = "MEDIUM"
    else:
        interpretation = "LOW"

    return json.dumps(
        {
            "risk_score": round(composite, 1),
            "interpretation": interpretation,
            "volume_risk": round(volume_risk, 3),
            "geo_risk": round(geo_risk, 3),
            "total_amount": total,
            "transaction_count": count,
        },
        indent=2,
    )


@tool
def calculate_transaction_velocity(customer_id: str, days: int = 30) -> str:
    """Calculate transaction frequency and volume velocity for a customer.

    Args:
        customer_id: The unique customer identifier.
        days: Look-back window in days (default 30).

    Returns:
        JSON string with velocity metrics: tx_count, total_volume, avg_daily_volume, max_single_day_count.
    """
    import json as _json
    import os
    from pathlib import Path

    data_dir = Path(os.getenv("DATA_DIR", "data/"))
    with open(data_dir / "transactions.json") as f:
        all_txns = _json.load(f)

    customer_txns = [t for t in all_txns if t["customer_id"] == customer_id]

    if not customer_txns:
        return _json.dumps({"error": f"No transactions for {customer_id}."})

    # Use the most recent transaction date as reference
    dates = [datetime.strptime(t["date"], "%Y-%m-%d") for t in customer_txns]
    ref_date = max(dates)
    cutoff = ref_date - timedelta(days=days)

    window_txns = [t for t, d in zip(customer_txns, dates) if d >= cutoff]

    if not window_txns:
        return _json.dumps({"tx_count": 0, "total_volume": 0, "avg_daily_volume": 0, "max_single_day_count": 0})

    total_volume = sum(t["amount"] for t in window_txns)
    avg_daily = total_volume / days

    # Count per day
    from collections import Counter
    day_counts = Counter(t["date"] for t in window_txns)
    max_single_day = max(day_counts.values())

    return _json.dumps(
        {
            "customer_id": customer_id,
            "window_days": days,
            "tx_count": len(window_txns),
            "total_volume": round(total_volume, 2),
            "avg_daily_volume": round(avg_daily, 2),
            "max_single_day_count": max_single_day,
        },
        indent=2,
    )
