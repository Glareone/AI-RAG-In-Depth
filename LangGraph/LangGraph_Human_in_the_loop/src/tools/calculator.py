"""Financial ratio calculators.

All functions accept raw financial statement numbers (floats / ints) and return
a dict of labelled ratios rounded to 4 decimal places.  They raise ValueError
if required inputs are zero or missing to avoid silent division-by-zero errors.
"""

from __future__ import annotations


def _round(value: float, decimals: int = 4) -> float:
    return round(value, decimals)


# ---------------------------------------------------------------------------
# Liquidity ratios
# ---------------------------------------------------------------------------

def calculate_liquidity_ratios(
    current_assets: float,
    current_liabilities: float,
    inventory: float = 0.0,
    cash_and_equivalents: float = 0.0,
) -> dict[str, float]:
    """Compute common liquidity ratios.

    Returns:
        current_ratio       — current assets / current liabilities
        quick_ratio         — (current assets − inventory) / current liabilities
        cash_ratio          — cash & equivalents / current liabilities
    """
    if current_liabilities == 0:
        raise ValueError("current_liabilities must be non-zero.")

    return {
        "current_ratio": _round(current_assets / current_liabilities),
        "quick_ratio": _round((current_assets - inventory) / current_liabilities),
        "cash_ratio": _round(cash_and_equivalents / current_liabilities),
    }


# ---------------------------------------------------------------------------
# Profitability ratios
# ---------------------------------------------------------------------------

def calculate_profitability_ratios(
    revenue: float,
    gross_profit: float,
    operating_income: float,
    net_income: float,
    total_assets: float,
    total_equity: float,
) -> dict[str, float]:
    """Compute common profitability ratios.

    Returns:
        gross_margin        — gross profit / revenue
        operating_margin    — operating income / revenue
        net_margin          — net income / revenue
        return_on_assets    — net income / total assets  (ROA)
        return_on_equity    — net income / total equity  (ROE)
    """
    if revenue == 0:
        raise ValueError("revenue must be non-zero.")
    if total_assets == 0:
        raise ValueError("total_assets must be non-zero.")
    if total_equity == 0:
        raise ValueError("total_equity must be non-zero.")

    return {
        "gross_margin": _round(gross_profit / revenue),
        "operating_margin": _round(operating_income / revenue),
        "net_margin": _round(net_income / revenue),
        "return_on_assets": _round(net_income / total_assets),
        "return_on_equity": _round(net_income / total_equity),
    }


# ---------------------------------------------------------------------------
# Leverage / solvency ratios
# ---------------------------------------------------------------------------

def calculate_leverage_ratios(
    total_debt: float,
    total_equity: float,
    total_assets: float,
    ebitda: float,
    interest_expense: float = 0.0,
) -> dict[str, float | str]:
    """Compute common leverage and solvency ratios.

    Returns:
        debt_to_equity      — total debt / total equity  (D/E)
        debt_to_assets      — total debt / total assets
        equity_multiplier   — total assets / total equity
        net_debt_to_ebitda  — total debt / ebitda  (if ebitda > 0)
        interest_coverage   — ebitda / interest expense  (if interest_expense > 0, else 'N/A')
    """
    if total_equity == 0:
        raise ValueError("total_equity must be non-zero.")
    if total_assets == 0:
        raise ValueError("total_assets must be non-zero.")
    if ebitda == 0:
        raise ValueError("ebitda must be non-zero.")

    result: dict[str, float | str] = {
        "debt_to_equity": _round(total_debt / total_equity),
        "debt_to_assets": _round(total_debt / total_assets),
        "equity_multiplier": _round(total_assets / total_equity),
        "net_debt_to_ebitda": _round(total_debt / ebitda),
    }

    if interest_expense and interest_expense != 0:
        result["interest_coverage"] = _round(ebitda / interest_expense)
    else:
        result["interest_coverage"] = "N/A"

    return result