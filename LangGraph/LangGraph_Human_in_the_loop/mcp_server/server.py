"""MCP server — financial document and calculator tools.

Runs as a standalone process. Supports two transports:
  stdio (default) — launched as a subprocess by the MCP client
  sse             — HTTP server, suitable for Docker / remote deployment

Usage:
    # stdio (local dev, client spawns this automatically)
    python -m mcp_server.server

    # SSE (separate process or Docker)
    MCP_TRANSPORT=sse python -m mcp_server.server
    MCP_TRANSPORT=sse MCP_SERVER_PORT=8000 python -m mcp_server.server

Tools exposed:
  Document tools:
    list_available_documents       list documents, optional ticker filter
    get_document                   fetch full text by doc_id
    get_document_info              fetch metadata by doc_id
    find_document_by_ticker_and_type  look up doc_id by ticker + doc_type
    preview_document               first N chars of a document

  Calculator tools:
    calc_liquidity_ratios          current ratio, quick ratio, cash ratio
    calc_profitability_ratios      gross/operating/net margin, ROA, ROE
    calc_leverage_ratios           D/E, debt-to-assets, net-debt/EBITDA, coverage
"""

import os

from mcp.server.fastmcp import FastMCP

from src.tools.calculator import (
    calculate_leverage_ratios,
    calculate_liquidity_ratios,
    calculate_profitability_ratios,
)
from src.tools.data_loader import (
    find_document,
    get_document_metadata,
    get_document_text,
    list_documents,
)
from src.tools.document_parser import extract_text_preview

mcp = FastMCP("financial-document-server")


# ---------------------------------------------------------------------------
# Document tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_available_documents(ticker: str = "") -> list[dict]:
    """List available financial documents.

    Args:
        ticker: Optional ticker symbol to filter results (e.g. 'ACM', 'TFI').
                Leave empty to list all documents.

    Returns:
        List of document metadata dicts (doc_id, ticker, doc_type, fiscal_period, description).
    """
    results = list_documents(ticker=ticker or None)
    return [
        {
            "doc_id": d["doc_id"],
            "ticker": d["ticker"],
            "company_name": d.get("company_name", ""),
            "doc_type": d["doc_type"],
            "fiscal_period": d.get("fiscal_period", ""),
            "description": d.get("description", ""),
        }
        for d in results
    ]


@mcp.tool()
def get_document(doc_id: str) -> str:
    """Fetch the full text of a financial document by doc_id.

    Args:
        doc_id: Document identifier from the registry (e.g. 'ACM-AR-2024').

    Returns:
        Full text content of the document.
    """
    return get_document_text(doc_id)


@mcp.tool()
def get_document_info(doc_id: str) -> dict:
    """Return structured metadata for a document.

    Args:
        doc_id: Document identifier (e.g. 'TFI-AR-2024').

    Returns:
        Metadata dict with ticker, company_name, doc_type, fiscal_period, filed_date, file_path.
    """
    meta = get_document_metadata(doc_id)
    if meta is None:
        return {"error": f"Document '{doc_id}' not found in registry."}
    return meta


@mcp.tool()
def find_document_by_ticker_and_type(ticker: str, doc_type: str) -> dict:
    """Find a document by ticker symbol and document type.

    Args:
        ticker:   Company ticker symbol (e.g. 'ACM').
        doc_type: Document type — one of: 'annual_report', 'earnings_call'.

    Returns:
        Metadata dict for the matching document, or an error dict if not found.
    """
    doc = find_document(ticker=ticker, doc_type=doc_type)
    if doc is None:
        return {"error": f"No document found for ticker='{ticker}' doc_type='{doc_type}'."}
    return doc


@mcp.tool()
def preview_document(doc_id: str, max_chars: int = 2000) -> str:
    """Return the first N characters of a document (useful for quick inspection).

    Args:
        doc_id:    Document identifier.
        max_chars: Maximum characters to return (default 2000).

    Returns:
        Truncated document text.
    """
    meta = get_document_metadata(doc_id)
    if meta is None:
        return f"Error: Document '{doc_id}' not found in registry."
    return extract_text_preview(meta["file_path"], max_chars=max_chars)


# ---------------------------------------------------------------------------
# Calculator tools
# ---------------------------------------------------------------------------

@mcp.tool()
def calc_liquidity_ratios(
    current_assets: float,
    current_liabilities: float,
    inventory: float = 0.0,
    cash_and_equivalents: float = 0.0,
) -> dict:
    """Calculate liquidity ratios from balance sheet figures.

    Args:
        current_assets:       Total current assets (USD).
        current_liabilities:  Total current liabilities (USD).
        inventory:            Inventory value (USD, default 0).
        cash_and_equivalents: Cash and cash equivalents (USD, default 0).

    Returns:
        Dict with current_ratio, quick_ratio, cash_ratio.
    """
    try:
        return calculate_liquidity_ratios(
            current_assets=current_assets,
            current_liabilities=current_liabilities,
            inventory=inventory,
            cash_and_equivalents=cash_and_equivalents,
        )
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool()
def calc_profitability_ratios(
    revenue: float,
    gross_profit: float,
    operating_income: float,
    net_income: float,
    total_assets: float,
    total_equity: float,
) -> dict:
    """Calculate profitability ratios from income statement and balance sheet figures.

    Args:
        revenue:          Total revenue (USD).
        gross_profit:     Gross profit (USD).
        operating_income: Operating income / EBIT (USD).
        net_income:       Net income (USD).
        total_assets:     Total assets (USD).
        total_equity:     Total shareholders' equity (USD).

    Returns:
        Dict with gross_margin, operating_margin, net_margin, return_on_assets, return_on_equity.
    """
    try:
        return calculate_profitability_ratios(
            revenue=revenue,
            gross_profit=gross_profit,
            operating_income=operating_income,
            net_income=net_income,
            total_assets=total_assets,
            total_equity=total_equity,
        )
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool()
def calc_leverage_ratios(
    total_debt: float,
    total_equity: float,
    total_assets: float,
    ebitda: float,
    interest_expense: float = 0.0,
) -> dict:
    """Calculate leverage and solvency ratios.

    Args:
        total_debt:        Total interest-bearing debt (USD).
        total_equity:      Total shareholders' equity (USD).
        total_assets:      Total assets (USD).
        ebitda:            EBITDA (USD).
        interest_expense:  Interest expense (USD, default 0 → interest_coverage = N/A).

    Returns:
        Dict with debt_to_equity, debt_to_assets, equity_multiplier,
        net_debt_to_ebitda, interest_coverage.
    """
    try:
        return calculate_leverage_ratios(
            total_debt=total_debt,
            total_equity=total_equity,
            total_assets=total_assets,
            ebitda=ebitda,
            interest_expense=interest_expense,
        )
    except ValueError as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))
    mcp.run(transport=transport, port=port)
