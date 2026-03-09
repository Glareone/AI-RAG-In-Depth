"""AnalystState — the shared state passed between all graph nodes."""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.agent.output import FinancialReport


class AnalystState(dict):
    """TypedDict-style state for the financial analyst graph.

    Fields
    ------
    messages          Conversation history (auto-merged by add_messages).
    ticker            Company ticker symbol supplied by the caller.
    doc_type          Document type: 'annual_report' | 'earnings_call'.
    document_id       doc_id resolved from the registry after plan_analysis.
    analysis_plan     Agent-proposed analysis plan (set in plan_analysis node).
    human_plan_feedback  Human feedback from interrupt #1.
    extracted_data    Raw financial data extracted from the document.
    computed_ratios   Dict of ratio results from the calculator tools.
    risk_flags        List of material risk strings (set in identify_risks node).
    human_risk_feedback  Human feedback from interrupt #2.
    report            Final structured FinancialReport (set in format_report node).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    ticker: str
    doc_type: str
    document_id: str | None
    analysis_plan: str
    human_plan_feedback: str
    extracted_data: str
    computed_ratios: dict
    risk_flags: list[str]
    human_risk_feedback: str
    report: FinancialReport | None