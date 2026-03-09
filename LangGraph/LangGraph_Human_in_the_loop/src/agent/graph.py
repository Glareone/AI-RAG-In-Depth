"""FinancialAnalystGraph — LangGraph with two human-in-the-loop interrupts.

Graph flow:
    START
      → plan_analysis       (LLM proposes plan)
      → [INTERRUPT #1]      (human reviews / adjusts plan)
      → extract_data        (LLM reads documents via MCP tools)
      → compute_ratios      (LLM calls calculator MCP tools)
      → identify_risks      (LLM identifies material risks)
      → [INTERRUPT #2]      (human reviews / overrides risk flags)
      → format_report       (LLM produces structured FinancialReport)
      → END
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import yaml
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agent.output import FinancialReport
from src.agent.state import AnalystState
from src.config.mcp_tools_settings import MCPToolsSettings
from src.config.settings import Settings
from src.mcp_tools.client import get_mcp_tools


def _load_prompts() -> dict:
    path = Path("src/agent/prompts.yaml")
    return yaml.safe_load(path.read_text())


def _build_llm(settings: Settings) -> ChatBedrockConverse:
    kwargs: dict = {"model": settings.bedrock_model_id, "region_name": settings.aws_region}
    if settings.aws_profile:
        kwargs["credentials_profile_name"] = settings.aws_profile
    return ChatBedrockConverse(**kwargs)


def _execute_tool_calls(response: AIMessage, tools_by_name: dict) -> list[ToolMessage]:
    """Synchronously execute all tool calls in an AIMessage and return ToolMessages."""
    results = []
    for tc in response.tool_calls:
        tool = tools_by_name.get(tc["name"])
        if tool is None:
            content = f"Tool '{tc['name']}' not found."
        else:
            try:
                content = str(tool.invoke(tc["args"]))
            except Exception as exc:
                content = f"Tool error: {exc}"
        results.append(ToolMessage(content=content, tool_call_id=tc["id"]))
    return results


def _compile_graph(settings: Settings, tools: list, prompts: dict):
    llm = _build_llm(settings)

    # Split tools by role
    doc_tools = [t for t in tools if not t.name.startswith("calc_")]
    calc_tools = [t for t in tools if t.name.startswith("calc_")]

    doc_tools_by_name = {t.name: t for t in doc_tools}
    calc_tools_by_name = {t.name: t for t in calc_tools}

    llm_with_doc_tools = llm.bind_tools(doc_tools)
    llm_with_calc_tools = llm.bind_tools(calc_tools)
    formatter = llm.with_structured_output(FinancialReport)

    analyst_system = SystemMessage(content=prompts["financial_analyst"])
    formatter_system = prompts["financial_formatter"]

    # ── Nodes ─────────────────────────────────────────────────────────────

    def plan_analysis(state: AnalystState) -> dict:
        """Propose an analysis plan, then pause for human review (interrupt #1)."""
        messages = [
            analyst_system,
            HumanMessage(
                content=(
                    f"Ticker: {state['ticker']}, Document type: {state['doc_type']}.\n"
                    "Propose a concise step-by-step analysis plan for this financial document. "
                    "List the key areas you will investigate and the ratios you will compute."
                )
            ),
        ]
        plan = llm.invoke(messages).content

        human_feedback = interrupt({
            "analysis_plan": plan,
            "instructions": (
                "Review the analysis plan above. "
                "Press Enter to approve, or type corrections / additional focus areas."
            ),
        })

        return {
            "analysis_plan": plan,
            "human_plan_feedback": human_feedback,
        }

    def extract_data(state: AnalystState) -> dict:
        """Read the financial document via MCP document tools."""
        plan_context = (
            f"Approved plan: {state['analysis_plan']}\n"
            f"Human notes: {state['human_plan_feedback']}"
        )
        messages = [
            analyst_system,
            HumanMessage(
                content=(
                    f"Ticker: {state['ticker']}, Document type: {state['doc_type']}.\n"
                    f"{plan_context}\n\n"
                    "Use the document tools to locate and read the relevant financial document. "
                    "Extract all key financial figures: revenue, gross profit, operating income, "
                    "net income, total assets, current assets, current liabilities, inventory, "
                    "cash and equivalents, total debt, total equity, EBITDA, interest expense. "
                    "Present the extracted figures clearly."
                )
            ),
        ]

        # LLM decides which tools to call; execute one round of tool calls
        response = llm_with_doc_tools.invoke(messages)
        if response.tool_calls:
            tool_results = _execute_tool_calls(response, doc_tools_by_name)
            messages += [response, *tool_results]
            # Second call: synthesise tool results into structured extraction
            synthesis = llm.invoke(messages)
            extracted = synthesis.content
        else:
            extracted = response.content

        return {"extracted_data": extracted}

    def compute_ratios(state: AnalystState) -> dict:
        """Compute financial ratios using MCP calculator tools."""
        messages = [
            analyst_system,
            HumanMessage(
                content=(
                    "Based on the extracted financial data below, call the calculator tools "
                    "to compute liquidity, profitability, and leverage ratios. "
                    "Use the exact figures from the extraction — do not estimate.\n\n"
                    f"Extracted data:\n{state['extracted_data']}"
                )
            ),
        ]

        response = llm_with_calc_tools.invoke(messages)
        computed: dict = {}

        if response.tool_calls:
            tool_results = _execute_tool_calls(response, calc_tools_by_name)
            # Collect ratio dicts from tool outputs
            for tr in tool_results:
                try:
                    import json
                    result = json.loads(tr.content.replace("'", '"'))
                    if isinstance(result, dict):
                        computed.update(result)
                except Exception:
                    pass

        return {"computed_ratios": computed}

    def identify_risks(state: AnalystState) -> dict:
        """Identify material risks, then pause for human review (interrupt #2)."""
        messages = [
            analyst_system,
            HumanMessage(
                content=(
                    "Based on the document analysis and computed ratios below, "
                    "identify all material risks. List each risk as a concise bullet point.\n\n"
                    f"Extracted data:\n{state['extracted_data']}\n\n"
                    f"Computed ratios:\n{state['computed_ratios']}"
                )
            ),
        ]
        response = llm.invoke(messages)
        risk_text = response.content
        risk_flags = [
            line.lstrip("•-* ").strip()
            for line in risk_text.splitlines()
            if line.strip() and line.strip()[0] in "•-*"
        ] or [risk_text]

        human_feedback = interrupt({
            "risk_flags": risk_flags,
            "instructions": (
                "Review the identified risk flags above. "
                "Press Enter to approve, or type overrides / additional risks."
            ),
        })

        return {
            "risk_flags": risk_flags,
            "human_risk_feedback": human_feedback,
        }

    def format_report(state: AnalystState) -> dict:
        """Produce the final structured FinancialReport."""
        analysis_summary = (
            f"Ticker: {state['ticker']}\n"
            f"Document type: {state['doc_type']}\n\n"
            f"Extracted data:\n{state['extracted_data']}\n\n"
            f"Computed ratios:\n{state['computed_ratios']}\n\n"
            f"Risk flags (after human review):\n"
            + "\n".join(f"• {r}" for r in state.get("risk_flags", []))
            + f"\n\nHuman risk feedback: {state.get('human_risk_feedback', '')}"
        )
        messages = [
            SystemMessage(content=formatter_system),
            HumanMessage(content=analysis_summary),
        ]
        report: FinancialReport = formatter.invoke(messages)
        return {"report": report}

    # ── Graph assembly ─────────────────────────────────────────────────────

    builder = StateGraph(AnalystState)
    builder.add_node("plan_analysis", plan_analysis)
    builder.add_node("extract_data", extract_data)
    builder.add_node("compute_ratios", compute_ratios)
    builder.add_node("identify_risks", identify_risks)
    builder.add_node("format_report", format_report)

    builder.add_edge(START, "plan_analysis")
    builder.add_edge("plan_analysis", "extract_data")
    builder.add_edge("extract_data", "compute_ratios")
    builder.add_edge("compute_ratios", "identify_risks")
    builder.add_edge("identify_risks", "format_report")
    builder.add_edge("format_report", END)

    return builder.compile(checkpointer=MemorySaver())


@asynccontextmanager
async def build_graph(settings: Settings) -> AsyncGenerator:
    """Async context manager — opens the MCP connection and yields a compiled graph.

    Usage::

        async with build_graph(settings) as graph:
            async for chunk in graph.astream(input, config):
                ...
    """
    mcp_settings = MCPToolsSettings()
    async with get_mcp_tools(settings=mcp_settings) as tools:
        graph = _compile_graph(settings, tools, _load_prompts())
        yield graph