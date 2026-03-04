from pathlib import Path

import boto3
import yaml
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.config import Settings
from src.tools import all_tools

from .output import AMLReport
from .state import AgentState

_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"


def _load_prompts() -> dict:
    with open(_PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def _make_bedrock_client(settings: Settings):
    session = boto3.Session(
        profile_name=settings.aws_profile or None,
        region_name=settings.aws_region,
    )
    return session.client("bedrock-runtime")


def build_graph(settings: Settings) -> CompiledStateGraph:
    prompts = _load_prompts()
    bedrock_client = _make_bedrock_client(settings)

    # ReAct LLM — tool calling enabled
    react_llm = ChatBedrock(
        model_id=settings.bedrock_model_id,
        client=bedrock_client,
        provider="anthropic",
    ).bind_tools(all_tools)

    # Formatter LLM — structured output, no tools
    formatter_llm = ChatBedrock(
        model_id=settings.bedrock_model_id,
        client=bedrock_client,
        provider="anthropic",
    ).with_structured_output(AMLReport)

    tool_node = ToolNode(all_tools)

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=prompts["aml_analyst_system"])] + list(messages)
        response = react_llm.invoke(messages)
        return {"messages": [response]}

    def formatter_node(state: AgentState) -> dict:
        messages = state["messages"] + [
            HumanMessage(content=prompts["aml_formatter"])
        ]
        report = formatter_llm.invoke(messages)
        return {"report": report}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "formatter"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("formatter", formatter_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "formatter": "formatter"})
    graph.add_edge("tools", "agent")
    graph.add_edge("formatter", END)

    return graph.compile(checkpointer=MemorySaver())
