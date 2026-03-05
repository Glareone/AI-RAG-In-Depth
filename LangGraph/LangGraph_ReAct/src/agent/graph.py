from pathlib import Path

import yaml
from botocore.client import BaseClient
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.config import Settings
from src.infrastructure import make_bedrock_client
from src.tools import all_tools

from .output import AMLReport, RiskAssessment
from .state import AgentState

_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"


def _load_prompts() -> dict:
    with open(_PROMPTS_PATH) as f:
        return yaml.safe_load(f)


class AMLAgentGraph:
    def __init__(self, settings: Settings, bedrock_client: BaseClient) -> None:
        prompts = _load_prompts()

        self._prompts = prompts
        self._tool_node = ToolNode(all_tools)

        self._react_llm = ChatBedrock(
            model=settings.bedrock_model_id,
            client=bedrock_client,
            provider="anthropic",
        ).bind_tools(all_tools)

        self._formatter_llm = ChatBedrock(
            model=settings.bedrock_model_id,
            client=bedrock_client,
            provider="anthropic",
        ).with_structured_output(AMLReport)

        self._assessor_llm = ChatBedrock(
            model=settings.bedrock_model_id,
            client=bedrock_client,
            provider="anthropic",
        ).with_structured_output(RiskAssessment)

    def _agent_node(self, state: AgentState) -> dict:
        messages = state.messages
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self._prompts["aml_analyst_system"])] + list(messages)
        response = self._react_llm.invoke(messages)

        # Extract tool result text as plain strings — ToolMessage objects cannot be sent
        # standalone to Claude (tool_result requires a preceding tool_use in the same turn).
        tool_results = [m for m in state.messages if isinstance(m, ToolMessage)]
        collected_data = (
            "\n\n".join(f"[{m.name}]:\n{m.content}" for m in tool_results)
            if tool_results
            else "No data collected yet."
        )

        assessor_messages = [
            SystemMessage(content=self._prompts["risk_assessor"]),
            HumanMessage(content=f"Data collected so far:\n\n{collected_data}\n\nAssess the current risk rate."),
        ]
        assessment = self._assessor_llm.invoke(assessor_messages)

        return {"messages": [response], "risk_rate": assessment.risk_rate}

    def _formatter_node(self, state: AgentState) -> dict:
        messages = list(state.messages) + [HumanMessage(content=self._prompts["aml_formatter"])]
        report = self._formatter_llm.invoke(messages)
        return {"report": report}

    def _should_continue(self, state: AgentState) -> str:
        last = state.messages[-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "formatter"

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("agent", RunnableLambda(self._agent_node))
        graph.add_node("tools", self._tool_node)
        graph.add_node("formatter", RunnableLambda(self._formatter_node))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", self._should_continue, {"tools": "tools", "formatter": "formatter"})
        graph.add_edge("tools", "agent")
        graph.add_edge("formatter", END)

        return graph.compile(checkpointer=MemorySaver())


def build_graph(settings: Settings) -> CompiledStateGraph:
    bedrock_client = make_bedrock_client(settings)
    return AMLAgentGraph(settings, bedrock_client).build()
