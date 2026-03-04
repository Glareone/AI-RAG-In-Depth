from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

from .output import AMLReport


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    case_id: str
    report: NotRequired[AMLReport | None]
