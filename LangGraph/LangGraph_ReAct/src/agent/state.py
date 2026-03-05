from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .output import AMLReport


class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    case_id: str = ""
    risk_rate: Literal["High", "Low"] | None = None
    report: AMLReport | None = None
