from typing import Literal

from pydantic import BaseModel, Field


class AMLReport(BaseModel):
    case_id: str
    risk_level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    recommendation: Literal["FILE_SAR", "ESCALATE_FOR_REVIEW", "CLOSE_NO_ACTION"]
    recommendation_rationale: str = Field(description="One sentence explaining the recommendation.")
    facts: list[str] = Field(description="Discrete AML facts extracted from the analysis.")
    analyst_summary: str = Field(description="2-3 sentence narrative summary of findings.")
