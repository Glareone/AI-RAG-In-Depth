from typing import Literal

from pydantic import BaseModel, Field


class RiskAssessment(BaseModel):
    risk_rate: Literal["High", "Low"] = Field(
        description=(
            "High — one or more clear AML indicators are present in the data seen so far "
            "(e.g. structuring, high-risk jurisdiction, PEP involvement, round-trip). "
            "Low — evidence is weak, ambiguous, or insufficient to indicate suspicious activity."
        )
    )


class AMLReport(BaseModel):
    case_id: str
    risk_level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    recommendation: Literal["FILE_SAR", "ESCALATE_FOR_REVIEW", "CLOSE_NO_ACTION"]
    recommendation_rationale: str = Field(description="One sentence explaining the recommendation.")
    facts: list[str] = Field(description="Discrete AML facts extracted from the analysis.")
    analyst_summary: str = Field(default="", description="2-3 sentence narrative summary of findings suitable for a compliance officer.")
