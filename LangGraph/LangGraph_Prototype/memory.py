"""
Memory and State management for the LangGraph document analysis workflow.
Defines the state schema and state-related utilities.
"""

from typing import Dict, List, Any, TypedDict, NotRequired


class DocumentState(TypedDict):
    """
    State schema for document processing workflow.

    Attributes:
        content: Raw document content as string
        chunks: Document split into chunks for processing
        candidate_rules: Rules that potentially apply based on keyword matching
        validated_rules: Rules validated by LLM using ReAct pattern
        confidence_score: Overall confidence in rule extraction (0.0-1.0)
        iteration_count: Number of processing iterations performed
        processed: Flag indicating if processing is complete
    """
    content: NotRequired[str]
    chunks: NotRequired[List[str]]
    candidate_rules: NotRequired[List[Dict[str, Any]]]
    validated_rules: NotRequired[List[Dict[str, Any]]]
    confidence_score: NotRequired[float]
    iteration_count: NotRequired[int]
    processed: NotRequired[bool]


def create_initial_state(content: str) -> DocumentState:
    """
    Create an initial state for document processing.

    Args:
        content: Raw document content

    Returns:
        DocumentState with initial values
    """
    return {
        "content": content,
        "chunks": [],
        "candidate_rules": [],
        "validated_rules": [],
        "confidence_score": 0.0,
        "iteration_count": 0,
        "processed": False
    }