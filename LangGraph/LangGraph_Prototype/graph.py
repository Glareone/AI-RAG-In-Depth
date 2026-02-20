"""
Graph construction and compilation for the LangGraph document analysis workflow.
Defines the workflow structure, nodes, edges, and conditional routing.
"""

from langgraph.graph import StateGraph, END
from memory import DocumentState
from nodes import (
    chunk_document,
    extract_candidate_rules,
    react_rule_validation,
    assess_confidence,
    should_continue_processing,
    enrich_rules
)


def create_workflow() -> StateGraph:
    """
    Create and configure the document analysis workflow graph.

    Returns:
        Configured StateGraph ready for compilation
    """
    # Initialize the workflow
    workflow = StateGraph(DocumentState)

    # Add nodes
    workflow.add_node("chunk", chunk_document)
    workflow.add_node("extract_candidates", extract_candidate_rules)
    workflow.add_node("validate_rules", react_rule_validation)
    workflow.add_node("assess_confidence", assess_confidence)
    workflow.add_node("enrich", enrich_rules)

    # Add edges
    workflow.add_edge("chunk", "extract_candidates")
    workflow.add_edge("extract_candidates", "validate_rules")
    workflow.add_edge("validate_rules", "assess_confidence")

    # Conditional routing based on confidence
    workflow.add_conditional_edges(
        "assess_confidence",
        should_continue_processing,
        {
            "continue": "enrich",
            "finish": END
        }
    )

    # Enrichment loops back to validation
    workflow.add_edge("enrich", "validate_rules")

    # Set the entry point
    workflow.set_entry_point("chunk")

    return workflow


def build_app():
    """
    Build and compile the document analysis application.

    Returns:
        Compiled LangGraph application ready for execution
    """
    workflow = create_workflow()
    return workflow.compile()


# Create the application instance
app = build_app()