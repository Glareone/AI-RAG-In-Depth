"""
Node functions for the LangGraph document analysis workflow.
Each function represents a processing step in the graph.
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage
from memory import DocumentState
from tools import get_sample_rules
from config import llm, VALIDATION_THRESHOLD, CONFIDENCE_THRESHOLD, MAX_ITERATIONS, EXPECTED_RULES_PER_DOCUMENT


def chunk_document(state: DocumentState) -> DocumentState:
    """
    Split document into manageable chunks.

    Args:
        state: Current document state

    Returns:
        Updated state with document chunks
    """
    content = state["content"]

    # Simple chunking by paragraphs (can be enhanced with semantic chunking)
    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]

    print(f"Document split into {len(chunks)} chunks")

    return {
        **state,
        "chunks": chunks,
        "iteration_count": state.get("iteration_count", 0)
    }


def extract_candidate_rules(state: DocumentState) -> DocumentState:
    """
    Extract potentially applicable rules using keyword matching.

    Args:
        state: Current document state with chunks

    Returns:
        Updated state with candidate rules
    """
    chunks = state["chunks"]
    candidate_rules = []
    rules = get_sample_rules()

    for chunk in chunks:
        chunk_lower = chunk.lower()

        for rule in rules:
            # Keyword matching (can be replaced with vector similarity)
            keyword_matches = sum(
                1 for keyword in rule["keywords"]
                if keyword.lower() in chunk_lower
            )

            if keyword_matches > 0:
                candidate_rules.append({
                    "rule_id": rule["id"],
                    "rule_description": rule["description"],
                    "chunk": chunk,
                    "keyword_matches": keyword_matches,
                    "confidence": keyword_matches / len(rule["keywords"])
                })

    # Remove duplicates and sort by confidence
    unique_candidates = []
    seen = set()
    for candidate in sorted(candidate_rules, key=lambda x: x["confidence"], reverse=True):
        key = (candidate["rule_id"], candidate["chunk"][:50])
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)

    print(f"Found {len(unique_candidates)} candidate rule applications")

    return {
        **state,
        "candidate_rules": unique_candidates
    }


def react_rule_validation(state: DocumentState) -> DocumentState:
    """
    Use ReAct pattern to validate rule applicability with LLM.

    Args:
        state: Current document state with candidate rules

    Returns:
        Updated state with validated rules
    """
    candidate_rules = state["candidate_rules"]
    validated_rules = []

    for candidate in candidate_rules:
        # ReAct prompt for rule validation
        prompt = f"""
        You are a financial document compliance expert. Analyze if the following rule applies to the given text chunk.

        RULE: {candidate['rule_description']}

        TEXT CHUNK: {candidate['chunk']}

        Think step by step:
        1. THOUGHT: What does this rule require?
        2. OBSERVATION: What do I see in the text?
        3. ACTION: Does the rule apply? (YES/NO)
        4. CONFIDENCE: Rate your confidence (0.0-1.0)

        Respond in this exact format:
        THOUGHT: [your reasoning]
        OBSERVATION: [what you observe]
        ACTION: [YES or NO]
        CONFIDENCE: [0.0-1.0]
        """

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content

            # Parse ReAct response
            lines = response_text.strip().split('\n')
            action_line = next((line for line in lines if line.startswith('ACTION:')), 'ACTION: NO')
            confidence_line = next((line for line in lines if line.startswith('CONFIDENCE:')), 'CONFIDENCE: 0.0')

            action = action_line.split(':', 1)[1].strip().upper()
            confidence = float(confidence_line.split(':', 1)[1].strip())

            if action == 'YES' and confidence > VALIDATION_THRESHOLD:
                validated_rules.append({
                    **candidate,
                    "validation_confidence": confidence,
                    "llm_reasoning": response_text
                })

        except Exception as e:
            print(f"Error in ReAct validation: {e}")
            continue

    print(f"Validated {len(validated_rules)} rules")

    return {
        **state,
        "validated_rules": validated_rules,
        "iteration_count": state["iteration_count"] + 1
    }


def assess_confidence(state: DocumentState) -> DocumentState:
    """
    Assess overall confidence in rule extraction.

    Args:
        state: Current document state with validated rules

    Returns:
        Updated state with confidence score
    """
    validated_rules = state["validated_rules"]

    if not validated_rules:
        confidence_score = 0.0
    else:
        avg_confidence = sum(
            rule["validation_confidence"] for rule in validated_rules
        ) / len(validated_rules)

        # Factor in number of rules found vs expected
        coverage_factor = min(len(validated_rules) / EXPECTED_RULES_PER_DOCUMENT, 1.0)
        confidence_score = avg_confidence * coverage_factor

    print(f"Overall confidence score: {confidence_score:.2f}")

    return {
        **state,
        "confidence_score": confidence_score
    }


def should_continue_processing(state: DocumentState) -> str:
    """
    Determine next step based on confidence and iteration count.

    Args:
        state: Current document state

    Returns:
        "continue" to continue processing, "finish" to end
    """
    confidence = state["confidence_score"]
    iterations = state["iteration_count"]

    # Continue if confidence is low and we haven't exceeded max iterations
    if confidence < CONFIDENCE_THRESHOLD and iterations < MAX_ITERATIONS:
        print("Confidence low, continuing processing...")
        return "continue"
    else:
        print("Processing complete!")
        return "finish"


def enrich_rules(state: DocumentState) -> DocumentState:
    """
    Attempt to find additional applicable rules (enrichment step).

    Args:
        state: Current document state

    Returns:
        Updated state with enriched rules
    """
    print("Attempting rule enrichment...")

    chunks = state["chunks"]
    current_rule_ids = {rule["rule_id"] for rule in state["validated_rules"]}
    rules = get_sample_rules()

    # Check if we missed any obvious rules
    for rule in rules:
        if rule["id"] not in current_rule_ids:
            # More aggressive matching for enrichment
            for chunk in chunks:
                if any(keyword.lower() in chunk.lower() for keyword in rule["keywords"]):
                    # Add to validated rules with lower confidence
                    state["validated_rules"].append({
                        "rule_id": rule["id"],
                        "rule_description": rule["description"],
                        "chunk": chunk,
                        "keyword_matches": 1,
                        "confidence": 0.6,
                        "validation_confidence": 0.6,
                        "llm_reasoning": "Added during enrichment phase"
                    })
                    break

    return state