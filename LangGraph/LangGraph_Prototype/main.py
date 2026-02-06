"""
Main entry point for the LangGraph document analysis application.
This module orchestrates the document processing workflow.
"""

from pathlib import Path
from memory import create_initial_state
from tools import read_file
from graph import app


def print_results(result: dict) -> None:
    """
    Print formatted results from the workflow execution.

    Args:
        result: Final state from workflow execution
    """
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    print(f"Total validated rules: {len(result['validated_rules'])}")
    print(f"Final confidence score: {result['confidence_score']:.2f}")
    print(f"Processing iterations: {result['iteration_count']}")

    print("\nValidated Rules:")
    for rule in result['validated_rules']:
        print(f"- {rule['rule_id']}: {rule['rule_description']}")
        print(f"  Confidence: {rule['validation_confidence']:.2f}")
        print(f"  Applied to: {rule['chunk'][:100]}...")
        print()


def run_analysis(file_path: str) -> dict:
    """
    Run document analysis workflow on a file.

    Args:
        file_path: Path to the document file

    Returns:
        Final workflow state with results
    """
    # Read document content
    content = read_file(file_path)

    if not content:
        raise ValueError(f"Could not read file: {file_path}")

    # Initialize the state
    initial_state = create_initial_state(content)

    # Run the workflow
    print("Starting document analysis workflow...")
    result = app.invoke(initial_state)

    return result


def main():
    """Main execution function."""
    # Example file path
    file_path = Path("data/example.txt")

    # Run the analysis
    result = run_analysis(str(file_path))

    # Print results
    print_results(result)


if __name__ == "__main__":
    main()