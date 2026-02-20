"""
Utility tools and functions for the LangGraph document analysis application.
Includes file operations and rule management.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def read_file(file_path: str) -> str:
    """
    Read content from a text file.

    Args:
        file_path: Path to the file to read

    Returns:
        File content as string, empty string on error

    Raises:
        None - errors are caught and logged
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""


def load_rules_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load rules from a JSON file.

    Args:
        file_path: Path to the JSON rules file

    Returns:
        List of rule dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading rules from JSON: {e}")
        return []


def get_sample_rules(rules_path: str = "data/sample_rules.json") -> List[Dict[str, Any]]:
    """
    Load rules for document analysis from JSON file.

    Args:
        rules_path: Path to the JSON rules file (default: data/sample_rules.json)

    Returns:
        List of rule dictionaries

    Raises:
        FileNotFoundError: If rules file doesn't exist
        ValueError: If rules file is empty or invalid
    """
    json_path = Path(rules_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")

    rules = load_rules_from_json(str(json_path))

    if not rules:
        raise ValueError(f"Rules file is empty or invalid: {rules_path}")

    return rules