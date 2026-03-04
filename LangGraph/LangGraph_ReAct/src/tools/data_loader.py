import json
import os
from pathlib import Path

from langchain_core.tools import tool

_DATA_DIR = Path(os.getenv("DATA_DIR", "data/"))


def _load_json(filename: str) -> list[dict]:
    path = _DATA_DIR / filename
    with open(path) as f:
        return json.load(f)


@tool
def get_transaction_details(transaction_id: str) -> str:
    """Fetch details for a specific transaction by its ID.

    Args:
        transaction_id: The unique transaction identifier (e.g. TXN-001).

    Returns:
        JSON string with transaction details or an error message.
    """
    transactions = _load_json("transactions.json")
    for txn in transactions:
        if txn["transaction_id"] == transaction_id:
            return json.dumps(txn, indent=2)
    return f"Transaction {transaction_id} not found."


@tool
def get_customer_profile(customer_id: str) -> str:
    """Fetch the KYC profile for a customer.

    Args:
        customer_id: The unique customer identifier (e.g. CUST-101).

    Returns:
        JSON string with customer profile or an error message.
    """
    customers = _load_json("customers.json")
    for cust in customers:
        if cust["customer_id"] == customer_id:
            return json.dumps(cust, indent=2)
    return f"Customer {customer_id} not found."


@tool
def get_case_details(case_id: str) -> str:
    """Fetch details for an ML-flagged AML case.

    Args:
        case_id: The unique case identifier (e.g. CASE-2024-001).

    Returns:
        JSON string with case details or an error message.
    """
    cases = _load_json("cases.json")
    for case in cases:
        if case["case_id"] == case_id:
            return json.dumps(case, indent=2)
    return f"Case {case_id} not found."


@tool
def search_transactions(case_id: str) -> str:
    """Find all transactions belonging to an AML case.

    Case is the root entity — transactions are looked up by case_id.

    Args:
        case_id: The unique case identifier (e.g. CASE-2024-001).

    Returns:
        JSON string with list of transactions for the case.
    """
    transactions = _load_json("transactions.json")
    case_txns = [t for t in transactions if t["case_id"] == case_id]
    if not case_txns:
        return f"No transactions found for case {case_id}."
    return json.dumps(case_txns, indent=2)


@tool
def search_transactions_by_customer(customer_id: str) -> str:
    """Find all transactions associated with a customer across all cases.

    Args:
        customer_id: The unique customer identifier (e.g. CUST-101).

    Returns:
        JSON string with list of transactions for the customer.
    """
    transactions = _load_json("transactions.json")
    customer_txns = [t for t in transactions if t["customer_id"] == customer_id]
    if not customer_txns:
        return f"No transactions found for customer {customer_id}."
    return json.dumps(customer_txns, indent=2)
