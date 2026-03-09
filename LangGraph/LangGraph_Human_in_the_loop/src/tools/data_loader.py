"""Load document registry and fetch document text by ticker / doc_type."""

import json
from pathlib import Path


def _registry_path() -> Path:
    return Path("data/metadata.json")


def load_registry() -> dict:
    """Return the full document registry as a dict."""
    with _registry_path().open() as f:
        return json.load(f)


def list_documents(ticker: str | None = None) -> list[dict]:
    """Return document metadata entries, optionally filtered by ticker."""
    registry = load_registry()
    docs = registry["documents"]
    if ticker:
        docs = [d for d in docs if d["ticker"].upper() == ticker.upper()]
    return docs


def get_document_metadata(doc_id: str) -> dict | None:
    """Return metadata for a single document by doc_id, or None if not found."""
    registry = load_registry()
    for doc in registry["documents"]:
        if doc["doc_id"] == doc_id:
            return doc
    return None


def get_document_text(doc_id: str) -> str:
    """Return the raw text content of a document by doc_id.

    Raises FileNotFoundError if the file doesn't exist.
    Raises ValueError if the doc_id is not in the registry.
    """
    meta = get_document_metadata(doc_id)
    if meta is None:
        raise ValueError(f"Document '{doc_id}' not found in registry.")
    file_path = Path(meta["file_path"])
    if not file_path.exists():
        raise FileNotFoundError(f"Document file not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def find_document(ticker: str, doc_type: str) -> dict | None:
    """Find the first document matching ticker + doc_type."""
    for doc in list_documents(ticker=ticker):
        if doc["doc_type"] == doc_type:
            return doc
    return None
