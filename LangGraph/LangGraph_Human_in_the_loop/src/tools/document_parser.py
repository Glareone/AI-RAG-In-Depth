"""Extract text from financial documents (PDF and plain-text files)."""

from pathlib import Path


def extract_text(file_path: str | Path) -> str:
    """Extract text from a file.

    Supports:
    - .txt / .md  — read directly
    - .pdf        — extract with pypdf (page-by-page join)

    Returns the extracted text as a single string.
    Raises ValueError for unsupported extensions.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # optional dep; graceful error if missing
        except ImportError as e:
            raise ImportError("pypdf is required to parse PDF files. Install with: pip install pypdf") from e

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    raise ValueError(f"Unsupported file type: '{suffix}'. Supported: .txt, .md, .pdf")


def extract_text_preview(file_path: str | Path, max_chars: int = 2000) -> str:
    """Return the first `max_chars` characters of extracted text."""
    return extract_text(file_path)[:max_chars]
