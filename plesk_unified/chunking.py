from typing import List, Dict


def chunk_by_chars(text: str, size: int = 1500, overlap: int = 200) -> List[str]:
    """Chunk text by fixed character window with overlap."""
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    step = max(1, size - overlap)
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_by_lines(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Chunk text by lines with optional overlap.

    `chunk_size` is number of lines per chunk. `overlap` is number of lines
    to overlap between consecutive chunks.
    """
    if not text:
        return []
    lines = text.splitlines()
    if not lines:
        return []
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(lines), step):
        chunk = "\n".join(lines[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def build_doc_records(filename: str, chunks: List[str], meta: Dict) -> List[Dict]:
    """Build a list of document dicts suitable for DB insertion.

    Each record includes `text`, `title`, `filename`, `category`, and `breadcrumb`.
    """
    records: List[Dict] = []
    for c in chunks:
        records.append(
            {
                "text": c,
                "title": meta.get("title"),
                "filename": filename,
                "category": meta.get("category"),
                "breadcrumb": meta.get("breadcrumb"),
            }
        )
    return records


def persist_batch(table, docs: List[Dict]):
    """Persist a batch of docs to `table`.

    `table` is expected to implement an `add(iterable)` method
    (LanceDB-like).

    This wrapper keeps the call site testable. Returns the result of
    `table.add` when present.
    """
    if not docs:
        return None
    if hasattr(table, "add"):
        return table.add(docs)
    # Fallback: try treating table as a callable
    return table(docs)
