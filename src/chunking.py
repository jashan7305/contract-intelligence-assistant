from dataclasses import dataclass
from pathlib import Path
from .data import resolve_doc_path, read_contract

@dataclass
class Passage:
    doc_path: str
    start: int
    end: int
    text: str
    chunk_id: str

def chunk_text(text, doc_path, chunk_chars=1200, overlap_chars=200):
    if chunk_chars <= 0 or overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("Require chunk_chars > overlap_chars >= 0")
    passages = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        # Prefer a nearby paragraph/newline boundary, but retain exact offsets.
        if end < len(text):
            boundary = text.rfind("\\n", start + chunk_chars // 2, end)
            if boundary > start:
                end = boundary
        if end <= start:
            end = min(len(text), start + chunk_chars)
        passages.append(Passage(str(doc_path), start, end, text[start:end], f"{doc_path}:{start}:{end}"))
        if end == len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return passages

def load_corpus_for_queries(rows, corpus_dir, chunk_chars=1200, overlap_chars=200):
    docs = {}
    for row in rows:
        for snip in row["snippets"]:
            fp = snip.get("file_path", "")
            resolved = resolve_doc_path(corpus_dir, fp)
            if resolved is not None:
                docs[str(resolved)] = read_contract(resolved)
    passages = []
    for doc_path, text in docs.items():
        passages.extend(chunk_text(text, doc_path, chunk_chars, overlap_chars))
    return passages
