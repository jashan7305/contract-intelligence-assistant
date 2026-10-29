import hashlib
import json
import random
from pathlib import Path

def mini(tests, n=194):
    """Reproduce the benchmark authors' deterministic mini-selection."""
    if len(tests) <= n:
        return tests
    def key(t):
        random.seed(t["snippets"][0]["file_path"])
        return random.random()
    return sorted(tests, key=key)[:n]

def assign_split(query: str) -> str:
    h = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if h % 10 < 3 else "train"

def load_benchmark(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tests = raw["tests"] if isinstance(raw, dict) else raw
    selected = mini(tests, 194)
    rows = []
    for i, test in enumerate(selected):
        query = (test.get("query") or test.get("question") or "").strip()
        snippets = test.get("snippets", [])
        if not query:
            continue
        rows.append({
            "id": str(test.get("id", i)),
            "query": query,
            "snippets": snippets,
            "split": assign_split(query),
            "answerable": True,
        })
    return rows

def read_contract(path):
    # Keep Python's default newline handling and UTF-8 decoding, as required by the brief.
    return Path(path).read_text(encoding="utf-8")

def resolve_doc_path(corpus_dir, file_path):
    """Try common LegalBench-RAG path representations without changing file contents."""
    base = Path(corpus_dir)
    p = Path(file_path)
    candidates = [p, base / p, base / p.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Some benchmark paths include a corpus prefix.
    parts = p.parts
    for i in range(len(parts)):
        candidate = base.joinpath(*parts[i:])
        if candidate.is_file():
            return candidate
    return None

def snippet_offsets(snippet):
    """Return a benchmark snippet's character offsets across mirror schemas."""
    span = snippet.get("span")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return span[0], span[1]
    return snippet.get("start", snippet.get("start_char")), snippet.get(
        "end", snippet.get("end_char")
    )

def snippet_answer(snippet):
    return snippet.get("answer", snippet.get("text", snippet.get("snippet", "")))

def validate_snippet_offsets(rows, corpus_dir):
    checks = []
    for row in rows:
        for snip in row["snippets"]:
            file_path = snip.get("file_path", "")
            start, end = snippet_offsets(snip)
            answer = snippet_answer(snip)
            doc = resolve_doc_path(corpus_dir, file_path)
            if doc is None or start is None or end is None:
                checks.append({"query_id": row["id"], "ok": False, "reason": "missing document or offsets"})
                continue
            text = read_contract(doc)
            start, end = int(start), int(end)
            actual = text[start:end]
            checks.append({
                "query_id": row["id"],
                "ok": 0 <= start <= end <= len(text) and actual == answer,
                "expected": answer,
                "actual": actual,
            })
    return checks
