import json
from pathlib import Path
from .data import load_benchmark, resolve_doc_path, snippet_offsets
from .chunking import load_corpus_for_queries
from .retrieval import Retriever
from .metrics import char_precision_recall_by_doc

def gold_spans(row, corpus_dir):
    gold = {}
    for snip in row["snippets"]:
        fp = snip.get("file_path", "")
        resolved = resolve_doc_path(corpus_dir, fp)
        start, end = snippet_offsets(snip)
        if resolved is not None and start is not None and end is not None:
            gold.setdefault(str(resolved), []).append((int(start), int(end)))
    return gold

def score_row(results, gold):
    pred = {}
    for r in results:
        pred.setdefault(r["doc_path"], []).append((r["start"], r["end"]))
    return char_precision_recall_by_doc(pred, gold)

def evaluate(config, split="train", save_path=None):
    rows = load_benchmark(config["data"]["benchmark_path"])
    rows = [r for r in rows if r["split"] == split]
    corpus_dir = config["data"]["corpus_dir"]
    # Index only contracts referenced by the selected split's queries. The union is limited
    # to this split; no ground-truth span is passed to the retriever.
    passages = load_corpus_for_queries(rows, corpus_dir,
        config["retrieval"]["chunk_chars"], config["retrieval"]["overlap_chars"])
    retriever = Retriever(passages, config["retrieval"]["method"],
                          config["retrieval"]["bm25_weight"], config["retrieval"]["tfidf_weight"])
    ks = config["evaluation"]["ks"]
    per_query = []
    for row in rows:
        results = retriever.search(row["query"], max(ks), config["retrieval"]["min_score"])
        gold = gold_spans(row, corpus_dir)
        per_query.append({"id": row["id"], "query": row["query"],
                          "results": results, "gold": gold, "answerable": True})
    metrics = {}
    for k in ks:
        scores = [score_row(x["results"][:k], x["gold"]) for x in per_query]
        metrics[f"char_precision@{k}"] = sum(x[0] for x in scores) / max(1, len(scores))
        metrics[f"char_recall@{k}"] = sum(x[1] for x in scores) / max(1, len(scores))

    # Evaluate abstention on the fixed, separately committed unanswerable set.
    unans_path = Path("data/unanswerable_queries.json")
    unans = json.loads(unans_path.read_text(encoding="utf-8")) if unans_path.exists() else []
    unans = [q for q in unans if q["split"] == split]
    # Include the unanswerable-query contracts in the searchable corpus only if they are
    # already part of the answerable benchmark corpus; they are not given gold spans.
    unans_results = []
    for row in unans:
        results = retriever.search(row["query"], max(ks), config["retrieval"]["min_score"])
        unans_results.append({"id": row["id"], "query": row["query"],
                              "results": results, "gold": {}, "answerable": False})
    answerable_abstentions = sum(not x["results"] for x in per_query)
    unanswerable_correct_abstentions = sum(not x["results"] for x in unans_results)
    metrics["answerable_false_abstention_rate"] = answerable_abstentions / max(1, len(per_query))
    metrics["unanswerable_correct_abstention_rate"] = (
        unanswerable_correct_abstentions / max(1, len(unans_results)))
    metrics["unanswerable_queries"] = len(unans)
    metrics["answerable_queries"] = len(per_query)

    report = {"split": split, "n_queries": len(rows), "metrics": metrics,
              "per_query": per_query, "unanswerable": unans_results}
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
