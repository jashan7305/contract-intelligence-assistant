"""Small, auditable train-only coordinate search over retrieval settings."""
import copy, csv, json
from pathlib import Path
from .config import load_config
from .evaluate import evaluate

CANDIDATES = [
    ("chunk_chars", 800), ("chunk_chars", 1200), ("chunk_chars", 1600),
    ("overlap_chars", 100), ("overlap_chars", 200),
    ("method", "tfidf"), ("method", "bm25"), ("method", "hybrid"),
    ("min_score", 0.0), ("min_score", 0.04), ("min_score", 0.08),
    ("min_score", 0.12), ("min_score", 0.2), ("min_score", 0.3),
    ("min_score", 0.4), ("min_score", 0.5),
    ("bm25_weight", 0.35), ("bm25_weight", 0.55), ("bm25_weight", 0.7),
]

def objective(report):
    # Primary metric: mean char recall at 5. Precision and abstention are guardrails.
    m = report["metrics"]
    return m["char_recall@5"], m["char_precision@5"]

def accepted(proposal, baseline, best):
    proposed = proposal["metrics"]
    baseline_metrics = baseline["metrics"]
    best_metrics = best["metrics"]
    return (
        proposed["char_recall@5"] > best_metrics["char_recall@5"]
        and proposed["char_precision@5"] >= best_metrics["char_precision@5"] - 0.02
        and proposed["unanswerable_correct_abstention_rate"]
        >= baseline_metrics["unanswerable_correct_abstention_rate"]
    )

def run_loop(config_path="config.yaml", iterations=5):
    cfg = load_config(config_path)
    Path("artifacts").mkdir(exist_ok=True)
    log_path = Path("artifacts/experiment_log.jsonl")
    current = copy.deepcopy(cfg)
    baseline = evaluate(current, "train", "artifacts/baseline_train.json")
    best_obj = objective(baseline)
    best_report = baseline
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"iteration": 0, "status": "baseline", "config": current,
            "metrics": baseline["metrics"], "objective": best_obj}) + "\n")
        for i in range(1, iterations + 1):
            key, value = CANDIDATES[(i - 1) % len(CANDIDATES)]
            proposal = copy.deepcopy(current)
            if key in ("chunk_chars", "overlap_chars", "method", "min_score", "bm25_weight"):
                proposal["retrieval"][key] = value
            # Ensure valid chunk geometry.
            if proposal["retrieval"]["overlap_chars"] >= proposal["retrieval"]["chunk_chars"]:
                proposal["retrieval"]["overlap_chars"] = 100
            report = evaluate(proposal, "train", f"artifacts/iteration_{i}_train.json")
            obj = objective(report)
            accept = accepted(report, baseline, best_report)
            record = {"iteration": i, "proposal": {key: value}, "reason": "coordinate-search candidate",
                "proposer": "predeclared parameter search", "metrics": report["metrics"],
                "objective": obj, "accepted": accept}
            if accept:
                current, best_obj, best_report = proposal, obj, report
                record["status"] = "accepted"
            else:
                record["status"] = "rejected"
            log.write(json.dumps(record) + "\n")
    Path("artifacts/final_config.yaml").write_text(
        __import__("yaml").safe_dump(current, sort_keys=False), encoding="utf-8")
    evaluate(current, "train", "artifacts/final_train.json")
    return current

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    run_loop(args.config, args.iterations)
