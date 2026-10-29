import json
import typer
from rich import print
from rich.panel import Panel
from .config import load_config
from .data import load_benchmark
from .chunking import load_corpus_for_queries
from .retrieval import Retriever
from .evaluate import evaluate

app = typer.Typer(help="Contract clause retrieval assistant")

def build_retriever(cfg):
    rows = load_benchmark(cfg["data"]["benchmark_path"])
    passages = load_corpus_for_queries(rows, cfg["data"]["corpus_dir"],
        cfg["retrieval"]["chunk_chars"], cfg["retrieval"]["overlap_chars"])
    return Retriever(passages, cfg["retrieval"]["method"],
        cfg["retrieval"]["bm25_weight"], cfg["retrieval"]["tfidf_weight"]), cfg

@app.command()
def ask(query: str, config: str = "config.yaml"):
    """Search the contract corpus for passages relevant to QUERY."""
    retriever, cfg = build_retriever(load_config(config))
    r = retriever.search(query, cfg["retrieval"]["top_k"], cfg["retrieval"]["min_score"])
    if not r:
        print("[yellow]No sufficiently relevant passage found.[/yellow]")
        raise typer.Exit()
    for i, item in enumerate(r, 1):
        print(Panel(f"[bold]Score:[/bold] {item['score']:.4f}\n[bold]Document:[/bold] {item['doc_path']}\n[bold]Offsets:[/bold] [{item['start']}:{item['end']}]\n\n{item['text'][:1800]}", title=f"Result {i}"))

@app.command()
def evaluate_cmd(split: str = typer.Option("train", "--split"), config: str = "config.yaml"):
    """Evaluate retrieval on the specified split."""
    cfg = load_config(config)
    out = evaluate(cfg, split, f"artifacts/evaluation_{split}.json")
    print(json.dumps({k:v for k,v in out.items() if k != "per_query"}, indent=2))

if __name__ == "__main__":
    app()
