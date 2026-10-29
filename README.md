# Contract Intelligence Assistant

## Overview

Contract Intelligence Assistant is an evidence-first passage retrieval system
for the LegalBench-RAG CUAD-mini benchmark. Given a contract question, it
ranks relevant contract passages and reports each source document and exact
character offsets. It does not generate legal conclusions or legal advice.

The measured final configuration uses 1,600-character overlapping chunks and
a hybrid BM25/TF-IDF ranker. The final test results are documented in
[`REPORT.md`](REPORT.md).

## Setup

### Requirements

- Python 3.10 or newer
- The CUAD benchmark and contract corpus
- Approximately 30 MB for the downloaded CUAD data, plus Python dependencies

No LLM or external model API is used, so no API key or local model
configuration is required.

### Option A: uv

Install [uv](https://docs.astral.sh/uv/) if necessary, then create the
environment and install the project with the application and test extras:

```bash
uv venv --python 3.10
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[app,test]"
```

To install from the locked dependency file instead:

```bash
uv sync --extra app --extra test
```

### Option B: pip

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[app,test]"
```

### Download the data

The benchmark mirror is `awinml/legalbench-rag` on Hugging Face. Download
`benchmarks/cuad.json` and the complete `corpus/cuad/` directory, preserving
these paths relative to the repository root:

```text
data/raw/benchmarks/cuad.json
data/raw/corpus/cuad/
```

The paths are configured in [`config.yaml`](config.yaml). The repository
intentionally does not include the raw benchmark or contract corpus.

For example, with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p data/raw
huggingface-cli download awinml/legalbench-rag \
  benchmarks/cuad.json corpus/cuad \
  --repo-type dataset --local-dir data/raw
```

If the mirror's JSON schema differs, inspect the downloaded file before
evaluation. The current loader supports both `snippets[].span: [start, end]`
and separate `start`/`end` (or `start_char`/`end_char`) fields.

## How to run

Run all commands from the repository root with the virtual environment active.

### Generate and inspect the fixed unanswerable set

Generate the prescribed 35-query set before tuning. Inspect it against the
indexed corpus and commit it before changing retrieval parameters:

```bash
python -m src.unanswerable
```

The resulting file is
[`data/unanswerable_queries.json`](data/unanswerable_queries.json). Do not
silently replace it after experiments begin.

### Validate benchmark offsets

The following command validates the selected benchmark rows and every gold
snippet against its original document:

```bash
python - <<'PY'
from src.config import load_config
from src.data import load_benchmark, validate_snippet_offsets

cfg = load_config("config.yaml")
rows = load_benchmark(cfg["data"]["benchmark_path"])
checks = validate_snippet_offsets(rows, cfg["data"]["corpus_dir"])
valid = sum(item["ok"] for item in checks)
print(f"{valid}/{len(checks)} spans valid")
assert valid == len(checks)
PY
```

### Search from the CLI

```bash
python -m src.cli ask "Does the agreement allow sublicensing?"
```

The CLI prints ranked passages, source document paths, character offsets, and
scores.

### Start the web app

```bash
streamlit run app.py
```

### Run evaluation

Evaluate only the training split during development:

```bash
python -m src.cli evaluate-cmd --split train
```

The command writes `artifacts/evaluation_train.json`. Do not evaluate the test
split while tuning.

### Run the improvement loop

The loop performs train-only coordinate search and writes one JSON record per
experiment to [`artifacts/experiment_log.jsonl`](artifacts/experiment_log.jsonl):

```bash
python -m src.improve --iterations 16
```

The acceptance rule requires improved train Recall@5, a Precision@5
guardrail, and unanswerable-query specificity no worse than the baseline.
The selected configuration is written to
[`artifacts/final_config.yaml`](artifacts/final_config.yaml).

### Run tests

```bash
pytest -q
```

## Architecture

Configuration is in [`config.yaml`](config.yaml). The data flow is:

1. [`src.data`](src/data.py) loads the mirror-compatible benchmark schema,
   selects the deterministic 194-query CUAD mini set, assigns prescribed
   train/test splits, resolves corpus paths, and validates gold offsets.
2. [`src.chunking`](src/chunking.py) loads only referenced contracts and
   creates overlapping character passages while preserving source offsets.
3. [`src.retrieval`](src/retrieval.py) builds BM25 and TF-IDF indexes and
   returns a weighted hybrid ranking.
4. [`src.metrics`](src/metrics.py) computes character-level precision and
   recall by document and span overlap.
5. [`src.evaluate`](src/evaluate.py) evaluates answerable queries and the
   fixed unanswerable set, writing detailed per-query JSON reports.
6. [`src.improve`](src/improve.py) tries declared train-only parameter
   candidates, applies guardrails, and logs accepted and rejected proposals.
7. [`src.cli`](src/cli.py) exposes search/evaluation commands, while
   [`app.py`](app.py) provides the Streamlit interface.

Retrieval never receives gold spans as input. Gold spans are used only by the
evaluation code.

## Reproducing results

After installing dependencies and downloading the data:

1. Generate and inspect the fixed unanswerable set:

   ```bash
   python -m src.unanswerable
   ```

2. Validate the 194 selected answerable queries and their gold spans using the
   validation command above. The recorded run validated 272/272 spans.

3. Run the train baseline and improvement loop:

   ```bash
   python -m src.improve --iterations 16
   ```

   This produces the baseline, iteration reports, final train report, final
   configuration, and experiment log under `artifacts/`. On a typical laptop,
   this takes roughly 1–3 minutes depending on Python and disk performance.

4. Run the final test evaluation exactly once, using the selected configuration:

   ```bash
   python - <<'PY'
   from src.config import load_config
   from src.evaluate import evaluate

   report = evaluate(
       load_config("artifacts/final_config.yaml"),
       split="test",
       save_path="artifacts/final_test.json",
   )
   print(report["metrics"])
   PY
   ```

   This takes roughly 10–30 seconds on a typical laptop. Do not rerun it
   while tuning; the reported run count is one.

5. Run the unit tests:

   ```bash
   pytest -q
   ```

The final recorded split sizes are 140 train and 54 test answerable queries,
with 26 train and 9 test unanswerable queries. The measured outputs are
preserved in `artifacts/` and summarized in [`REPORT.md`](REPORT.md).

## Assumptions

- **Dataset selection:** The assignment's deterministic `mini` function and
  SHA-256 split assignment are authoritative, producing 194 selected
  answerable queries. This avoids changing the benchmark distribution.
- **Mirror schema:** The downloaded mirror uses `snippets[].span`; the loader
  also accepts common separate offset field names to avoid relying on a
  single mirror representation.
- **Document resolution:** Benchmark paths may include a `cuad/` prefix while
  local files live below `data/raw/corpus/cuad/`. Path resolution tries both
  representations and the basename without modifying document contents.
- **Text offsets:** Contracts are read as UTF-8 text with Python's default
  newline handling. Offsets are interpreted as Python character offsets, as
  required by the benchmark.
- **Unanswerable queries:** The 35 generated questions are treated as a fixed
  evaluation set after manual corpus inspection. Lexical absence does not prove
  semantic unanswerability, so specificity results are reported but not
  treated as calibrated certainty.
- **Retrieval objective:** Character-level overlap is used because the
  benchmark provides gold character spans. It favors broad evidence passages
  and is not equivalent to answer correctness.
- **Abstention:** A query is considered abstained only when no retrieved
  passage meets `retrieval.min_score`. The threshold is a train-tuned heuristic,
  not a probability.
- **Test discipline:** The test split is run once after tuning. Test metrics
  must not influence parameter selection.
- **System scope:** The assistant returns evidence, not synthesized legal
  advice; human review is required for any legal use.

## AI tool usage

AI assistance was used to scaffold and debug the Python implementation,
inspect the benchmark schema, add mirror-compatible offset handling, improve
the train-only acceptance rule, and draft this documentation. Shell commands,
`curl`, Python, `pytest`, and Git were used to download/validate data, run
experiments, and verify results. No LLM was used by the retrieval system, and
no contract text was sent to an external inference API.
