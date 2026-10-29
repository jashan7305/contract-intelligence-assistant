# Contract Intelligence Assistant

Evidence-first contract passage retrieval for the LegalBench-RAG CUAD-mini task. The system ranks passages and returns document paths and character offsets; it does not generate legal conclusions.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -e ".[app,test]"
```

Download the LegalBench-RAG dataset from `awinml/legalbench-rag` and place the files so these paths exist:

- `data/raw/benchmarks/cuad.json`
- `data/raw/corpus/cuad/`

Only CUAD is used. The implementation reproduces the 194-query selection function in the assignment and indexes only documents referenced by those selected queries. Run offset validation before trusting metrics. The benchmark schema may vary by mirror; inspect the actual JSON and adapt `src.data.validate_snippet_offsets` if its snippet text/offset field names differ.

## Commands

```bash
# Generate the fixed unanswerable set; commit it before running the improvement loop.
python -m src.unanswerable

# Search from the CLI
python -m src.cli ask "Does the agreement allow sublicensing?"

# Evaluate only train during development
python -m src.cli evaluate-cmd --split train

# Streamlit demo
streamlit run app.py

# Tests
pytest -q

# Five or more train-only experiments
python -m src.improve --iterations 5
```

Do not run the test split during tuning. Run it once for final reporting and record the count in `REPORT.md`.

## Architecture

1. `src.data`: deterministic mini selection, split assignment, path resolution and offset validation.
2. `src.chunking`: overlapping character chunks with exact source offsets.
3. `src.retrieval`: BM25, TF-IDF, or a weighted hybrid ranker.
4. `src.metrics` / `src.evaluate`: character-span overlap scoring and per-query diagnostics.
5. `src.improve`: train-only coordinate search and JSONL experiment logging.
6. `src.cli` / `app.py`: command-line and Streamlit interfaces.

Pipeline settings are in `config.yaml`.

## Reproducibility

Run commands from the repository root. Save the final test report under `artifacts/` and include the exact command, query count, timestamp, and number of test runs in the report. Do not use ground-truth spans as retrieval-time inputs.

## Assumptions and limitations

- Character chunks are used as a simple baseline; clause-aware chunking is a candidate improvement.
- Retrieval returns evidence passages rather than synthesized answers.
- The configured score threshold is a heuristic, not a calibrated probability. It must be tuned on train, including the unanswerable train set.
- File paths in the benchmark may differ from paths in the local corpus. Validate them before interpreting metrics.
- The unanswerable queries are proposed examples, not guaranteed to be absent from every contract. Manually check them against the indexed corpus before committing the set; after that initial commit, do not change them.
- This is a prototype and must not be relied on for legal advice.

## AI tool usage

Disclose the AI tools used to scaffold code, debug issues, and draft documentation. The author must review, understand, test, and be able to modify all generated code before submission.


## Important implementation checks before submission

This repository is a runnable scaffold, not a claim of completed benchmark results. Before reporting numbers:
1. Run offset validation and confirm all 194-query snippets resolve and match `document[start:end]`.
2. Confirm the mirror's exact schema for `file_path`, `start`, `end`, and answer text; adapt the compatibility accessors if needed.
3. Manually inspect the 35 proposed unanswerable queries against the indexed corpus, then commit `data/unanswerable_queries.json` before any tuning.
4. The starter improvement loop performs a small parameter search on train retrieval metrics. Extend its acceptance rule to use the unanswerable train metrics and ensure every required experiment is logged.
5. The evaluation currently treats a returned passage as answerable when it passes the configured threshold. Calibrate that threshold on train only and document that synthetic unanswerable queries are not guaranteed to be truly absent.
