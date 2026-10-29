# Evaluation Report

## Summary
The final system uses 1,600-character overlapping chunks and a BM25/TF-IDF
hybrid ranker. On the 54 answerable CUAD test queries, it achieved
character Recall@5 of 0.2184 and Precision@5 of 0.0125 (Recall@10 0.3653,
Precision@10 0.0128). The fixed test set also contains 9 unanswerable
queries; the system abstained on 0/9, so test specificity was 0.0%. The test
split was evaluated exactly once.

## System design
The corpus is split into 1,600-character passages with 200 characters of
overlap. A BM25 score normalized per query is blended with word unigram/bigram
TF-IDF cosine similarity using weights 0.55/0.45. The minimum score is 0.08
and up to 10 passages are returned. The search index contains only documents
referenced by the selected split's benchmark queries, without passing gold
spans to retrieval. Results retain the source path and exact character
offsets. The mirror uses `snippets[].span`; the loader supports that schema
and the alternate separate start/end fields. BM25-only, TF-IDF-only, chunk
sizes 800/1200/1600, overlap 100/200, and score thresholds through 0.5 were
tried.

## LLM use
No LLM was used at retrieval or evaluation time. The system is deterministic
and local after dataset download, so there is no model API cost or prompt
latency and no contract text is sent to an inference provider. AI assistance
was used to scaffold/debug the implementation and draft this report; the
code and measured outputs were reviewed and tested.

## Evaluation setup
- Dataset: LegalBench-RAG-mini, CUAD only.
- Query selection: benchmark authors' `mini` function, n=194.
- Split: prescribed SHA-256 `assign_split`.
- Train answerable queries: 140
- Test answerable queries: 54
- Unanswerable train/test counts: 26/9
- Test evaluation runs: 1
- Offset validation: 272 valid / 272 checked

The 35 proposed unanswerable queries were inspected against all 462 indexed
contract files: none occurred as an exact query string. They were then fixed
in `data/unanswerable_queries.json` before tuning. This lexical check cannot
prove semantic unanswerability.

## Results

| Configuration | Split | Queries | Precision@1 | Recall@1 | Precision@3 | Recall@3 | Precision@5 | Recall@5 | Precision@10 | Recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Train | 140 | 0.0032 | 0.0106 | 0.0064 | 0.0580 | 0.0168 | 0.1875 | 0.0173 | 0.3821 |
| Final | Train | 140 | 0.0053 | 0.0309 | 0.0151 | 0.1390 | 0.0179 | 0.2888 | 0.0147 | 0.4493 |
| Baseline | Test | 54 | 0.0047 | 0.0441 | 0.0101 | 0.1656 | 0.0125 | 0.2184 | 0.0128 | 0.3653 |
| Final | Test | 54 | 0.0047 | 0.0441 | 0.0101 | 0.1656 | 0.0125 | 0.2184 | 0.0128 | 0.3653 |

The train/test Recall@5 gap is 0.0704 (0.2888 versus 0.2184); the
Precision@5 gap is 0.0054 (0.0179 versus 0.0125).

## Abstention
At the final threshold (0.08), train specificity was 0/26 = 0.0% and test
specificity was 0/9 = 0.0%. False-abstention on answerable queries was 0/140
= 0.0% on train and 0/54 = 0.0% on test. Threshold candidates 0.0, 0.04,
0.08, 0.12, 0.2, 0.3, 0.4, and 0.5 were evaluated on train only. Acceptance
required higher Recall@5, Precision@5 no more than 0.02 below the current
best, and unanswerable specificity no lower than baseline. No tested
threshold improved abstention without violating the recall objective.

## Iteration history
The primary objective was train character Recall@5. Guardrails were
Precision@5 within 0.02 of the current best and unanswerable specificity at
least the baseline. The baseline and 16 proposals are recorded in
[`artifacts/experiment_log.jsonl`](artifacts/experiment_log.jsonl).

| Iteration | Proposal | Result | Recall@5 | Precision@5 | Specificity |
|---:|---|---|---:|---:|---:|
| 0 | baseline, 1200 chars | baseline | 0.1875 | 0.0168 | 0.0% |
| 1 | 800-char chunks | rejected | 0.0885 | 0.0105 | 0.0% |
| 3 | 1600-char chunks | accepted | 0.2888 | 0.0179 | 0.0% |
| 4 | 100-char overlap | rejected | 0.2747 | 0.0193 | 0.0% |
| 6 | TF-IDF | rejected; specificity 10/26 | 0.1647 | 0.0114 | 38.5% |
| 7 | BM25 | rejected | 0.2617 | 0.0166 | 0.0% |
| 13 | threshold 0.2 | rejected | 0.2888 | 0.0179 | 0.0% |
| 16 | threshold 0.5 | rejected | 0.0826 | 0.0091 | 0.0% |

## Error analysis
The final test run exposed the following concrete top-1 failures. Predicted
and gold offsets are shown as `[start:end]`.

| Query ID | Group | Predicted document/offset | Gold document/offset | Diagnosis / next experiment |
|---|---|---|---|---|
| 8 | right document, wrong section | HPIL cooperation `[2800:4400]` | same document `[18172:18336]` | Long contract; add clause-aware sections and rerank termination terms. |
| 14 | right document, wrong section | Theravance service `[0:1600]` | same document `[32015:32250]` | Governing-law vocabulary is weak; add heading-aware retrieval. |
| 18 | right document, wrong section | Theravance service `[0:1600]` | same document `[16945:17247]` | IP ownership is remote; use larger context/reranking. |
| 21 | right document, wrong section | Cano sponsorship `[0:1600]` | same document `[4159:4489]` | Query identifies document but not clause; add section titles. |
| 23 | partial span coverage | Cano sponsorship `[0:1600]` | same document `[1116:1204]` | Gold clause is near a chunk edge; tune boundary/overlap. |
| 27 | right document, wrong section | Vitamin Shoppe sponsorship `[0:1600]` | same document `[54279:54600]` | Liability clause is far from lead terms; add document-level reranking. |
| 34 | right document, wrong section | Duo strategic alliance `[0:1600]` | same document `[20426:21063]` | Non-compete appears late; use clause-aware indexing. |
| 37 | query vocabulary mismatch | Duo strategic alliance `[0:1600]` | same document `[36711:37219]` | Anti-assignment synonym mismatch; expand legal synonyms. |
| 38 | query vocabulary mismatch | Duo strategic alliance `[0:1600]` | same document `[11393:11877]` | Revenue/profit-sharing terms need query expansion. |
| 42 | cross-reference/definition dependence | Duo strategic alliance `[0:1600]` | same document `[12612:12792]` | Post-termination obligations need cross-reference expansion. |
| 43 | right document, wrong section | Duo strategic alliance `[0:1600]` | same document `[32028:32397]` | Uncapped liability clause is remote; add liability reranking. |
| 49 | partial span coverage | Flotek supply `[0:1600]` | same document `[16222:16306]` | Short late clause; increase recall before precision. |

These are predominantly right-document/wrong-section failures, not wrong
document failures. No answerable query abstained, but all 9 synthetic
unanswerable test queries also returned a passage, which is an incorrect
abstention decision.

## Limitations and next steps
The benchmark is small (194 selected answerable queries) and character
overlap rewards broad chunks while producing low precision. The 35
unanswerable queries are synthetic and exact-string absence does not establish
semantic absence; the 0/9 specificity result is therefore not a calibrated
claim. Threshold calibration needs harder negatives and a held-out validation
set. Clause-aware chunking, heading/section indexing, legal synonym
expansion, cross-reference and definition expansion, and a cross-encoder
reranker are the next experiments. Local sparse retrieval is inexpensive and
has no external privacy transfer, but latency and memory should be measured
on the production corpus. This prototype must not be used for legal reliance
without stronger abstention evaluation, human review, broader contracts, and
security/privacy controls.
