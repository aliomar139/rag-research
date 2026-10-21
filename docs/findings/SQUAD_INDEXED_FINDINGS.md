# SQuAD indexed execution: verified findings

Received and verified 21 October 2026. The subsequent indexed TREC run is also
received and verified. Next action: use [FINAL_RESULTS.md](../FINAL_RESULTS.md)
and [the TREC findings](TREC_COVID_INDEXED_FINDINGS.md) in the write-up. No
additional Kaggle run or upload is required for the reported evidence.

## What changed

Precomputed BM25 postings replace repeated whole-corpus token scoring. Static
fusion, standalone BM25 and every cascade use this same implementation. Neural
models, Ridge weights, features and all four thresholds remain frozen. The
study measures retrieval execution; it does not evaluate generated answers.

## Evaluation scope

The full 20,958-document SQuAD title/paragraph corpus is retained. The evaluated
256 of 10,570 dev questions are the first IDs after sorting by
SHA256(`squad` + NUL + query ID), as declared before this run's outcomes.
All eight methods use these questions with three randomized warmed timing
repetitions on a Tesla T4 and one CPU thread. One timing session is recorded.
Index preparation takes 9.43 seconds; query means exclude preparation.

The preselected primary is **ridge11_r99**, assessed against a 90% recall
floor. Its 99% calibration target is reported separately. The other settings
are secondary. The earlier full-query experiment keeps its original r90
primary. This is an engineering follow-up on inspected dataset families.

## Measured results

**Table 1. Indexed retrieval on the fixed 256-question sample.** Retention is
the ratio of mean Recall@100 to static fusion. Fusion Recall@100 is 1.0 on
this sample. Means include actual query encoding and conditional execution.

| Method | Recall retained | nDCG@10 | MAP@10 | Mean ms | Mean time saved | Early exits |
|:--|--:|--:|--:|--:|--:|--:|
| Static fusion | 100.00% | 0.80931 | 0.77059 | 29.55 | Reference | 0.00% |
| BM25 | 95.70% | 0.77869 | 0.75038 | 1.03 | 96.50% | N/A |
| Dense | 95.70% | 0.68778 | 0.63040 | 10.82 | 63.37% | N/A |
| SPLADE | 98.83% | 0.85440 | 0.82831 | 16.96 | 42.61% | N/A |
| Cascade nominal 90% (secondary) | 99.22% | 0.80092 | 0.76757 | 18.59 | 37.08% | 39.84% |
| Cascade nominal 92% (secondary) | 100.00% | 0.80287 | 0.76757 | 21.31 | 27.89% | 30.47% |
| Cascade nominal 95% (secondary) | 100.00% | 0.80277 | 0.76464 | 24.82 | 15.99% | 17.97% |
| **Cascade nominal 99% (primary)** | **100.00%** | **0.81020** | **0.77082** | **28.31** | **4.19%** | **5.47%** |

The primary saves 4.19% mean time, with a query-bootstrap 95% interval of
1.90% to 7.04%. Its 14 exits preserve every sampled parent hit. The secondary
r90 saves 37.08% (31.58% to 42.69%), retaining 99.22% (98.05% to 100.00%).
It retrieves the annotated parent for 254 questions versus fusion's 256.
Secondary r92 saves 27.89% (22.91% to 33.04%) with all 256 parents retained.
All nominal calibration targets pass on this sample.

Zero sampled recall loss and bootstrap intervals of [1, 1] for r92/r95/r99
are empirical properties of these observations. They do not guarantee zero
loss on unmeasured questions. Intervals resample questions and do not capture
all hardware/session variation. The reported reduction is in mean time:
r90 p95 is 31.48 ms versus fusion's 31.37 ms, so a p95 improvement is not shown.

## What the evidence supports

Indexed execution makes conditional neural skipping save meaningful mean time
against equally optimized fusion at the more permissive secondary settings.
This is a measured system result. The conservative primary has a smaller
4.19% gain. Selecting r90 after observing the run does not make it primary.

BM25 alone already exceeds the requested 90% recall floor and is the fastest
listed method at 1.03 ms. It also retains over 90% of fusion nDCG and MAP.
If that floor alone determines the choice, this cascade adds unnecessary
cost on SQuAD. SPLADE leads nDCG and MAP and takes 16.96 ms. Its 253 parent
hits are fewer than r90's 254 and r92/r95/r99's 256, so it does not dominate
every cascade in all three quality metrics.

The result supports an execution improvement and a measured quality/time
trade-off. It does not establish a major new algorithm, general superiority
over standalone retrievers, a universal retention guarantee, or a speedup
over the original paper's Elasticsearch/ELSER implementation.

## Comparison with the original backend

All 2,048 indexed first-pass rankings match the original run **exactly** on
the same 256 questions: ordered document IDs, scores, routes, neural-call
decisions and per-query recall. Maximum score difference is zero for every
method. The query sample independently matches the declared hash selection.

**Table 2. Same questions across the original and indexed timing runs.**
The original has one complete pass; indexed has three. Ratios include
between-run variation and are not an interleaved backend experiment.

| Method | Original same-query mean ms | Indexed mean ms | Observed ratio |
|:--|--:|--:|--:|
| Static fusion | 147.01 | 29.55 | 4.98x |
| BM25 | 117.55 | 1.03 | 113.74x |
| Dense | 10.83 | 10.82 | 1.00x |
| SPLADE | 17.16 | 16.96 | 1.01x |
| Cascade nominal 90% | 134.92 | 18.59 | 7.26x |

Do not compare the old full-query fusion mean of 148.26 ms with the new
sample mean as if query sets matched. Do not attribute the sample's perfect
fusion recall to indexing: the same questions have identical old rankings.
Full-query quality remains in [SQUAD_FINDINGS.md](SQUAD_FINDINGS.md); the
37.08% indexed time saving has been measured only on these 256 questions.

## Artifact verification and reproduction

Input `artifacts/downloads/squad_indexed_cascade_outputs.zip`: 9,976,117 bytes, SHA256
`e1a07593ce7d16eb7660b95c975f8d241a925918e0a567942792d2101dfe0bfd`.
Results are extracted under `cascaded-blended-rag/results/indexed_squad_q256/`.

Experiment 22 validates 2,048 retrieval records and 6,144 timings, frozen
source/weight/protocol hashes, all query coverage, conditional ranking
identity, branch counts and three repetitions. Experiment 29 independently
reconstructs the exact qrels from raw SQuAD dev data and reproduces all
Recall@100/nDCG@10/MAP@10 values. Experiment 30 checks hash selection and
exact equality against the saved full-query original run.

```powershell
& '.\cascaded-blended-rag\venv\Scripts\python.exe' '.\cascaded-blended-rag\experiments\22_import_paper_benchmark.py' '.\artifacts/downloads/squad_indexed_cascade_outputs.zip'
& '.\cascaded-blended-rag\venv\Scripts\python.exe' '.\cascaded-blended-rag\experiments\29_verify_squad_metrics.py' --result-dir '.\cascaded-blended-rag\results\indexed_squad_q256'
& '.\cascaded-blended-rag\venv\Scripts\python.exe' '.\cascaded-blended-rag\experiments\30_compare_indexed_execution.py'
```

These commands read completed artifacts and regenerate checks; they perform
no retrieval, model training, download or GPU benchmark. Verification files
are `local_verification.json`, `independent_metric_verification.json` and
`paired_backend_comparison.json` in the result directory. The combined CSV
labels this result as an indexed sample.

## Next action

The full-query indexed TREC run is now received. Primary r99 retains 96.68%
recall and saves 19.04% mean time against equally indexed fusion; SPLADE
remains a stronger recall/time baseline with lower MAP. Use the completed
findings in the write-up. No further Kaggle run is required for this scope.

## Five-part claim review

1. Contribution: score-preserving execution improvement and conditional mean
   time savings are measured; algorithmic novelty remains unestablished.
2. Clarity: primary/secondary status, sampled/full quality and original/indexed
   timing scopes are explicit. Nominal target is not observed retention.
3. Strength: three warmed repetitions, all baselines, exact saved-rank agreement
   and independent scoring support the bounded result; one session limits it.
4. Completeness: all eight methods and metrics remain reported; modern adaptive
   baselines, memory/update costs and generated-answer evaluation are absent.
5. Soundness: frozen policies are retained; faster constant baselines prevent
   a claim that the cascade is the best solution for a 90% SQuAD floor.
