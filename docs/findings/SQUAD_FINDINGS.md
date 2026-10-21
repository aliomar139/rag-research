# SQuAD: full-query quality, one completed timing pass

Received 21 October 2026: `artifacts/downloads/squad_one_pass_outputs.zip`.

## Verified scope

All 10,570 official SQuAD 1.1 dev questions were scored against 20,958 unique
title/paragraph documents from official train and dev. This is the declared
paragraph-retrieval adaptation, not an exact reproduction of the original
paper's index. Relevance is the annotated parent paragraph; answer text,
generated-answer EM and F1 are not evaluated.

The original job planned three warmed randomized passes. It was stopped after
one complete pass and 4,649 additional query repetitions. The completed pass
supplies headline quality and timings; additional records are preserved
separately and show no changed ranking IDs. The execution amendment records
the runtime decision before quality scoring. Do not describe this as three
completed timing repetitions or pool the partial second pass unevenly.

Local validation covers 84,560 first-pass retrieval records and timings,
frozen source/weights/protocol, branch counts, exact conditional branch
agreement and all supplemental-record counts. The local raw SQuAD dev data
reconstruct the exact Kaggle qrels SHA256:
`75e26c40fc8389017381b89939bfc4bc74da6681c9aff3e648398b4e723f0bbc`.
An independent parent-label calculation reproduces Recall@100, nDCG@10 and
MAP@10 for all eight methods within 1e-10.

Evidence is in `cascaded-blended-rag/results/paper_squad_one_pass/`, including
`local_verification.json` and `independent_metric_verification.json`.

## Frozen cascade results

**Table 1. Original slow-BM25 backend, one T4 timing pass. Retention is relative to static fusion Recall@100.**

| Training setting | Recall retained | Mean latency | Mean time saved | Early exits |
|:--|--:|--:|--:|--:|
| 90% (original primary) | 99.38% | 139.19 ms | 6.12% | 32.77% |
| 92% | 99.50% | 140.87 ms | 4.98% | 26.83% |
| 95% | 99.70% | 143.38 ms | 3.29% | 18.48% |
| 99% | 99.95% | 146.92 ms | 0.91% | 5.83% |
| Static fusion | 100.00% | 148.26 ms | Reference | 0% |

Every nominal target is met on this dataset. The original primary r90 retains
99.38% with a query-bootstrap interval of 99.22–99.54%; mean time reduction
is 6.12%, interval 5.93–6.31%. These intervals reflect variation across queries
from one complete pass, not stability across repeated timing sessions.

The quality outcome is stronger than the failed transfer targets on SCIDOCS
and TREC-COVID. Its execution saving is still small with the old backend.

## Standalone baselines

**Table 2. Recall and ranking metrics increase with better retrieval; lower mean latency is preferable.**

| Method | Recall@100 | Retained recall | nDCG@10 | MAP@10 | Mean latency |
|:--|--:|--:|--:|--:|--:|
| Static fusion | 0.99243 | 100.00% | 0.82897 | 0.79284 | 148.26 ms |
| BM25 | 0.96377 | 97.11% | 0.78831 | 0.75540 | 119.45 ms |
| Dense | 0.95393 | 96.12% | 0.68247 | 0.63339 | 10.72 ms |
| SPLADE | 0.98600 | 99.35% | 0.85367 | 0.82425 | 17.24 ms |
| Cascade r90 | 0.98628 | 99.38% | 0.81896 | 0.78335 | 139.19 ms |

SPLADE retains 99.35% of fusion recall at 8.60x fusion speed, with better
nDCG and MAP than fusion and every cascade. Cascade r90 has slightly higher
observed recall than SPLADE, so SPLADE does not strictly dominate every
cascade on every metric. That small recall difference is not evidence of
useful practical superiority for the much slower cascade.

BM25 alone already meets the user's 90% recall floor. It also retains over
90% of fusion nDCG and MAP. The score-preserving indexed backend should keep
these quality values while changing runtime. A useful gate contribution must
therefore be evaluated beyond the fact that it beats fusion on latency; a
constant cheap retriever is a necessary baseline.

## Indexed follow-up received

The short indexed follow-up has returned and passed artifact and independent
metric checks. See [SQUAD_INDEXED_FINDINGS.md](SQUAD_INDEXED_FINDINGS.md).
It measures 256 fixed-hash questions, the full corpus and three timing passes.
Primary r99 saves 4.19%; secondary r90 saves 37.08% with 99.22% retention.
The full-query indexed TREC run is also received. The next action is the
write-up using [FINAL_RESULTS.md](../FINAL_RESULTS.md); no further Kaggle run
is required to report the completed evidence.

The indexed study's conservative r99 primary was selected before these full
SQuAD outcomes were inspected and remains fixed, judged against the user's
90% floor. All four policies stay reported. Its sample metrics are not
full-query metrics. The earlier primary r90 and its full-query result above
are unchanged. Do not rerun the old long benchmark for this step.

## Reproduce local validation

```powershell
& '.\cascaded-blended-rag\venv\Scripts\python.exe' '.\cascaded-blended-rag\experiments\22_import_paper_benchmark.py' '.\artifacts/downloads/squad_one_pass_outputs.zip'
& '.\cascaded-blended-rag\venv\Scripts\python.exe' '.\cascaded-blended-rag\experiments\29_verify_squad_metrics.py'
```
