# Indexed TREC-COVID: verified execution improvement

Received and verified 21 October 2026. Next: review the result in
[FINAL_RESULTS.md](../FINAL_RESULTS.md) and use the bounded claim below in the
write-up. No further Kaggle run or source upload is required to report it.

## Main result

The preselected **ridge11_r99** policy reduces mean retrieval time from
**91.28 to 73.90 ms**, saving **19.04%**, while retaining **96.68%** of
optimized fusion Recall@100. The query-bootstrap 95% intervals are
**9.47% to 30.11%** for mean time saved and **93.58% to 98.75%** for recall
retained. Both lower bounds meet the follow-up's positive-time-saving and
90%-retention criteria in this measured setting.

This passes the user's 90% floor. It misses the policy's original nominal
99% calibration target. All four nominal calibration targets still fail on
TREC-COVID; the indexing change preserves their retrieval results exactly.

## Scope and fixed decisions

The run covers **all 50 official BEIR queries and all 171,332 documents**,
eight methods and three warmed randomized timing repetitions, one Tesla T4
session and one CPU thread. Query limit 0 means all queries. Preparation
takes 62.91 seconds using existing corpus representations and is excluded
from query latency. Neural query inference executes during timing; old query
rankings are used only for the comparison after timing.

Static fusion, standalone BM25 and every cascade use the same indexed BM25
implementation. Models, features, weights and thresholds remain unchanged.
The follow-up's r99 primary was declared before indexed outcomes using
already-inspected collections, including original TREC quality. This is an
engineering follow-up, not a new independent quality-generalization claim.
The original TREC experiment retains its r90 primary and failed 90% floor.

## All measured methods

**Table 1. Indexed TREC-COVID, full corpus and all queries.** Retention is
the ratio of mean Recall@100 scores to static fusion, whose Recall@100 is
0.12250. Higher quality and lower mean latency are preferable.

| Method | Recall retained | nDCG@10 | MAP@10 | Mean ms | Mean time saved | Early exits |
|:--|--:|--:|--:|--:|--:|--:|
| Static fusion | 100.00% | 0.76323 | 0.01861 | 91.28 | Reference | 0% |
| BM25 | 71.38% | 0.54969 | 0.01145 | 8.11 | 91.11% | N/A |
| Dense | 69.81% | 0.47254 | 0.01054 | 30.41 | 66.68% | N/A |
| SPLADE | 104.69% | 0.72899 | 0.01662 | 52.77 | 42.19% | N/A |
| Nominal 90%, secondary | 84.53% | 0.65115 | 0.01499 | 40.41 | 55.74% | 62% |
| Nominal 92%, secondary | 84.53% | 0.65315 | 0.01499 | 41.99 | 54.00% | 60% |
| Nominal 95%, secondary | 91.95% | 0.70561 | 0.01742 | 55.66 | 39.02% | 44% |
| **Nominal 99%, primary** | **96.68%** | **0.72281** | **0.01769** | **73.90** | **19.04%** | **22%** |

The primary exits on 11 queries, avoiding 33 Dense and 33 SPLADE calls across
three repetitions. Each neural branch runs 117 times versus fusion's 150.
Its 1.24x speedup is relative to equally optimized fusion. Primary p95 is
99.41 ms versus fusion's 97.66 ms, so a p95 improvement is not demonstrated.
The result concerns mean retrieval latency with prepared indexes.

The secondary r95 point saves 39.02% (26.54% to 51.94%) and retains 91.95%
(86.88% to 95.76%). Its lower recall interval crosses below 90%, and its
point retention misses 95%. It is a useful reported trade-off, not the primary
or a guaranteed 90% operating point. r90/r92 fail the requested floor even
at their point estimates.

## Standalone comparison and interpretation

BM25 and Dense alone fail the 90% recall floor. SPLADE alone exceeds that
floor, takes 52.77 ms, and has higher observed recall and nDCG than every
cascade. It is the fastest listed method whose point recall meets the floor.
The cascade is therefore not the best measured choice when recall and mean
time alone determine the decision.

The primary cascade has higher MAP@10 than SPLADE: 0.01769 versus 0.01662,
a descriptive relative difference of 6.43%. No significance claim is made
for this difference. SPLADE does not dominate every method on every quality
metric. Conversely, this MAP advantage cannot justify replacing the original
recall-only floor with a new selection objective after seeing the results.

Primary Recall@100, nDCG@10 and MAP@10 retain 96.68%, 94.70% and 95.09% of
fusion, respectively. Retention is relative quality: absolute Recall@100 is
0.11844 versus fusion's 0.12250. The corpus has many relevant documents per
topic; a top-100 ranking cannot retrieve them all. No generated-answer
quality, production-load or universal retention guarantee was evaluated.

## What the backend change achieved

Every method's 50 first-pass rankings match the old run exactly: document
IDs and order, scores, routes, calls and per-query recall. The maximum score
difference is zero for all eight methods. Scores and routing quality were
preserved while the earlier first-stage cost was reduced.

**Table 2. Same queries across original and indexed runs.** Both have three
complete timing passes. These are separate runs, not interleaved randomized
backend timings; ratios include run variability.

| Method | Original mean ms | Indexed mean ms | Observed ratio |
|:--|--:|--:|--:|
| BM25 | 1410.14 | 8.11 | 173.78x |
| Static fusion | 1493.23 | 91.28 | 16.36x |
| Dense | 30.34 | 30.41 | 1.00x |
| SPLADE | 52.98 | 52.77 | 1.00x |
| Primary r99 | 1475.73 | 73.90 | 19.97x |

The cross-run ratio of old to new r99 time is not the cascade contribution
against a fair optimized reference. **19.04% less mean time than indexed
fusion** is that comparison. Indexed execution is an engineering improvement;
eager sparse scoring does not establish new algorithmic priority.

## Safe wording for the report

> On all 50 BEIR TREC-COVID queries with the full corpus, the frozen
> conservative cascade reduced mean retrieval latency by 19.04% relative to
> equally optimized static fusion while retaining 96.68% of its Recall@100.
> Standalone SPLADE remained faster with higher recall and nDCG, whereas the
> cascade retained higher MAP@10. The result demonstrates an execution
> trade-off without establishing overall method superiority.

The system is our MiniLM/SPLADE/RRF adaptation. It does not establish an
acceleration of the original paper's Elasticsearch/ELSER implementation.
TREC and SQuAD are now measured within that scope; full NQ/HotpotQA remain
pending and resource dependent. Further experiments are not required to
report these completed observations and their limits.

## Evidence and local reproduction

Input `artifacts/downloads/trec_covid_indexed_cascade_outputs.zip`, 697,499 bytes. SHA256:
`92857af16ed9944a90494c79ddabef3286cff3f0f1c49a274ba2f1dfbd882529`.
Local evidence directory: `cascaded-blended-rag/results/indexed_trec-covid_q0/`.

Experiment 22 validates all 400 retrieval records, 1,200 timings, frozen
source/weight/protocol/model fingerprints, branch counts and conditional
ranking equality. Experiment 30 independently checks all-query coverage,
asset equality and exact old/new rankings. Experiment 31 reads the local
raw qrels with SHA256
`10669ab7d526cb04f52079139fd88c3d467a0776441b046567f540582798982b`
and independently reproduces all per-query recalls and aggregate
Recall@100/nDCG@10/MAP@10 values. The original TREC report has also received
that independent metric verification. Timing instrumentation is not rerun.

```powershell
& '.\cascaded-blended-rag\venv\Scripts\python.exe' '.\cascaded-blended-rag\experiments\22_import_paper_benchmark.py' '.\artifacts/downloads/trec_covid_indexed_cascade_outputs.zip'
& '.\cascaded-blended-rag\venv\Scripts\python.exe' '.\cascaded-blended-rag\experiments\30_compare_indexed_execution.py' --old-dir '.\cascaded-blended-rag\results\paper_trec-covid' --indexed-dir '.\cascaded-blended-rag\results\indexed_trec-covid_q0'
& '.\cascaded-blended-rag\venv\Scripts\python.exe' '.\cascaded-blended-rag\experiments\31_verify_beir_metrics.py'
```

Already executed successfully. These read completed artifacts, with no
retrieval, training, downloads or GPU benchmark. The combined CSV explicitly
labels all 50 TREC queries separately from the 256-question SQuAD sample.

## Five-part review

1. Contribution: measured score-preserving execution improvement and 19.04%
   conditional mean saving; no novel algorithm or best-retriever claim.
2. Clarity: primary success at 90%, nominal 99% failure, full-query scope and
   mean versus p95 are all distinguished.
3. Strength: repeated measured execution, independent raw-label scoring and
   exact old/new ranking agreement; one session and 50 topics limit inference.
4. Completeness: all eight methods and thresholds remain visible; modern
   adaptive baselines, answer evaluation and full NQ/HotpotQA are absent.
5. Soundness: no retuning; equal indexing for reference and proposed method.
   Inspected-data selection and SPLADE's stronger recall/time result prevent
   an independent-generalization or practical-superiority claim.
