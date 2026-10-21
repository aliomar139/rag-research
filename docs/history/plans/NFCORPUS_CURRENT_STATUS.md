> **Historical document.** This file predates the consolidated submission and may contain superseded claims or commands. Use [the current manuscript](../../FINAL_MANUSCRIPT.md) for conclusions.

> **Superseded on 20 October 2026.** This document preserves earlier project context.
> The completed experiment sequence and next steps are in [FINAL_RESULTS.md](../../FINAL_RESULTS.md);
> the current manuscript is [FINAL_MANUSCRIPT.md](../../FINAL_MANUSCRIPT.md).
> Earlier requests for more Kaggle runs or tuning below are historical.

# NFCorpus: current status and next actions

## Active plan: four quality budgets (2026-10-20)

**The NFCorpus benchmark is now completed.** The downloaded result is
`artifacts/downloads/nfcorpus_retention_benchmark_outputs.zip`; verified extracted artifacts are
in `cascaded-blended-rag/results/retention_tradeoff_dev/`. Read
`docs/findings/NFCORPUS_BENCHMARK_FINDINGS.md` for the outcome and baseline comparison.
The actual Kaggle source dataset was named
`nfcorpus-retention-tradeoff-20261020-v1.2`.

Original-gate 90/92/95/99% targets retain 92.33/94.74/96.23/99.27% dev recall,
at 17.07/18.18/19.67/23.14 ms, versus static RRF's 23.61 ms. Dense alone retains
99.77% recall at 6.62 ms, so the cascade does not establish best recall/latency
performance against the strong baselines. SPLADE has better nDCG/MAP.

**Current next action:** use
`notebooks/kaggle_scidocs_confirmation_v1/START_HERE.md` for one predeclared
full-corpus SCIDOCS confirmation. It keeps the same original gate weights and
all four thresholds, uses r92 as the primary development-selected candidate,
and reports all baselines/targets. No model or threshold tuning on SCIDOCS.
Download `artifacts/downloads/scidocs_confirmation_outputs.zip`, then consolidate the two datasets
into the final report with the strong-baseline and transfer limitations.

The following paragraphs preserve the completed preparation context:

The user now accepts a quality/speed trade-off. Evaluate 90%, 92%, 95%, and
99% recall retention instead of continuing to optimize only the 99% constraint.
Use `notebooks/kaggle_retention_tradeoff_v1/START_HERE.md` as the current run
guide and its `RUN_IN_KAGGLE.py` cell. The source dataset title is
`nfcorpus-retention-tradeoff-20261020-v1`.

Train-only nested curves and frozen policies are prepared in
`results/nfcorpus_retention_train_curve.json` and
`results/nfcorpus_frozen_retention_routers.json`. For the original Ridge gate,
90/92/95/99% targets give 42.93/36.10/24.05/4.32% exits and
90.26/92.19/95.43/99.42% pooled retention. Folds can miss their nominal target;
these are development estimates, not guarantees or latency results.

Next: one Kaggle run measures all four targets for the 11- and 69-feature
routers, alongside standalone BM25/Dense/SPLADE and static RRF. The returned
ZIP contains the complete quality/speed curve, uncertainty summaries, and
count-matched random-exit quality references. NFCorpus dev remains inspected
development evidence. An untouched confirmation is still needed before
claiming independent generalization. Defer further risk-objective/model
tuning while this requested budget comparison is completed.

Corpus file received: `artifacts/downloads/nfcorpus_corpus_for_features.jsonl` at the workspace
root. Its SHA256 matches the corpus used for the saved train labels. When the
user supplies a filename, treat it as confirmation that the downloaded file
is in the project folder; locate it before asking for another upload.

## Completed locally

`experiments/15_compare_nfcorpus_coverage_features.py` added 58 corpus/query/
BM25 features to the existing 11. They cover score concentration, weighted
query-token coverage, content-word coverage, title/abstract agreement, and
document length. Feature computation uses no relevance labels or neural
rankings. All 2,590 BM25 top-100 rankings matched the saved ranking IDs.

The four fixed model specifications and nested folds match the earlier CPU
comparison. Each threshold is selected on inner-fold labels and scored on
outer-fold queries from train. Dev and test were not opened.

| Model | Original exits | Enriched exits | Enriched recall/static | Folds meeting 99% |
|:--|--:|--:|--:|--:|
| Ridge | 4.32% | 6.87% | 99.33% | 4/5 |
| Depth-4 tree | 2.82% | 3.51% | 99.39% | 3/5 |
| ExtraTrees-64 | 7.10% | 7.57% | 98.95% | 4/5 |
| Histogram gradient boosting | 4.02% | 5.64% | 99.32% | 4/5 |

The richer features give a modest improvement. No enriched candidate meets
the 99% target in every outer fold. ExtraTrees misses the pooled target.
Computing the extra features averaged 0.468 ms locally (p95 0.727 ms), in
addition to BM25 and model inference. This is not a Kaggle end-to-end timing
measurement and does not establish a speedup.

Twelve local tests passed, including ranking ties, empty/OOV queries, coverage
values, and the earlier oracle/report checks. The full train run also checked
every saved BM25 ranking against the current implementation.

## What to do now

1. Follow `notebooks/kaggle_retention_tradeoff_v1/START_HERE.md` for the new
   four-target benchmark, using its source ZIP and exact dataset title.
2. Save the returned `artifacts/downloads/nfcorpus_retention_benchmark_outputs.zip` in the project
   folder. The original corpus and train-label inputs are already local.
3. Follow the active four-budget benchmark plan above. The previously proposed
   risk-objective change is deferred; no more model/feature tuning is needed
   before measuring the requested four operating points.

## Reproduce this completed comparison

From `C:\Users\user\Desktop\rag-research-alone` in PowerShell:

```powershell
& '.\cascaded-blended-rag\venv\Scripts\python.exe' -u '.\cascaded-blended-rag\experiments\15_compare_nfcorpus_coverage_features.py'
```

This reuses the validated coverage feature cache when available and overwrites
the local comparison JSON. It does not contact Kaggle or load neural models.
