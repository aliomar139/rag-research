# One benchmark for 90%, 92%, 95%, and 99% retention

## What is ready

All eight policies are frozen using train data: four targets for the original
11-feature Ridge router and four for the 69-feature router. Kaggle does not
retrain models or select thresholds. Portable JSON weights avoid pickle/
scikit-learn version incompatibilities. Policies are targets, not guarantees.

Train-only nested CV for the original router gives:

| Target | Early exits | Recall retained | Folds meeting target |
|:--|--:|--:|--:|
| 90% | 42.93% | 90.26% | 3/5 |
| 92% | 36.10% | 92.19% | 2/5 |
| 95% | 24.05% | 95.43% | 3/5 |
| 99% | 4.32% | 99.42% | 4/5 |

These are development quality estimates, not speed measurements. Previously
inspected NFCorpus dev is used for the next measured development curve.

## Do these steps

1. Create a private Kaggle dataset with this exact title/slug:
   `nfcorpus-retention-tradeoff-20261020-v1`.
2. Upload **only** `nfcorpus_retention_tradeoff_source_v1.zip` from this folder.
3. Attach that dataset to a notebook. Enable Internet and a GPU. Paste
   `RUN_IN_KAGGLE.py` into one cell and run it. It locates ZIP or extracted
   source files without assuming a particular Kaggle mount layout, copies
   them into writable `/kaggle/working`, and reuses prior corpus indexes if
   the same session still has them. Budget roughly 10-30 minutes for setup,
   indexing if needed, and the benchmark; slow downloads can extend this.
4. Download `artifacts/downloads/nfcorpus_retention_benchmark_outputs.zip` from
   `/kaggle/working/nfcorpus_retention_tradeoff_v1/results/` and save it in the
   local project folder. Giving that filename confirms it is downloaded.

## What the run does

- Executes 12 methods on the same full corpus and dev queries: BM25, Dense,
  SPLADE, static RRF, and all eight frozen cascade policies. Every method
  uses the identical BM25 implementation where applicable.
- Times three repeats after warmup with randomized query/method order. Query
  results/features are not shared between methods. Early exits do not call
  Dense or SPLADE; branch counts are recorded and checked.
- Reports Recall@100 retention, nDCG@10, MAP@10, mean/p50/p95/p99 latency,
  skipped calls, and paired query-bootstrap intervals. A count-matched random
  exit reference checks recall quality; its latency is not measured.
- Produces `quality_speed.csv`, PNG/SVG plots, first-pass rankings/decisions,
  raw query timing samples, and a complete report in one result ZIP.
- Reads dev relevance grades only after routes/rankings/timings are fixed.
  It never reads test qrels. Dev has already been inspected and remains
  development evidence; this run is not an untouched final evaluation.

## How this leads to finishing the project

After the ZIP is returned, check whether any policy has useful measured speed
at acceptable observed retention, beats the random exit reference, and offers
a useful trade-off against standalone Dense/SPLADE as well as static RRF.
Report every requested target, including failures. Then freeze the selected
operating point for an independent confirmation on untouched data and use the
confirmed evidence for the final paper/report. A favorable contribution is
an empirical outcome, not guaranteed by lowering the retention target.
