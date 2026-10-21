# Train-only router diagnostics

The previous dev run retained 99.27% of static recall and exited on 13/324
queries. Mean latency fell only 0.46%. This diagnostic investigates the train
exit opportunities before another policy change.

1. Upload `nfcorpus_train_diagnostics_source_v1.zip` from this folder as a new
   private Kaggle dataset titled `nfcorpus-train-router-diagnostics-20261020-v1`.
2. Attach that dataset to a notebook. Enable Internet and a GPU. An existing
   session can reuse the previous full-corpus indexes; a new session can
   download and rebuild them.
3. Paste all of `RUN_IN_KAGGLE.py` into one notebook cell and run it. Budget
   roughly 10-30 minutes including setup; cached indexes shorten the run.
4. Download `artifacts/downloads/nfcorpus_train_diagnostics_outputs.zip` from
   `/kaggle/working/nfcorpus_train_diagnostics_v1/results/` and send it back.

## What it saves

- `nfcorpus_train_router_diagnostics.json`: Ridge versus oracle exit rates.
- `nfcorpus_train_router_examples.jsonl`: training labels, features, query IDs,
  and BM25/static top-100 document IDs. Saved incrementally and validated by
  source/data hashes before reuse.
- `nfcorpus_train_router_decisions.jsonl`: per-query OOF predictions and Ridge/
  oracle routing decisions.
- `nfcorpus_train_threshold_frontier.json`: train threshold candidates.
- `nfcorpus_train_examples_manifest.json`: provenance for reusable examples.

Only train qrels are opened. The primary oracle uses the same clipped loss
budget as the existing Ridge gate; another oracle credits queries where BM25
beats static RRF. Both use known training labels and are retrospective upper
bounds, not attainable speedup claims. The Ridge threshold is selected on all
train labels using OOF predictions; the report is a development diagnostic.
