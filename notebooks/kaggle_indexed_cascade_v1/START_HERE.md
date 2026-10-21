# Indexed-BM25 follow-up: short, matched timing benchmark

## Current action

Let the current full SQuAD run finish. Download
`squad_paper_benchmark_outputs.zip` first. Keep that Kaggle session and its
working files: they contain the corpus indexes needed by this short follow-up.

## Prepared upload

File:

```text
C:\Users\user\Desktop\rag-research-alone\notebooks\kaggle_indexed_cascade_v1\indexed_cascade_source_20261021_b8e6_v1.zip
```

New private Kaggle dataset title:

```text
indexed-cascade-followup-20261021-b8e6-v1
```

The title differs from earlier project datasets. Upload the ZIP only.

## Five steps, after the current job finishes

1. Download and preserve the full original SQuAD result ZIP.
2. Upload the new source ZIP above as a private dataset. Add it to the same
   notebook; keep the previous source input attached too.
3. Paste `PASTE_CELL.py` into a new cell. Keep `DATASET = "squad"` and
   `QUERY_LIMIT = 256`. Run it with GPU and Internet enabled.
4. The runner reuses existing Dense/SPLADE corpus indexes. It measures all
   eight methods on 256 queries selected by a fixed hash, using the full
   corpus and three repeats. Budget 5–20 minutes if the indexes remain cached;
   this is an estimate, not an end-to-end measured runtime.
5. Download `artifacts/downloads/squad_indexed_cascade_outputs.zip` and put it in the local
   workspace root. Send the filename in chat.

The launcher refuses to start when the original result or its indexes are
missing. It does not silently regenerate expensive corpus representations.
If you closed the original session, preserve the result and report that the
working files are gone; the short-cache path then needs different setup.

## What changes

BM25 precomputes term contributions into postings. The current rank_bm25
formula, tokenization, repeated query terms, epsilon-adjusted IDFs, Best
Fields max, float64 score accumulation and corpus-order ties are retained.
Frozen gate weights, thresholds and neural models are unchanged. Static
fusion and standalone BM25 use the same optimized scorer as the cascades.
The old source files and old results are preserved.

This does not fix cross-dataset threshold calibration. It tests whether the
first-stage runtime hid useful cascade savings. For this new follow-up only,
the existing conservative `ridge11_r99` policy is primary, judged against the
user's 90% retention floor. It was selected using the already inspected
NFCorpus/SCIDOCS/TREC-COVID evidence, before the indexed follow-up or current
full SQuAD outcomes were inspected. No model or threshold is refitted.
The earlier studies keep their original primary policies. All four policies
remain reported, including whether they meet their original calibration
labels. Meeting a 90% floor must not be described as achieving a 99% target.

## What is already measured locally

Full TREC-COVID: 171,332 documents, all 50 queries, three randomized paired CPU
passes. Reference BM25 averaged 908.70 ms and indexed BM25 6.74 ms (134.85x).
Every full-corpus document score and top-100 order matched exactly in every
query repetition. This is a local CPU engineering result, not a Kaggle GPU
cascade result or a novel BM25 algorithm. Machine details are saved in
`cascaded-blended-rag/results/indexed_bm25_cpu_trec-covid.json`.

## Reading the follow-up result correctly

- SQuAD quality and timings from this follow-up cover the fixed 256-query
  sample, not all 10,570 questions. The earlier full evaluation stays visible.
  The document corpus is complete. The hash sample is independent of grades.
- The old timings are not subtracted or spliced into the new result. All
  eight methods receive new paired measurements on the same query sample.
- Previous BM25 top-100 IDs and scores are compared exactly after timing;
  reference rankings are never served as inference results.
- Only corpus representations are reused. Query encoders and retrieval run
  for every timed method; the measured cascade skips neural calls on exits.
- Retention, uncertainty, nDCG/MAP and standalone baselines remain required.
  An indexed scorer speedup alone does not establish a routing contribution.

Set `QUERY_LIMIT = 0` only for a later full-query follow-up if needed; it is
not the short default and may take much longer. Outputs are separated by
query-limit directory so a full run cannot reuse a partial run's checkpoint.
For TREC-COVID use `DATASET = "trec-covid"` only if that original run and its
indexes still exist in this same Kaggle session. All 50 topics then fit below
the 256-query cap.
