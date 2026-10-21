# Frozen SCIDOCS confirmation

## Why this run

On NFCorpus dev the cascade reduces latency versus full fusion, but Dense alone
is faster and retains more Recall@100 than every cascade. SPLADE has stronger
nDCG@10/MAP@10. The final write-up must include these baselines. This run checks
the frozen method on a new collection; it does not try to repair the dev result.

SCIDOCS has 1,000 official test queries and a roughly 25K-document corpus.
Source and official archive MD5:
https://github.com/beir-cellar/beir#-available-datasets

## Do these steps

1. Create a new private Kaggle dataset titled
   `scidocs-frozen-cascade-confirmation-20261020-v1`.
2. Upload only `scidocs_frozen_confirmation_source_v1.zip` from this folder.
3. Attach that dataset to a notebook, enable Internet and GPU, and paste
   `RUN_IN_KAGGLE.py` into one cell. Budget roughly 30-60 minutes for a first
   run, depending on corpus indexing and downloads. It uses the full corpus
   and all 1,000 queries; neither depends on relevance labels.
4. Download `artifacts/downloads/scidocs_confirmation_outputs.zip` from
   `/kaggle/working/scidocs_frozen_confirmation_v1/results/`, save it in the
   local project folder, and give its filename.

## Frozen protocol

- Original 11-feature Ridge gate only, at all four 90/92/95/99% thresholds.
  Weights and thresholds are byte-identical to those in the NFCorpus run.
- Primary candidate: the 92% target as a development-selected compromise;
  report all four outcomes regardless of target misses. The 69-feature
  ablation remains in the NFCorpus results and is not selected for transfer.
- No fitting, recalibration, threshold search, or query/corpus subsampling on
  SCIDOCS. Official archive checksum is verified. Query IDs select the split;
  grades are parsed only after all routes/rankings/timings are complete.
- Four baselines (BM25, Dense, SPLADE, static RRF), real conditional neural
  calls, three randomized warm timing repeats, paired bootstrap summaries,
  and count-matched random-exit recall references match the earlier run.
- Exclude the identical query/document ID from each branch before computing
  routing features or fusion. Request 101 candidates so 100 remain after
  removal. This follows BEIR's self-match convention:
  https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py
- Include every result, whether favorable or unfavorable. A failed transfer
  target is a result; do not tune on SCIDOCS and call it untouched again.
- No NFCorpus test labels are read. RAG answer quality is not measured.

The source bundle includes `scidocs_confirmation_protocol.json`, frozen
weights, source checksums, and the NFCorpus development result summary. After
this confirmation, consolidate the measured evidence into the final report
and paper with its limitations; avoid further development runs unless an
implementation error requires correction.
