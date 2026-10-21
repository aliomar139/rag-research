# TREC-COVID: frozen paper-dataset extension

> **Original backend result, retained as measured.** The indexed follow-up
> is now received: r99 retains the same 96.68% recall and saves 19.04% mean
> time against equally optimized fusion. All rankings match exactly and raw
> qrels independently reproduce all metrics for both runs. See
> [TREC_COVID_INDEXED_FINDINGS.md](TREC_COVID_INDEXED_FINDINGS.md).

## Outcome

The primary nominal 90% policy retains 84.53% of fusion Recall@100 and saves
3.56% mean retrieval time. It fails the requested 90% floor. The secondary
nominal 95% policy retains 91.95%, but saves only 2.43% mean time, and its
retention interval crosses below 90%. This does not demonstrate a major
quality-preserving acceleration.

## Protocol and verification

The downloaded `artifacts/downloads/trec_covid_paper_benchmark_outputs.zip` covers all 50 official
BEIR test queries and 171,332 documents. All original NFCorpus-trained weights
and 90/92/95/99% thresholds are unchanged. Three warmed randomized repetitions
ran on a Tesla T4 in one timing session. Corpus/index preparation took 71.99
minutes and is excluded from the query latencies below.

Local checks validate 400 first-pass records and 1,200 query timings, exact
source/protocol/weight hashes, query coverage, branch counts and per-query
cascade equality with its selected baseline branch. All rankings are reported
stable between repeats. Raw qrels are absent from this ZIP: local checks
reaggregate saved recall and do not independently recompute nDCG/MAP or rerun
clock instrumentation. Evidence is in
`cascaded-blended-rag/results/paper_trec-covid/`.

## All frozen operating points

**Table 1. Recall retention and measured query latency relative to static fusion. All nominal calibration targets fail.**

| Target | Observed retention | Mean latency | Mean time saved | Early exits |
|:--|--:|--:|--:|--:|
| 90% (primary) | 84.53% | 1,440.00 ms | 3.56% | 62% |
| 92% | 84.53% | 1,443.98 ms | 3.30% | 60% |
| 95% | 91.95% | 1,456.90 ms | 2.43% | 44% |
| 99% | 96.68% | 1,475.73 ms | 1.17% | 22% |
| Static fusion | 100.00% | 1,493.23 ms | Reference | 0% |

The primary policy's retention interval is 78.33–90.10% and its mean-time
reduction interval is 2.61–4.58%. The nominal 95% policy has retention interval
86.88–95.76% and mean-time reduction interval 1.58–3.36%. The nominal 99%
policy exceeds a 90% floor even at the lower retention interval bound
(93.58%), but saves only 1.17% mean time. These are paired query-bootstrap
95% intervals, not universal guarantees. With only 50 queries, quality
uncertainty remains material.

The 95% policy is a predeclared secondary point, not a retrospectively selected
primary success. Reporting its observed retention is valid; claiming that it
met its 95% calibration target is not.

## Standalone baselines

**Table 2. Higher quality metrics and lower latency are preferable.**

| Method | Recall@100 | Retention | nDCG@10 | MAP@10 | Mean latency |
|:--|--:|--:|--:|--:|--:|
| Static fusion | 0.12250 | 100.00% | 0.76323 | 0.01861 | 1,493.23 ms |
| BM25 | 0.08744 | 71.38% | 0.54969 | 0.01145 | 1,410.14 ms |
| Dense | 0.08552 | 69.81% | 0.47254 | 0.01054 | 30.34 ms |
| SPLADE | 0.12825 | 104.69% | 0.72899 | 0.01662 | 52.98 ms |
| Cascade, nominal 95% | 0.11265 | 91.95% | 0.70561 | 0.01742 | 1,456.90 ms |

SPLADE has higher observed Recall@100 and lower mean latency than full fusion
and every cascade. It runs 28.19 times faster than fusion. Fusion retains
higher nDCG and MAP than SPLADE. The 95% and 99% cascades also retain higher
MAP than SPLADE, so SPLADE does not dominate every listed method on every
metric. Its recall-retention interval relative to fusion is 97.75–112.87%,
so the observed recall advantage is not established throughout that interval.

BM25 alone consumes 1,410.14 ms against fusion's 1,493.23 ms. The primary
policy skips 93 Dense and 93 SPLADE invocations over three passes, but still
pays the BM25 cost on every query. This full-corpus scoring implementation
limits savings; the result does not characterize optimized inverted-index
BM25 or the original paper's Elasticsearch runtime.

Absolute Recall@100 is low partly because many documents are judged relevant
per topic and only 100 are returned. Retention is a relative ratio. Hit@10 is
100% for fusion, SPLADE and all four cascades; this means finding at least one
relevant document for each topic, not recovering every relevant document or
preserving answer quality. nDCG/MAP expose ranking differences that hit rate
alone misses.

## Next action

SQuAD original and indexed results, plus indexed TREC, are now received.
Use [FINAL_RESULTS.md](../FINAL_RESULTS.md) for the current write-up and evidence.
No new Kaggle run or upload is required to report these completed observations.

NQ/HotpotQA full-corpus evaluation remains resource dependent and pending.
The dataset extension is an evaluation of our MiniLM/SPLADE/RRF adaptation;
it is not an exact reproduction or direct speed comparison with the original
Blended RAG implementation.

## Repeat the local import

```powershell
& '.\cascaded-blended-rag\venv\Scripts\python.exe' '.\cascaded-blended-rag\experiments\22_import_paper_benchmark.py'
```

This only validates saved results and regenerates the combined table.
