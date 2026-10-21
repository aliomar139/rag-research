# Indexed BM25: measured execution improvement

21 October 2026. The original and indexed SQuAD results are received and
verified. The CPU experiment below preceded that follow-up; its new result
is recorded in [SQUAD_INDEXED_FINDINGS.md](SQUAD_INDEXED_FINDINGS.md).

## What changed

The earlier scorer visits every document's Python term-frequency dictionary
for each query token. The new scorer precomputes per-term document
contributions and reads their postings at query time. It also selects the
top 100 with partitioning and an explicit corpus-order tie rule.

The BM25 formula is unchanged: same regular-expression tokenizer,
rank_bm25 epsilon-adjusted IDFs, per-field length normalization, duplicate
query tokens, float64 accumulation, Best Fields maximum, and tie ordering.
The original files are preserved. New code is `src/indexed_bm25.py`.

## Actual local CPU measurements

**Table 1. Paired CPU execution benchmark, one thread, three randomized passes. These are local CPU measurements, not Kaggle cascade measurements.**

| Dataset | Full corpus | Queries | Old scorer mean | Indexed mean | Scorer speedup |
|:--|--:|--:|--:|--:|--:|
| TREC-COVID | 171,332 | 50 | 908.70 ms | 6.74 ms | 134.85x |
| NFCorpus train | 3,633 | 2,590 | 4.50 ms | 0.13 ms | 34.59x |

All full-corpus score vectors and top-100 IDs/order match exactly on every
query repetition. Maximum absolute score difference is zero: 150 paired
query repetitions on TREC-COVID and 7,770 on NFCorpus train.
The TREC corpus SHA256 matches the full corpus used in the Kaggle report.

Reference field-index construction took 17.62 seconds on TREC-COVID; converting
to postings took an additional 26.85 seconds. NFCorpus took 0.34 and 0.63
seconds respectively. Index construction is offline work excluded from query
latency. Additional storage/memory and performance on corpus updates have
not been evaluated.

Environment: local Intel64 Family 6 Model 154 Stepping 4, Python 3.12.10,
NumPy 2.5.4, one CPU thread. Kaggle used different hardware/software; the
134.85x number must not be computed against or described as its old 1410 ms
BM25 latency. Machine-readable results are under `cascaded-blended-rag/results/`:

- `indexed_bm25_cpu_trec-covid.json`
- `indexed_bm25_cpu_nfcorpus.json`

## Why this changes the next experiment

The old TREC-COVID BM25 stage consumed almost all static-fusion time. Making
that stage cheap could let neural skipping save a meaningful fraction of
the optimized pipeline. Both static fusion and the cascade must receive
the same index improvement before comparison.

This change does not repair the gate's quality calibration. With identical
rankings and features, its full-query retrieval quality is unchanged. The
indexed SQuAD comparison is now measured on 256 questions, as reported below.
Both indexed follow-ups are now measured, as reported below. No new algorithmic novelty
is established by precomputing term scores: eager sparse scoring is studied in
[BM25S](https://arxiv.org/abs/2407.03618).

## Fast follow-up protocol, now executed on SQuAD

The combined `notebooks/kaggle_fast_finish_v1/` workflow was used after the
original SQuAD result was downloaded in the same Kaggle session.
It reuses corpus indexes and refuses
missing/mismatched caches instead of starting document encoding again.

The default uses 256 questions selected by SHA256(dataset + NUL + query ID),
the full corpus, all eight methods and three repetitions. Query selection
uses no grades. Its quality scores cover that sample; they cannot be labeled
as full 10,570-question scores. Timing estimates use only the new measurements,
not old-stage subtraction. Previous BM25 top-100 IDs and scores are compared
after timing without serving cached rankings as inference results.

For this follow-up only, `ridge11_r99` is primary, judged against a 90%
retention floor. It was chosen using already inspected datasets before the
indexed experiment or current full SQuAD outcomes were inspected. The four
thresholds/weights remain unchanged. Original nominal-target misses stay
reported and the earlier experiments retain their original primary settings.

## Verified indexed SQuAD outcome

On the declared 256-question sample, indexed fusion takes 29.55 ms. Primary
r99 retains 100% sample recall and saves 4.19% mean time; secondary r90
retains 99.22% and saves 37.08%. Secondary r92 retains 100% and saves 27.89%.
The primary remains r99. All 2,048 rankings and scores exactly match the old
run on the same questions, as do routes and recall. Independent scoring
matches every reported quality metric. Three timing passes are complete.

BM25 alone retains 95.70% at 1.03 ms and is the fastest listed option for a
90% floor. SPLADE leads nDCG/MAP at 16.96 ms. The cascade therefore has a
measured mean time advantage against fusion without demonstrated superiority
over the relevant constant baselines. The sample cannot establish full-query
indexed quality or latency, and r90 p95 does not improve.

Indexed TREC is now also measured and verified: primary r99 retains 96.68%
recall and saves 19.04% mean time against equally optimized fusion over all
50 queries. SPLADE is faster with higher recall/nDCG and lower MAP. See
[the complete findings](TREC_COVID_INDEXED_FINDINGS.md). Next is the write-up;
no additional Kaggle run is required for this scope. Direct-selector training
remains contingent; its required branch-specific labels are absent
and the general strategy-selection idea has
[prior art](https://arxiv.org/abs/2109.10739).

## Claim review

- **Measured:** faster local BM25 execution with exact score/ranking agreement
  on the stated corpora and queries.
- **Measured:** indexed SQuAD sample execution and quality, with equal indexing
  for fusion, standalone BM25 and all gates.
- **Not established:** new BM25 algorithm, novel query routing, universal 90%
  quality guarantee, superiority over optimized engines, generated-answer
  preservation or production performance.

The implementation contribution is concrete. The follow-up demonstrates
conditional mean time savings; a superior routing method remains unestablished.


## Five-part review

1. Contribution: measured compatible execution optimization; no priority or
   superior-routing claim is made.
2. Clarity: local scorer measurements and the separate indexed SQuAD sample
   pipeline measurements are labeled explicitly.
3. Experimental strength: paired repeated CPU timing and exact full-score
   agreement on two corpora; one local hardware setting, no memory/update study.
4. Completeness: the completed SQuAD GPU follow-up includes equally optimized
   fusion, all gates and standalone retrievers. It uses a declared sample.
5. Soundness: no quality improvement is inferred from score-preserving code.
   The conservative primary is preselected using prior inspected evidence;
   earlier primary failures and all secondary results remain reported.
