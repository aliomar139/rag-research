> **Historical document.** This snapshot predates the consolidated submission. Use [the current manuscript](../FINAL_MANUSCRIPT.md) for conclusions.

> **Latest result: TREC-COVID completed and verified.** The primary 90% policy retains 84.53% of recall and saves 3.56% mean time. The secondary 95% policy retains 91.95% but saves only 2.43%. SPLADE has higher observed recall at 28.19x fusion speed; fusion has higher nDCG/MAP. Read [TREC_COVID_FINDINGS.md](../findings/TREC_COVID_FINDINGS.md). Next: run the same Kaggle cell with `DATASET = "squad"`; no new upload. Earlier pending-TREC statements below are historical.

> **Update: original-paper dataset extension is now prepared.** NFCorpus/SCIDOCS findings below remain unchanged. The next run is full-corpus TREC-COVID, followed by SQuAD. Code also supports full NQ/HotpotQA on a suitably provisioned machine. See [START_HERE.md](../../notebooks/kaggle_paper_datasets_v1/START_HERE.md). New results are pending.

# Final results and completion steps

Updated 20 October 2026. Experimental work for this study is complete.

## Read this first

The cascade saves retrieval time compared with running all three retrievers.
Its calibrated quality targets do not transfer reliably to SCIDOCS, and the
standalone Dense baseline is a better recall/latency choice on both measured
collections. Present the project as an empirical study of conditional
retrieval, calibration transfer and baseline choice. The results do not support
a claim that the cascade is the best retrieval system.

## What we tried

Run BM25, then use a small Ridge model to decide whether to return its results
or run Dense and SPLADE too. The fallback combines all three rankings with
reciprocal rank fusion (RRF). The purpose was to save neural retrieval calls
while retaining 90%, 92%, 95% or 99% of full fusion's Recall@100.

Retention is the ratio of mean Recall@100 scores, not agreement between two
document lists. For example, 109.65% means a method has higher mean recall
than fusion; its absolute recall is still below 100%.

## Development and frozen confirmation

The original 11-feature Ridge model and all four thresholds were fitted using
NFCorpus train. NFCorpus dev was already inspected development evidence.
The 92% policy was selected as the primary candidate before SCIDOCS was scored.
SCIDOCS used the frozen weights and thresholds, without refitting or threshold
changes. Each collection used its full corpus, a Tesla T4 and three randomized,
warmed timing repetitions. Timing comparisons apply within each run.

**Table 1. Frozen original gate: measured quality and mean retrieval latency.**

| Training target | NFCorpus dev retention | NFCorpus mean | SCIDOCS retention | SCIDOCS mean | SCIDOCS target met? |
|:--|--:|--:|--:|--:|:--|
| 90% | 92.33% | 17.07 ms | 88.67% | 149.43 ms | No |
| 92% (primary) | 94.74% | 18.18 ms | 90.01% | 150.69 ms | No |
| 95% | 96.23% | 19.67 ms | 92.10% | 152.87 ms | No |
| 99% | 99.27% | 23.14 ms | 97.44% | 157.22 ms | No |
| Full fusion | 100.00% | 23.61 ms | 100.00% | 159.51 ms | Reference |

All four nominal targets fail on SCIDOCS. The primary policy retains 90.01%
and reduces mean latency by 5.53% (1.06x speedup), with 35% early exits.
Its paired query-bootstrap 95% intervals are 88.27–91.61% for retention and
5.00–6.08% for mean latency reduction. Even a 90% quality floor is uncertain.
Do not rename the nominal 95% policy a successful calibrated 92% policy after
seeing its 92.10% result.

## Strong baselines determine the practical outcome

**Table 2. SCIDOCS test, 1,000 queries and 25,657 documents. Quality metrics increase with better retrieval; latency decreases.**

| Method | Recall@100 | nDCG@10 | MAP@10 | Mean latency | Speedup vs fusion |
|:--|--:|--:|--:|--:|--:|
| Full fusion | 0.46523 | 0.18591 | 0.10976 | 159.51 ms | 1.00x |
| BM25 | 0.32378 | 0.13534 | 0.07626 | 132.51 ms | 1.20x |
| Dense | 0.51012 | 0.21640 | 0.12937 | 10.14 ms | 15.73x |
| SPLADE | 0.37342 | 0.15856 | 0.09245 | 16.39 ms | 9.73x |
| Cascade, nominal 92% | 0.41873 | 0.17082 | 0.09932 | 150.69 ms | 1.06x |

Dense alone exceeds full fusion and every cascade in all three quality metrics
and is much faster on SCIDOCS. On NFCorpus dev it retains 99.77% of fusion recall
at 6.62 ms (3.57x speedup), beating every cascade in recall and latency.
SPLADE has the best NFCorpus nDCG and MAP, so the preferred retriever depends on
the quality metric. Neither collection establishes the cascade as the best
practical operating point.

## What we learned

1. Real conditional execution works. The primary SCIDOCS policy skips 1,050
   Dense and 1,050 SPLADE calls across three passes, but saves only 5.53% mean
   time. Branch-call savings and wall-clock savings are different quantities.
2. The first stage limits the savings. BM25 alone takes 132.51 ms compared
   with 159.51 ms for fusion. These averages describe this full-corpus scoring
   implementation; they do not establish that BM25 is intrinsically slow or
   predict an optimized inverted-index implementation.
3. Calibration weakens across datasets. The primary policy retains 94.74%
   on NFCorpus dev and 90.01% on SCIDOCS. Its count-matched offline random-exit
   reference averages 89.42% on SCIDOCS, compared with 89.54% on NFCorpus dev.
   The learned-versus-random gap therefore narrows. Random routing latency
   was not measured, and this comparison is not a significance test.

## Defensible contribution statement

> We implement and evaluate a BM25-first conditional hybrid retriever under
> four recall-retention budgets. Repeated measurements and a frozen
> cross-dataset confirmation show that reduced neural execution can yield
> limited latency savings when the first stage dominates runtime, while
> retention calibration can fail to transfer. Strong standalone retrievers
> expose the limits of evaluating a cascade only against full fusion.

This is an implementation and empirical contribution. Method novelty,
superiority over the wider literature, production efficiency, energy savings
and preserved generated-answer quality have not been established. Whether
this scope satisfies a course or publication requirement depends on that
requirement; these results cannot guarantee acceptance.

## Evidence and reproducibility

- Original downloaded ZIPs remain at the workspace root.
- SCIDOCS extracted evidence:
  `cascaded-blended-rag/results/scidocs_confirmation/`.
- NFCorpus extracted evidence:
  `cascaded-blended-rag/results/retention_tradeoff_dev/`.
- Combined machine-readable table:
  `cascaded-blended-rag/results/final_quality_speed.csv`.
- Frozen weights and predeclared protocol:
  `cascaded-blended-rag/results/nfcorpus_frozen_retention_routers.json` and
  `cascaded-blended-rag/results/scidocs_confirmation_protocol.json`.

Local artifact checks cover all 8,000 SCIDOCS first-pass records and 24,000
timings, query coverage, saved recall aggregation, exits, neural calls,
100 unique non-self document IDs per ranking, ZIP CRCs and exact frozen
weights/protocol/source fingerprints. All later rankings are reported stable.
The downloaded ZIP does not contain raw qrels, so these checks do not
independently recompute relevance metrics from raw grades or rerun timing
instrumentation. See `local_verification.json` in the extracted directory.

To repeat this local consolidation only, run from the project root:

```powershell
& '.\cascaded-blended-rag\venv\Scripts\python.exe' '.\cascaded-blended-rag\experiments\19_finalize_confirmation.py'
```

No Kaggle dataset upload or GPU run is needed for consolidation.

## Finish in this order

1. Use `docs/FINAL_MANUSCRIPT.md` as the current manuscript.
2. Use the tables above and the contribution statement in the project report
   and presentation. Retain the failed targets and standalone baselines.
3. Keep the source bundles, downloaded results and frozen protocol with the
   project submission. The existing Kaggle instructions are reproduction
   instructions, not a new experiment queue.

NFCorpus test was inspected earlier and remains excluded from the final
calibration work. SCIDOCS is now inspected; further changes evaluated on it
would be development work. Historical NQ simulations and SciFact experiments
remain in the results ledger and are not combined with these latency claims.
