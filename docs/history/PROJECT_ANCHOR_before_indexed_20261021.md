> **Historical document.** This snapshot predates the consolidated submission. Use [the current manuscript](../FINAL_MANUSCRIPT.md) for conclusions.

> **Latest: TREC-COVID verified, SQuAD next.** All four nominal targets miss. Nominal 95% retains 91.95% with only 2.43% mean savings. SPLADE leads recall and is 28.19x faster than fusion, while fusion leads nDCG/MAP. See [findings](../findings/TREC_COVID_FINDINGS.md).

> **Active work:** original-paper dataset extension requested after the completed NFCorpus/SCIDOCS sequence. Prepared [Kaggle source and instructions](../../notebooks/kaggle_paper_datasets_v1/START_HERE.md); next action is TREC-COVID, with frozen original weights and 90% primary target. No new benchmark result yet.

> **Current status: final experiment sequence completed, 20 October 2026.**
> Start with [final results](../FINAL_RESULTS.md) and the [current manuscript](../FINAL_MANUSCRIPT.md).
> All four retention targets fail in frozen SCIDOCS transfer. Dense alone has higher
> SCIDOCS quality and 15.73x fusion speed. No further Kaggle run is queued.
> The remaining text is historical context.

# Project Anchor: Cascaded Adaptive Blended RAG

**Updated:** 2026-10-19  
**Status:** SciFact threshold gate failed its recall constraint. A frozen Ridge gate transferred to NFCorpus reduced mean latency but lost substantial Recall@100.

## Current verified evidence

Artifact: [`cascaded-blended-rag/results/cascade_heldout_scifact.json`](../../cascaded-blended-rag/results/cascade_heldout_scifact.json), generated on a Tesla T4.

- Corpus: 5,183 SciFact documents; 809 train queries; 300 held-out test queries.
- Threshold rule: BM25 Best Fields top score, score gap, and MaxIDF; 90 candidates searched on train queries.
- Constraint: train Recall@100 at least 98% of static RRF (static train Recall@100 = 0.95838; target ? 0.93920).
- Selection: no threshold met the constraint; the runner used static fallback.
- Held-out static RRF: Recall@100 0.9610, nDCG@10 0.6984, MAP@10 0.6538, mean latency 53.63 ms.
- Fallback path: same retrieval metrics, 53.69 ms mean latency, 0% early exits, zero neural calls skipped.
- Timing: one measured T4 pass; repeat-run uncertainty is not available.

**Interpretation:** the execution protocol is cleaner and the comparison is matched, but the method did not meet its objective. Do not claim current latency savings, preserved answer quality, zero leakage across the full project, or a validated production cascade.

Cross-dataset artifact: [`cascaded-blended-rag/results/ridge_transfer_nfcorpus.json`](../../cascaded-blended-rag/results/ridge_transfer_nfcorpus.json), Tesla T4, 3,633 full-corpus documents, 323 official test queries, two warmed timing passes.

| System | Recall@100 | nDCG@10 | MAP@10 | Mean latency | Exits |
|:--|--:|--:|--:|--:|--:|
| Static RRF | 0.3147 | 0.3452 | 0.1304 | 25.04 ms | 0% |
| SciFact-trained Ridge cascade | 0.2681 | 0.3203 | 0.1235 | 13.56 ms | 68.11% |

The cascade saved 45.8% mean latency while retaining 85.2% of static Recall@100. It is a cross-dataset quality/speed trade-off, not a successful quality-preserving cascade. The NFCorpus test set has now been inspected; do not tune against it and reuse it as final evaluation.

## Historical evidence (exploratory only)

- NQ: BM25 hit@20 0.844; static RRF 0.972; SPLADE 0.982 on the notebook?s 500-query qrels-filtered pool. The cascade exits and latencies are simulated/calculated.
- Legacy SciFact CV: Recall@100 0.9001, nDCG@10 0.6613, 84% exits, and estimated 30.20 ms. The script used single-field BM25 cache and an unmatched static baseline.
- SQuAD/TREC-COVID runs are static retriever or generator results, not cascade QA results.

See [`docs/history/results_ledger.md`](results_ledger.md) for complete provenance.

## Next work

1. Completed: `cascaded-blended-rag/results/threshold_frontier_scifact.json` records all 90 fixed-threshold candidates; none meets the train Recall@100 target.
2. Completed: `cascaded-blended-rag/results/router_training_scifact.jsonl` contains 809 train-only examples with BM25 features and per-query escalation-gain labels.
3. Completed first learned-router attempt: it skipped 70.67% of neural calls but reduced Recall@100 to 0.9346 and increased mean latency to 61.48 ms. The router prediction step averaged about 21 ms per query.
4. Completed: nested train-only CV compared compact predictors; local profiling found Ridge inference around 0.14 ms/query.
5. Completed transfer probe: frozen Ridge on NFCorpus reduces mean latency but loses too much Recall@100. Next, diagnose/recalibrate on a development split and evaluate the revised policy on another untouched collection.

**Kaggle GPU:** required for fresh Dense/SPLADE query retrieval. Existing encoding caches are reusable only while their Kaggle working files and session persist.
