# Results Ledger — Existing Runs

This ledger preserves existing project results while separating execution evidence from evaluation validity. “Executed” means code ran and produced the recorded output; it does not by itself mean the protocol is unbiased or the metric is a live end-to-end measurement.

## Kaggle Natural Questions notebook

Source: [executed reproduction notebook](../notebooks/reproduction.ipynb), cells 1–3. The notebook loads 500 BEIR NQ queries and reports the executed outputs below.

### Retriever ablation

| Retriever | Hit@5 | Hit@20 | nDCG@10 |
|:--|--:|--:|--:|
| BM25 | 0.6700 | 0.8440 | 0.5433 |
| Dense MiniLM | 0.8060 | 0.9600 | 0.6751 |
| SPLADE | 0.8900 | 0.9820 | 0.7589 |
| Three-way RRF | 0.8300 | 0.9720 | 0.7094 |

**Protocol caveats:** The code builds `pool_corpus` from the first 15,000 corpus records and then adds every document appearing in the selected queries’ qrels. It uses these qrels to define the candidate corpus. This is not a clean full-corpus BEIR test. The code computes all three branch rankings for all queries before applying the cascade decision.

### Default cascade run

| Thresholds τhigh, τgap, τidf | Exits | Hit@5 | Hit@20 | nDCG@10 | Reported latency |
|:--|--:|--:|--:|--:|--:|
| 22, 3.5, 3.0 | 92.4% | 0.6940 | 0.8800 | 0.5625 | 14.3 ms |

The notebook reports static RRF at Hit@5 0.8300, Hit@20 0.9720, nDCG@10 0.7094, and 67.0 ms; BM25 at 0.6700, 0.8440, 0.5433, and 10.0 ms. The 10/12/45 ms component latencies are constants in the notebook, so the cascade latency is calculated, not timed.

### Threshold sensitivity sweep

| Configuration | Thresholds τhigh, τgap, τidf | Exits | Hit@5 | Hit@20 | nDCG@10 | Calculated latency |
|:--|:--|--:|--:|--:|--:|--:|
| Conservative | 28, 6, 3.5 | 73.2% | 0.730 | 0.918 | 0.6115 | 25.3 ms |
| Balanced | 25, 5, 3.5 | 84.2% | 0.710 | 0.894 | 0.5851 | 19.0 ms |
| Moderate | 23, 4, 3.0 | 90.0% | 0.702 | 0.884 | 0.5704 | 15.7 ms |
| Aggressive | 21, 3, 3.0 | 94.2% | 0.692 | 0.872 | 0.5584 | 13.3 ms |

**Interpretation:** The executed ranking results show a quality–exit-rate trade-off on this qrels-filtered candidate pool. These are in-sample threshold sweep values from the same 500 queries, not held-out threshold generalization. Because the notebook precomputes all branches, exits are simulated and do not demonstrate skipped encoder calls.

## SciFact cached-results experiment

Sources: `cascaded-blended-rag/results/cv_cascaded_results.json` and `experiments/07_tune_cascade_cv.py`.

| System | Recall@100 | nDCG@10 | Reported mean latency | Exits |
|:--|--:|--:|--:|--:|
| Static RRF | 0.9610 | 0.7032 | 90.87 ms | 0% |
| Cascade | 0.9001 | 0.6613 | 30.20 ms | 84% |

**Protocol caveats:** The script reads the single-field BM25 cache, uses the test query/qrels collection for its threshold cross-validation, and compares held-out cascade averages with a full-query static baseline. Latencies are reconstructed from cached component timings; the script does not execute a conditional neural fallback. Keep this as legacy exploratory evidence, not the final held-out result.

## Other saved retrieval results

Sources: `squad_replication_results.json`, `trec_covid_replication_results.json`.

| Dataset / metric | BM25 | Dense | SPLADE | Three-way RRF |
|:--|--:|--:|--:|--:|
| SQuAD v1.1 Hit@10 (500 questions) | 0.906 | 0.954 | 0.988 | 0.964 |
| SQuAD v1.1 nDCG@10 | 0.7630 | 0.7576 | 0.8648 | 0.8298 |
| TREC-COVID nDCG@10 | 0.4998 | 0.5904 | 0.6403 | 0.6463 |

These are static retriever results, not cascade evaluations. The SQuAD FLAN-T5-XXL generation scores are a separate experiment and do not measure answer quality under cascade versus static retrieval.

## Next accepted result

Use the held-out experiment in `cascaded-blended-rag/experiments/08_evaluate_cascade_heldout.py`. It tunes only on SciFact train qrels, uses Best Fields BM25, evaluates static and cascade on the same test queries, executes neural branches conditionally, synchronizes GPU timing, and saves per-query route/timing data. Its output must be reviewed before updating manuscript claims.

