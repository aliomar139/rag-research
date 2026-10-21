> **Historical document.** This snapshot predates the consolidated submission. Use [the current manuscript](../FINAL_MANUSCRIPT.md) for conclusions.

# Cascaded Adaptive Blended RAG

This repository studies whether a BM25-first confidence gate can skip dense and SPLADE retrieval in a hybrid retriever. The current implementation is a research prototype; the latest held-out SciFact run did **not** find a threshold that met its training quality constraint, so it fell back to static retrieval and saved no neural calls.

## Latest corrected result

Source: [`cascaded-blended-rag/results/cascade_heldout_scifact.json`](../../cascaded-blended-rag/results/cascade_heldout_scifact.json), evaluated on a Tesla T4.

| System | Recall@100 | nDCG@10 | MAP@10 | Mean latency | Neural calls skipped |
|:--|--:|--:|--:|--:|--:|
| Static BM25 + Dense + SPLADE RRF | 0.9610 | 0.6984 | 0.6538 | 53.63 ms | 0% |
| Gate with static fallback | 0.9610 | 0.6984 | 0.6538 | 53.69 ms | 0% |

Thresholds were selected using 809 SciFact train queries under a 98%-of-static-Recall@100 constraint and evaluated on 300 held-out test queries over all 5,183 corpus documents. None of the 90 tested threshold combinations met the constraint. The fallback therefore ran Dense and SPLADE for all test queries. Latencies are from a single measured pass and are hardware-specific.

The result does not demonstrate an acceleration benefit. The saved training frontier (`cascaded-blended-rag/results/threshold_frontier_scifact.json`) contains 90 candidates; none meets the recall target. See the results ledger for the trade-off and limits. Next, diagnose gate calibration, then evaluate any revised policy on untouched queries. See [the results ledger](results_ledger.md) and [improvement plan](plans/improvement_plan.md).

## Repository contents

- `cascaded-blended-rag/src/`: BM25, Dense, SPLADE, and routing implementations.
- `cascaded-blended-rag/experiments/`: retrieval baselines, legacy exploratory CV, and held-out evaluation.
- `cascaded-blended-rag/results/`: saved baseline and legacy experiment outputs.
- `notebooks/legacy/reproduction.ipynb`: original executed NQ and SQuAD run; preserve as historical evidence.
- `notebooks/legacy/heldout_scifact_kaggle.ipynb`: Kaggle launch instructions for the corrected evaluation.
- `artifacts/source_bundles/cascade_source_bundle.zip`: source bundle for Kaggle.
- `docs/history/results_ledger.md`: metrics, source artifacts, and validity notes.
- `docs/history/drafts/cascaded_adaptive_blended_rag_paper.md`: current manuscript draft.
- `docs/history/drafts/project_focused_literature_review.md`: related-work analysis based on downloaded papers.

## Run the SciFact evaluation

From the repository root, activate the project environment and run:

```powershell
python cascaded-blended-rag/experiments/08_evaluate_cascade_heldout.py
```

The script downloads SciFact if needed, builds or validates retriever caches, calibrates thresholds on train qrels, and evaluates on held-out test qrels. A GPU is recommended for the neural retrievers. The output is written to `cascaded-blended-rag/results/cascade_heldout_scifact.json`.

## Historical results

The original NQ notebook reports a conservative setting with 73.2% simulated exits and 25.3 ms calculated latency. It built its candidate pool using relevant documents from the evaluated queries and computed all neural rankings before routing. Those values are exploratory; they do not measure live skipped calls or full-corpus NQ performance. The older SciFact CV artifact also used a single-field cache and reconstructed latency. See the results ledger for details.
