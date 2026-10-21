> **Historical document.** This file predates the consolidated submission and may contain superseded claims or commands. Use [the current manuscript](../../FINAL_MANUSCRIPT.md) for conclusions.

> **Superseded on 20 October 2026.** This document preserves earlier project context.
> The completed experiment sequence and next steps are in [FINAL_RESULTS.md](../../FINAL_RESULTS.md);
> the current manuscript is [FINAL_MANUSCRIPT.md](../../FINAL_MANUSCRIPT.md).
> Earlier requests for more Kaggle runs or tuning below are historical.

# Explaining the Hybrid Retrieval Project

## Short version

This project tests whether BM25 can act as a first-stage gate in a hybrid retriever. If its ranking looks reliable, the system returns BM25 results; otherwise, it invokes Dense and SPLADE and merges the rankings with reciprocal rank fusion. The first clean held-out SciFact run found that none of the tested gates met the stated quality constraint. It therefore used the static retriever for every test query and saved no neural calls. The current gate has not shown a speed benefit.

## What the system does

1. Run Best Fields BM25 over title and abstract.
2. Inspect the top BM25 score, the gap between the first two scores, and the maximum IDF among query terms.
3. Return BM25 early only if the selected thresholds are met.
4. Otherwise, run MiniLM dense retrieval and SPLADE, then combine all three rankings with RRF.

The router is a heuristic. Its scores depend on the dataset, BM25 implementation, and tokenizer, so thresholds need careful calibration.

## What the latest experiment found

The corrected SciFact run searched the full 5,183-document corpus. It used 809 train queries to select from 90 threshold combinations and 300 separate test queries for final evaluation. The target was at least 98% of static training Recall@100. No tested threshold reached that target, so the program used static fallback.

On test queries, static RRF reached Recall@100 0.9610, nDCG@10 0.6984, and MAP@10 0.6538 at 53.63 ms mean latency on a Tesla T4. The fallback path returned the same rankings, invoked Dense and SPLADE on every query, and measured 53.69 ms. This is not a speedup. The timings come from one pass and need repeated measurements.

## Questions you may be asked

### Did the cascade reduce latency?

Not in the corrected SciFact test. None of the tested gates met the training quality target, so every query used the full static path. The implementation supports conditional execution, but the current evaluation does not demonstrate savings.

### Why did the earlier project documents claim 60?70% savings?

Those figures came from exploratory results. The NQ notebook selected part of its corpus using documents judged relevant to the same evaluation queries, computed all neural rankings before applying the route, and used fixed latency assumptions. The old SciFact CV used a single-field BM25 cache and estimated latency from cached timings. Those results do not establish live neural-call savings.

### Does this prove that early exit cannot work?

No. None of the 90 tested settings met the chosen target. The saved training frontier shows Recall@100 from 0.9111 to 0.9286 at exit rates from 78.74% to 96.42%. Its highest recall is still below the 0.93922 target. The estimated latency values are not held-out measurements. Next, inspect score calibration before redesigning the gate.

### Does the project show preserved QA answer quality?

No. The existing FLAN-T5-XXL result is a separate SQuAD experiment. There is no matched static-versus-cascade generation evaluation.

### What would strengthen the project?

- Save and inspect every train-only threshold candidate's recall, exit rate, and estimated latency.
- Compare calibrated gates and simple baselines, including BM25-only and SPLADE-only.
- Select the policy on development data and evaluate it on untouched queries.
- Repeat latency measurements and report variability.
- Run QA only if making claims about answer quality.

## Evidence to cite

Use [`results_ledger.md`](../results_ledger.md) for source artifacts, metrics, and protocol caveats. The corrected result file is `../cascaded-blended-rag/results/cascade_heldout_scifact.json`.
