> **Historical document.** This file predates the consolidated submission and may contain superseded claims or commands. Use [the current manuscript](../../FINAL_MANUSCRIPT.md) for conclusions.

> **Superseded on 20 October 2026.** This document preserves earlier project context.
> The completed experiment sequence and next steps are in [FINAL_RESULTS.md](../../FINAL_RESULTS.md);
> the current manuscript is [FINAL_MANUSCRIPT.md](../../FINAL_MANUSCRIPT.md).
> Earlier requests for more Kaggle runs or tuning below are historical.

# Hybrid Retrieval Project Specification

## Current research question

Can a BM25-first confidence gate reduce the number of Dense and SPLADE queries while retaining a predeclared fraction of static hybrid retrieval quality?

## System under study

The static retriever runs Best Fields BM25, dense MiniLM, and SPLADE, then combines the three ranked lists with reciprocal rank fusion (RRF, k=60). The cascade runs BM25 first. It may return the BM25 ranking when the score and query-specificity thresholds indicate confidence; otherwise it runs Dense and SPLADE and fuses all rankings.

The current confidence features are:

- Best Fields BM25 top score, s1
- Gap between the first and second BM25 scores, s1 - s2
- Maximum IDF of a query term

This is a heuristic design whose thresholds depend on the collection and BM25 score scale.

## Current evidence

The corrected SciFact evaluator uses the full 5,183-document corpus, 809 train queries, and 300 held-out test queries. It searches 90 threshold combinations using the train split and requires at least 98% of static training Recall@100. No candidate met the constraint. Static fallback was therefore used on every test query.

On the Tesla T4, static RRF achieved Recall@100 0.9610, nDCG@10 0.6984, and MAP@10 0.6538 at 53.63 ms mean latency. The fallback path had the same retrieval metrics, 53.69 ms mean latency, and zero early exits. The single-pass latency difference is not a speedup. The current method has not demonstrated practical efficiency gains.

The historical NQ notebook and SciFact CV results are exploratory. Their protocol limitations are detailed in [`results_ledger.md`](../results_ledger.md).

## Main technical gaps

1. The run JSON does not retain metrics for each of the 90 train-only candidates.
2. The BM25 score scale may not be well calibrated for the selected thresholds.
3. The saved 90-point training frontier has no configuration meeting the 98% relative-recall constraint. Its highest recall is 0.9286 at a 79.48% exit rate; estimated candidate latency is not held-out evidence.
4. SPLADE alone slightly exceeds RRF on test nDCG@10 and MAP@10, so it is an important baseline.
5. Latency was measured once. Repeated timing and uncertainty estimates remain necessary.
6. No matched QA evaluation shows whether retrieval changes affect generated answers.

## Next experiment

Inspect `cascaded-blended-rag/results/threshold_frontier_scifact.json`, compare calibrated routing features with BM25-only and SPLADE-only baselines, then select a policy without test labels. Evaluate revised policies on untouched queries with repeated latency measurements. (Recall@100, exit rate, estimated latency), compare normalized or calibrated routing features with BM25-only and SPLADE-only baselines, then select a policy without test labels. Evaluate any revised policy on untouched queries and measure latency over repeated runs. Use the GPU when new Dense or SPLADE rankings must be computed; retain and reuse existing encoding caches when the Kaggle session allows it.

## Project documents

- [`results_ledger.md`](../results_ledger.md): evidence provenance and protocol notes.
- [`improvement_plan.md`](../plans/improvement_plan.md): staged evaluation and writing work.
- [`project_focused_literature_review.md`](project_focused_literature_review.md): related-work positioning.
- [`cascaded_adaptive_blended_rag_paper.md`](cascaded_adaptive_blended_rag_paper.md): current manuscript draft.
