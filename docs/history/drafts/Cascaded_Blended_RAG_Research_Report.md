> **Historical document.** This file predates the consolidated submission and may contain superseded claims or commands. Use [the current manuscript](../../FINAL_MANUSCRIPT.md) for conclusions.

> **Superseded on 20 October 2026.** This document preserves earlier project context.
> The completed experiment sequence and next steps are in [FINAL_RESULTS.md](../../FINAL_RESULTS.md);
> the current manuscript is [FINAL_MANUSCRIPT.md](../../FINAL_MANUSCRIPT.md).
> Earlier requests for more Kaggle runs or tuning below are historical.

# BM25-First Routing for Hybrid Retrieval

## Project research report

**Author:** [Add author names and affiliations]  
**Updated:** 19 October 2026

## Executive summary

This project tests whether a lexical confidence gate can avoid running dense and learned-sparse retrievers on queries that BM25 can handle. The first corrected, held-out SciFact run did not find a tested gate that met its predeclared training quality constraint. The system therefore used static fallback on every test query and skipped no neural calls. The current implementation has not demonstrated a speed or compute benefit.

The result is useful because it comes from a cleaner protocol than the earlier experiments: the full SciFact corpus was searched, thresholds were selected on train queries, and static and cascade paths were compared on the same held-out test queries. It also shows that the proposed score, margin, and IDF signals need further analysis before any efficiency claim is justified.

## 1. Research question

Hybrid retrieval combines lexical matching, dense semantic search, and learned sparse retrieval. This can improve document ranking, but executing all retrievers for every query may add unnecessary work. The project asks whether a BM25-first routing rule can identify queries for which returning the BM25 ranking is adequate, while sending uncertain queries to dense and SPLADE retrieval.

The current rule uses the top two BM25 scores and query-term IDF. It exits when the top score exceeds a threshold, or when the top-score margin and maximum query IDF both exceed their thresholds. Otherwise, it escalates to dense MiniLM and SPLADE and fuses the three result lists with reciprocal rank fusion (RRF, k=60).

## 2. Evaluation protocol

The corrected experiment uses BEIR SciFact with 5,183 corpus documents, 809 train queries, and 300 held-out test queries. BM25 uses the Best Fields rule, taking the maximum of title and abstract BM25 scores. Dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2`; sparse neural retrieval uses `naver/splade-cocondenser-ensembledistil`.

The experiment searches 90 threshold combinations using only train qrels. A candidate qualifies only if its train Recall@100 is at least 98% of static three-way RRF train recall. Among qualifying candidates, the runner would select the one with the lowest estimated latency. No candidate qualified, so the runner used static fallback for every test query. Final metrics were calculated on the untouched test qrels. Latency was measured on a Tesla T4 with CUDA synchronization, but each path was timed in one pass; the figures are preliminary hardware-specific measurements.

## 3. Held-out SciFact results

**Table 1. Retrieval quality and measured latency on the same 300 test queries.**

| System | Recall@100 | nDCG@10 | MAP@10 | Mean latency | P50 | P95 | P99 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Static three-way RRF | 0.9610 | 0.6984 | 0.6538 | 53.63 ms | 51.21 ms | 82.38 ms | 89.35 ms |
| Cascade with static fallback | 0.9610 | 0.6984 | 0.6538 | 53.69 ms | 51.21 ms | 80.02 ms | 90.39 ms |

The cascade exits on 0 of 300 test queries. It skips 0 Dense calls and 0 SPLADE calls; each neural retriever is invoked 300 times. Its mean latency differs from static RRF by 0.06 ms, which is not evidence of a speedup. Repeated runs are needed to estimate timing variability.

**Table 2. Single-retriever results on the held-out SciFact queries.**

| Retriever | Recall@100 | nDCG@10 | MAP@10 |
|:--|--:|--:|--:|
| Best Fields BM25 | 0.8706 | 0.6369 | 0.5908 |
| Dense MiniLM | 0.9250 | 0.6451 | 0.5959 |
| SPLADE | 0.9320 | 0.7024 | 0.6590 |

SPLADE alone slightly exceeds three-way RRF on nDCG@10 and MAP@10 in this run. It should be included as a strong baseline in further routing experiments.

## 4. Interpretation

The current gate did not meet its stated objective under the chosen quality constraint. The data do not show that lexical-first routing is impossible; they show that none of the 90 tested combinations of this particular gate met the training constraint on SciFact. The saved train-only diagnostic evaluates all 90 configurations. None reaches 0.93922. Training recall ranges from 0.9111 to 0.9286; the highest-recall point reaches 0.9286 with a 79.48% training exit rate. Candidate latency values are estimated, not held-out measurements. This does not show that lexical-first routing is impossible; it shows this gate and threshold grid did not meet the training constraint on SciFact., so we cannot tell how close the best candidate came or what quality loss would be associated with more exits.

The earlier NQ results remain exploratory. Their candidate pool contains the first 15,000 documents plus documents judged relevant to the same 500 evaluation queries. The notebook also computes all three retrieval branches before applying its gate. Its reported 73.2% conservative exit rate and 25.3 ms latency therefore do not measure skipped neural calls or full-corpus NQ performance. The older SciFact CV output is also historical: it uses a single-field BM25 cache, compares unmatched aggregates, and reconstructs latency from cached timings.

The project has no matched static-versus-cascade generation experiment. Existing SQuAD generator scores measure a separate setup and cannot support a claim about downstream answer quality under this gate.

## 5. Literature position

Hybrid retrieval and adaptive retrieval are established areas. The downloaded literature includes query-complexity routing, retrieval-strategy selection, post-retrieval escalation, passage-budget control, and adaptive execution of hybrid retrieval. The specific project question is narrower: whether BM25 candidate evidence can safely gate the Dense and SPLADE branches of a Blended RAG-style retriever. The present result does not establish a novel or effective policy; any contribution should be framed as an empirical study until stronger evidence is obtained. See the project-focused literature review for paper-by-paper positioning.

## 6. Next work

1. Extend the evaluator to save all 90 train-only candidates with Recall@100, exit rate, and estimated latency.
2. Inspect the saved training quality/exit frontier and diagnose feature calibration before revising the gate.?exit frontier before choosing another constraint.
3. Check the score and margin distributions under Best Fields BM25; test normalized features or a calibrated confidence model against simple baselines.
4. Include BM25-only, SPLADE-only, static RRF, and the candidate cascade on identical splits.
5. Select any revised policy using development data only, then evaluate it on untouched queries.
6. Repeat latency measurements and report variability. Run matched QA generation only if the paper will claim answer-quality preservation.

A fresh GPU run is required if new Dense or SPLADE rankings must be computed. If the Kaggle session and cache files remain available, the document embeddings and sparse matrix may be reused.

## References

- Sawarkar, K., Mangal, A., and Solanki, S. R. ?Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers.? IEEE MIPR, 2024.
- Robertson, S. and Zaragoza, H. ?The Probabilistic Relevance Framework: BM25 and Beyond.? *Foundations and Trends in Information Retrieval*, 2009.
- Formal, T., Piwowarski, B., and Clinchant, S. ?SPLADE: Sparse Lexical and Expansion Model for Information Retrieval.? SIGIR, 2021.
- Jeong, S. et al. ?Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.? NAACL, 2024.
- Thakur, N. et al. ?BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.? NeurIPS Datasets and Benchmarks, 2021.

## Evidence artifacts

- `cascaded-blended-rag/results/cascade_heldout_scifact.json`
- `notebooks/legacy/reproduction.ipynb`
- `cascaded-blended-rag/results/cv_cascaded_results.json`
- `docs/history/results_ledger.md`
- `docs/history/drafts/project_focused_literature_review.md`
