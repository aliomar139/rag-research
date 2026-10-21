> **Historical document.** This file predates the consolidated submission and may contain superseded claims or commands. Use [the current manuscript](../../FINAL_MANUSCRIPT.md) for conclusions.

> **Superseded on 20 October 2026.** This document preserves earlier project context.
> The completed experiment sequence and next steps are in [FINAL_RESULTS.md](../../FINAL_RESULTS.md);
> the current manuscript is [FINAL_MANUSCRIPT.md](../../FINAL_MANUSCRIPT.md).
> Earlier requests for more Kaggle runs or tuning below are historical.

# Evaluating BM25-First Early Exit for Hybrid Retrieval: A Held-Out SciFact Study

> **Draft status:** Updated with the first corrected held-out SciFact run. The current heuristic did not satisfy its training quality constraint and produced no early exits on the test set. This paper reports that negative result and makes no claim of an efficiency gain.

**A research paper draft**  
**Authors:** [Add author names and affiliations]

## Abstract

Hybrid retrieval combines lexical, dense, and learned sparse search, but running every branch for every query can add cost. We evaluate a BM25-first gate that exits on a high top score or on a large score margin paired with a high-IDF query term; otherwise, it invokes dense MiniLM and SPLADE and fuses their rankings with reciprocal rank fusion. On the full BEIR SciFact corpus, thresholds were selected from 809 training queries under a predeclared constraint of at least 98% of static training Recall@100, then evaluated on 300 held-out test queries. None of the 90 tested threshold combinations met the training constraint, so the system used static fallback. Static RRF achieved test Recall@100 of 0.9610 and nDCG@10 of 0.6984 at a measured mean latency of 53.63 ms on a Tesla T4. The fallback path had identical retrieval metrics, skipped no neural calls, and measured 53.69 ms mean latency. This evaluation finds no acceleration benefit from the current gate under the chosen quality constraint. Earlier Natural Questions cascade values are exploratory because the candidate pool was built using evaluation qrels and neural branches were precomputed.

**Keywords:** retrieval-augmented generation, hybrid retrieval, query routing, early exit, BM25, SPLADE

## 1. Introduction

Retrieval-augmented generation systems depend on retrieving useful evidence before generating an answer. Lexical retrieval is effective when query and document wording overlap, dense retrieval can match semantically similar text with different wording, and learned sparse retrieval combines term matching with model-based expansion. Blended RAG combines these complementary signals with reciprocal rank fusion (RRF) [1]. Its static execution pattern, however, runs each retriever for every query, regardless of whether an inexpensive result is already strong.

This paper studies a narrower systems question: can an initial lexical result identify a useful subset of queries for which neural retrieval can be skipped? The motivation comes from an ablation on 500 Natural Questions (NQ) queries. BM25 reaches a hit rate at 20 of 84.4%, compared with 97.2% for three-way BM25, dense, and SPLADE fusion. This gap shows both sides of the problem: lexical retrieval can cover many queries, while hybrid retrieval still recovers relevant documents that BM25 misses. The ablation motivates selective routing; it does not establish that every BM25 hit is redundant to neural retrieval or that the cascade preserves the fused system’s quality.

We implement a sequential cascade that computes BM25 first and makes an early-exit decision from its leading scores and the query’s maximum inverse document frequency (MaxIDF). Queries that do not satisfy the gate escalate to dense and SPLADE retrieval, followed by three-way RRF. The corrected SciFact runner executes the branches conditionally and measures end-to-end retrieval latency. Older NQ outputs use precomputed branch rankings and fixed latency assumptions; they are discussed only as exploratory evidence.

This study makes three contributions:

1. **An implementation of BM25-first conditional retrieval.** A lexical confidence gate can return a BM25 ranking or escalate to dense and learned-sparse retrieval with RRF.
2. **A matched held-out SciFact evaluation.** Threshold selection uses train qrels, while static and cascade paths are compared on the same untouched test queries and full corpus.
3. **A negative result under a stated quality target.** None of the tested gates met the 98% training Recall@100 constraint, and the fallback avoided no neural calls. The result identifies a limitation of the current signal and threshold grid rather than demonstrating a speedup.

## 2. Related Work

### 2.1 Hybrid retrieval

Lexical methods such as BM25 rank documents using term statistics [2]. Dense retrievers encode queries and documents into vectors, while learned sparse models such as SPLADE expand text into weighted lexical representations [3]. Rank fusion combines ranked lists without requiring their raw scores to share a scale; RRF is a widely used form of this approach [4]. Blended RAG applies hybrid retrieval to retrieval-augmented generation by combining semantic search with hybrid query-based retrieval [1]. Our implementation follows this broad pattern with BM25, MiniLM dense vectors, and SPLADE, fused with RRF using a rank constant of 60.

### 2.2 Adaptive retrieval

Adaptive-RAG selects retrieval strategies according to estimated query complexity [5]. That work addresses when and how much retrieval to perform at the QA-system level. The present study considers a different control point: after a lexical retrieval pass, the system decides whether to pay for additional retriever branches. The distinction is about mechanism and evaluation scope; this project does not compare against Adaptive-RAG experimentally and does not claim that early-exit routing is a new general principle.

## 3. Method

### 3.1 System overview

For a query $q$, the cascade first obtains a ranked BM25 list $L_B(q)$. It computes a routing decision from the top two BM25 scores and lexical specificity. An early-exit decision returns the BM25 ranking. Otherwise, the system obtains dense and SPLADE rankings and fuses all three lists. Thus, BM25 is used both as a retrieval branch in the fallback and as the evidence source for the routing decision.

```text
Query
  │
  ▼
BM25 retrieval ──► score/margin/MaxIDF gate
                         │
              confident ─┴─ uncertain
                  │              │
                  ▼              ▼
             Return BM25    Dense + SPLADE
                                  │
                                  ▼
                              3-way RRF
```

### 3.2 Confidence gate

Let $s_1$ and $s_2$ denote the first and second BM25 scores, with $s_2=0$ if there is only one candidate. The margin is $\Delta=s_1-s_2$. For query tokens $Q$, the maximum IDF is

\[
\operatorname{MaxIDF}(Q)=\max_{t\in Q}\log\left(\frac{N-df(t)+0.5}{df(t)+0.5}+1\right),
\]

where $N$ is the number of indexed documents and $df(t)$ is the number containing token $t$. Empty queries have MaxIDF zero. The implemented gate is

\[
\operatorname{route}(q)=
\begin{cases}
\text{early exit}, & s_1\geq\tau_{high},\\
\text{early exit}, & \Delta\geq\tau_{gap}\;\land\;\operatorname{MaxIDF}(Q)\geq\tau_{idf},\\
\text{escalate}, & \text{otherwise}.
\end{cases}
\]

The first condition treats a sufficiently large leading BM25 score as adequate confidence. The second requires both a distinct score margin and a specific query term, reducing reliance on the margin alone. These rules are heuristic thresholds; their scores depend on the collection, tokenizer, and BM25 implementation, so the numerical thresholds should not be interpreted as universal constants.

### 3.3 Fallback fusion

For each retriever $m\in\{B,D,S\}$, let $\operatorname{rank}_m(d)$ be the one-based rank of document $d$, where $B$, $D$, and $S$ denote BM25, dense MiniLM, and SPLADE. The fallback score is

\[
\operatorname{RRF}(d)=\sum_{m\in\{B,D,S\}}\frac{1}{60+\operatorname{rank}_m(d)},
\]

with documents absent from a list contributing zero. The cascade returns the BM25 ranking on early exit and the fused ranking on escalation. In the reported evaluation implementation, top-100 cached lists are used to build the fusion run.

## 4. Experimental Setup

### 4.1 Datasets and implementations

The corrected evaluation uses the full 5,183-document BEIR SciFact corpus, 809 train queries, and 300 held-out test queries. It uses Best Fields BM25, `sentence-transformers/all-MiniLM-L6-v2` with FAISS inner-product search, and `naver/splade-cocondenser-ensembledistil`. The runner executes BM25 first and invokes Dense and SPLADE only after escalation. Components are warmed before timing, and CUDA is synchronized around timed calls. The measurements come from one Kaggle run on a Tesla T4. Earlier NQ runs use 500 queries but remain exploratory because their corpus pool includes judged relevant documents from those queries.

### 4.2 Metrics and protocols

The train split is used to select among 90 threshold combinations. A candidate is eligible only if its train Recall@100 is at least 98% of static train Recall@100; among eligible candidates, the runner selects the lowest estimated mean latency. No candidate met this constraint, so the test cascade used static fallback. Test metrics are Recall@100, nDCG@10, and MAP@10. Static and cascade latency are measured on the same 300 test queries; mean, p50, p95, and p99 are reported. Each path was measured once, so the latency figures are preliminary and have no repeated-run uncertainty estimate.

### 4.3 Scope of comparison

The project’s base-paper reproduction also includes SQuAD v1.1 retrieval and generation, TREC-COVID, and SciFact. These experiments provide context for the retrieval stack, but the cascade is not evaluated on SQuAD or TREC-COVID in the available result artifacts. The NQ SQuAD generator result (79.00 EM and 81.59 token F1 using FLAN-T5-XXL) evaluates a separate QA setup; it is not a downstream evaluation of the cascade. We therefore restrict cascade claims to the measured retrieval metrics and reported cost estimates.

## 5. Results

### 5.1 NQ branch ablation

Table 1 reports the available NQ ablation. SPLADE has the highest individual hit rate and nDCG@10 among the listed branches. Three-way RRF improves hit rate@20 over BM25 by 12.8 percentage points, showing that neural retrieval contributes relevant candidates beyond the lexical branch. BM25’s 84.4% hit rate@20 nevertheless identifies a substantial region in which a lexical-only answer may be viable.

**Table 1. NQ retrieval ablation on 500 queries.**

| Retriever | Hit rate@5 | Hit rate@20 | nDCG@10 |
|:--|--:|--:|--:|
| BM25 | 67.0% | 84.4% | 0.5433 |
| Dense MiniLM | 80.6% | 96.0% | 0.6751 |
| SPLADE | 89.0% | 98.2% | 0.7589 |
| Three-way RRF | 83.0% | 97.2% | 0.7094 |

The fused system does not exceed each component on every metric: SPLADE alone has higher reported NQ hit rates and nDCG@10 than the RRF run. Accordingly, the motivation for routing here is the cost-quality behavior of this specific implementation, not a claim that fusion is uniformly the strongest NQ ranking.

### 5.2 Held-out SciFact evaluation

**Table 2. Held-out SciFact results on the same 300 test queries. Latency was measured once on a Tesla T4.**

| System | Recall@100 | nDCG@10 | MAP@10 | Mean latency (ms) | Early exits |
|:--|--:|--:|--:|--:|--:|
| Static three-way RRF | 0.9610 | 0.6984 | 0.6538 | 53.63 | 0.0% |
| Cascade (static fallback) | 0.9610 | 0.6984 | 0.6538 | 53.69 | 0.0% |

The quality constraint was not met by any of the 90 training candidates. Static train Recall@100 was 0.95838, making the target approximately 0.93920. The runner selected no threshold and escalated every test query. Dense and SPLADE were each invoked 300 times; zero neural calls were skipped. The 0.06 ms difference between measured path means is not evidence of a speedup and is within the uncertainty expected from a single timing pass.

The single-retriever results provide context: Best Fields BM25 reaches Recall@100 0.8706, dense MiniLM 0.9250, and SPLADE 0.9320. SPLADE alone has slightly higher nDCG@10 (0.7024) and MAP@10 (0.6590) than three-way RRF in this run. Future gate comparisons should therefore include SPLADE-only, not only static fusion and BM25.

### 5.3 NQ threshold sweep

**Table 3. NQ quality–cost sweep on 500 queries.**

| Configuration | Thresholds $(\tau_{high},\tau_{gap},\tau_{idf})$ | Early exits | Mean latency (ms) | Hit rate@20 |
|:--|:--|--:|--:|--:|
| Static three-way RRF | — | 0.0% | 67.0 | 97.2% |
| Conservative | (28, 6, 3.5) | 73.2% | 25.3 | 91.8% |
| Balanced | (25, 5, 3.5) | 84.2% | 19.0 | 89.4% |
| Moderate | (23, 4, 3.0) | 90.0% | 15.7 | 88.4% |
| BM25 only | — | 100.0% | 10.0 | 84.4% |

The NQ sweep shows a monotonic trade-off across the listed configurations: more early exits correspond to lower reported mean latency and lower hit rate@20. The conservative point reduces reported latency by 62.2% compared with the static result while lowering hit rate@20 by 5.4 percentage points. Because the available summary does not specify fold-based threshold selection for NQ, these values should be read as a threshold sweep, not a held-out generalization estimate.

## 6. Discussion

The corrected result does not support the current gate as a useful accelerator under the 98% training recall constraint. Although the runner implements conditional neural execution, no tested threshold qualified, so every held-out query used static fallback. The tested BM25 score, margin, and MaxIDF features, combined with the current threshold grid, were not sufficient to identify safe early exits on this collection.

Train-only frontier (`cascaded-blended-rag/results/threshold_frontier_scifact.json`): all 90 candidates were evaluated on the 809 training queries. None meets the 0.93922 Recall@100 target. Training Recall@100 ranges from 0.9111 to 0.9286; exit rates range from 78.74% to 96.42%. The best-recall point is 0.9286 at 79.48% exits, still 0.0106 below target. Candidate latency is estimated on training data and is not evidence of held-out savings. The thresholds may also be poorly scaled for Best Fields BM25 on SciFact. Test latency was measured in one pass, without repeated-run confidence intervals or energy measurements. The NQ sweep is not held-out efficiency evidence because its corpus pool uses evaluation qrels and its neural branches were precomputed. No cascade-versus-static answer-quality experiment exists. SPLADE alone slightly exceeds static RRF on two SciFact ranking metrics., so we cannot assess how close the best gate came to the quality constraint or whether a weaker target would provide a useful operating point. The thresholds may also be poorly scaled for Best Fields BM25 on SciFact. Latency was measured in one pass, without repeated-run confidence intervals or energy measurements. The NQ sweep is not a held-out efficiency result because its corpus pool uses evaluation qrels and its neural branches were precomputed. No cascade-versus-static answer-quality experiment exists. Finally, SPLADE alone slightly exceeds static RRF on two SciFact ranking metrics, making the choice of fallback a live design question.

The next experiment should recalibrate the gate features, compare with strong single-retriever baselines, then evaluate any selected policy on untouched queries. Measure latency over repeated runs. Any RAG-quality claim requires matched generator settings. quality?exit?latency frontier, compare calibrated gates and strong single-retriever baselines, then evaluate any selected policy on untouched queries. Latency should be measured over repeated runs. Any claim about RAG quality requires a matched generator evaluation with identical prompts and context budgets.

## 7. Conclusion

We evaluated a BM25-first gate for a hybrid retriever that combines BM25, dense MiniLM, and SPLADE with RRF. On a full-corpus held-out SciFact split, none of the 90 tested thresholds met the predeclared 98% training Recall@100 constraint. Static fallback therefore ran all branches on all test queries, reaching Recall@100 of 0.9610, nDCG@10 of 0.6984, and mean latency of 53.69 ms, effectively matching the static baseline. The current evidence does not establish an efficiency benefit.

The next study should recalibrate or redesign the confidence signal, include SPLADE-only as a strong baseline, and evaluate any revised policy on untouched queries with repeated latency measurements. quality?exit frontier, recalibrate or redesign the confidence signal, include SPLADE-only as a strong baseline, and evaluate any revised policy on untouched queries with repeated latency measurements.

## References

1. K. Sawarkar, A. Mangal, and S. R. Solanki. “Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers.” arXiv:2404.07220, 2024. https://arxiv.org/abs/2404.07220
2. S. Robertson and H. Zaragoza. “The Probabilistic Relevance Framework: BM25 and Beyond.” *Foundations and Trends in Information Retrieval*, 3(4):333–389, 2009. https://doi.org/10.1561/1500000019
3. T. Formal, B. Piwowarski, and S. Clinchant. “SPLADE: Sparse Lexical and Expansion Model for Information Retrieval.” In *Proceedings of SIGIR 2021*, 2021. https://doi.org/10.1145/3404835.3463098
4. G. V. Cormack, C. L. A. Clarke, and S. Büttcher. “Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.” In *Proceedings of SIGIR 2009*, pp. 758–759, 2009. https://doi.org/10.1145/1571941.1572114
5. S. Jeong et al. “Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.” In *Proceedings of NAACL 2024*, pp. 7036–7050, 2024. https://aclanthology.org/2024.naacl-long.389/
6. N. Thakur et al. “BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.” In *NeurIPS Datasets and Benchmarks*, 2021. https://openreview.net/forum?id=wCu6T5xFjeJ

## Author self-review (drafting checklist)

- **Contribution:** The current result is a negative evaluation of one heuristic and threshold grid; a strong standalone novelty claim is not yet established.
- **Writing clarity:** Gate variables, fallback behavior, data splits, and metrics are defined.
- **Experimental strength:** The corrected comparison is matched and held out, but no gate met the quality constraint. Latencies have no repeated-run uncertainty estimate.
- **Evaluation completeness:** No frontier table, transfer study, adaptive baseline, energy analysis, or cascade QA comparison is available.
- **Method soundness:** Conditional execution is implemented, but the tested heuristic did not activate on the held-out test set. Calibration and transfer remain unresolved.

## Claim-evidence map

- **Claim:** BM25 alone reaches 84.4% hit rate@20 on the evaluated NQ sample. **Evidence:** `kaggle_nq_and_xxl_results.json`, 500-query ablation. **Status:** Supported for the exploratory, qrels-filtered pool only.
- **Claim:** The corrected SciFact gate reduced neural calls under the 98% training recall constraint. **Evidence:** `cascaded-blended-rag/results/cascade_heldout_scifact.json` reports null thresholds, static fallback, 0% exits, and 0 calls skipped. **Status:** Not supported; no tested threshold met the constraint.
- **Claim:** Static RRF on held-out SciFact reached Recall@100 0.9610 and nDCG@10 0.6984, with 53.63 ms mean latency on a Tesla T4. **Evidence:** `cascaded-blended-rag/results/cascade_heldout_scifact.json`, 300 test queries. **Status:** Supported for this single measured pass and setup.
- **Claim:** The corrected cascade is faster than static RRF. **Evidence:** Mean latencies are 53.69 ms vs. 53.63 ms, with no neural calls skipped. **Status:** Not supported.
- **Claim:** Cascaded retrieval preserves downstream answer quality. **Evidence:** No cascade QA experiment is present. **Status:** Not evaluated.
