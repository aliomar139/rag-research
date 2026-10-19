# Cascaded Adaptive Blended Retrieval for Efficient Retrieval-Augmented Generation

> **Draft status:** This manuscript predates the corrected held-out evaluation. Treat its empirical results as provisional; see `results_ledger.md` and `improvement_plan.md`. It will be rewritten after the corrected GPU-backed run.

**A research paper draft**  
**Authors:** [Add author names and affiliations]

## Abstract

Hybrid retrieval combines lexical, dense, and learned sparse search to improve the chance that a relevant document appears near the top of a ranking. A static hybrid pipeline pays for every retrieval branch on every query, even when a cheaper lexical result may already be adequate. We study whether sequential confidence-based routing can reduce this cost in a Blended RAG-style system. Our cascade first runs BM25 and exits when its top score is high, or when a large score margin coincides with a query containing a high-IDF term; otherwise, it runs dense MiniLM and SPLADE retrieval and merges the three rankings with reciprocal rank fusion. On 500 Natural Questions queries, BM25 alone achieves 84.4% hit rate at 20, while static three-way fusion achieves 97.2%. A conservative cascade configuration exits on 73.2% of queries and reports 91.8% hit rate at 20, with mean latency of 25.3 ms compared with 67.0 ms for the reported static run. On BEIR SciFact, five-fold cross-validation selects the same threshold triplet in each fold and yields mean Recall@100 of 0.9001 and nDCG@10 of 0.6613, with 84.0% early exits and an estimated mean latency of 30.20 ms versus 90.87 ms for static fusion. The SciFact latency is reconstructed from cached component timings rather than measured in a live end-to-end cascade. These results indicate a promising quality–cost trade-off, while leaving live latency, broader generalization, and downstream answer quality for the cascade unverified.

**Keywords:** retrieval-augmented generation, hybrid retrieval, query routing, early exit, BM25, SPLADE

## 1. Introduction

Retrieval-augmented generation systems depend on retrieving useful evidence before generating an answer. Lexical retrieval is effective when query and document wording overlap, dense retrieval can match semantically similar text with different wording, and learned sparse retrieval combines term matching with model-based expansion. Blended RAG combines these complementary signals with reciprocal rank fusion (RRF) [1]. Its static execution pattern, however, runs each retriever for every query, regardless of whether an inexpensive result is already strong.

This paper studies a narrower systems question: can an initial lexical result identify a useful subset of queries for which neural retrieval can be skipped? The motivation comes from an ablation on 500 Natural Questions (NQ) queries. BM25 reaches a hit rate at 20 of 84.4%, compared with 97.2% for three-way BM25, dense, and SPLADE fusion. This gap shows both sides of the problem: lexical retrieval can cover many queries, while hybrid retrieval still recovers relevant documents that BM25 misses. The ablation motivates selective routing; it does not establish that every BM25 hit is redundant to neural retrieval or that the cascade preserves the fused system’s quality.

We implement a sequential cascade that computes BM25 first and makes an early-exit decision from its leading scores and the query’s maximum inverse document frequency (MaxIDF). Queries that do not satisfy the gate escalate to dense and SPLADE retrieval, followed by three-way RRF. The approach is evaluated using cached component retrieval results on SciFact and NQ. The experiments report retrieval quality, early-exit frequency, and latency estimates derived from component timings.

This work makes three contributions:

1. **A sequential early-exit design for a Blended RAG-style retriever.** BM25 confidence is used to decide whether dense and learned sparse retrieval should run.
2. **A branch ablation and routing evaluation.** The project reports single-branch and fused retrieval results on NQ, plus five-fold threshold selection on SciFact and a threshold sweep on NQ.
3. **A measured account of the current evidence boundary.** We distinguish retrieval outcomes from answer quality and estimated latency from end-to-end cascade timing, allowing the current results to be interpreted without treating them as a deployment validation.

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

The project evaluates BEIR SciFact using 300 test queries and Natural Questions using 500 queries from the BEIR setup. The SciFact corpus contains title and abstract fields. The retrievers use a BM25 implementation, `sentence-transformers/all-MiniLM-L6-v2` with FAISS inner-product search for dense retrieval, and `naver/splade-cocondenser-ensembledistil` for learned sparse retrieval. The available source code records cached top-100 component results and per-query component timings. NQ runs are reported from a Kaggle environment with two NVIDIA T4 GPUs; the SciFact artifacts report local evaluation.

### 4.2 Metrics and protocols

NQ reports hit rate at 5 and 20 and nDCG@10 for retriever ablations, plus hit rate at 20, early-exit percentage, and mean latency for a threshold sweep. SciFact reports Recall@100 and nDCG@10. Five-fold cross-validation partitions the 300 query IDs into five folds after shuffling with seed 42. For each fold, the threshold grid is searched over the other four folds, minimizing summed cached component latency subject to training Recall@100 being at least 98% of the full-fusion training recall. The chosen threshold is then evaluated on the held-out fold. The reported static latency is a sequential sum of cached BM25, dense, and SPLADE times. Cascade latency is calculated by adding BM25 time for exits and all three component times for escalations. These values estimate sequential retrieval cost; they do not include an observed live router run or necessarily represent a common hardware/runtime configuration across datasets.

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

### 5.2 SciFact cross-validation

**Table 2. SciFact five-fold results.** Cascade results are means over held-out folds; baseline values are computed over the full query set in the result script.

| System | Recall@100 | nDCG@10 | Mean latency (ms) | Early exits |
|:--|--:|--:|--:|--:|
| Static three-way RRF | 0.9610 | 0.7032 | 90.87 | 0.0% |
| Cascaded retrieval | 0.9001 | 0.6613 | 30.20* | 84.0% |

\* Estimated by summing cached component timings conditional on the routing outcome.

The cascade reduces reported mean retrieval cost by 66.8% relative to the 90.87 ms summed baseline and avoids neural branches on 84.0% of held-out queries. Mean held-out Recall@100 is 0.9001, or 93.66% of the full-set baseline recall; nDCG@10 is 0.6613. This is a measurable trade-off, not evidence that recall is preserved within the 2% training constraint on held-out data. The five folds select the same fallback triplet, \((\tau_{high},\tau_{gap},\tau_{idf})=(26,4,3.5)\), in the saved result artifact. The fold-level variation is not reported, so uncertainty around the mean cannot be assessed from the available summary.

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

The results support the feasibility of lexical-first routing under the tested conditions. BM25 provides a low-cost first-stage ranking, while score magnitude, score separation, and query specificity offer simple signals for deciding when to escalate. The NQ sweep makes the operating trade-off explicit, and SciFact cross-validation provides an initial check that a threshold selected without the held-out fold can reduce the estimated retrieval cost.

Several issues limit the strength of the evidence. First, the gate’s features and BM25 thresholds are collection-dependent, and the experiment does not test calibration transfer between datasets. Second, SciFact latency is reconstructed from independently cached component times, rather than measured with the cascade executing online; this omits or may misrepresent routing, scheduling, batching, and system overhead. The NQ latency setup is also not directly comparable to SciFact. Third, the evaluation does not include per-query confidence intervals, significance tests, tail latency, GPU energy, or an end-to-end answer-quality comparison for cascaded versus static retrieval. Fourth, the static baseline and cascade metrics on SciFact are not aggregated identically in the saved code: the baseline is evaluated on all queries while the cascade is averaged over held-out folds. A matched fold-level baseline would make the comparison cleaner. Finally, the NQ branch ablation shows that SPLADE alone outperforms the fused run on the listed retrieval metrics, so the choice of fallback and fusion configuration merits further study.

## 7. Conclusion

We presented a BM25-first cascade for a three-branch hybrid retriever. The gate exits on a high BM25 score or on a joint margin-and-specificity condition, and otherwise escalates to dense and SPLADE retrieval with RRF. On the project’s NQ sweep, the conservative configuration reports 73.2% early exits and a 62.2% lower mean latency than static fusion, at a 5.4-point reduction in hit rate@20. On SciFact, five-fold threshold selection reports 84.0% early exits and an estimated 66.8% latency reduction, with held-out mean Recall@100 of 0.9001 versus the reported static baseline of 0.9610. These results motivate further study of adaptive hybrid retrieval but do not yet establish live latency savings or downstream QA preservation.

The next evaluation should run the router and retrievers online under a shared hardware and serving setup, report fold-level and per-query uncertainty, compare matched fold baselines, and measure end-to-end answer quality. It should also evaluate threshold transfer across collections and compare alternative gates and fallback rankings, including SPLADE-only and learned or calibrated confidence models.

## References

1. K. Sawarkar, A. Mangal, and S. R. Solanki. “Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers.” arXiv:2404.07220, 2024. https://arxiv.org/abs/2404.07220
2. S. Robertson and H. Zaragoza. “The Probabilistic Relevance Framework: BM25 and Beyond.” *Foundations and Trends in Information Retrieval*, 3(4):333–389, 2009. https://doi.org/10.1561/1500000019
3. T. Formal, B. Piwowarski, and S. Clinchant. “SPLADE: Sparse Lexical and Expansion Model for Information Retrieval.” In *Proceedings of SIGIR 2021*, 2021. https://doi.org/10.1145/3404835.3463098
4. G. V. Cormack, C. L. A. Clarke, and S. Büttcher. “Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.” In *Proceedings of SIGIR 2009*, pp. 758–759, 2009. https://doi.org/10.1145/1571941.1572114
5. S. Jeong et al. “Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.” In *Proceedings of NAACL 2024*, pp. 7036–7050, 2024. https://aclanthology.org/2024.naacl-long.389/
6. N. Thakur et al. “BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.” In *NeurIPS Datasets and Benchmarks*, 2021. https://openreview.net/forum?id=wCu6T5xFjeJ

## Author self-review (drafting checklist)

- **Contribution:** The paper presents a specific early-exit extension and an initial empirical trade-off. Its novelty is scoped to this implementation and evaluation; priority claims are avoided.
- **Writing clarity:** Gate variables, fallback behavior, datasets, and metric definitions are specified. Threshold values are identified as collection-dependent.
- **Experimental strength:** Results show substantial estimated cost reduction with measurable retrieval loss. The available summary does not provide confidence intervals or significance tests.
- **Evaluation completeness:** The cascade has two dataset evaluations, but no fair online latency benchmark, adaptive baseline comparison, tail-latency analysis, or cascade QA evaluation.
- **Method soundness:** The gate is reproducible from source, though score calibration and dataset transfer remain unvalidated. Latency evidence is explicitly labeled as an estimate where appropriate.

## Claim–evidence map

- **Claim:** BM25 alone reaches 84.4% hit rate@20 on the evaluated NQ sample. **Evidence:** `kaggle_nq_and_xxl_results.json`, 500-query ablation. **Status:** Supported for this sample and setup.
- **Claim:** The conservative NQ cascade configuration reports 73.2% early exits, 25.3 ms mean latency, and 91.8% hit rate@20. **Evidence:** Project result tables in the detailed report and repository anchor. **Status:** Supported as reported; the summary artifact does not preserve query-level cascade timing data.
- **Claim:** SciFact cross-validation reports 84.0% early exits and mean Recall@100 of 0.9001. **Evidence:** `cv_cascaded_results.json` and `07_tune_cascade_cv.py`. **Status:** Supported by saved aggregate, with an unmatched full-set baseline comparison noted.
- **Claim:** SciFact latency falls 66.8%. **Evidence:** Cached component-latency arithmetic in `07_tune_cascade_cv.py`; the result JSON stores the percentage. **Status:** Supported as a reconstructed estimate, not live end-to-end timing.
- **Claim:** The cascade preserves downstream answer quality. **Evidence:** No cascade QA experiment is present. **Status:** Needs evidence; excluded as a result claim.

