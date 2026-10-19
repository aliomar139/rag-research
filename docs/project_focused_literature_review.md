# Project-Focused Literature Review: Lexical Early Exit in Hybrid RAG

## Purpose and scope

This review positions the repository’s specific question: whether a BM25-first gate can skip dense and learned-sparse retrieval for some queries in a Blended RAG-style system. It draws on the downloaded papers in `Similar-Papers/downloaded_papers/`, prioritizing work that changes retrieval strategy, gates retrieval, uses a post-retrieval cascade, or adapts hybrid retrieval execution. It is a focused narrative review, not a systematic review of all 106 downloaded PDFs. The downloaded set contains work on many RAG topics, and download count alone does not establish relevance or exhaustive screening.

The comparison distinguishes four different decisions that are often grouped together under “adaptive RAG”:

1. **Whether to retrieve at all**, based on model uncertainty or another pre-retrieval signal.
2. **Which retrieval strategy or engine to use**, based on query characteristics or observed response quality.
3. **Whether to escalate after an initial retrieval**, based on what that retrieval returned.
4. **How much of a hybrid retrieval process to execute**, while preserving or trading off ranking quality.

The project’s intended decision is closest to the third: run lexical retrieval, then decide whether neural branches are needed. The implemented experiment must be interpreted carefully because its SciFact CV script reads the single-field BM25 cache, while the project framing describes a Best Fields gate.

## 1. Static hybrid retrieval and the project baseline

Sawarkar et al.’s Blended RAG combines lexical, dense, and sparse-encoder retrieval with hybrid query strategies and rank fusion [1]. It is the appropriate architectural starting point for this project because the proposed cascade keeps a three-branch fallback and changes when that fallback runs. The project’s baseline uses BM25, `all-MiniLM-L6-v2`, and SPLADE with RRF (rank constant 60).

The repository’s reproduction should be described as an **independent implementation of a Blended RAG-style pipeline**, not as a strict reproduction of every original system detail. The implementation uses an open SPLADE model and local retrievers, while the reference work includes its own model and infrastructure choices. Dataset scope, candidate pools, metrics, and retrieval implementation also affect direct numerical comparisons.

The project’s own ablations show why adaptive execution is worth investigating, but they do not establish that the neural branches are redundant whenever BM25 finds a relevant document. On the notebook’s 500-query, qrels-filtered NQ pool, BM25 achieves hit rate@20 of 84.4%, while three-way RRF achieves 97.2%. SPLADE alone reaches 98.2% hit rate@20 and nDCG@10 of 0.7589; the fused run reaches 97.2% and 0.7094. Thus, BM25 covers many queries, but it leaves room for neural retrieval to add relevant candidates, and this particular fusion configuration does not outperform SPLADE alone on the reported NQ metrics. These values are descriptive of the notebook’s restricted pool, not a full-corpus BEIR evaluation.

## 2. Adaptive retrieval and strategy routing

Adaptive-RAG predicts query complexity and selects among no retrieval, single-step retrieval, and more involved retrieval strategies [2]. MBA-RAG uses a bandit framework to choose retrieval strategies while balancing answer quality and retrieval cost [3]. Unsupervised Query Routing learns to select a search engine using automatically constructed response-quality signals [4]. MoR learns to select and combine a portfolio of heterogeneous retrievers [5]. SymRAG routes among symbolic, neural, and hybrid pathways using query and system signals [6].

These systems establish that allocating retrieval effort by query is a broader research direction. They are not direct replications of this project’s mechanism: their choices concern retrieval paradigms, engines, or reasoning pathways, rather than a BM25 score-and-margin gate over the branches of this specific hybrid pipeline. They should therefore be used to position the work in adaptive retrieval, not dismissed as irrelevant because they lack a multi-field index.

TARG asks whether external retrieval is needed, using uncertainty from a short model-generated draft before retrieval [7]. It shares the goal of avoiding unnecessary retrieval work, but the signal and decision point differ: TARG uses generator uncertainty to bypass retrieval, whereas this project uses lexical retrieval evidence to bypass additional neural retrievers. The comparison matters because it prevents a broad claim that selective retrieval or early gating is itself new.

## 3. Cascades and post-retrieval decisions

The closest conceptual comparison in the downloaded collection is *The Coverage Illusion* [8]. It studies a cheapest-first, post-retrieval cascade for query augmentation: run a lower-cost workflow first and escalate to LLM-based augmentation when retrieval does not produce documents. In its production case study, only 27.8% of real queries required augmentation under the reported deferral policy, and the cascade reduced latency relative to always using HyDE. This is a different trigger and system—the escalation is for query augmentation, not from BM25 to dense-plus-SPLADE retrieval—but it demonstrates that conditional post-retrieval escalation and cost-ordered cascades have been studied in RAG. Accordingly, the project should not claim to introduce sequential early exit as a general idea.

The project’s differentiating question is more specific: can the top BM25 score, the gap between the first two BM25 candidates, and query-term IDF decide when to return a lexical ranking versus invoke neural branches? That distinction is meaningful only if evaluated against the actual execution path and baselines that use the same queries, fields, and hardware.

## 4. Adaptive hybrid execution and confidence signals

SAGE learns a per-query passage budget from lightweight initial-retrieval features, including score distributions, rank gaps, and lexical signals, to improve a quality–latency trade-off under SLOs [9]. This supports the general plausibility of using retrieval-side signals for resource allocation. It does **not** independently validate this project’s particular conjunction of MaxIDF and BM25 margin, nor does choosing a passage count establish that dense and sparse encoder calls can be skipped.

Exact Adaptive Hybrid Retrieval (EAHR) addresses the cost of hybrid RRF by adaptively requesting deeper portions of dense and sparse rankings, using bounds to preserve the ordered top-K of complete-list RRF [10]. This is a close systems comparison because it adapts hybrid retrieval execution itself. Its guarantee and objective differ from the project’s: EAHR aims to reproduce a declared fused ranking exactly while adapting list depth; the BM25 cascade can return a different ranking on exit and trades retrieval quality against cost. EAHR should be treated as a direct neighboring method, while noting that the downloaded version is a 2026 preprint.

DAT dynamically changes sparse–dense weights after assessing retrieval results [11]. Since it evaluates results from both retrieval branches, it is reasonable to distinguish it from a system intended to avoid invoking those branches. The comparison should remain specific to DAT’s mechanism; it does not justify a blanket statement that all adaptive fusion methods save zero compute.

## 5. What the project has actually measured

### 5.1 Retriever ablations

**Table 1. Natural Questions retrieval results in the saved 500-query result artifact.**

| Retriever | Hit@5 | Hit@20 | nDCG@10 |
|:--|--:|--:|--:|
| BM25 | 0.670 | 0.844 | 0.5433 |
| Dense MiniLM | 0.806 | 0.960 | 0.6751 |
| SPLADE | 0.890 | 0.982 | 0.7589 |
| Three-way RRF | 0.830 | 0.972 | 0.7094 |

This is useful motivation for an adaptive policy, but not a direct test of the gate. Hit@20 measures whether a relevant document appears in the first 20; it does not measure answer quality, whether fusion improved the rank of a BM25 hit, or whether the generator can answer from the BM25-only context. SPLADE’s stronger values also mean the project should compare its cascade against a strong single retriever, not only against static fusion and BM25.

### 5.2 Cascade evaluations

The saved SciFact result artifact reports five-fold mean cascade Recall@100 = 0.9001, nDCG@10 = 0.6613, estimated mean latency = 30.20 ms, and 84% early exits. The accompanying script searches thresholds on four folds subject to a training recall constraint of 98% of full fusion, then evaluates the selected gate on the held-out fold. It selects the same threshold triplet in all five folds. The comparison in the script is not fully matched: static metrics are evaluated on the full 300-query set, while cascade metrics are averaged over held-out folds. The 90.87 ms static latency is hard-coded as a sum of component averages, and cascade latency is reconstructed from cached component timings rather than observed by executing a live cascade. In addition, this script reads `bm25_single_cache.json`, not `bm25_best_cache.json`.

The executed Kaggle notebook contains both a default NQ cascade run and a sensitivity sweep. Its default run reports 92.4% exits, hit@20 of 88.0%, nDCG@10 of 0.5625, and 14.3 ms estimated latency; the static run reports hit@20 of 97.2% and nDCG@10 of 0.7094. The sweep reports the conservative setting (28, 6, 3.5) at 73.2% exits, hit@20 of 91.8%, nDCG@10 of 0.6115, and 25.3 ms; progressively more aggressive settings increase exits and reduce retrieval metrics. These are executed notebook outputs, not just figures in a summary. However, the latency values are constants assigned in the notebook (10 ms BM25, 12 ms dense, 45 ms SPLADE), not timings measured by the cascade.

The Kaggle notebook also limits the NQ corpus to the first 15,000 documents plus every document judged relevant to the selected 500 queries. This constructs the evaluation pool using the evaluation qrels and is not a clean held-out retrieval protocol; it also differs substantially from searching the full BEIR NQ collection. In addition, the notebook computes BM25, dense, SPLADE, and static RRF rankings for all 500 queries before applying the cascade logic to those in-memory rankings. The reported exit rate therefore measures the proportion of queries the rule would route to BM25; it does not demonstrate that the notebook actually skipped neural model invocations. Finally, the notebook evaluates several threshold settings on the same 500-query set and supplies no independent NQ validation split. The sweep is useful for exploratory analysis but not an unbiased estimate of a selected threshold’s generalization.

Neither cascade evaluation establishes preserved answer quality. The FLAN-T5-XXL SQuAD result in the project is a separate generator experiment and is not a comparison between cascaded and static retrieval. The current cascade evidence is therefore retrieval quality in an exploratory NQ pool and a SciFact cached-results simulation, not end-to-end QA quality or measured production savings.

### 5.3 Other reproduction results are context, not cascade evidence

The repository’s SQuAD retrieval result has hit@10 of 0.906 for BM25, 0.954 for dense, 0.988 for SPLADE, and 0.964 for fusion. Its TREC-COVID result has nDCG@10 of 0.6463 for fusion, compared with 0.4998 BM25, 0.5904 dense, and 0.6403 SPLADE. These results show that branch complementarity and the best single retriever vary by dataset. They do not validate the cascade, which has not been evaluated on these datasets in the saved artifacts.

## 6. Project-specific synthesis and defensible positioning

The literature supports the general premise that retrieval work can be allocated adaptively. The downloaded comparisons also show that this premise has multiple established forms: query-complexity routing, engine or retriever selection, retrieval gating, cheapest-first workflow escalation, passage-budget selection, and adaptive hybrid-list execution. The project’s potentially distinctive angle is narrower: a **lexical-first gate that uses BM25 candidate evidence to decide whether to run the dense and learned-sparse branches of a Blended RAG-style retriever**.

The current evidence supports these statements:

- BM25 has substantial hit@20 on the evaluated NQ subset, while neural branches and fusion can add further retrieval coverage.
- A thresholded BM25 gate can produce a range of exit rates and retrieval-quality outcomes in offline evaluation.
- On SciFact, the saved cross-validation summary reports 84% early exits alongside lower recall and nDCG than the static run, with latency calculated from cached timings.

The current evidence does not support claims that the method is the first adaptive RAG system, that it preserves accuracy within 1.5–2%, that it has validated downstream QA quality, or that the reported latency gains are measured live end-to-end. Nor does it establish preservation of Best Fields behavior in the SciFact cascade, given the cache used by the CV script.

The research contribution should therefore be framed as an **initial empirical study of a simple lexical early-exit policy for a hybrid retriever**, with its retrieval-quality/estimated-cost trade-off and methodological limits stated directly. Any stronger novelty claim needs a documented search protocol and explicit comparison against the closest neighboring mechanisms above.

## References

1. K. Sawarkar, A. Mangal, and S. R. Solanki. “Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers.” IEEE MIPR, 2024. [arXiv:2404.07220](https://arxiv.org/abs/2404.07220). [Base Paper PDF](docs/Blended-Rag.pdf).
2. S. Jeong et al. “Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.” NAACL, 2024. [ACL Anthology](https://aclanthology.org/2024.naacl-long.389/).
3. X. Tang et al. “MBA-RAG: a Bandit Approach for Adaptive Retrieval-Augmented Generation through Question Complexity.” COLING, 2025. [ACL Anthology](https://aclanthology.org/2025.coling-main.218/).
4. F. Mu et al. “Unsupervised Query Routing for Retrieval Augmented Generation.” 2025. [arXiv:2501.07793](https://arxiv.org/abs/2501.07793).
5. J. S. Kalra et al. “MoR: Better Handling Diverse Queries with a Mixture of Sparse, Dense, and Human Retrievers.” EMNLP, 2025. [ACL Anthology](https://aclanthology.org/2025.emnlp-main.601/).
6. S. B. Hakim et al. “SymRAG: Efficient Neuro-Symbolic Retrieval Through Adaptive Query Routing.” NeSy, PMLR 284, 2025. [Proceedings](https://proceedings.mlr.press/v284/).
7. Y. Wang, L. Wei, and H. Ling. “Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG.” TMLR, 2026. [OpenReview](https://openreview.net/pdf?id=L8gYtUZfVU).
8. Z. Hussain and K. Nielbo. “The Coverage Illusion: From Pre-retrieval Routing Failure to Post-retrieval Cascades in a Production RAG System.” 2026 preprint. [arXiv:2605.27220](https://arxiv.org/abs/2605.27220).
9. M. F. Raza, S. Yang, and S. M. Srinivasan. “SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems.” 2026 preprint. [arXiv:2608.08237](https://arxiv.org/abs/2608.08237).
10. C. Zhang. “Exact Adaptive Hybrid Retrieval Without Fixed Top-L Cutoffs.” 2026 preprint. [arXiv:2608.07152](https://arxiv.org/abs/2608.07152).
11. H.-L. Hsu and J. Tzeng. “DAT: Dynamic Alpha Tuning for Hybrid Retrieval in Retrieval-Augmented Generation.” 2025 preprint. [arXiv:2503.23013](https://arxiv.org/abs/2503.23013).

## Evidence files consulted

- `cascaded-blended-rag/results/kaggle_nq_and_xxl_results.json`
- `cascaded-blended-rag/results/cv_cascaded_results.json`
- `cascaded-blended-rag/results/squad_replication_results.json`
- `cascaded-blended-rag/results/trec_covid_replication_results.json`
- `cascaded-blended-rag/experiments/07_tune_cascade_cv.py`
- `cascaded-blended-rag/src/cascaded_router.py`
- `Similar-Papers/download_report.md` and `Similar-Papers/downloaded_papers_index.csv`

