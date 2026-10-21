# Presentation speaker notes

Suggested talk: 15–20 minutes on slides 1–19. Slides 20–23 are appendices.

## 1. Conditional Hybrid Retrieval

We begin with the Blended RAG paper we chose, then explain our reproduction, its limits and results. We next describe the indexed implementation and compare its results with our first version. The expected contribution is a careful measurement of query cost under recall constraints, not a new retrieval algorithm. This is a bounded engineering study. No new dataset or model run was performed for this presentation. Suggested talk length: 15–20 minutes for slides 1–19. Slides 20–23 are appendices.

## 2. The original Blended RAG paper

Sawarkar, Mangal and Solanki study keyword, dense-vector and learned-sparse retrieval with hybrid field queries. Their implementation uses Elasticsearch and ELSER. The published TREC nDCG and SQuAD answer values are reference results from that paper. Our system did not reproduce them. The paper motivates complementary retrieval signals, while this project investigates the runtime cost of always executing them. CoQA appears in the paper’s discussion of metadata-poor documents. Source: https://arxiv.org/html/2404.07220v2

## 3. What we set out to contribute

The expected contribution was an inspectable test of whether a BM25-first gate can reduce query time while respecting an explicit recall target. We also wanted to distinguish the gain from indexed lexical execution from the additional gain due to routing, and to compare the cascade with simpler retrievers. These are empirical engineering questions. Conditional retrieval and indexed sparse scoring already have prior work, so we make no algorithmic novelty claim.

## 4. First baseline from our adaptation

These are the first baseline results from our Python adaptation on the overlapping paper datasets. Using the same datasets lets us compare our own implementation stages. It does not reproduce the original Elasticsearch and ELSER system or support a direct speed comparison with it. On TREC-COVID the r90 setting retains 84.53 percent recall and saves 3.56 percent mean time. On full-query SQuAD retrieval it retains 99.38 percent and saves 6.12 percent, with one complete timing pass. In the TREC run, BM25 itself takes 1410.14 milliseconds out of 1493.23 milliseconds for fusion. This showed that the gate paid most of the cost before deciding whether to skip neural retrieval. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 5. Reproduction and adaptation boundaries

We reproduce the use of complementary retrieval signals and Best Fields lexical scoring, then make declared substitutions. Our accessible Python implementation uses MiniLM, SPLADE and RRF. The original paper studies a different search engine, sparse model and query formulations. We also construct an explicit SQuAD paragraph corpus. These differences prevent direct equivalence of quality or latency numbers. Source: https://arxiv.org/html/2404.07220v2. Project method: docs/FINAL_MANUSCRIPT.md, sections 3–5.

## 6. Recall retention and absolute recall

Retention is the ratio of mean Recall@100 values. It is not the overlap of two result lists and it does not imply the same quality for every query. In TREC, many documents can be relevant for a topic. Recovering at least one relevant item gives a hit, while recall asks how much of the relevant set was recovered. Our primary retains 96.68 percent of a reference recall near 0.12250, giving absolute recall near 0.11844. nDCG evaluates ranking order and relevance, while EM and F1 evaluate generated answers. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 7. The live conditional decision

Every cascade query first executes Best Fields BM25. Eleven features from the query and initial lexical ranking feed a standardized Ridge regressor. The regressor predicts clipped recall gain from escalation. A small predicted gain returns BM25 immediately. A larger prediction executes Dense and SPLADE and fuses those rankings with the already-computed BM25 result. Inference does not use evaluation relevance labels. Methods do not borrow precomputed neural query results from one another. Sources: experiments 12 and 25; frozen router artifact.

## 8. Four frozen recall budgets

The calibration uses five-fold out-of-fold predictions on 2590 NFCorpus train queries, seed 42. A Ridge model with alpha 10 predicts nonnegative gain. Each policy limits the sum of observed clipped losses over the queries selected for exit. The final exported model fits all training queries. Training calibration is an empirical procedure and does not guarantee transfer to a new distribution. All four policies remain frozen in the reported follow-ups. Source: results/nfcorpus_frozen_retention_routers.json and experiment 12.

## 9. Completed evaluation scope

NFCorpus dev is inspected development evidence. SCIDOCS is frozen confirmation. Original paper-dataset runs declared r90 primary. The indexed follow-up declared r99 primary against a 90 percent floor, using already-inspected TREC quality evidence. It therefore tests compatible execution rather than new independent TREC generalization. Timings include query encoding, retrieval and routing with GPU synchronization, one CPU thread and randomized order. Corpus encoding, index preparation, model loading, downloads and answer generation are outside the timing window. Source: frozen protocols and main manuscript section 5.

## 10. TREC-COVID primary result

The conservative r99 policy was preselected as primary for the indexed engineering follow-up. It takes 73.90 milliseconds against indexed fusion at 91.28 milliseconds, giving 1.2351 times speedup, rounded to 1.24. This 1.24x isolates routing because both methods use indexed BM25. Against our earlier, non-indexed fusion run, the same cascade is 20.21x faster across separate runs; that ratio combines indexing and routing. The indexing-only comparison is 16.36x. These are comparisons within our implementation, not against the original paper system. The cascade retains 96.68 percent of fusion recall, with an interval of 93.58 to 98.75 percent. Mean time saved is 19.04 percent against indexed fusion. The primary p95 is slightly higher than indexed fusion, so the claim concerns mean time. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 11. TREC-COVID operating points

All four frozen settings remain visible. r95 is the fastest secondary point above 90 percent observed retention. Its retention interval is 86.88 to 95.76 percent and crosses the floor. r92 and r90 both retain 84.53 percent, so their greater speed does not meet the requested quality budget. The 99/95/92/90 names refer to training calibration, and every nominal target still misses on TREC. The fair additional-routing comparison is against the same indexed fusion backend. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 12. Indexing and routing combined

This progression divides our original, non-indexed fusion latency by indexed method latency. For primary r99, 20.21x is the combined observed ratio: 16.36x comes from indexed fusion alone, and 1.24x compares indexed cascade against indexed fusion. These separate runs include session variation, so they describe measured configurations rather than a causal decomposition. The larger settings reach 26.83x/35.56x/36.96x against our original fusion but do not meet the requested quality floor. These are not comparisons with the original Blended RAG system. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 13. SQuAD sampled cascade results

The indexed SQuAD result uses 256 fixed SHA-selected questions with the full 20958-document corpus. The fastest secondary setting r90 takes 18.59 milliseconds and retains 99.22 percent of fusion recall, producing 1.59 times speedup over indexed fusion. Its retention interval is 98.05 to 100 percent. The primary retains 100 percent sampled recall at 1.04 times. Original evidence covers 10570 questions with one complete pass plus separately preserved partial repetitions. Cross-run ratios use the same 256 questions, not the full-query original mean. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 14. Standalone retrievers remain strong

The static fusion reference is useful for measuring what the cascade preserves, but it is not automatically the best efficient action. On TREC, SPLADE has higher recall and nDCG than the primary at lower mean time. The primary has higher MAP, 0.01769 versus 0.01662. On sampled SQuAD, BM25 alone exceeds the 90 percent recall floor at 1.03 milliseconds, and SPLADE has the highest nDCG and MAP. Thus a fusion-relative cascade gain does not establish overall superiority under a recall/time objective. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 15. Earlier transfer findings

Earlier runs are part of the evidence, not discarded failures. The NFCorpus development primary r92 retains 94.74 percent and saves 23 percent mean time. Frozen SCIDOCS r92 retains 90.01 percent and saves only 5.53 percent, with a retention interval crossing 90. Every nominal target fails on SCIDOCS and TREC. Dense is a stronger recall/time action than the cascades on NFCorpus and SCIDOCS. These observations motivate the first-stage investigation and limit any universal transfer claim. Source: final_quality_speed.csv and the original benchmark reports.

## 16. Saved evidence and compatibility

We compare every saved first-pass result across original and indexed backends on the same query IDs. The 400 TREC records and 2048 sampled SQuAD records match in ranked IDs, scores, routes and recall. Independent scorers reproduce quality against exact raw relevance labels, and frozen source and weight hashes identify the executed code. These checks validate saved artifacts and quality calculations. They do not independently verify the GPU clock or establish production behavior. Source: experiments 22, 29, 30 and 31; recorded local verification and metric-check reports.

## 17. Implementation and reproducibility

The code remains organized around retrieval modules, numbered experiments, frozen result protocols and compact saved reports. The indexed backend changes how lexical scores execute while preserving score definitions. Benchmark 25 executes the indexed cascade; import and scoring helpers check returned evidence. The publication builders only consume saved reports. They do not run neural encoding, training or query timing. The source-sharing ZIP excludes local datasets, environments and bulk query rankings. Their original exports are identified separately in the artifact catalog. Source: repository module indexes and reproduction guide.

## 18. What to improve in a next study

The completed study does not evaluate full current NQ, HotpotQA or CoQA. Earlier NQ used a relevance-informed candidate pool and simulated routing, so it cannot establish full-corpus savings for this cascade. A next study should predeclare a new evaluation protocol, test generated-answer EM/F1 and modern adaptive baselines, then measure multiple hardware types, concurrent load, energy and index memory/update costs. It should retain strong standalone retrievers as baselines.

## 19. Overall review

The project met its bounded engineering aim: it optimized the lexical stage, compared routing against an equally indexed reference, and reported a real quality/time trade-off. Against our earlier fusion run, the TREC primary is 20.21 times faster at 96.68 percent indexed-fusion recall retention. The fastest indexed SQuAD setting is 7.91 times faster at 99.22 percent on the same 256-question sample. The additional routing-only gains are 1.24 times over indexed fusion on TREC and 1.59 times on SQuAD. The findings are limited by small query scopes in some results, one GPU type, previously inspected TREC quality, and stronger standalone alternatives. The work supports an implementation finding, not a new algorithm or overall superiority. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 20. Appendix: TREC-COVID quality metrics

These are the complete indexed quality values and mean times for the same evaluated query set. Recall retained uses mean Recall@100 as the denominator. The SQuAD sample is not the full-query evaluation. Values are rounded for display; source reports retain full precision. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 21. Appendix: SQuAD sample quality metrics

These are the complete indexed quality values and mean times for the same evaluated query set. Recall retained uses mean Recall@100 as the denominator. The SQuAD sample is not the full-query evaluation. Values are rounded for display; source reports retain full precision. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 22. Appendix: primary references

Primary sources verified for this release: https://arxiv.org/html/2404.07220v2
https://arxiv.org/abs/2109.10739
https://arxiv.org/abs/2407.03618
https://arxiv.org/abs/2104.08663
https://arxiv.org/abs/1606.05250
https://arxiv.org/abs/2107.05720
https://arxiv.org/abs/1908.10084. The main manuscript provides full bibliographic details. Sources for each measured claim appear in the corresponding speaker notes and the shared results JSON.

## 23. Appendix: rebuilding the report

To rebuild figures, run experiment 32 from the repository root using the documented analysis requirements. To rebuild the release figures, paper and HTML, use the publication tools and requirements. The editable presentation builder uses the bundled Codex Artifact Tool runtime. Headless browser rendering creates the showcase PDF. Exact commands and required environment variables appear in docs/submission/README.md. No new Kaggle run is necessary to read or regenerate current result plots. Full record-level rescoring needs original matching result archives separately.
