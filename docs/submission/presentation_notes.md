# Presentation speaker notes

Suggested talk: 15–20 minutes on slides 1–18. Slides 19–22 are appendices.

## 1. Conditional Hybrid Retrieval

This presentation develops the completed research showcase. The central question is whether conditional retrieval saves actual query time while preserving a chosen fraction of fusion recall. We will distinguish the original Blended RAG paper from our adaptation, explain the metric differences, and separate the gain from indexing from the extra gain from routing. The work is a bounded engineering study. No new dataset or model run was performed for this presentation. Suggested talk length: 15–20 minutes for slides 1–18. Slides 19–22 are appendices.

## 2. The original Blended RAG paper

Sawarkar, Mangal and Solanki study keyword, dense-vector and learned-sparse retrieval with hybrid field queries. Their implementation uses Elasticsearch and ELSER. The published TREC nDCG and SQuAD answer values are reference results from that paper. Our system did not reproduce them. The paper motivates complementary retrieval signals, while this project investigates the runtime cost of always executing them. CoQA appears in the paper’s discussion of metadata-poor documents. Source: https://arxiv.org/html/2404.07220v2

## 3. The first-stage bottleneck

The initial TREC implementation spent 1410.14 milliseconds in BM25 while fusion took 1493.23 milliseconds. The gate had already paid most of the runtime before deciding whether to skip neural branches. Neural-call reduction alone therefore could not establish a useful speedup. The backend optimization addresses this execution bottleneck and is applied to reference and proposed methods equally. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 4. Reproduction and adaptation boundaries

We reproduce the use of complementary retrieval signals and Best Fields lexical scoring, then make declared substitutions. Our accessible Python implementation uses MiniLM, SPLADE and RRF. The original paper studies a different search engine, sparse model and query formulations. We also construct an explicit SQuAD paragraph corpus. These differences prevent direct equivalence of quality or latency numbers. Source: https://arxiv.org/html/2404.07220v2. Project method: docs/FINAL_MANUSCRIPT.md, sections 3–5.

## 5. Recall retention and absolute recall

Retention is the ratio of mean Recall@100 values. It is not the overlap of two result lists and it does not imply the same quality for every query. In TREC, many documents can be relevant for a topic. Recovering at least one relevant item gives a hit, while recall asks how much of the relevant set was recovered. Our primary retains 96.68 percent of a reference recall near 0.12250, giving absolute recall near 0.11844. nDCG evaluates ranking order and relevance, while EM and F1 evaluate generated answers. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 6. The live conditional decision

Every cascade query first executes Best Fields BM25. Eleven features from the query and initial lexical ranking feed a standardized Ridge regressor. The regressor predicts clipped recall gain from escalation. A small predicted gain returns BM25 immediately. A larger prediction executes Dense and SPLADE and fuses those rankings with the already-computed BM25 result. Inference does not use evaluation relevance labels. Methods do not borrow precomputed neural query results from one another. Sources: experiments 12 and 25; frozen router artifact.

## 7. Four frozen recall budgets

The calibration uses five-fold out-of-fold predictions on 2590 NFCorpus train queries, seed 42. A Ridge model with alpha 10 predicts nonnegative gain. Each policy limits the sum of observed clipped losses over the queries selected for exit. The final exported model fits all training queries. Training calibration is an empirical procedure and does not guarantee transfer to a new distribution. All four policies remain frozen in the reported follow-ups. Source: results/nfcorpus_frozen_retention_routers.json and experiment 12.

## 8. Completed evaluation scope

NFCorpus dev is inspected development evidence. SCIDOCS is frozen confirmation. Original paper-dataset runs declared r90 primary. The indexed follow-up declared r99 primary against a 90 percent floor, using already-inspected TREC quality evidence. It therefore tests compatible execution rather than new independent TREC generalization. Timings include query encoding, retrieval and routing with GPU synchronization, one CPU thread and randomized order. Corpus encoding, index preparation, model loading, downloads and answer generation are outside the timing window. Source: frozen protocols and main manuscript section 5.

## 9. TREC-COVID primary result

The conservative r99 policy was preselected as primary for the indexed engineering follow-up. It takes 73.90 milliseconds against indexed fusion at 91.28 milliseconds, giving 1.2351 times speedup, rounded to 1.24. It retains 96.68 percent of fusion recall, with a query-bootstrap interval of 93.58 to 98.75 percent. Mean time saved is 19.04 percent, interval 9.47 to 30.11. This meets the user’s 90 percent floor for this measured setting and misses the policy’s nominal 99 percent target. The primary p95 is slightly higher than fusion, so the claim concerns mean time. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 10. TREC-COVID operating points

All four frozen settings remain visible. r95 is the fastest secondary point above 90 percent observed retention. Its retention interval is 86.88 to 95.76 percent and crosses the floor. r92 and r90 both retain 84.53 percent, so their greater speed does not meet the requested quality budget. The 99/95/92/90 names refer to training calibration, and every nominal target still misses on TREC. The fair additional-routing comparison is against the same indexed fusion backend. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 11. Indexing and routing combined

This larger progression divides original fusion latency by the indexed method latency. It combines indexing and routing across different runs. Indexing alone gives 16.36 times, followed by r99 at 20.21, r95 at 26.83, r92 at 35.56 and r90 at 36.96. These are configurations, not scaling measurements as corpus size grows. The maximum is below the requested quality floor. The r95 secondary is above 90 percent at its point estimate but its interval crosses the floor. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 12. SQuAD sampled cascade results

The indexed SQuAD result uses 256 fixed SHA-selected questions with the full 20958-document corpus. The fastest secondary setting r90 takes 18.59 milliseconds and retains 99.22 percent of fusion recall, producing 1.59 times speedup over indexed fusion. Its retention interval is 98.05 to 100 percent. The primary retains 100 percent sampled recall at 1.04 times. Original evidence covers 10570 questions with one complete pass plus separately preserved partial repetitions. Cross-run ratios use the same 256 questions, not the full-query original mean. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 13. Standalone retrievers remain strong

The static fusion reference is useful for measuring what the cascade preserves, but it is not automatically the best efficient action. On TREC, SPLADE has higher recall and nDCG than the primary at lower mean time. The primary has higher MAP, 0.01769 versus 0.01662. On sampled SQuAD, BM25 alone exceeds the 90 percent recall floor at 1.03 milliseconds, and SPLADE has the highest nDCG and MAP. Thus a fusion-relative cascade gain does not establish overall superiority under a recall/time objective. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 14. Earlier transfer findings

Earlier runs are part of the evidence, not discarded failures. The NFCorpus development primary r92 retains 94.74 percent and saves 23 percent mean time. Frozen SCIDOCS r92 retains 90.01 percent and saves only 5.53 percent, with a retention interval crossing 90. Every nominal target fails on SCIDOCS and TREC. Dense is a stronger recall/time action than the cascades on NFCorpus and SCIDOCS. These observations motivate the first-stage investigation and limit any universal transfer claim. Source: final_quality_speed.csv and the original benchmark reports.

## 15. Saved evidence and compatibility

We compare every saved first-pass result across original and indexed backends on the same query IDs. The 400 TREC records and 2048 sampled SQuAD records match in ranked IDs, scores, routes and recall. Independent scorers reproduce quality against exact raw relevance labels, and frozen source and weight hashes identify the executed code. These checks validate saved artifacts and quality calculations. They do not independently verify the GPU clock or establish production behavior. Source: experiments 22, 29, 30 and 31; recorded local verification and metric-check reports.

## 16. Implementation and reproducibility

The code remains organized around retrieval modules, numbered experiments, frozen result protocols and compact saved reports. The indexed backend changes how lexical scores execute while preserving score definitions. Benchmark 25 executes the indexed cascade; import and scoring helpers check returned evidence. The publication builders only consume saved reports. They do not run neural encoding, training or query timing. The source-sharing ZIP excludes local datasets, environments and bulk query rankings. Their original exports are identified separately in the artifact catalog. Source: repository module indexes and reproduction guide.

## 17. Limits and deferred evaluation

The completed study does not evaluate full current NQ, HotpotQA or CoQA. The old NQ candidate pool is qrel-informed and the routing is simulated, so it cannot support the current method’s full-corpus savings. Generated-answer EM/F1, modern adaptive-retrieval baselines, concurrency, energy and index memory/update costs are unmeasured. One GPU type, small numbers of sessions and known TREC quality restrict generalization. A later study should preserve this release, declare its protocol and quality objective, and retain strong standalone actions.

## 18. Conclusion

The main result is that compatible first-stage indexing makes the frozen cascade deliver measurable extra mean-time savings against an equally optimized fusion reference. TREC primary gives 1.24 times at 96.68 percent retention. The fastest sampled SQuAD cascade gives 1.59 times at 99.22 percent. Every secondary point, uncertainty and standalone comparison remains visible. The contribution is execution optimization and empirical understanding of its quality/time limits. It does not establish overall superiority or a new routing principle. The current project can be reported at this scope while remaining datasets are tested later. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 19. Appendix: TREC-COVID quality metrics

These are the complete indexed quality values and mean times for the same evaluated query set. Recall retained uses mean Recall@100 as the denominator. The SQuAD sample is not the full-query evaluation. Values are rounded for display; source reports retain full precision. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 20. Appendix: SQuAD sample quality metrics

These are the complete indexed quality values and mean times for the same evaluated query set. Recall retained uses mean Recall@100 as the denominator. The SQuAD sample is not the full-query evaluation. Values are rounded for display; source reports retain full precision. Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.

## 21. Appendix: primary references

Primary sources verified for this release: https://arxiv.org/html/2404.07220v2
https://arxiv.org/abs/2109.10739
https://arxiv.org/abs/2407.03618
https://arxiv.org/abs/2104.08663
https://arxiv.org/abs/1606.05250
https://arxiv.org/abs/2107.05720
https://arxiv.org/abs/1908.10084. The main manuscript provides full bibliographic details. Sources for each measured claim appear in the corresponding speaker notes and the shared results JSON.

## 22. Appendix: rebuilding the report

To rebuild figures, run experiment 32 from the repository root using the documented analysis requirements. To rebuild the release figures, paper and HTML, use the publication tools and requirements. The editable presentation builder uses the bundled Codex Artifact Tool runtime. Headless browser rendering creates the showcase PDF. Exact commands and required environment variables appear in docs/submission/README.md. No new Kaggle run is necessary to read or regenerate current result plots. Full record-level rescoring needs original matching result archives separately.
