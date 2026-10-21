# Results Ledger — Existing Runs

This ledger preserves existing project results while separating execution evidence from evaluation validity. “Executed” means code ran and produced the recorded output; it does not by itself mean the protocol is unbiased or the metric is a live end-to-end measurement.

## Kaggle Natural Questions notebook

Source: [executed reproduction notebook](../../notebooks/legacy/reproduction.ipynb), cells 1–3. The notebook loads 500 BEIR NQ queries and reports the executed outputs below.

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

## Corrected held-out SciFact run

Source: [`cascaded-blended-rag/results/cascade_heldout_scifact.json`](../../cascaded-blended-rag/results/cascade_heldout_scifact.json), produced by `experiments/08_evaluate_cascade_heldout.py` on Kaggle with a Tesla T4. It uses all 5,183 SciFact corpus documents, 809 train queries for threshold selection, and 300 held-out test queries. It uses Best Fields BM25 consistently and compares static three-way RRF with the cascade on the same test queries. This is the first live conditional-path evaluation in the project.

| System | Recall@100 | nDCG@10 | MAP@10 | Mean latency | P50 | P95 | P99 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Static three-way RRF | 0.9610 | 0.6984 | 0.6538 | 53.63 ms | 51.21 ms | 82.38 ms | 89.35 ms |
| Cascade (static fallback) | 0.9610 | 0.6984 | 0.6538 | 53.69 ms | 51.21 ms | 80.02 ms | 90.39 ms |

| Single retriever | Recall@100 | nDCG@10 | MAP@10 |
|:--|--:|--:|--:|
| Best Fields BM25 | 0.8706 | 0.6369 | 0.5908 |
| Dense MiniLM | 0.9250 | 0.6451 | 0.5959 |
| SPLADE | 0.9320 | 0.7024 | 0.6590 |

**Selection outcome:** the predeclared training constraint was 98% of static training Recall@100. Static training recall was 0.95838, so the target was approximately 0.93920. None of the tested thresholds met it. The runner therefore selected no threshold and used static RRF for all test queries. The test cascade had 0% early exits, skipped 0 Dense calls and 0 SPLADE calls, and invoked each neural retriever 300 times. Its latency is effectively equal to static RRF; this run demonstrates no retrieval-cost reduction from the current gate under the stated quality constraint.

**Interpretation and limits:** the protocol is held out and matched, and latency is measured on the Kaggle T4 with CUDA synchronization. It is a single timing pass, so treat percentiles and mean as initial hardware-specific measurements. Train-only frontier (`cascaded-blended-rag/results/threshold_frontier_scifact.json`): all 90 candidates were evaluated on the 809 training queries. None meets the 0.93922 Recall@100 target. Training Recall@100 ranges from 0.9111 to 0.9286; exit rates range from 78.74% to 96.42%. The best-recall point is 0.9286 at 79.48% exits, still 0.0106 below target. Candidate latency is estimated on training data and is not evidence of held-out savings. The result is evidence that the tested gate/grid did not meet its objective, not evidence that no possible gate can. the protocol is held out and matched, and latency is measured on the Kaggle T4 with CUDA synchronization. It is a single timing pass, so treat percentiles and mean as initial hardware-specific measurements. The output does not include the per-threshold training quality/exit table, so it cannot show the best achievable quality–cost point outside the 98% constraint. The result is evidence that the tested gate/grid did not meet its objective, not evidence that no possible gate can.

## Train-only threshold frontier

Train-only frontier (`cascaded-blended-rag/results/threshold_frontier_scifact.json`): all 90 candidates were evaluated on the 809 training queries. None meets the 0.93922 Recall@100 target. Training Recall@100 ranges from 0.9111 to 0.9286; exit rates range from 78.74% to 96.42%. The best-recall point is 0.9286 at 79.48% exits, still 0.0106 below target. Candidate latency is estimated on training data and is not evidence of held-out savings.

**Next:** inspect feature calibration and add stronger routing baselines, then select any revised policy on development data only. Evaluate the revised policy on untouched queries with repeated latency measurements. A GPU is needed only for fresh Dense or SPLADE rankings; reuse Kaggle caches if available., add a train-only threshold diagnostic that records each grid point's Recall@100, exit rate, and estimated latency. Plot the training quality–exit frontier and state the trade-off transparently. If no useful point exists, redesign or recalibrate the gate, then evaluate the revised rule on the untouched SciFact test queries or a newly reserved test split. A GPU is needed for fresh neural retrieval passes; the SciFact corpus encoding caches may be reused only if the same Kaggle session and files are preserved.



## Learned escalation-risk router: first held-out attempt

Source: `results/cascaded-blended-rag/results/learned_cascaded-blended-rag/results/cascade_heldout_scifact.json`, produced on a Tesla T4 using a Random Forest trained on 809 train-only examples. The routing threshold was calibrated from five-fold out-of-fold training predictions.

| System | Recall@100 | nDCG@10 | MAP@10 | Mean latency | Early exits |
|:--|--:|--:|--:|--:|--:|
| Static three-way RRF | 0.9610 | 0.6984 | 0.6538 | 56.01 ms | 0% |
| Learned gate + static escalation | 0.9346 | 0.6694 | 0.6200 | 61.48 ms | 70.67% |

The learned gate skipped 212 Dense and 212 SPLADE test-query invocations, but test Recall@100 retained only 97.25% of static RRF and mean latency was 5.47 ms slower. Average router computation was about 21.04 ms per query, which erased the savings from skipped neural work. Training OOF calibration predicted 0.9842 of static recall with a 76.76% exit rate; the held-out result did not reproduce that quality level. This is an implementation diagnostic, not a successful efficiency result.

**Engineering follow-up:** profile feature extraction and model inference separately; compare much smaller/cheaper predictors and calibrate with a larger safety margin using training data only. After choosing an implementation, evaluate it on a fresh untouched split. Do not tune a new threshold against this already inspected test result.

## Compact router comparison (train-only nested CV)

`results/compact_router_cv_scifact.json` compares compact predictors using nested cross-validation over the 809 SciFact training queries. Each outer fold's threshold was selected only from its inner training folds to target 99% of static BM25+Dense+SPLADE RRF recall. The original held-out test qrels were not used. Values below are pooled outer-fold results; inference timings are local one-row model prediction only, not end-to-end Kaggle latency.

| Predictor | Early exits | Recall@100 / static | Local prediction |
|:--|--:|--:|--:|
| Depth-2 decision tree | 49.69% | 99.22% | 0.060 ms |
| Ridge | 49.20% | 99.36% | 0.144 ms |
| 32-tree ExtraTrees | 55.25% | 99.49% | 1.188 ms |
| 300-tree Random Forest reference | 60.94% | 99.32% | 9.779 ms |

Fold results vary, and some outer folds fall below the 99% target. Treat this as a development comparison, not a guarantee. A local microbenchmark measured feature construction at about 0.006 ms, the 300-tree predictor at about 9.6 ms, Ridge at about 0.14 ms, and a depth-2 tree at about 0.06 ms per query. These CPU-only figures are not directly comparable to the prior Kaggle route timings; the recorded ~21 ms route time combines feature construction and prediction. All 809 SciFact training queries have already participated in nested CV/model comparison, and the original 300-query test set has already been inspected, so neither is a new untouched final test. The local TREC-COVID fast loader is also unsuitable as an independent test corpus because it selects relevant documents using qrels. A valid next evaluation needs an independent dataset with an unfiltered corpus and a predeclared protocol. That retrieval evaluation needs Kaggle GPU access for Dense/SPLADE unless matching validated caches can be reused.

## Cross-dataset transfer evaluation: SciFact-trained Ridge on NFCorpus

Artifact: `results/cascaded-blended-rag/results/ridge_transfer_nfcorpus.json`; script: `experiments/11_evaluate_ridge_transfer_nfcorpus.py`. It freezes a `StandardScaler + Ridge(alpha=10)` router and threshold using SciFact train examples, then evaluates on the official BEIR NFCorpus test split. Train-only calibration gave threshold 0.06379, 55.25% OOF exits, and 99.36% pooled OOF recall relative to static RRF. The evaluator used the full 3,633-document corpus, read relevance grades after routing, validated the dataset archive MD5, used a corpus-specific cache directory, and measured two warmed timing passes on a Tesla T4.

| System | Recall@100 | nDCG@10 | MAP@10 | Mean latency | p50 | p95 | p99 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Static RRF | 0.3147 | 0.3452 | 0.1304 | 25.04 ms | 23.44 ms | 35.02 ms | 38.54 ms |
| Frozen Ridge cascade | 0.2681 | 0.3203 | 0.1235 | 13.56 ms | 8.89 ms | 33.23 ms | 38.05 ms |

The router exited on 220/323 queries (68.11%) and skipped 220 Dense plus 220 SPLADE calls per measured pass. Mean latency fell 45.8%, while Recall@100 fell 0.0466 absolute, retaining 85.2% of static RRF. nDCG retained 92.8%; MAP retained 94.8%. p95 latency improved only 5.1%, and p99 1.3%. The cascade is materially faster on average but fails the near-static-recall goal under cross-dataset transfer. Treat this as a negative transfer/quality trade-off result, not a successful quality-preserving speedup. The test set is now inspected; do not tune this result and present it as held out again.

BEIR lists NFCorpus as a 3.6K-document corpus with train/dev/test splits ([official BEIR dataset list](https://github.com/beir-cellar/beir/wiki/Datasets-available)). The source archive is `cascade_source_bundle_nfcorpus_ridge_v1.zip`; its Kaggle dataset name was `cascade-source-bundle-nfcorpus-ridge-v1`. The script downloads NFCorpus itself and writes the result to `results/cascaded-blended-rag/results/ridge_transfer_nfcorpus.json`.

## NFCorpus train calibration and dev check

Artifact: `cascaded-blended-rag/results/nfcorpus_train_calibrated_dev.json` (also preserved at the workspace root). Script: `experiments/12_calibrate_nfcorpus_train_check_dev.py`. The Ridge gate and threshold use 2,590 official train queries; the check uses 324 official dev queries and the full 3,633-document corpus on a Tesla T4. The result records no test access. Five-fold train OOF calibration selected threshold 0.0629951 with 6.45% exits and 99.11% relative recall. Threshold selection uses all train labels, so those OOF figures are development calibration figures rather than an independent held-out estimate.

| System | Recall@100 | nDCG@10 | MAP@10 | Mean latency | p95 latency | Dev exits |
|:--|--:|--:|--:|--:|--:|--:|
| Static RRF | 0.30337 | 0.31872 | 0.10129 | 25.34 ms | 35.28 ms | 0 |
| Train-calibrated Ridge | 0.30116 | 0.31568 | 0.09984 | 25.22 ms | 36.20 ms | 13/324 |

Dev recall retains 99.27% of static RRF. Only 4.01% of queries exit; mean latency falls 0.46%, and p95 becomes 2.62% slower. Treat this as negligible observed acceleration; the result contains no uncertainty estimate establishing a timing benefit. Dev is now inspected. The next diagnostic uses train only to compare the router's exits with label-informed oracle exits, exporting per-query data for further development without another dev/test pass.

## NFCorpus train oracle diagnostic and CPU model comparison

Artifacts: workspace-root `artifacts/downloads/nfcorpus_train_diagnostics_outputs.zip`, plus `results/nfcorpus_train_router_diagnostics.json` and `results/nfcorpus_train_router_comparison.json`. The diagnostic export contains 2,590 train labels, BM25/static rankings, features, and OOF routing predictions. Record counts, policy summaries, and collector/retriever source fingerprints were checked locally.

Ridge's train-calibrated OOF policy exits on 167 queries (6.45%). The label-informed oracle exits on 1,421 (54.86%) under the identical clipped loss budget; a second oracle crediting BM25 wins exits on 1,595 (61.58%). BM25 matches or exceeds static recall on 1,123 train queries; Ridge escalates 1,041 of them. These are retrospective label-informed opportunities, not achievable router performance or latency claims.

Four fixed model specifications were then compared locally with five outer and four inner folds (seed 17), using the same 11 features and clipped loss target. Inner OOF predictions select thresholds; outer folds score the fixed policies. All queries come from train. CPU thread pools were limited to one for the local timing check.

| Predictor | Nested exits | Recall / static | Folds meeting 99% | Local one-row prediction |
|:--|--:|--:|--:|--:|
| Ridge alpha 10 | 4.32% | 99.42% | 4/5 | 0.143 ms |
| Depth-4 tree, leaf minimum 25 | 2.82% | 99.70% | 5/5 | 0.060 ms |
| ExtraTrees-64, depth 8, leaf minimum 10 | 7.10% | 98.99% | 3/5 | 2.328 ms |
| Histogram gradient boosting, 100 iterations, 7 leaves | 4.02% | 99.50% | 4/5 | 0.517 ms |

The tested model changes retain low exit rates; ExtraTrees narrowly misses the pooled target and adds more local inference overhead. This motivates testing richer cheap features, rather than claiming that the oracle's exits are attainable or that no model can recover them. These comparisons remain development results; selecting a model from them does not create an independent final evaluation. Next, use the hash-matched corpus and saved labels to add BM25/document coverage features on CPU before another retrieval evaluation.

## NFCorpus BM25 coverage feature extension (train only)

Script `experiments/15_compare_nfcorpus_coverage_features.py` uses the hash-matched corpus and saved train labels. It adds 58 features to the original 11: normalized score gaps/concentration, query-token and IDF-weighted coverage, content-word coverage, title/abstract agreement, and document lengths. No static ranking or relevance label is read by feature extraction. All 2,590 recomputed BM25 top-100 document lists matched the saved labels' BM25 rankings exactly. Model specifications, outer/inner folds, target, and seed match the original-feature comparison.

Artifact: `results/nfcorpus_train_coverage_comparison.json`. Per-query features are preserved in `results/nfcorpus_train_coverage_features.jsonl` with a corpus/source/input manifest.

| Predictor | Enriched exits | Recall / static | Folds meeting 99% | Local one-row prediction |
|:--|--:|--:|--:|--:|
| Ridge | 6.87% | 99.33% | 4/5 | 0.157 ms |
| Depth-4 tree | 3.51% | 99.39% | 3/5 | 0.058 ms |
| ExtraTrees-64 | 7.57% | 98.95% | 4/5 | 2.332 ms |
| Histogram gradient boosting | 5.64% | 99.32% | 4/5 | 0.521 ms |

Extra feature computation averaged 0.468 ms locally, with p95 0.727 ms; these CPU figures are not end-to-end Kaggle timings. Ridge's exit rate improves by 2.55 percentage points, while one outer fold remains below target (98.94% relative recall). No enriched candidate meets the target in all outer folds. The evidence does not justify a claim of useful acceleration or a quality guarantee. Twelve synthetic/local tests passed in addition to the full-ranking checks. The next train-only hypothesis is to change the risk objective from mean clipped loss to harm-probability/cost-aware scoring before another neural retrieval evaluation.

## Requested retention budgets: 90%, 92%, 95%, 99% (train-only preparation)

The user has accepted a broader relative-recall/latency trade-off; the risk-objective experiment is deferred. Script `16_prepare_retention_tradeoff.py` reuses the same saved train labels and the same five outer/four inner folds (seed 17). Each fold fits its Ridge gate once, then calibrates each requested threshold on inner OOF predictions. The portable final models use all train labels, with thresholds from five-fold OOF predictions (seed 42). No dev/test relevance labels are used.

| Target | 11-feature exits | 11-feature retention | Folds meeting target | 69-feature exits | 69-feature retention | Folds meeting target |
|:--|--:|--:|--:|--:|--:|--:|
| 90% | 42.93% | 90.26% | 3/5 | 42.97% | 90.89% | 3/5 |
| 92% | 36.10% | 92.19% | 2/5 | 36.25% | 93.08% | 4/5 |
| 95% | 24.05% | 95.43% | 3/5 | 25.79% | 95.69% | 3/5 |
| 99% | 4.32% | 99.42% | 4/5 | 6.87% | 99.33% | 4/5 |

Artifacts: `results/nfcorpus_retention_train_curve.json`, `results/nfcorpus_frozen_retention_routers.json`, and per-query train predictions. These are development estimates; threshold targets are not guarantees, and exits are not latency savings. The benchmark uses a shared stable vectorized BM25 implementation for all systems, with rankings verified identical to the saved original BM25 rankings. This removes a first-stage implementation confound from comparisons between feature families.

The prepared Kaggle benchmark measures the four baselines plus eight frozen policies on previously inspected dev, with real conditional execution, warmup, three randomized repeats, branch counters, quality metrics, timing samples, and query-bootstrap intervals. It also provides offline random-exit recall references at identical exit counts. The completed measured outcomes are recorded below. Independent untouched confirmation and explicit comparison against standalone neural baselines remain necessary for final empirical claims.

## NFCorpus four-target measured dev benchmark (completed)

Artifact: workspace-root `artifacts/downloads/nfcorpus_retention_benchmark_outputs.zip`; extracted files in `results/retention_tradeoff_dev/`. It covers all 324 dev queries and 3,633 documents on Tesla T4, with three randomized warmed repeats. All 3,888 first-pass records, 11,664 query timings, branch counts, frozen weights, and source fingerprint were validated. No ranking changed between repeats. Source dataset title: `nfcorpus-retention-tradeoff-20261020-v1.2`.

| Method | Recall@100 | Retention | Mean latency | Mean latency reduction | Exits |
|:--|--:|--:|--:|--:|--:|
| Static RRF | 0.30337 | 100% | 23.61 ms | 0% | 0% |
| Original gate r90 | 0.28010 | 92.33% | 17.07 ms | 27.70% | 39.20% |
| Original gate r92 | 0.28742 | 94.74% | 18.18 ms | 23.00% | 32.10% |
| Original gate r95 | 0.29192 | 96.23% | 19.67 ms | 16.70% | 23.77% |
| Original gate r99 | 0.30116 | 99.27% | 23.14 ms | 2.00% | 4.01% |
| Dense alone | 0.30266 | 99.77% | 6.62 ms | 71.98% | N/A |
| SPLADE alone | 0.29654 | 97.75% | 10.26 ms | 56.55% | N/A |
| BM25 alone | 0.20472 | 67.48% | 6.44 ms | 72.74% | N/A |

Original r92's paired query-bootstrap recall interval is 92.66-96.92%, with latency reduction 19.33-26.71%. Original r90's recall interval (89.47-95.04%) crosses 90%, so the observed point is not a high-confidence 90% floor. Tail savings are limited: r92 p95 is 32.79 ms versus static's 33.97 ms. Intervals resample queries with repeat-averaged times and do not represent all session/hardware variation.

The original gate preserves more recall than random exits at equal exit count: r92 retains 94.74% versus the random assignments' mean 89.54%. At r90 it skips 381 Dense and 381 SPLADE calls across repeats. Random-policy latency was not measured. The 69-feature r90/r92/r95/r99 policies retain 94.00/94.86/96.49/98.75% at 17.64/18.79/20.33/23.24 ms; its r99 misses the nominal target and its timing interval includes no improvement.

**Baseline implication:** Dense has higher Recall@100 and lower mean latency than every cascade point. SPLADE has higher nDCG@10 (0.33164) and MAP@10 (0.10852) than static RRF and all cascades; it also has better recall and latency than the r90/r92/r95 cascade points. These results demonstrate a controlled trade-off against full fusion and better routing than random assignment, but do not establish superiority over strong single retrievers or generated-answer quality.

Detailed interpretation is in `docs/findings/NFCORPUS_BENCHMARK_FINDINGS.md`. For a fixed confirmation, the original r92 gate is the development-selected primary candidate, and all original-gate r90/r92/r95/r99 thresholds remain byte-identical. The predeclared SCIDOCS run uses full corpus/all queries, strong baselines, and self-match exclusion; no SCIDOCS relevance grades have been inspected during this preparation. After confirmation, finalize claims with all outcomes rather than pursuing further favorable retuning.


## 2026-10-20: frozen SCIDOCS confirmation completed

Downloaded `artifacts/downloads/scidocs_confirmation_outputs.zip`; extracted to
`cascaded-blended-rag/results/scidocs_confirmation/`. Local script 19 validates
8,000 first-pass records, 24,000 timings, frozen assets and benchmark-source
hashes, exits and branch counts. Raw qrels are absent from the result ZIP;
local checks reaggregate saved per-query recall rather than recomputing grades.

Full 25,657-document corpus, 1,000 queries, Tesla T4, three warmed randomized
repeats. The original NFCorpus-trained Ridge gate and all four thresholds
remain unchanged. SCIDOCS grades are not used to select policies.

| Policy target | Observed retention | Mean ms | Early exits | Target met |
|:--|--:|--:|--:|:--|
| 90% | 88.67% | 149.43 | 39.90% | No |
| 92% primary | 90.01% | 150.69 | 35.00% | No |
| 95% | 92.10% | 152.87 | 27.10% | No |
| 99% | 97.44% | 157.22 | 11.00% | No |

Static fusion: recall 0.46523, mean 159.51 ms. Primary mean savings 5.53%;
retention interval 88.27?91.61%. BM25 mean 132.51 ms limits savings in this
implementation. Dense: recall 0.51012, nDCG 0.21640, MAP 0.12937, mean 10.14 ms
(15.73x fusion speed). Dense exceeds fusion and all cascades in these quality
metrics and latency. The final sequence does not establish a superior cascade
or transferable retention guarantees. No retuning on SCIDOCS is queued.

Current summary: `docs/FINAL_RESULTS.md`. Current paper:
`docs/FINAL_MANUSCRIPT.md`. Earlier pending-confirmation plans are completed.


## 2026-10-20: original-paper dataset extension prepared (not executed)

User requested evaluation on the datasets used by Blended RAG. Experiment 20
supports full BEIR TREC-COVID, NQ and HotpotQA and an explicitly declared
SQuAD 1.1 paragraph-retrieval conversion (all train+dev paragraphs, all dev
questions). Original weights and all thresholds remain frozen. The primary
policy for this new extension is nominal 90%, selected before its outcomes;
this does not change the earlier SCIDOCS primary policy or results.

Upload bundle: `artifacts/source_bundles/kaggle_paper_datasets_v1/paper_datasets_frozen_source_20261020_a7c4_v1.zip`.
Unique new Kaggle title: `paper-datasets-frozen-r90-20261020-a7c4-v1`.
The first run is TREC-COVID, followed by a separate SQuAD run using the same
bundle. NQ/HotpotQA require explicit large-corpus opt-in and a suitably
provisioned machine; full execution within Kaggle resources is not claimed.
No corpus/query reductions are made. Syntax, ZIP CRCs, archive paths and
manifest hashes checked locally; no new retrieval benchmark executed locally.
The frozen protocol is `cascaded-blended-rag/results/paper_datasets_protocol.json`.

Eight methods and three warmed randomized timing repeats; all targets and
standalone baselines reported. Checkpoints can resume completed query
repetitions while working files remain; timing sessions are recorded.
The adapted system uses MiniLM/SPLADE/RRF, not exact Elasticsearch/ELSER
reproduction. Success against fusion at a 90% point estimate alone does not
establish major novelty, a confidence floor or superiority over single methods.
Keep all previous unfavorable findings visible in the final report.


## 2026-10-20: TREC-COVID extension measured and imported

Received `artifacts/downloads/trec_covid_paper_benchmark_outputs.zip` (693,171 bytes), archive
SHA256 `769e02d83978b0d3f116e48fcc99ea93f52633a0990f85f245f582ab8014a3df`.
Full corpus 171,332; all 50 official test queries; three warmed randomized T4
passes in one timing session. Frozen weights, source and extension protocol
match. Script 22 validates 400 records, 1,200 timings, saved recall aggregation,
counts and exact conditional ranking equality with BM25 or static fallback.
Raw qrels and independent timing instrumentation are not verified locally.

Targets 90/92/95/99 retain 84.53/84.53/91.95/96.68% and save
3.56/3.30/2.43/1.17% mean retrieval time. All miss their nominal target.
Primary r90 fails the 90% floor. Secondary r95 exceeds a 90% point floor,
but its retention interval is 86.88?95.76%; mean-time savings are small.
r99 has retention lower bound 93.58% but saves only 1.17% mean time.

Static mean 1493.23 ms, recall 0.12250, nDCG 0.76323, MAP 0.01861.
BM25 mean 1410.14 ms, limiting skip savings in this implementation.
SPLADE mean 52.98 ms, recall 0.12825 (104.69% retention), nDCG 0.72899,
MAP 0.01662, speedup 28.19x. SPLADE beats fusion and every cascade in observed
recall and latency, but does not beat fusion or r95/r99 on every metric.
Its recall interval relative to fusion crosses 100%. No major acceleration
claim is established. All previous findings remain reported.

Detailed result: `docs/findings/TREC_COVID_FINDINGS.md`. Next is full declared SQuAD
paragraph evaluation using the same bundle and `DATASET = "squad"`.


## 2026-10-21: indexed BM25 implemented; CPU execution benchmark completed

While the original SQuAD Kaggle job runs, implemented `src/indexed_bm25.py`
without changing the frozen sources or policies. It precomputes the existing
rank_bm25 field contributions and uses stable top-k selection. This is a
known engineering approach, not a claimed new BM25 algorithm.

Experiment 24 measured three paired, warmed, randomized CPU repetitions.
Full TREC-COVID corpus SHA matches the Kaggle corpus; all 50 queries:
old scorer 908.70 ms, indexed 6.74 ms, 134.85x. Full NFCorpus corpus SHA
matches training; all 2,590 train queries: 4.50 ms versus 0.13 ms, 34.59x.
All complete document score vectors and top100 IDs/order match exactly on
150 TREC and 7,770 NFCorpus query repetitions; maximum score difference zero.
Reports: `results/indexed_bm25_cpu_trec-covid.json` and
`results/indexed_bm25_cpu_nfcorpus.json` under `cascaded-blended-rag/`.
Local one-thread Intel CPU, Python 3.12.10, NumPy 2.5.4; these timings must
not be combined with old Kaggle stage times into a claimed measured pipeline.
No neural models or GPU cascade benchmark were run locally.

Prepared experiment 25 and source package via experiment 26. The short
follow-up uses the completed original Kaggle run's existing neural indexes,
full corpus, 256 questions chosen by fixed SHA256, all eight methods and
three timing repeats. Both static and cascaded retrieval get indexed BM25.
The launcher refuses missing/mismatched corpus caches before document encoding.
Previous BM25 top100 scores and IDs are compared after timing; old per-query
rankings are not served during timed inference. Sample metrics are not
full-query metrics. End-to-end indexed measurements remain pending.

For this NEW execution follow-up only, primary is the unchanged conservative
`ridge11_r99` threshold, assessed against the user's 90% retention floor.
This selection uses already inspected NFCorpus/SCIDOCS/TREC evidence and
precedes inspection of indexed results or the currently running full SQuAD
outcomes. All four targets remain reported, including original target misses.
Prior studies keep their original primaries. No refit or threshold adjustment.
Protocol: `results/indexed_cascade_protocol.json`.

Bundle: `artifacts/source_bundles/kaggle_indexed_cascade_v1/indexed_cascade_source_20261021_b8e6_v1.zip`.
New title: `indexed-cascade-followup-20261021-b8e6-v1`.
Run only after downloading the original SQuAD ZIP and keep that session's
working files. Detailed findings and scope: `docs/findings/INDEXED_BM25_FINDINGS.md`.

Training-data audit: current train bundles contain BM25/static recall labels
and original features, but no Dense/SPLADE per-query train labels. A new
query-only Dense/SPLADE selector therefore needs additional supervised data;
its saved TREC oracle illustration is neither trained nor independently
validated. Prior work already studies strategy selection (arXiv:2109.10739)
and eager sparse scoring (arXiv:2407.03618). Defer selector training until
matched indexed execution establishes whether a larger redesign is needed.


## 2026-10-21: checkpoint-only runtime shortcut prepared (not yet executed)

User reports original SQuAD timing at repeat 2/3, query 3600/10570. First-pass
rankings therefore cover every query; remaining execution repeats timing.
Prepared experiment 27 to stop only a lingering matching original subprocess,
read the full first pass, preserve additional records, and reuse the exact
hash-verified original scorer/export code without neural inference. Output
is separate: `artifacts/downloads/squad_one_pass_outputs.zip`. The original protocol is preserved;
an execution amendment explicitly declares one scored pass instead of three,
before grades are scored. Partial later records are supplemental, not unevenly
pooled into headline means. Timing confidence is correspondingly limited.

Combined source package: `artifacts/source_bundles/kaggle_fast_finish_v1/fast_finish_indexed_20261021_d4a9_v1.zip`.
New Kaggle title: `fast-finish-indexed-20261021-d4a9-v1`.
Default ACTION is `finalize_original`; after download, `indexed_followup` uses
the same source package and existing corpus caches for the short indexed run.
Experiment 22 now recognizes and validates this explicit runtime amendment.
Syntax/package checks completed. The live user checkpoint has not been read
or executed locally; no result or completed amendment is claimed yet.


## 2026-10-21: SQuAD one-pass export received and independently scored

Received `artifacts/downloads/squad_one_pass_outputs.zip`, 357,768,875 bytes; archive SHA256
`12f926600e0da66729f3778b5dd8632bcc19a7220e25f0adf79f615211a39eac`.
Full 20,958-document title/paragraph corpus and all 10,570 dev questions.
One full T4 timing pass retained; 4,649 second-pass query repetitions preserved
as supplemental data. Runtime amendment is explicit, before quality scoring.
Original checkpoint remains preserved. No changed ranking IDs observed in
those additional records. Do not claim completion of three timing passes.

Experiment 22 validates 84,560 first-pass retrieval records and timings,
source/weight/protocol/finalizer fingerprints, query coverage, scores, branch
counts, conditional ranking equality and all additional-record counts.
Experiment 29 reconstructs exact qrels from local raw SQuAD dev 1.1 data;
SHA256 matches Kaggle. Independent parent-label recall/nDCG/MAP matches every
method within 1e-10. Saved under `results/paper_squad_one_pass/`.

Nominal 90/92/95/99 settings retain 99.38/99.50/99.70/99.95%, saving
6.12/4.98/3.29/0.91% mean time. Every nominal target is met. Primary r90
recall interval 99.22?99.54%; time-saving interval 5.93?6.31%. Query bootstrap
from one complete timing pass does not establish repeat-run stability.

Static recall 0.99243, nDCG 0.82897, MAP 0.79284, mean 148.26 ms.
SPLADE recall 0.98600 (99.35% retention), nDCG 0.85367, MAP 0.82425,
mean 17.24 ms (8.60x fusion speed). r90 recall is slightly higher than SPLADE,
but nDCG/MAP are lower and time much larger. BM25 alone retains 97.11% recall
at 119.45 ms; it already meets a 90% floor. Score-preserving indexing therefore
must also be compared with this cheap constant baseline. Dense retains 96.12%
at 10.72 ms, with lower ranking metrics. No superior cascade claim follows.

Next: same combined Kaggle cell, `ACTION = "indexed_followup"`; no new upload,
full corpus, fixed-hash 256 questions, all eight methods, three timing passes,
existing neural indexes. Conservative r99 primary remains as selected before
SQuAD outcome inspection; all settings reported. Detailed result:
`docs/findings/SQUAD_FINDINGS.md`. Indexed GPU result is still pending.

## 2026-10-21: indexed SQuAD sample received and verified

Received `artifacts/downloads/squad_indexed_cascade_outputs.zip`, 9,976,117 bytes, SHA256
`e1a07593ce7d16eb7660b95c975f8d241a925918e0a567942792d2101dfe0bfd`.
Full 20,958-document corpus, 256 of 10,570 questions selected by the declared
SHA256 rule, all eight methods, three warmed randomized T4 timing passes,
one session. Preparation 9.43 seconds, excluded from query timings. No
threshold, model or feature changes. Equal indexed BM25 for fusion and gates.

Preselected primary r99: 100% sampled recall retained, 28.31 ms, 4.19% mean
saving (query-bootstrap interval 1.90% to 7.04%), 14/256 exits. Secondary r90:
99.22% retained (98.05% to 100%), 18.59 ms, 37.08% saving (31.58% to 42.69%),
102 exits. Secondary r92/r95 retain 100% at 21.31/24.82 ms, saving
27.89/15.99%. All nominal targets pass. Primary remains r99. Sample zero-loss
intervals are not population guarantees; r90 p95 does not improve.

Indexed fusion 29.55 ms. BM25 alone retains 95.70% at 1.03 ms, already
meeting 90%, and is fastest. SPLADE retains 98.83% at 16.96 ms and leads
nDCG/MAP. Thus mean savings against equally optimized fusion are real, but
best-method status for a 90% floor and algorithmic novelty are unestablished.

Experiment 22 imports indexed archives and validates all 2,048 retrieval
records, 6,144 timings, frozen fingerprints, coverage, branch identities and
counts. Experiment 29 accepts a result directory and independently reproduces
all metrics using exact raw-label qrels. New experiment 30 verifies declared
hash selection and exact old/new equality of all methods' ordered IDs,
scores, routes and recalls on these questions; maximum score difference zero.
Saved under `results/indexed_squad_q256/`; combined CSV labels the sample.

Same-question original/indexed means: fusion 147.01/29.55 ms; BM25
117.55/1.03 ms; Dense 10.83/10.82 ms; SPLADE 17.16/16.96 ms. Old one pass
versus new three passes, across runs rather than an interleaved backend
experiment. Sample timings do not establish full-query indexed gains.
All original full-query evidence is preserved. Detailed findings and scope:
`docs/findings/SQUAD_INDEXED_FINDINGS.md`.

Next user action: paste `notebooks/kaggle_fast_finish_v1/CHECK_TREC_CACHE.py`
into the current Kaggle notebook and send output. Read-only path check,
seconds, no upload, no model run. Indexed TREC timing is pending; do not
assume its old corpus caches remain available or start re-encoding blindly.

## 2026-10-21: TREC corpus cache files confirmed; indexed run cell prepared

User reports `TREC_CACHE_PRESENT` at
`/kaggle/working/paper_datasets_frozen_v1/data/processed_paper_trec-covid`.
This confirms file presence, not independently validated cache contents.
Prepared `notebooks/kaggle_fast_finish_v1/RUN_TREC_INDEXED.py` for the same
Kaggle session and existing source package. No replacement upload or dataset
title. It invokes unchanged experiment 25 on all 50 queries and 171,332
documents, eight methods and three timing passes, with required existing
neural caches. Runner validates fingerprints before neural initialization
and refuses missing/mismatched caches; document encoding is not a fallback.
Previous BM25 rankings are compared after timing if their JSONL is present.

Frozen r99 remains the indexed study's primary against the 90% floor; all
four thresholds are reported. No new tuning or new-generalization claim.
Output directory is separate: `results/indexed_trec-covid_all50_v1/` in the
combined Kaggle project. Expected export:
`artifacts/downloads/trec_covid_indexed_cascade_outputs.zip`. This is a prepared invocation, not
a received result. No retrieval benchmark has been run locally in this step.

## 2026-10-21: full-query indexed TREC received and independently verified

Received `artifacts/downloads/trec_covid_indexed_cascade_outputs.zip`, 697,499 bytes, SHA256
`92857af16ed9944a90494c79ddabef3286cff3f0f1c49a274ba2f1dfbd882529`.
All 50 official queries and 171,332 documents, eight methods, three warmed
randomized passes in one T4 session with one CPU thread. Existing corpus
representations reused; preparation 62.91 seconds, excluded from query times.
No model, weight, feature, threshold or frozen source change.

Indexed fusion 91.28 ms. Preselected r99 primary: 96.68% recall retained,
73.90 ms, 19.04% mean saving; retention interval 93.58% to 98.75%, time-saving
interval 9.47% to 30.11%. Both lower bounds pass the indexed follow-up's
90%-floor/positive-saving criteria. Nominal 99% still fails; all nominal TREC
targets still fail. 11/50 exits avoid 33 invocations per neural branch across
three passes. Primary p95 99.41 ms versus fusion 97.66 ms, so no p95 gain.

Secondary r90/r92: 84.53% retained, 55.74/54.00% mean saving, failing 90%.
Secondary r95: 91.95% retained at 55.66 ms, 39.02% saving; retention interval
86.88% to 95.76%, crossing 90%. It remains secondary. This engineering
follow-up uses already-inspected quality and does not restore an independent
generalization status to TREC or change the original extension's r90 primary.

Standalone BM25 71.38% retained at 8.11 ms; Dense 69.81% at 30.41 ms; SPLADE
104.69% at 52.77 ms. SPLADE beats the primary in recall/nDCG and mean time.
Primary MAP is higher: 0.01769 versus 0.01662, a descriptive 6.43% relative
difference without a significance claim. Do not claim all-metric SPLADE
dominance or change the original recall/time objective after these results.
Primary retained fractions of fusion nDCG/MAP are 94.70%/95.09%.

Experiment 22 validates 400 records, 1,200 timings, all frozen hashes, coverage,
conditional rankings and calls. Experiment 30 checks all-query coverage and
exact old/new ranking IDs, order, scores, routes and recall for every method;
maximum score difference zero. Old first-pass reference SHA256
`ad6605c44b698cb5e4a734934e86ec3d739765642158930d06bcfba47d449d38`.
New experiment 31 independently scores exact local raw qrels (SHA256
`10669ab7d526cb04f52079139fd88c3d467a0776441b046567f540582798982b`)
and reproduces all per-query recall and all aggregate quality values for
original and indexed TREC. No retrieval, training or GPU timing ran locally.

Evidence under `results/indexed_trec-covid_q0/`; q0 means all queries.
The combined CSV now distinguishes all-query indexed TREC from sampled
indexed SQuAD. Cross-run old/indexed fusion means 1493.23/91.28 ms (16.36x),
BM25 1410.14/8.11 ms (173.78x); neural means are nearly unchanged. Both runs
have three passes but are not interleaved backend timings. The fair routing
comparison is 19.04% against equally indexed fusion, not those larger ratios.

Next: write up the completed engineering results using
`docs/findings/TREC_COVID_INDEXED_FINDINGS.md`, `docs/FINAL_RESULTS.md` and the updated
manuscript/claim audit. No further Kaggle run or upload is required to report
these findings. Novel algorithm, best-method and original ELSER acceleration
claims remain unsupported; full NQ/HotpotQA remain resource-dependent pending
work rather than silently completed experiments.

## 2026-10-21: consolidated manuscript and full speedup progression exported

User requested a growing presentation including every speedup through the
highest, rather than only 19.04% mean savings. Consolidated FINAL_MANUSCRIPT
into one report with current abstract/conclusion, method, protocol, all main
results and prior-work limits. Previous manuscript preserved under
`docs/history/FINAL_MANUSCRIPT_before_consolidation_20261021.md`.

New experiment 32 reads completed reports to export PNG/SVG figures, CSV
tables and source fingerprints under `docs/figures/`. It performs no inference,
training, scoring or timing. `docs/SPEEDUP_SUMMARY.md` contains every gate,
both comparison references and the complete standalone baseline tables.

TREC configuration progression versus original fusion: 1.00x original,
16.36x indexed fusion, 20.21x r99, 26.83x r95, 35.56x r92, 36.96x r90.
Corresponding indexed-cascade gains over equally indexed fusion are
1.24x/1.64x/2.17x/2.26x, with retained recall
96.68%/91.95%/84.53%/84.53%. Largest points remain visible with failed 90%
floor; r95's interval crosses below 90%; primary remains r99.

Same-query SQuAD sample progression versus original fusion:
4.98x indexed fusion, 5.19x r99, 5.92x r95, 6.90x r92, 7.91x r90.
Indexed cascade gains 1.04x/1.19x/1.39x/1.59x with retained recall
100%/100%/100%/99.22%. Original sample timing comes from its one complete
pass; indexed timing uses three. These ratios are separate from the old
full-10,570-query means. No scaling-to-larger-corpus claim.

Additional standalone speeds are listed separately, including BM25's
184.02x over original TREC fusion at 71.38% fusion recall. The BM25 component
ratio over its own original scorer is 173.78x, with identical rankings.
Neither number becomes a 90%-quality cascade contribution.

Primary sources checked for the consolidated report: Blended RAG
arXiv:2404.07220v2 (models and dataset relationship), strategy selection
arXiv:2109.10739, BM25S arXiv:2407.03618, BEIR arXiv:2104.08663.
The claim audit includes the revised outline and answered five-part review.
The finish checklist requests report-format/author details and review; no
additional Kaggle run or upload is needed for the completed scope.

## 2026-10-21: repository organized for submission and sharing

Kept numbered experiment/core source paths to preserve recorded fingerprints.
Grouped current findings, review documents, historical drafts/plans, literature
catalogs and legacy notebooks under indexed folders. The migration CSV records
original paths and hashes before Markdown navigation updates. Four root exports
were removed only after equality with the canonical result copy was verified;
generated Python caches were removed.

Original download ZIPs, source bundles, binary report exports, raw datasets and
bulk per-query evidence remain local and ignored. Nine old data/archive/working-
note entries were removed from Git tracking without deleting local data. The
original archive inventory is in `docs/reproducibility/artifact_catalog.json`.

Added repository navigation/size/secret-pattern/syntax/fingerprint checks, a
manifested sharing ZIP builder, dependency groups and explicit input-restoration
helpers. This preparation performs no new model inference, training, timing or
unit-test execution. Current scientific claims and measured reports are preserved.
See `CHANGELOG.md` and `docs/reproducibility/README.md` for the current entry points.

## 2026-10-21: final research communication release

The completed study now has an 11-page academic paper, offline HTML showcase,
matching 9-page showcase PDF, and an editable 22-slide presentation with
speaker notes and a presentation PDF. The narrative follows the original
paper, motivation, reproduction/adaptation boundary, metric differences,
results and implementation. A preserved pre-release manuscript remains in
history. The current release review records claim support and self-review.

Every cascade setting and both timing references remain visible, alongside
all standalone speedups and their quality. Full current NQ, HotpotQA, CoQA
and generated-answer EM/F1 remain deferred. No retrieval, neural encoding,
training or new timing run was performed to create the material release.

Publication sources and exact rebuilding commands are in
`docs/submission/README.md`. The HTML embeds licensed fonts and chart assets;
the PPTX contains editable tables/charts and speaker notes. The presentation
PDF preserves inspected slide renders. Frozen benchmark source and policy
fingerprints remain unchanged.
