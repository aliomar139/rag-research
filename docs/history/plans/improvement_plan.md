> **Historical document.** This file predates the consolidated submission and may contain superseded claims or commands. Use [the current manuscript](../../FINAL_MANUSCRIPT.md) for conclusions.

> **Superseded on 20 October 2026.** This document preserves earlier project context.
> The completed experiment sequence and next steps are in [FINAL_RESULTS.md](../../FINAL_RESULTS.md);
> the current manuscript is [FINAL_MANUSCRIPT.md](../../FINAL_MANUSCRIPT.md).
> Earlier requests for more Kaggle runs or tuning below are historical.

# Improvement Plan: Cascaded Blended RAG

## Starting point

Start with evaluation validity and provenance. The executed Kaggle notebook is important evidence and should be preserved, but its cascade cells currently simulate routing after computing all three retrievers for every query. The reported NQ stage latencies are hard-coded constants. The notebook also constructs its 15,119-document pool by including all documents judged relevant to the selected 500 evaluation queries. These choices make the current results useful for exploration, but not a valid measurement of live neural-call savings or a clean full-corpus held-out result.

The SciFact experiment has related issues: the CV script routes from `bm25_single_cache.json`, although the project describes Best Fields BM25; the static baseline is aggregated across all queries while the cascade is averaged over held-out folds; and latency is reconstructed from cached component timings. These need to be reconciled before revising paper claims.

## Work order

### Phase 1 — Freeze and label the existing evidence

- Preserve the executed notebook and its outputs as the original exploratory run.
- Record its exact query sample, candidate-pool construction, thresholds, retrieval metrics, and latency assumptions.
- Separate measured wall-clock execution from arithmetic latency estimates throughout project documents.
- Reconcile each number against its source artifact; remove unsupported numbers and claims.

**Completion condition:** One results ledger identifies every reported value, its source, protocol, and limitations.

### Phase 2 — Correct the evaluation protocol

- Remove query-label-dependent corpus construction. Evaluate against the full declared corpus, or define a corpus slice independently of evaluation qrels.
- Separate threshold development from final evaluation. Use validation folds for threshold selection and keep a held-out test set untouched; avoid sweeping thresholds and reporting the same queries as the final result.
- Decide whether the intended first stage is single-field BM25 or Best Fields BM25, then use that same variant in threshold tuning and evaluation.
- On SciFact, compute the static baseline on the exact same held-out queries as the cascade in every fold.
- Save fold-level metrics and per-query routing decisions, rankings, and timings.

**Completion condition:** A deterministic protocol with no qrel-informed candidate pool, no threshold selection on the final test set, and matched baselines.

### Phase 3 — Implement and time a real conditional cascade

- Wire BM25, the router, and the conditional dense/SPLADE fallback into one query-time path.
- Verify in instrumentation that early-exit queries do not call dense or SPLADE.
- Measure actual per-query elapsed time using warmups, repeated runs, and correct GPU synchronization; report p50, p95, and p99 as well as the mean.
- Record branch invocations, GPU time, and device details. Keep estimated costs clearly separate from observed latency.
- Fingerprint embedding and sparse-index caches by model, corpus, and document ordering.

**Completion condition:** The cascade is executed online and its skipped work and latency are directly observed.

### Phase 4 — Evaluate quality and operating trade-offs

- Compare BM25, dense, SPLADE, static RRF, and cascade on identical query/corpus splits. Include SPLADE as a strong single-retriever baseline because it exceeded RRF on the saved NQ retrieval metrics.
- Report hit/recall at useful cutoffs, nDCG, and a latency–quality curve. State the quality constraint before threshold tuning.
- Add per-query or bootstrap uncertainty and fold-level results.
- Evaluate downstream QA with the same generator, prompts, and context budget for static and cascaded retrieval; keep the existing SQuAD generator result separate unless it is rerun as a matched comparison.
- Test threshold transfer between at least one development collection and an untouched target collection.

**Completion condition:** The main claims are supported by matched retrieval and, if claiming RAG quality, matched QA results.

### Phase 5 — Consolidate the project narrative

After the evaluation is stable, rewrite the project-facing material from the results ledger: project-focused literature review, research paper, full report, README, project anchor, and oral-defense guide. Regenerate the DOCX/PDF versions from the approved source and verify they agree. Keep the original exploratory Kaggle notebook available and add a clean final reproduction notebook/script for the corrected experiments.

**Completion condition:** All documents use the same method description, metrics, threshold provenance, and limitations.

## Updated result and next work

The first corrected SciFact run is complete and recorded in `cascaded-blended-rag/results/cascade_heldout_scifact.json`. It uses the full 5,183-document corpus, 809 train queries for threshold selection, and 300 held-out test queries. On the Tesla T4, static RRF reaches Recall@100 0.9610, nDCG@10 0.6984, and MAP@10 0.6538 at 53.63 ms mean measured latency. The 98%-of-static-training-recall constraint was not met by any tested gate. The runner correctly used static fallback: 0% test exits, 0 neural calls skipped, and 53.69 ms mean latency. Thus, the current gate has not demonstrated useful acceleration under the declared quality target.

### NFCorpus train/dev recalibration (completed)

- Use `notebooks/kaggle_nfcorpus_train_dev_v2/START_HERE.md` as the single Kaggle run guide.
- Upload only `artifacts/source_bundles/kaggle_nfcorpus_train_dev_v2/cascade_source_bundle_nfcorpus_train_dev_v1.zip`.
- Kaggle dataset slug: `nfcorpus-train-calibrated-dev-check-20261019-v2`. Keep the older `cascade-source-bundle-nfcorpus-ridge-v1` dataset unchanged.
- The bundle contains the train-calibration/dev-check script. It selects the router and threshold on NFCorpus `train`, evaluates on `dev`, and does not read `test.tsv`.
- Completed result: `cascaded-blended-rag/results/nfcorpus_train_calibrated_dev.json` records 2,590 train queries, 324 dev queries, and the full 3,633-document corpus. Dev Recall@100 retains 99.27% of static RRF (0.30116 vs. 0.30337); only 13 queries exit (4.01%). Mean latency falls 0.46% (25.22 vs. 25.34 ms), while p95 rises 2.62%. This is a conservative quality result with negligible observed acceleration. Dev is now inspected and is not an untouched final test.

### Train-only router diagnostics (completed)

- Guide and ready-to-paste cell: `notebooks/kaggle_nfcorpus_train_diagnostics_v1/START_HERE.md` and `RUN_IN_KAGGLE.py`.
- Upload `nfcorpus_train_diagnostics_source_v1.zip` from that folder using dataset slug `nfcorpus-train-router-diagnostics-20261020-v1`.
- Script `experiments/13_diagnose_nfcorpus_train_router.py` saves resumable per-query train labels, features, OOF predictions, and routing decisions. It compares Ridge exits with retrospective oracle exits using the same 99% recall budget, with a second oracle allowing BM25 wins to offset losses.
- This diagnostic opens train qrels only. It reuses existing validated full-corpus indexes when available. Download `artifacts/downloads/nfcorpus_train_diagnostics_outputs.zip` afterward; saved examples support later router work without repeated retrieval.
- Completed archive: `artifacts/downloads/nfcorpus_train_diagnostics_outputs.zip` at the workspace root. All 2,590 per-query records and policy metrics were validated against the report and source fingerprints. Ridge exits on 167 queries (6.45%); the same-budget retrospective oracle exits on 1,421 (54.86%). Ridge escalates 1,041 queries where BM25 matches or beats static recall. The oracle is an unattainable label-informed upper bound, not a speedup measurement.
- CPU-only nested CV on the cached labels is recorded in `results/nfcorpus_train_router_comparison.json` using `experiments/14_compare_nfcorpus_train_routers.py`. Ridge: 4.32% exits, 99.42% retained recall, 4/5 folds meeting the target. Depth-4 tree: 2.82%, 99.70%, 5/5 folds. ExtraTrees-64: 7.10%, 98.99%, 3/5 folds. Histogram gradient boosting: 4.02%, 99.50%, 4/5 folds. These four tested model changes do not close the oracle gap with the current 11 features.
- Completed: the hash-matched corpus is now at workspace-root `artifacts/downloads/nfcorpus_corpus_for_features.jsonl`. Script `experiments/15_compare_nfcorpus_coverage_features.py` adds 58 BM25/document coverage features and checks all 2,590 saved BM25 top-100 rankings exactly. The matched nested comparison is saved in `results/nfcorpus_train_coverage_comparison.json`.
- With 69 total features, Ridge exits rise from 4.32% to 6.87% with 99.33% pooled relative recall (4/5 folds meeting the target). Tree exits are 3.51% (99.39%, 3/5 folds), ExtraTrees 7.57% (98.95%, 4/5 folds), and histogram boosting 5.64% (99.32%, 4/5 folds). Local extra feature computation averages 0.468 ms. This does not establish useful acceleration or quality stability across all folds.
- Current handoff: `docs/history/plans/NFCORPUS_CURRENT_STATUS.md`. The 69-feature change alone showed modest benefit. The user has now requested a broader 90/92/95/99% retention comparison; defer risk-objective/model tuning and use the next benchmark below. Twelve feature/diagnostic tests passed; dev/test were not opened during feature development.

### Four retention targets (completed)

- `experiments/16_prepare_retention_tradeoff.py` has completed locally on cached train labels. Both 11- and 69-feature Ridge gates are frozen at 90%, 92%, 95%, and 99% targets. No dev labels selected these thresholds.
- The original gate's nested-CV exits are 42.93%, 36.10%, 24.05%, and 4.32%, with pooled retention 90.26%, 92.19%, 95.43%, and 99.42%. The 69-feature exits are 42.97%, 36.25%, 25.79%, and 6.87%, with retention 90.89%, 93.08%, 95.69%, and 99.33%. Individual folds sometimes miss their target.
- Upload the source ZIP in `notebooks/kaggle_retention_tradeoff_v1/` under the unique Kaggle title `nfcorpus-retention-tradeoff-20261020-v1`, then use that folder's `RUN_IN_KAGGLE.py`.
- Script `17_benchmark_retention_tradeoff.py` measures three warmed, randomized repetitions of four baselines and eight frozen policies on already-inspected NFCorpus dev. It records real branch counts and query-time latency, all requested quality metrics, paired query-bootstrap intervals, first-pass rankings, a CSV and plots, plus an offline count-matched random-exit recall reference.
- Download `artifacts/downloads/nfcorpus_retention_benchmark_outputs.zip`. Judge contribution against strong single retrievers and random skipping as well as static RRF. Preserve target misses and uncertainty. If a useful point exists, freeze it for an independent untouched confirmation before final claims; do not relabel the dev curve as untouched evidence.

- Completed archive at the workspace root; extracted records in `results/retention_tradeoff_dev/`. All 3,888 rankings/decisions, 11,664 timing samples, branch counts, source hash and frozen weights were validated. Original-gate observed retention is 92.33/94.74/96.23/99.27%, with mean latency 17.07/18.18/19.67/23.14 ms. Static RRF is 23.61 ms. Target r92 saves 23.00% mean latency with a 19.33-26.71% paired query-bootstrap interval.
- Critical baseline: Dense alone retains 99.77% of static recall at 6.62 ms (3.57x static speed). It is faster and has higher Recall@100 than every cascade point. SPLADE has stronger nDCG/MAP and beats the 90/92/95% cascade points on all measured quality metrics and latency. The cascade can claim savings versus full fusion, not general superiority over single retrievers. See `docs/findings/NFCORPUS_BENCHMARK_FINDINGS.md`.

### Final frozen confirmation (prepared; no SCIDOCS grades inspected)

- Upload `artifacts/source_bundles/kaggle_scidocs_confirmation_v1/scidocs_frozen_confirmation_source_v1.zip` under Kaggle title `scidocs-frozen-cascade-confirmation-20261020-v1`; use the adjacent `RUN_IN_KAGGLE.py`.
- Run the full official SCIDOCS corpus and all 1,000 test queries with the checksum-verified BEIR archive. Primary candidate: original 11-feature r92; report original-gate r90/r92/r95/r99 plus all four single/static baselines. The frozen weights and thresholds are identical to the NF run. The 69-feature ablation is not selected for transfer.
- The protocol fixes three timing repeats and disallows refitting or recalibration on SCIDOCS. It includes BEIR-style query/document self-ID exclusion before routing/fusion, with 101 candidates requested to retain 100 after exclusion. Query IDs select the split; grade values score results only after all routing/timing is complete.
- Download `artifacts/downloads/scidocs_confirmation_outputs.zip`; then consolidate both datasets, including target misses and simpler-baseline wins, into the paper/report. An unfavorable result must be reported rather than tuned and presented as untouched again.

### Next implementation step

1. Completed: `cascaded-blended-rag/results/threshold_frontier_scifact.json` records all 90 fixed-threshold candidates. None meets the 0.93922 train Recall@100 target.
2. Completed: `cascaded-blended-rag/results/router_training_scifact.jsonl` contains 809 train-only examples with BM25 features and per-query recall gain from Dense+SPLADE escalation.
3. Completed first learned-router attempt: `results/cascaded-blended-rag/results/learned_cascaded-blended-rag/results/cascade_heldout_scifact.json` shows 70.67% exits but lower recall (0.9346 vs 0.9610) and slower latency (61.48 vs 56.01 ms). The Random Forest router alone averaged about 21 ms per query.
4. Completed: `results/compact_router_cv_scifact.json` compares shallow trees, Ridge, ExtraTrees, and the 300-tree reference with nested train-only CV and a 99% recall target. Ridge predicts in 0.14 ms locally with 49.20% exits and 99.36% pooled relative recall; ExtraTrees-32 gives 55.25% exits and 99.49% relative recall at 1.19 ms. Fold results vary, and neither result is a held-out guarantee.
5. Completed: the frozen SciFact-trained Ridge gate was evaluated on the official full NFCorpus test corpus. It exited on 68.11% of queries and cut mean latency 45.8%, but Recall@100 retained only 85.2% of static RRF (0.2681 vs. 0.3147); p95/p99 latency improved only 5.1%/1.3%. This is not a quality-preserving result. The NFCorpus test set is now inspected, so do not tune on it. Next, use a development split to diagnose or recalibrate the router, freeze the revised protocol, and evaluate on another untouched collection. Keep the current JSON as a transfer-failure case study.

## Current project position

Treat the existing NQ cascade sweep as exploratory: its pool uses qrels from the selected evaluation queries, its neural branches are precomputed before routing, and its latency is calculated from fixed component costs. Treat the SciFact cross-validation output as legacy exploratory evidence because it used a single-field BM25 cache and unmatched aggregate baselines. The new held-out SciFact run corrects those protocol issues but finds no qualifying gate in its tested threshold grid. Do not claim measured neural-call savings or answer-quality preservation for the current method.

