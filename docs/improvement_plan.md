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

## Kaggle GPU checkpoint

The corrected first evaluation path is now prepared for SciFact. **A Kaggle GPU run is needed now** to execute the neural retrievers and obtain the first held-out measurement. This run uses the complete SciFact corpus, selects thresholds using train qrels, and evaluates on separate test queries. It does not correct the earlier NQ corpus protocol; those notebook results remain exploratory until a full-corpus NQ rerun is designed and executed. The first SciFact latency report is a measured pass, and should be treated as an initial hardware-specific estimate; repeated timing runs and uncertainty analysis remain follow-up improvements.

Run `notebooks/heldout_scifact_kaggle.ipynb` with `notebooks/cascade_source_bundle.zip` attached as a Kaggle dataset input. Enable Internet and select a GPU accelerator (a single T4 is sufficient). Download the resulting `cascade_heldout_scifact.json` and return it here before we revise quantitative claims in the paper and project reports.

## Current project position

Treat the existing NQ cascade sweep as exploratory: on the notebook’s qrels-filtered pool, the conservative setting reports 73.2% early exits, hit@20 91.8%, nDCG@10 0.6115, and 25.3 ms calculated from fixed component costs; static RRF reports hit@20 97.2% and nDCG@10 0.7094. Treat SciFact CV as a cached-results simulation until the BM25 variant and matched-fold baseline are corrected. Do not claim preserved answer quality or measured live latency from these runs.

The held-out SciFact runner and Kaggle handoff are prepared, but no corrected result exists until that notebook is executed. The existing 07 experiment remains a historical artifact; 08 is the intended next evaluation entry point.

