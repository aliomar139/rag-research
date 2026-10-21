> **Historical document.** This file predates the consolidated submission and may contain superseded claims or commands. Use [the current manuscript](../../FINAL_MANUSCRIPT.md) for conclusions.

# Agent Handoff: Cascaded Adaptive Blended RAG

**Prepared:** 2026-10-19  
**Purpose:** Give a new agent enough project and conversation context to continue without repeating the Kaggle setup or overstating results.

## Read this first

The user is trying to turn the RAG project into a sound, reproducible implementation. Keep the work focused on the routing method, retrieval behavior, measurement, and evidence. Paper claims can change; implementation quality is the priority.

The user has ADHD and asked for the `i-have-adhd` response style. Put the next action first, use short numbered steps when needed, state where the work stands, give concrete estimates when possible, and end with one actionable instruction. They prefer direct Python scripts over notebooks for running experiments. When asking them to upload a Kaggle dataset, always give the exact dataset name expected by the code and make sure it cannot be confused with an old upload. Tell them plainly when a GPU is actually needed. Keep them updated while working; they explicitly asked to see progress instead of being left waiting.

## Current project locations

The IDE has been pointed at `C:\Users\user\Desktop\rag-research-alone`. The writable workspace for this agent is `C:\Users\user\Desktop\M2-Porjects\rag-research`. The latter has been the staging repo; selected scripts, results, and notes have been copied into `rag-research-alone` with explicit permission. The user wants to continue in `rag-research-alone`, so treat that as the active project and sync any required deliverables there.

The active IDE tab shown in the latest messages is `cascaded-blended-rag/results/threshold_frontier_scifact.json`.

## Project in one paragraph

The project studies a cascaded retriever for RAG: begin with BM25, estimate whether the query needs further retrieval, and conditionally call Dense and SPLADE. Static Reciprocal Rank Fusion (RRF) is the quality baseline. The intended contribution is an adaptive retrieval path that can skip neural retrieval work while keeping retrieval quality near the static system. Earlier runs showed why live conditional execution and matched measurement matter: simulated or qrel-filtered candidate pools cannot establish real savings. The corrected SciFact work fixed several protocol issues but has not yet demonstrated a safe speedup.

## Conversation history that affects the next work

1. The user first asked for a review of the whole project, including Markdown and DOCX material, and a research paper. Later they asked for a project-focused literature review based on the downloaded papers, then shifted priority to implementation improvement.
2. The user asked to preserve attention to the reproduction notebook because it records a Kaggle run. During Kaggle setup, the attached source dataset appeared under nested paths such as `/kaggle/input/datasets/<owner>/cascade-source-bundle/cascaded-blended-rag/`; code that searched only for a top-level ZIP failed. A recursive file discovery approach was needed.
3. The user explicitly rejected notebook-only instructions and asked for plain Python code to run. They later cleared old cells to avoid rerunning them in the wrong order.
4. An early SciFact attempt ran with GPU disabled. The user enabled a Tesla T4, but Dense and SPLADE caches were absent then, so those cells had to run again. SPLADE document encoding took about 55 minutes on CPU; the Numba warning after that was not itself a hang. Do not casually tell the user to repeat corpus encoding. Check whether the exact corpus/model/order cache is present first.
5. The user eventually produced `cascaded-blended-rag/results/cascade_heldout_scifact.json`, `cascaded-blended-rag/results/threshold_frontier_scifact.json`, and `cascaded-blended-rag/results/learned_cascaded-blended-rag/results/cascade_heldout_scifact.json`. These artifacts are in the active repo root. The latest additions were the threshold frontier and learned-router result.
6. The user asked to focus on implementation and contribution, not fixed paper claims. They also asked whether old datasets should be deleted. Advice given: keep old uploads unless the code names a different new dataset explicitly; avoid ambiguous duplicate attachments.
7. A compact router model comparison was run on the 809 SciFact training examples. The script and result were copied into `rag-research-alone`. The user agreed not to run Kaggle until told.
8. The assistant said it would prepare an evaluation on a fresh split. Inspection then showed a validity problem: every one of the 809 SciFact training examples had already appeared in nested CV/model comparison, and the original 300-query SciFact test set had already been inspected. Do not call a random split of those rows an untouched final holdout. The TREC-COVID fast loader was also checked and rejected for a clean external evaluation because it selects relevant corpus documents from qrels.
9. The user asked whether they should just wait, then asked for clear current observations and next steps, and finally asked for this Markdown handoff. Keep working autonomously and keep sending brief progress messages.
10. After the handoff, the assistant prepared official BEIR NFCorpus as an external transfer test, based on BEIR's published listing of a 3.6K-document corpus with official train/dev/test splits. The user uploaded the source bundle under the new dataset name and ran the evaluator. Kaggle initially lacked `ranx`; installing `ranx>=0.3.20` fixed the import, and the user downloaded the result.

## Verified results

### Corrected threshold-gate evaluation (experiment 08)

Artifact: `cascaded-blended-rag/results/cascade_heldout_scifact.json`. This run uses the full SciFact corpus of 5,183 documents, 809 train queries for threshold selection, and 300 test queries. It applies Best Fields BM25 consistently and compares the static system with the cascade on the same test queries. The evaluation ran on a Tesla T4.

- Static three-way RRF: Recall@100 **0.9610**, nDCG@10 **0.6984**, MAP@10 **0.6538**, mean latency **53.63 ms**.
- The predeclared training constraint was 98% of static training Recall@100. Static train recall was 0.95838, making the target about 0.9392. None of the 90 threshold candidates met it.
- The runner used static fallback on test: 0% exits, no Dense/SPLADE calls skipped, mean latency 53.69 ms.
- This corrected run supports the conclusion that the tested threshold grid did not meet its quality constraint. It does not show a successful acceleration.

The training-only fixed threshold frontier is in `cascaded-blended-rag/results/threshold_frontier_scifact.json`. Its candidates reached Recall@100 0.9111–0.9286, with exit rates 78.74–96.42%; the highest-recall point still missed the target by about 0.0106.

### First learned-router held-out attempt (experiment 09)

Artifact: `cascaded-blended-rag/results/learned_cascaded-blended-rag/results/cascade_heldout_scifact.json`. A 300-tree Random Forest estimated per-query recall gain from escalation; the threshold came from five-fold OOF predictions over the same 809 train examples.

| System | Recall@100 | nDCG@10 | MAP@10 | Mean latency | Test exits |
|:--|--:|--:|--:|--:|--:|
| Static RRF | 0.9610 | 0.6984 | 0.6538 | 56.01 ms | 0% |
| Learned gate | 0.9346 | 0.6694 | 0.6200 | 61.48 ms | 70.67% (212/300) |

Within this run, the cascade was 5.47 ms slower on average and retained about 97.25% of static Recall@100. It skipped 212 Dense and 212 SPLADE calls, but the recall drop makes that an unsuccessful operating point. The route-time field averaged about 21.04 ms. One pass only; the latency is hardware-specific. The static baseline is 56.01 ms in this script, whereas experiment 08 reports 53.63 ms, so compare systems within each run rather than treating these timings as identical.

Training OOF calibration estimated 76.76% early exits and roughly 98.42% of static recall, but test retention fell to 97.25%. This gap is an important calibration/generalization failure.

### Compact-model training comparison (experiment 10)

Files: `cascaded-blended-rag/experiments/10_compare_router_models.py` and `cascaded-blended-rag/results/compact_router_cv_scifact.json`. This CPU-only nested CV uses all 809 SciFact training examples, inner-fold threshold calibration, and a 99% pooled recall-ratio target. It does not read test qrels.

| Model | Pooled outer-fold exits | Pooled Recall@100 ratio | Local one-row prediction |
|:--|--:|--:|--:|
| Ridge | 49.20% | 99.36% | 0.144 ms |
| Depth-2 decision tree | 49.69% | 99.22% | 0.060 ms |
| 32-tree ExtraTrees | 55.25% | 99.49% | 1.188 ms |
| 300-tree Random Forest reference | 60.94% | 99.32% | 9.779 ms |

Some outer folds fell below 99%, so pooled results do not promise the target on unseen queries. Ridge is a simple low-cost candidate; ExtraTrees-32 exits more often at a modestly higher prediction cost. Neither has been evaluated as an end-to-end final policy on a clean test set. Do not pick one solely because its pooled score is the highest.

### Local timing profile

A quick local microbenchmark on a representative query and synthetic top-10 BM25 score list measured feature construction at roughly **0.0055 ms**, Random Forest prediction at **9.56 ms**, Ridge at **0.14 ms**, and a depth-2 tree at **0.056 ms**. This isolates the Python feature function and model call; it is not a full query benchmark and does not reproduce Kaggle/T4 timing. It suggests the old model prediction is a large part of route overhead, while feature construction itself is cheap. The original 21 ms route time includes both and cannot be attributed to feature construction alone.

## Dataset and protocol cautions

- Do not reuse the SciFact test set to tune a new threshold or select a new router. Its metrics have already been inspected.
- Do not describe a newly sampled subset of the 809 training queries as a clean untouched test: all rows took part in the nested model comparison and selection.
- `trec_covid_fast_loader.py` does not build a qrel-blind candidate corpus. It reads qrels, selects up to 25 highly relevant documents per query, and adds corpus records to a 5,000-document pool. That pool is not a clean corpus-level external test.
- The repository has local TREC-COVID data and files under `data/processed_trec_covid`, but cache metadata/document ordering must be validated before reuse. Presence of `.npy` or `.npz` files alone does not prove a match. Existing loader also selects the evaluation pool using qrels.
- The earlier NQ cascade is exploratory: qrels informed the candidate pool, all neural rankings were computed before routing, and latency was simulated. It cannot support live skipped-call claims.
- User wants to know exactly when GPU use is necessary. Dense/SPLADE query retrieval can use GPU; corpus encoding can be reused only when fingerprints, model names, and document ordering match.

## Current code and file layout

In `rag-research-alone`:

- Root artifacts: `cascaded-blended-rag/results/cascade_heldout_scifact.json`, `cascaded-blended-rag/results/learned_cascaded-blended-rag/results/cascade_heldout_scifact.json`, `cascaded-blended-rag/results/threshold_frontier_scifact.json`, and `cascaded-blended-rag/results/router_training_scifact.jsonl`.
- Experiment source: `cascaded-blended-rag/experiments/08_evaluate_cascade_heldout.py`, `09_evaluate_learned_router.py`, and `10_compare_router_models.py`.
- Compact CV result: `cascaded-blended-rag/results/compact_router_cv_scifact.json`.
- Notes: `docs/history/results_ledger.md`, `docs/history/plans/improvement_plan.md`, and `.local/PROJECT_ANCHOR.md`.
- Existing source bundle/notebooks are under `notebooks/`; the user prefers direct Python instructions for the next run.

**Path mismatch to resolve before rerunning experiment 10:** the active repo root contains `cascaded-blended-rag/results/router_training_scifact.jsonl`, while experiment 10 expects it at `cascaded-blended-rag/results/router_training_scifact.jsonl`. Experiment 09 has the same `results/` expectation. The prior model comparison output exists, but the script may fail from the current alone-repo layout unless the file is copied into the expected location or the script is changed to accept/find the root artifact. The Kaggle source bundle may have had a separate `results/` copy; inspect before assuming.

## Cross-dataset transfer result

Artifact: `cascaded-blended-rag/results/ridge_transfer_nfcorpus.json` (the user's downloaded copy was `rag-research-alone/cascaded-blended-rag/results/ridge_transfer_nfcorpus.json`). Script: `cascaded-blended-rag/experiments/11_evaluate_ridge_transfer_nfcorpus.py`. The Ridge model and threshold came from SciFact train examples only; the script used the full official NFCorpus corpus and test split, then scored relevance grades after routing. It performed two warmed timing passes on a Tesla T4. Dataset details are in the [official BEIR dataset list](https://github.com/beir-cellar/beir/wiki/Datasets-available).

| System | Recall@100 | nDCG@10 | MAP@10 | Mean latency | p50 | p95 | p99 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Static RRF | 0.3147 | 0.3452 | 0.1304 | 25.04 ms | 23.44 ms | 35.02 ms | 38.54 ms |
| Frozen Ridge cascade | 0.2681 | 0.3203 | 0.1235 | 13.56 ms | 8.89 ms | 33.23 ms | 38.05 ms |

It exited on 220/323 queries (68.11%) and skipped one Dense and one SPLADE call per early-exit query in each pass. Mean latency fell 45.8%; Recall@100 retained 85.2% of static RRF, a 0.0466 absolute drop. p95 improved 5.1% and p99 1.3%. This is a speed/quality trade-off and a negative cross-dataset transfer result, not a quality-preserving efficiency result. The test set is now inspected; don't retune and reuse it as a final test.

The source archive was `artifacts/source_bundles/cascade_source_bundle_nfcorpus_ridge_v1.zip`; Kaggle dataset name was `cascade-source-bundle-nfcorpus-ridge-v1`. The code initially failed because Kaggle lacked `ranx`; `pip install ranx>=0.3.20` resolved it. The actual script ran with GPU enabled and wrote `cascaded-blended-rag/results/ridge_transfer_nfcorpus.json`.

## Next work for the agent

1. Keep the NFCorpus test output as a fixed diagnostic; do not tune its threshold or use it again as a held-out evaluation.
2. Use NFCorpus train/dev or another development source to examine feature shift and design a better calibration strategy, clearly separating any development metrics from final test results.
3. Freeze the revised method, then select another untouched evaluation collection and validate its corpus/split/cache protocol before the next Kaggle request.
4. Update project claims only after the next evaluation supports them; the present transfer result does not support quality preservation.

## User-facing coordination

The user asked for status messages while the agent works. Send a short commentary update before longer tool work and after a meaningful finding; don't go silent. The NFCorpus transfer evaluator has run and the user downloaded the result. Current conclusion: the compact gate saves mean time but loses too much Recall@100 on transfer. No Kaggle action is pending until a new protocol is ready.

## Honest current conclusion

The project has a credible problem and working retrieval components, and its live evaluations now expose clear implementation limits. The frozen Ridge gate transferred to NFCorpus and reduced mean latency, but it lost 14.8% of static Recall@100. The next milestone is to diagnose cross-domain calibration on development data, freeze a revised policy, and test it on a different untouched collection.
