# Final results

This folder contains the materials prepared for review and presentation.
The figures, tables, and PDFs use the saved project reports; opening them does
not require a Kaggle session or model downloads.

## Materials

| File | What it contains |
|:--|:--|
| [Research paper](conditional_retrieval_paper.pdf) | 9-page report with method, evaluation, limitations and references |
| [HTML showcase](research_showcase.html) | Ten-section presentation; use the on-screen arrows or Left/Right keys. “Read full page” switches to continuous reading. |
| [Presentation PDF](research_presentation.pdf) | 23-slide presentation at 3840 × 2160 slide render resolution |
| [Showcase PDF](research_showcase.pdf) | Ten-page printable version of the HTML showcase |
| [Editable presentation](research_presentation.pptx) | PowerPoint deck with editable text, tables, charts and speaker notes |
| [Speaker notes](presentation_notes.md) | Talk track and evidence references for the presentation |
| [Release manifest](RELEASE_MANIFEST.json) | File hashes and the scope of the released evidence |

Authors: Ali Omar, Abdallah Mestrah and Abbas Yazbek. Supervisor: Dr. Ahmad
Abboud. Lebanese University.

## Experiment code

The numbered scripts are in [`../cascaded-blended-rag/experiments/`](../cascaded-blended-rag/experiments/).
They preserve the experiment numbering used by the reports and handoff notes.

| # | Script | Purpose |
|--:|:--|:--|
| 01 | [01_baseline_bm25.py](../cascaded-blended-rag/experiments/01_baseline_bm25.py) | BM25 baseline |
| 02 | [02_baseline_dense.py](../cascaded-blended-rag/experiments/02_baseline_dense.py) | Dense MiniLM baseline |
| 03 | [03_baseline_splade.py](../cascaded-blended-rag/experiments/03_baseline_splade.py) | SPLADE baseline |
| 04 | [04_baseline_blended_rag.py](../cascaded-blended-rag/experiments/04_baseline_blended_rag.py) | Blended retrieval baseline |
| 05 | [05_replicate_squad.py](../cascaded-blended-rag/experiments/05_replicate_squad.py) | Initial SQuAD reproduction |
| 06 | [06_replicate_trec_covid.py](../cascaded-blended-rag/experiments/06_replicate_trec_covid.py) | Initial TREC-COVID reproduction |
| 07 | [07_tune_cascade_cv.py](../cascaded-blended-rag/experiments/07_tune_cascade_cv.py) | Cross-validated cascade tuning |
| 08 | [08_evaluate_cascade_heldout.py](../cascaded-blended-rag/experiments/08_evaluate_cascade_heldout.py) | Held-out cascade evaluation |
| 09 | [09_evaluate_learned_router.py](../cascaded-blended-rag/experiments/09_evaluate_learned_router.py) | Learned router evaluation |
| 10 | [10_compare_router_models.py](../cascaded-blended-rag/experiments/10_compare_router_models.py) | Router model comparison |
| 11 | [11_evaluate_ridge_transfer_nfcorpus.py](../cascaded-blended-rag/experiments/11_evaluate_ridge_transfer_nfcorpus.py) | Ridge transfer evaluation on NFCorpus |
| 12 | [12_calibrate_nfcorpus_train_check_dev.py](../cascaded-blended-rag/experiments/12_calibrate_nfcorpus_train_check_dev.py) | Router calibration and development check |
| 13 | [13_diagnose_nfcorpus_train_router.py](../cascaded-blended-rag/experiments/13_diagnose_nfcorpus_train_router.py) | NFCorpus training diagnostics |
| 14 | [14_compare_nfcorpus_train_routers.py](../cascaded-blended-rag/experiments/14_compare_nfcorpus_train_routers.py) | NFCorpus router comparison |
| 15 | [15_compare_nfcorpus_coverage_features.py](../cascaded-blended-rag/experiments/15_compare_nfcorpus_coverage_features.py) | Coverage feature comparison |
| 16 | [16_prepare_retention_tradeoff.py](../cascaded-blended-rag/experiments/16_prepare_retention_tradeoff.py) | Prepare recall-retention policies |
| 17 | [17_benchmark_retention_tradeoff.py](../cascaded-blended-rag/experiments/17_benchmark_retention_tradeoff.py) | Benchmark retention policies |
| 18 | [18_confirm_scidocs_frozen.py](../cascaded-blended-rag/experiments/18_confirm_scidocs_frozen.py) | Frozen SCIDOCS confirmation |
| 19 | [19_finalize_confirmation.py](../cascaded-blended-rag/experiments/19_finalize_confirmation.py) | Finalize confirmation results |
| 20 | [20_benchmark_paper_datasets.py](../cascaded-blended-rag/experiments/20_benchmark_paper_datasets.py) | Evaluate the paper-dataset extensions |
| 21 | [21_package_paper_datasets.py](../cascaded-blended-rag/experiments/21_package_paper_datasets.py) | Package the paper-dataset run |
| 22 | [22_import_paper_benchmark.py](../cascaded-blended-rag/experiments/22_import_paper_benchmark.py) | Import saved paper-dataset results |
| 23 | [23_analyze_direct_routing_headroom.py](../cascaded-blended-rag/experiments/23_analyze_direct_routing_headroom.py) | Analyze direct-routing headroom |
| 24 | [24_benchmark_indexed_bm25_cpu.py](../cascaded-blended-rag/experiments/24_benchmark_indexed_bm25_cpu.py) | Check indexed BM25 scoring on CPU |
| 25 | [25_benchmark_indexed_cascade.py](../cascaded-blended-rag/experiments/25_benchmark_indexed_cascade.py) | Run the indexed cascade benchmark |
| 26 | [26_package_indexed_cascade.py](../cascaded-blended-rag/experiments/26_package_indexed_cascade.py) | Package the indexed cascade run |
| 27 | [27_finalize_first_completed_pass.py](../cascaded-blended-rag/experiments/27_finalize_first_completed_pass.py) | Finalize the first completed benchmark pass |
| 28 | [28_package_fast_finish.py](../cascaded-blended-rag/experiments/28_package_fast_finish.py) | Package the completed fast-finish results |
| 29 | [29_verify_squad_metrics.py](../cascaded-blended-rag/experiments/29_verify_squad_metrics.py) | Independently verify SQuAD metrics |
| 30 | [30_compare_indexed_execution.py](../cascaded-blended-rag/experiments/30_compare_indexed_execution.py) | Compare original and indexed execution |
| 31 | [31_verify_beir_metrics.py](../cascaded-blended-rag/experiments/31_verify_beir_metrics.py) | Independently verify BEIR metrics |
| 32 | [32_prepare_report_assets.py](../cascaded-blended-rag/experiments/32_prepare_report_assets.py) | Prepare figures and tables from saved results |

The shared implementation lives in [`../cascaded-blended-rag/src/`](../cascaded-blended-rag/src/).
Frozen protocols and compact reports are under
[`../cascaded-blended-rag/results/`](../cascaded-blended-rag/results/).
The [reproduction guide](../docs/reproducibility/README.md) explains how to
restore matching query records and run the versioned Kaggle workflows.

## Reading the comparisons

On TREC-COVID, the primary indexed cascade is 20.21× faster than our earlier
fusion run and retains 96.68% of indexed fusion Recall@100. The 20.21× ratio
combines indexing and routing across separate runs; routing alone adds 1.24×
over indexed fusion. On the fixed 256-question SQuAD sample, the fastest
cascade is 7.91× faster than our earlier fusion run, or 1.59× over indexed
fusion, at 99.22% retained recall.

These are comparisons within our Python implementation. They do not establish
the same speedup in the original Elasticsearch / ELSER system. Full current
NQ, HotpotQA and CoQA evaluations, answer-generation EM/F1, production load,
and index memory/update costs remain future work. The study reports an
engineering contribution, not an overall superiority or algorithmic novelty
claim.

## Rebuild materials

From the repository root, regenerate figures, the paper PDF and HTML showcase
from saved reports with:

```powershell
python cascaded-blended-rag/experiments/32_prepare_report_assets.py
python tools/build_publication_figures.py
python tools/build_research_materials.py
```

The editable deck builder and PDF renderers are documented in
[`tools/README.md`](../tools/README.md). Full benchmark dependencies are listed
in [`../requirements/full-benchmark.txt`](../requirements/full-benchmark.txt).
