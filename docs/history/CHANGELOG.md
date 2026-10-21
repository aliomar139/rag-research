# Changes since `6ef8ace`

This working tree consolidates the research conducted after the last commit
and prepares its source, reports and compact evidence for sharing. No new
retrieval, training or timing experiment was run during the cleanup.

## Current study

- Finalized a 9-page paper with explicit original-paper/adaptation and
  metric distinctions, plus a 10-section HTML/PDF showcase and a 23-slide editable
  deck. [Final materials](../../final_results/README.md) include speaker notes,
  reproducible sources and licensed offline fonts. Remaining datasets and
  generated-answer metrics remain deferred.
- Retained experiments 09–32, frozen NFCorpus-trained routing weights,
  benchmark protocols, compact result reports and independent metric checks.
- Added the indexed BM25 implementation and its measured original/indexed
  comparisons. TREC-COVID covers all 50 queries; indexed SQuAD covers the
  declared 256-question sample. Full NQ/HotpotQA results remain pending.
- Consolidated conclusions in [the manuscript](../FINAL_MANUSCRIPT.md),
  [results](../FINAL_RESULTS.md) and [complete speedup tables](../SPEEDUP_SUMMARY.md).
  Failed quality thresholds, uncertainty and stronger standalone baselines
  remain visible.

## Navigation and layout

- Kept the original `cascaded-blended-rag`, `docs`, `notebooks` and
  `Similar-Papers` structure. Numbered experiments and frozen source paths
  retain their original locations for provenance.
- Grouped current interpretations under `docs/findings`, submission checks
  under `docs/review`, and superseded drafts/plans under `docs/history`.
  Historical documents carry a notice directing readers to the current report.
- Grouped literature catalogs and downloader reports into named folders;
  moved early notebooks into `notebooks/legacy`.
- Added folder indexes and a [reproduction guide](../reproducibility/README.md).
  The [migration table](../reproducibility/layout_migration.csv) records moved
  paths and the four byte-identical root exports that were removed.

## Local files and sharing

- Moved downloaded result ZIPs, original source bundles and obsolete binary
  report exports into ignored local artifact folders. Unique evidence is
  preserved locally; generated Python caches and four verified duplicate
  exports were removed.
- Removed raw data, the old source ZIP and local working notes from Git
  tracking without deleting existing local copies. Added ignore rules for
  datasets, indexes, bulk rankings, environments, archives and credentials.
- Added byte-preserving Git attributes for frozen code/JSON. Recorded source
  and weight fingerprints are checked without rewriting the original assets.
- Added standard-library repository checks and a submission ZIP builder with
  a per-file SHA256 manifest. The shared archive excludes Git metadata and
  local bulk artifacts. The [artifact catalog](../reproducibility/artifact_catalog.json)
  records original local filenames, sizes and fingerprints.

## Helper compatibility

Result import/finalization helpers locate archives under `artifacts/downloads`;
the final CSV builder includes the indexed runs. The combined Kaggle packager
can find its archived source bundle. The literature downloader follows its
new catalog/report folders. Optional input-restoration tools document how to
recover ignored labels and historical train-input paths.

Benchmark source, measured report JSON, frozen protocols and router weights
were preserved. Dependency files distinguish plotting, neural benchmarking
and optional literature downloads; they do not claim to lock the original
Kaggle environment.

## Before formal submission

Apply author/course details and the required report template, then review
the final manuscript. The repository has no selected license. See the
[submission checklist](../review/PROJECT_FINISH_CHECKLIST.md).
