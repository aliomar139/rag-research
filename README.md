# Conditional Hybrid Retrieval

A BM25-first Ridge gate chooses whether to return a lexical ranking or run
Dense MiniLM and SPLADE before reciprocal rank fusion. This repository contains
the compatible indexed BM25 implementation, frozen benchmark protocols,
measured results and the submission manuscript.

## Start here

The [final material bundle](docs/submission/README.md) includes the research
paper, offline HTML/PDF showcase and editable presentation with speaker notes.

| Reader goal | File |
|:--|:--|
| Understand the project and its development | [Plain-language explanation](docs/adhd-explanation.md) |
| Read the submission report | [Consolidated manuscript](docs/FINAL_MANUSCRIPT.md) |
| Inspect every speedup and quality trade-off | [Full speedup tables](docs/SPEEDUP_SUMMARY.md) |
| Reproduce plots or audit evidence | [Reproduction guide](docs/reproducibility/README.md) |
| Understand changes since the last commit | [CHANGELOG.md](CHANGELOG.md) |

## Measured result

**TREC-COVID: all 50 queries, 171,332 documents, three warmed T4 timing passes.**

| Configuration | Recall retained | Speedup vs our original fusion | Speedup vs equally indexed fusion |
|:--|--:|--:|--:|
| Indexed fusion | 100.00% | 16.36x | 1.00x |
| r99, primary | 96.68% | 20.21x | 1.24x |
| r95, secondary | 91.95% | 26.83x | 1.64x |
| r92, secondary | 84.53% | 35.56x | 2.17x |
| r90, secondary | 84.53% | 36.96x | 2.26x |

The largest two cascade points fail the 90% floor. The r95 point's recall
interval crosses below 90%; the conservative primary's lower bound is 93.58%.
Original-implementation ratios combine indexing and routing across runs.
The extra cascade benefit is the last column, using the same indexed scorer.

On the declared 256-question SQuAD sample, the fastest cascade retains 99.22%
recall at 1.59x over indexed fusion, or 7.91x over same-query original fusion.
SQuAD's full corpus remains in use; the indexed query sample is not the full
10,570-question evaluation. Standalone SPLADE or Dense/BM25 often provides a
better recall/time choice. Earlier target failures remain reported.

These are retrieval results from our MiniLM/SPLADE/RRF adaptation. They do
not establish acceleration of the original paper's ELSER system, preserved
answer quality or overall method superiority. Full NQ/HotpotQA remain pending.

![Measured speedup progression](docs/figures/speedup_progression.png)

## Repository layout

```text
cascaded-blended-rag/
  src/                 Retrieval and routing implementation
  experiments/         Numbered scripts, kept at their provenance paths
  results/             Compact reports, frozen assets and validation records
  data/                Local-only datasets and generated indexes
  tests/               Existing focused checks

docs/
  FINAL_MANUSCRIPT.md   Current report
  FINAL_RESULTS.md      Results entry point
  SPEEDUP_SUMMARY.md    Complete speedup tables
  adhd-explanation.md   Plain-language project history
  submission/          Paper PDF, HTML/PDF showcase and editable slide deck
  findings/            Individual experiment interpretation
  figures/             PNG/SVG plots and CSV source values
  reproducibility/     Reproduction, artifact inventory and layout migration
  review/              Claim audit and submission checklist
  history/             Superseded drafts, plans and chronological ledger

notebooks/             Versioned Kaggle workflows and labeled legacy notebooks
Similar-Papers/        Literature catalogs and optional downloader
requirements/          Analysis, benchmark and optional literature dependencies
tools/                 Sharing audits, ZIP builder and explicit input restoration
artifacts/             Ignored local downloads, old exports and submission ZIP
```

Detailed indexes: [documentation](docs/README.md),
[code](cascaded-blended-rag/README.md), [Kaggle](notebooks/README.md).

## Regenerate figures without a GPU

From the repository root, using Python 3.12 or a compatible newer interpreter:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/analysis.txt
.\.venv\Scripts\python.exe cascaded-blended-rag/experiments/32_prepare_report_assets.py
```

On Linux/macOS use `.venv/bin/python` instead. The checked-in reports are
sufficient for both figures and all speedup tables; no dataset, model download
or new timing run is needed. Full benchmark dependencies are in `requirements.txt`.

## Check and package for sharing

```powershell
python tools/check_repository.py
python tools/build_submission.py
```

The audit checks current navigation, shareable sizes, secret patterns, syntax
and frozen fingerprints. It does not run the unit tests or retrieval.
The sharing ZIP is written to
`artifacts/submission/rag_research_submission_20261021.zip`, with a SHA256
manifest for its files. It includes source, compact evidence, figures and
labeled history. Raw corpora, large rankings, downloaded papers, original
result ZIPs, environments and local working notes are excluded.

Original local artifacts are retained and inventoried in
[artifact_catalog.json](docs/reproducibility/artifact_catalog.json).
Receive matching ZIPs separately for full record-level revalidation; the
[reproduction guide](docs/reproducibility/README.md) explains how.
`.gitattributes` preserves exact frozen code/JSON bytes across platforms.
A repository license file has not been selected.
