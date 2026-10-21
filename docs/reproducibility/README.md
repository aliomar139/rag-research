# Reproduce the submitted results

## 1. Regenerate figures without a GPU or raw datasets

Use Python 3.12 or a compatible newer interpreter. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/analysis.txt
.\.venv\Scripts\python.exe cascaded-blended-rag/experiments/32_prepare_report_assets.py
```

On Linux/macOS use `.venv/bin/python` in place of the Windows interpreter
path. These commands rebuild the figures and complete speedup tables from
the shared compact reports. They do not remeasure performance.

## 2. Audit the repository and create the sharing ZIP

These tools use only Python's standard library and Git. They also accept an
extracted submission ZIP containing `SUBMISSION_MANIFEST.json`.

```powershell
python tools/check_repository.py
python tools/build_submission.py
```

The audit checks shareable file sizes, secret patterns, current navigation,
Python syntax and frozen source/weight fingerprints. It does not execute
unit tests, models or timing experiments. Historical navigation notices are
reported separately. The built ZIP and manifest go to `artifacts/submission/`.

`.gitattributes` disables newline conversion for Python, JSON and recorded
data formats because some original frozen assets use LF and others CRLF.
Their exact bytes must survive a Windows/Linux checkout. Do not reformat
frozen scripts or edit protocol JSON to make a changed source pass its hash.

## 3. Rerun record-level artifact checks

Large rankings and original ZIPs are not committed. Obtain the matching
Kaggle result exports separately and put them under `artifacts/downloads/`.
[artifact_catalog.json](artifact_catalog.json) records original names, sizes
and SHA256 values; it is an inventory, not a claim of public artifact hosting.

```powershell
python cascaded-blended-rag/experiments/22_import_paper_benchmark.py artifacts/downloads/trec_covid_indexed_cascade_outputs.zip
python cascaded-blended-rag/experiments/22_import_paper_benchmark.py artifacts/downloads/squad_indexed_cascade_outputs.zip
```

Importing reconstructs the large per-query files in their original ignored
result folders. To compare original/indexed rankings, also import the original
TREC/SQuAD exports, then use experiment 30 with the appropriate directories.
The complete commands and validity limits are in
[the TREC findings](../findings/TREC_COVID_INDEXED_FINDINGS.md) and
[the SQuAD findings](../findings/SQUAD_INDEXED_FINDINGS.md).

For independent raw-label scoring, explicitly restore official label inputs:

```powershell
python tools/prepare_verification_data.py --dataset both
python cascaded-blended-rag/experiments/31_verify_beir_metrics.py
python cascaded-blended-rag/experiments/29_verify_squad_metrics.py --result-dir cascaded-blended-rag/results/indexed_squad_q256
```

The restoration command performs network downloads only when the label files
are absent. For TREC it downloads the verified official archive but extracts
only test qrels. It does not encode documents or run retrieval. Metric checks
require the per-query records restored from the matching result ZIPs.

## 4. Reproduce retrieval itself

Install `requirements/full-benchmark.txt` for the full neural environment, then follow the
versioned [Kaggle workflows](../../notebooks/README.md). The final combined
workflow uses experiments 25/27 with frozen assets inside the source package.
Experiments 26 and 28 rebuild the indexed and combined upload packages:

```powershell
python cascaded-blended-rag/experiments/26_package_indexed_cascade.py
python cascaded-blended-rag/experiments/28_package_fast_finish.py
```

Packagers write ignored ZIPs under their named notebook folders. Archived
original packages are retained locally under `artifacts/source_bundles/`.
Rebuilt ZIPs can contain updated navigation documents; the archived originals
have separate catalog hashes. Reported Python/Torch/NumPy, model revisions,
GPU and source hashes are in the saved benchmark reports. Dependency ranges
are installation requirements, not an exact Kaggle environment lockfile.

Fresh neural corpus encoding needs a GPU and can take substantially longer
than cached query timing. Strict-cache follow-ups deliberately stop on missing
indexes rather than silently rebuilding them. NQ/HotpotQA full-corpus runs
remain uncompleted, resource-dependent work.

Historical scripts 14-16 expect two root-level train inputs. If their original
exports have been received under `artifacts/downloads/`, the optional command
`python tools/restore_local_inputs.py` recreates ignored compatibility copies.
Their frozen science code does not need a path edit.

## 5. Understand the two speedup references

Original-fusion comparisons combine indexing and routing across separate
runs. Indexed-fusion comparisons measure the extra cascade gain with equal
indexing. SQuAD uses the same 256 query IDs for cross-run ratios. The growing
configuration sequence does not measure performance as corpus size increases.
See [all speedups](../SPEEDUP_SUMMARY.md) for every point and failed floor.
