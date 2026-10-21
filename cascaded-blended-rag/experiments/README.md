# Experiment map

The numeric filenames preserve chronology and existing imports. A physical
move of frozen scripts would invalidate their recorded paths or hashes.

| Stage | Scripts | Purpose |
|:--|:--|:--|
| Exploration | 01-11 | Baselines, legacy cached experiments, corrected SciFact and transfer |
| Train-only development | 12-16 | NFCorpus calibration, diagnostics, features and four budgets |
| Measured frozen benchmarks | 17, 18, 20, 24, 25 | Development, SCIDOCS, original/indexed paper datasets and CPU scorer |
| Evidence and reporting | 19, 22, 23, 29-32 | Import, diagnostics, independent scoring, compatibility and figures |
| Packaging/runtime utility | 21, 26-28 | Source bundles and explicit first-pass timing amendment |

## Reviewer entry point

Run experiment **32** to regenerate the published figures and speedup tables
from compact reports. This needs analysis dependencies, no dataset or GPU.

Experiments **22**, **29**, **30** and **31** inspect full saved evidence and
raw relevance labels. Matching large inputs are local-only; see
[the evidence guide](../../docs/reproducibility/README.md).

Experiments **12**, **17**, **18**, **20**, **25**, **27** and the retrieval
helpers have recorded fingerprints. Their exact bytes remain preserved.
Never interpret old legacy scripts or their outputs as a new held-out result.
