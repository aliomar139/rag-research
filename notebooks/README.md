# Kaggle workflows

The versioned workflow names remain recognizable from the original project.
Runners are Python cells; `.zip` upload products are generated/local artifacts.

| Folder | Role |
|:--|:--|
| `kaggle_fast_finish_v1/` | Final original-SQuAD export and indexed follow-ups |
| `kaggle_indexed_cascade_v1/` | Indexed execution workflow before the combined utility |
| `kaggle_paper_datasets_v1/` | Original-backend paper-dataset extension |
| `kaggle_retention_tradeoff_v1/`, `kaggle_scidocs_confirmation_v1/` | Frozen development and transfer benchmarks |
| `kaggle_nfcorpus_*/`, `legacy/` | Earlier calibration/diagnostics and exploratory notebooks |

All final runs have completed within the scope documented in
[the results](../docs/FINAL_RESULTS.md). These are reproduction instructions,
not a request to start another GPU run.

Original source ZIPs were moved to `artifacts/source_bundles/` and excluded
from Git. Experiments 21, 26 and 28 can rebuild paper/indexed/combined source
packages into their original workflow folders; these new ZIPs are ignored too.
Rebuilt package navigation files may differ from the original archived ZIPs;
the frozen retrieval code, policies and protocol fingerprints remain the
scientific references. See [reproducibility](../docs/reproducibility/README.md).
