# Result evidence

## Main reports

| Directory | Evaluation |
|:--|:--|
| `indexed_trec-covid_q0/` | Indexed TREC, all 50 queries; q0 means no query limit |
| `indexed_squad_q256/` | Indexed SQuAD fixed 256-question sample |
| `paper_trec-covid/`, `paper_squad_one_pass/` | Original implementation, including SQuAD's explicit timing amendment |
| `retention_tradeoff_dev/` | Inspected NFCorpus dev, both feature families |
| `scidocs_confirmation/` | Frozen SCIDOCS confirmation |

`final_quality_speed.csv` combines all 52 method rows with separate scope
labels. Protocol and frozen-router JSON files at this directory's root are
shared assets required by validators and package builders. Root-level
SciFact/NFCorpus diagnostic reports preserve earlier results.

## Large evidence kept local

Per-query rankings, partial repetitions, training feature tables and retrieval
caches remain on disk but are ignored. The original exported ZIPs are under
`artifacts/downloads/` at repository root. Share them separately when full
record-level inspection is needed. Fingerprints and sizes are in
`docs/reproducibility/artifact_catalog.json`.

`local_verification.json` records artifact consistency checks;
`independent_metric_verification.json` records raw-label metric checks;
`paired_backend_comparison.json` records old/new ranking agreement and
same-query timing ratios. These records are shared, but are not a substitute
for receiving the underlying large artifacts when rerunning those checks.
