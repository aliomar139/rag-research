# Retrieval code and experiment records

| Directory | Role |
|:--|:--|
| [src](src/README.md) | Retriever implementations, routing and dataset helpers |
| [experiments](experiments/README.md) | Numbered historical and final runners |
| [results](results/README.md) | Compact reports, frozen assets and local bulk records |
| [data](data/README.md) | Local datasets and generated indexes |
| [tests](tests/README.md) | Existing focused checks; not executed during submission cleanup |

Numbered scripts stay at their recorded locations because protocols and
cross-script imports depend on those paths. Several source/protocol assets
have exact-byte fingerprints, so `.gitattributes` disables newline conversion
for code and JSON. Avoid mass-formatting frozen sources.

`data_loader.py`, `squad_loader.py`, `trec_covid_fast_loader.py` and
`README_nfcorpus_train_dev.md` support historical workflows. The current
full-corpus paper-dataset path is `src/paper_datasets.py`; the historical
fast loader is not the source of final full-corpus evidence.
