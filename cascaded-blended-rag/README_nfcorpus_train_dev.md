# NFCorpus train calibration and dev check

This source bundle calibrates the cascade gate from the official NFCorpus
`train` split and checks the frozen gate on `dev`. The script never reads
`test.tsv`.

## Kaggle dataset contents

- `experiments/12_calibrate_nfcorpus_train_check_dev.py`
- `src/` retriever and cache modules
- `requirements_nfcorpus_train_dev.txt`

## Run on Kaggle

Use a GPU runtime if available. Install the listed requirements, then run the
script from the project folder. It downloads the official BEIR NFCorpus archive
if it is not already present. The full corpus is used for retrieval indexes;
split qrels select query IDs and provide train/dev relevance labels.

The result is written to `results/nfcorpus_train_calibrated_dev.json`. Train
labels select the router and threshold using five-fold out-of-fold predictions
under a 99% of static Recall@100 constraint. Dev routes and rankings are fixed
before dev relevance grades are loaded. Report dev as a calibration check, not
as a final untouched test result.
