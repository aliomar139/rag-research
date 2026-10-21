# Local data

Downloaded corpora and annotations live in `raw/`. Dense/sparse indexes use
the original `processed*` directory names. These inputs and caches are ignored
and excluded from the submission ZIP; local copies are retained.

The current benchmark obtains official full corpora via `src/paper_datasets.py`
or the named BEIR loader. It checks declared dimensions and fingerprints and
does not select the corpus by relevance labels. Historical fast loaders have
different restrictions and do not supply final paper-dataset evidence.

See [the reproduction guide](../../docs/reproducibility/README.md) to restore
inputs. Figure generation uses compact reports and requires no raw data.
