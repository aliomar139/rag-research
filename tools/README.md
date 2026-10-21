# Sharing and input utilities

| Command | Effect |
|:--|:--|
| `python tools/check_repository.py` | Audit sharing hygiene, paths, syntax and frozen bytes |
| `python tools/build_submission.py` | Build a source/report ZIP with a file manifest |
| `python tools/prepare_verification_data.py --dataset both` | Explicitly obtain and verify official raw label inputs |
| `python tools/restore_local_inputs.py` | Restore ignored compatibility copies for historical training scripts |

The first two commands do not run tests, benchmarks, models or network
downloads. The preparation command downloads public dataset files when
needed. Large artifacts and generated outputs remain excluded from sharing.

## Publication builders

The final artifact sources and exact runtime commands are documented in
[the submission guide](../docs/submission/README.md). `build_publication_figures.py`
exports release charts; `build_research_materials.py` creates the paper PDF
and HTML; the JavaScript builders render the showcase and author the editable
deck. These tools consume recorded results without new retrieval experiments.
