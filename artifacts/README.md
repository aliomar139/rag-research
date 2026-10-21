# Local artifacts and sharing boundary

This README is shared. The generated/downloaded child folders are ignored.

| Local folder | Purpose |
|:--|:--|
| `downloads/` | Original downloaded Kaggle result ZIPs and feature-corpus input |
| `source_bundles/` | Archived source ZIPs used for Kaggle runs |
| `source_snapshots/` | Obsolete extracted upload copy, retained locally |
| `legacy_exports/` | Earlier generated Word/PDF/HTML report exports |
| `submission/` | Newly built sharing ZIP, manifest and repository audit |

The full archive inventory is in
[artifact_catalog.json](../docs/reproducibility/artifact_catalog.json), with
relative paths, byte sizes and SHA256 fingerprints. Per-query bulk records
also remain in their original `cascaded-blended-rag/results/` directories,
but `.gitignore` excludes them from Git and the sharing ZIP.

Compact reports, frozen weights/protocols, validation records, CSV tables and
figures are included in the repository. A recipient who needs to rerun full
record-level checks must receive the matching result ZIPs separately and put
them in `artifacts/downloads/`. No public result-download URL is asserted.

Build the source/report sharing ZIP from the repository root:

```powershell
python tools/build_submission.py
```

The ZIP includes a file-by-file `SUBMISSION_MANIFEST.json`. It excludes Git
metadata, environments, raw corpora, downloaded papers and local working notes.
