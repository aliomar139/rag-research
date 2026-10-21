"""Combine the indexed follow-up with the checkpoint-only first-pass exporter."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    destination = ROOT / "notebooks/kaggle_fast_finish_v1"
    indexed = ROOT / "notebooks/kaggle_indexed_cascade_v1/indexed_cascade_source_20261021_b8e6_v1.zip"
    if not indexed.is_file():
        indexed = ROOT / "artifacts/source_bundles/kaggle_indexed_cascade_v1" / indexed.name
    with zipfile.ZipFile(indexed) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()
                   if name not in ("package_manifest.json", "README.md", "RUN_INDEXED_CASCADE_IN_KAGGLE.py")}
    entries["experiments/27_finalize_first_completed_pass.py"] = (PROJECT / "experiments/27_finalize_first_completed_pass.py").read_bytes()
    entries["RUN_FAST_FINISH_IN_KAGGLE.py"] = (destination / "RUN_IN_KAGGLE.py").read_bytes()
    entries["README.md"] = (destination / "START_HERE.md").read_bytes()
    for name, raw in entries.items():
        if name.endswith(".py"):
            ast.parse(raw.decode("utf-8"), filename=name)
    ast.parse((destination / "PASTE_CELL.py").read_text(encoding="utf-8"))
    manifest = {name: sha(raw) for name, raw in entries.items()}
    entries["package_manifest.json"] = (json.dumps(manifest, indent=2)+"\n").encode("utf-8")
    output = destination / "fast_finish_indexed_20261021_d4a9_v1.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in entries.items():
            if "\\" in name or ".." in Path(name).parts or name.startswith("/"):
                raise ValueError("Invalid archive path")
            archive.writestr(name, raw)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        for name, expected in manifest.items():
            if sha(archive.read(name)) != expected:
                raise ValueError("Manifest mismatch: "+name)
    print("UPLOAD:", output)
    print("KAGGLE TITLE: fast-finish-indexed-20261021-d4a9-v1")
    print("BYTES:", output.stat().st_size)
    print("SHA256:", sha(output.read_bytes()))
    print("Syntax/package checks complete. No live checkpoint or GPU experiment executed locally.")


if __name__ == "__main__":
    main()
