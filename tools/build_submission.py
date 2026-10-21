"""Create a reviewable source/report ZIP without local data, Git metadata or caches."""
import argparse
import json
from pathlib import Path
import zipfile

from _repository import ROOT, git_files, sha256
from check_repository import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/submission/rag_research_submission_20261021.zip")
    args = parser.parse_args()
    check = audit()
    if check["errors"]:
        raise SystemExit("Repository audit failed. Run tools/check_repository.py for details.")
    names = git_files()
    manifest = {"snapshot_date": "2026-10-21", "files": [
        {"path": name, "bytes": (ROOT / name).stat().st_size, "sha256": sha256(ROOT / name)} for name in names],
        "scope": "Source, current reports, figures, compact evidence and labeled history. "
                 "Raw downloads, per-query bulk evidence, environments and local working notes are excluded."}
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/submission"):
        raise SystemExit("Choose an output inside artifacts/submission/.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in manifest["files"]:
            info = zipfile.ZipInfo(entry["path"], date_time=(2026, 10, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (ROOT / entry["path"]).read_bytes())
        archive.writestr("SUBMISSION_MANIFEST.json", json.dumps(manifest, indent=2)+"\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Submission archive CRC failure")
    (output.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("Submission ZIP:", output)
    print("Files:", len(names), "Bytes:", output.stat().st_size)
    print("SHA256:", sha256(output))


if __name__ == "__main__":
    main()
