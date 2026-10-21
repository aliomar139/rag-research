"""Explicitly download only raw labels needed by the saved-metric verifiers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

from _repository import ROOT, sha256

RAW = ROOT / "cascaded-blended-rag/data/raw"
RESULTS = ROOT / "cascaded-blended-rag/results"


def fetch(url, path):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+".part")
    print("Downloading:", url, flush=True)
    with urllib.request.urlopen(url, timeout=180) as source, partial.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    partial.replace(path)


def prepare(dataset):
    if dataset == "squad":
        report = json.loads((RESULTS / "indexed_squad_q256/indexed_cascade_benchmark.json").read_text())
        target = RAW / "squad_dev_v1.1.json"
        expected = report["dataset_preparation"]["source_sha256"]["dev"]
        fetch("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json", target)
    else:
        report = json.loads((RESULTS / "indexed_trec-covid_q0/indexed_cascade_benchmark.json").read_text())
        target = RAW / "trec-covid/qrels/test.tsv"
        expected = report["evaluation_qrels_sha256"]
        if not target.is_file():
            archive_path = RAW / "trec-covid.zip"
            fetch(report["dataset_preparation"]["source_url"], archive_path)
            digest = hashlib.md5()
            with archive_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != report["dataset_preparation"]["official_archive_md5"]:
                raise ValueError("Official TREC archive checksum mismatch; existing file kept for inspection")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as archive:
                with archive.open("trec-covid/qrels/test.tsv") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    if sha256(target) != expected:
        raise ValueError(f"Raw label fingerprint differs from the measured run: {target}")
    print("Verified raw labels:", target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("trec-covid", "squad", "both"), required=True)
    args = parser.parse_args()
    for dataset in (("trec-covid", "squad") if args.dataset == "both" else (args.dataset,)):
        prepare(dataset)


if __name__ == "__main__":
    main()
