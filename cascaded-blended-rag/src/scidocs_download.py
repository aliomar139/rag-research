"""Checksum-verified official BEIR SCIDOCS files; no grade parsing here."""

import hashlib
from pathlib import Path
import urllib.request
import zipfile

DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip"
DATASET_MD5 = "38121350fc3a4d2f48850f6aff52e4a9"


def ensure_scidocs(raw_dir):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "scidocs.zip"
    if not archive_path.is_file():
        print(f"Downloading official SCIDOCS archive: {DATASET_URL}", flush=True)
        temporary = raw_dir / "scidocs.zip.part"
        with urllib.request.urlopen(DATASET_URL, timeout=180) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024*1024):
                handle.write(chunk)
        if hashlib.md5(temporary.read_bytes()).hexdigest() != DATASET_MD5:
            raise ValueError("Downloaded SCIDOCS archive failed the official MD5 check")
        temporary.replace(archive_path)
    if hashlib.md5(archive_path.read_bytes()).hexdigest() != DATASET_MD5:
        raise ValueError("Cached SCIDOCS archive failed the official MD5 check")
    dataset_dir = raw_dir / "scidocs"
    # Extract only named members: no untrusted archive paths are followed.
    with zipfile.ZipFile(archive_path) as archive:
        for relative in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            content = archive.read("scidocs/"+relative)
            target = dataset_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(content).digest():
                target.write_bytes(content)
    return dataset_dir
