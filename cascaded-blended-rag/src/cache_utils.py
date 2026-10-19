import hashlib
import json
import os
from typing import Dict


def corpus_fingerprint(corpus: Dict[str, dict]) -> str:
    """Hash ordered document IDs and text fields used by the retrievers."""
    digest = hashlib.sha256()
    for doc_id, doc in corpus.items():
        digest.update(str(doc_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(doc.get("title", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(doc.get("abstract", doc.get("text", ""))).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def cache_matches(index_path: str, metadata_path: str, fingerprint: str, model_name: str) -> bool:
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("corpus_fingerprint") == fingerprint and metadata.get("model_name") == model_name


def write_cache_metadata(metadata_path: str, fingerprint: str, model_name: str) -> None:
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"corpus_fingerprint": fingerprint, "model_name": model_name}, f, indent=2)

