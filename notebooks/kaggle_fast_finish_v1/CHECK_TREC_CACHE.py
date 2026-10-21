"""Paste into the current Kaggle notebook. Read-only; no model execution."""
from pathlib import Path

root = Path("/kaggle/working")
required = ("dense_doc_embeddings.npy", "dense_doc_embeddings.npy.json",
            "scifact_splade_index.npz", "scifact_splade_index.npz.json")
folders = sorted(p for p in root.rglob("processed_paper_trec-covid") if p.is_dir())
if not folders:
    print("TREC_CACHE_ABSENT")
for folder in folders:
    missing = [name for name in required if not (folder / name).is_file()]
    print("TREC_CACHE_PRESENT" if not missing else "TREC_CACHE_INCOMPLETE", str(folder))
    if missing:
        print("Missing:", missing)
print("Copy this output back into the chat. No retrieval or encoding was started.")
