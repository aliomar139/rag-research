"""Paste into the existing Kaggle notebook; reuse the uploaded source and indexes."""
from pathlib import Path
import os
import subprocess
import sys
from IPython.display import FileLink, display

project = Path("/kaggle/working/fast_finish_indexed_v1")
original = Path("/kaggle/working/paper_datasets_frozen_v1")
runner = project / "experiments/25_benchmark_indexed_cascade.py"
cache = original / "data/processed_paper_trec-covid"
required = [
    runner,
    project / "artifacts/indexed_cascade_protocol.json",
    project / "artifacts/nfcorpus_frozen_retention_routers.json",
    original / "data/raw/trec-covid.zip",
    *[cache / name for name in (
        "dense_doc_embeddings.npy", "dense_doc_embeddings.npy.json",
        "scifact_splade_index.npz", "scifact_splade_index.npz.json")],
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing files; paste this error into the chat:\n" + "\n".join(missing))

import torch
if not torch.cuda.is_available():
    raise RuntimeError("A GPU is required. No benchmark has started; paste this error into the chat.")

output_dir = project / "results/indexed_trec-covid_all50_v1"
command = [
    sys.executable, "-u", str(runner),
    "--dataset", "trec-covid", "--query-limit", "0", "--repeats", "3",
    "--data-dir", str(original / "data"), "--output-dir", str(output_dir),
    "--require-existing-neural-cache",
]
reference = original / "results/paper_trec-covid/retrieval_first_pass.jsonl"
if reference.is_file():
    command += ["--reference-first-pass", str(reference)]
    print("Previous BM25 rankings will also be checked after timing.", flush=True)
else:
    print("Previous rankings absent: optional cross-run comparison skipped.", flush=True)

print("TREC-COVID: all 50 queries, full 171,332-document corpus, 8 methods, 3 passes.", flush=True)
print("Existing corpus indexes required; missing/mismatched indexes stop the run.", flush=True)
subprocess.run(command, cwd=project, check=True)

output = output_dir / "trec_covid_indexed_cascade_outputs.zip"
if not output.is_file():
    raise FileNotFoundError(f"Expected export missing: {output}")
print("DOWNLOAD:", output)
os.chdir("/kaggle/working")
display(FileLink(str(output.relative_to(Path("/kaggle/working")))))
