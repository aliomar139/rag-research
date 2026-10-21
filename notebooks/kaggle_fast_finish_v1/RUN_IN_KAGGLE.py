ACTION = globals().get("ACTION", "finalize_original")

from pathlib import Path
import os
import shutil
import subprocess
import sys

project = Path("/kaggle/working/fast_finish_indexed_v1")
project.mkdir(parents=True, exist_ok=True)
if Path(PACKAGE_ROOT).resolve() != project.resolve():
    shutil.copytree(PACKAGE_ROOT, project, dirs_exist_ok=True)
original = Path("/kaggle/working/paper_datasets_frozen_v1")
full = original / "results/paper_squad/squad_paper_benchmark_outputs.zip"
one_pass = original / "results/paper_squad_one_pass/squad_one_pass_outputs.zip"
if ACTION == "finalize_original":
    subprocess.run([sys.executable, "-u", str(project / "experiments/27_finalize_first_completed_pass.py"),
                    "--project", str(original)], cwd=project, check=True)
    output = full if full.is_file() else one_pass
elif ACTION == "indexed_followup":
    previous = full.parent if full.is_file() else one_pass.parent
    if not full.is_file() and not one_pass.is_file():
        raise FileNotFoundError("Run ACTION = 'finalize_original' and download its result first.")
    cache = original / "data/processed_paper_squad"
    required = [previous / "retrieval_first_pass.jsonl", cache / "dense_doc_embeddings.npy",
                cache / "dense_doc_embeddings.npy.json", cache / "scifact_splade_index.npz",
                cache / "scifact_splade_index.npz.json"]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("The original working files or corpus caches are missing. Keep the same Kaggle session.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Keep the GPU enabled for the indexed follow-up.")
    output_dir = project / "results/indexed_squad_q256"
    subprocess.run([
        sys.executable, "-u", str(project / "experiments/25_benchmark_indexed_cascade.py"),
        "--dataset", "squad", "--query-limit", "256", "--repeats", "3",
        "--data-dir", str(original / "data"), "--output-dir", str(output_dir),
        "--reference-first-pass", str(previous / "retrieval_first_pass.jsonl"),
        "--require-existing-neural-cache",
    ], cwd=project, check=True)
    output = output_dir / "squad_indexed_cascade_outputs.zip"
else:
    raise ValueError("ACTION must be finalize_original or indexed_followup")
print("DOWNLOAD AND PUT IN YOUR LOCAL PROJECT ROOT:", output)
from IPython.display import FileLink, display
os.chdir("/kaggle/working")
display(FileLink(str(output.relative_to(Path("/kaggle/working")))))
