# First run: TREC-COVID. This cell also supports the other paper datasets.
DATASET = globals().get("DATASET", "trec-covid")  # Later: "squad", "nq", or "hotpotqa"
ALLOW_LARGE_CORPUS = globals().get("ALLOW_LARGE_CORPUS", False)

from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile

root = Path("/kaggle/input")
project = Path("/kaggle/working/paper_datasets_frozen_v1")
script_name = "20_benchmark_paper_datasets.py"
bundle_name = "paper_datasets_frozen_source_20261020_a7c4_v1.zip"
project.mkdir(parents=True, exist_ok=True)
scripts = sorted(root.rglob(script_name))
bundles = sorted(root.rglob(bundle_name))
if len(scripts) > 1 or (not scripts and len(bundles) > 1):
    raise RuntimeError("Multiple copies of this bundle are attached. Keep one copy in Add Input.")
if scripts:
    shutil.copytree(scripts[0].parent.parent, project, dirs_exist_ok=True)
elif bundles:
    with zipfile.ZipFile(bundles[0]) as archive:
        for name in archive.namelist():
            if "\\" in name or name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("Unsafe archive entry: "+name)
        archive.extractall(project)
else:
    mounted = [str(p) for p in root.iterdir()] if root.exists() else []
    raise FileNotFoundError(
        "Use Add Input to attach paper-datasets-frozen-r90-20261020-a7c4-v1. "
        f"Expected {bundle_name} or {script_name}. Mounted folders: {mounted}"
    )
if DATASET not in ("trec-covid", "squad", "nq", "hotpotqa"):
    raise ValueError("DATASET must be trec-covid, squad, nq, or hotpotqa")
if DATASET in ("nq", "hotpotqa") and not ALLOW_LARGE_CORPUS:
    raise RuntimeError("NQ/HotpotQA need full multi-million-document indexes. Start with trec-covid; see START_HERE.md.")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(project / "requirements.txt")], check=True)
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Enable a GPU in Kaggle Settings, then rerun this cell.")
command = [sys.executable, "-u", str(project / "experiments" / script_name),
           "--dataset", DATASET, "--repeats", "3"]
if ALLOW_LARGE_CORPUS:
    command.append("--allow-large-corpus")
subprocess.run(command, cwd=project, check=True)
output = project / "results" / ("paper_"+DATASET) / (DATASET.replace("-", "_")+"_paper_benchmark_outputs.zip")
print("DOWNLOAD AND PUT IN YOUR LOCAL PROJECT ROOT:", output)
from IPython.display import FileLink, display
os.chdir("/kaggle/working")
display(FileLink(str(output.relative_to(Path("/kaggle/working")))))
