# Paste this entire file into ONE Kaggle notebook cell.
from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile

root = Path("/kaggle/input")
script_name = "17_benchmark_retention_tradeoff.py"
bundle_name = "nfcorpus_retention_tradeoff_source_v1.zip"
scripts = list(root.rglob(script_name))
bundles = list(root.rglob(bundle_name))
project = Path("/kaggle/working/nfcorpus_retention_tradeoff_v1")
project.mkdir(parents=True, exist_ok=True)

if scripts:
    shutil.copytree(scripts[0].parent.parent, project, dirs_exist_ok=True)
elif bundles:
    with zipfile.ZipFile(bundles[0]) as archive:
        archive.extractall(project)
else:
    raise FileNotFoundError(
        "Add Input: nfcorpus-retention-tradeoff-20261020-v1. "
        f"That dataset must contain {bundle_name} or its extracted files."
    )

for required in ("requirements.txt", "experiments/"+script_name,
                 "artifacts/nfcorpus_frozen_retention_routers.json"):
    if not (project / required).is_file():
        raise FileNotFoundError(f"Incomplete source dataset: missing {required}")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(project / "requirements.txt")],
    check=True,
)
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Enable a GPU accelerator in Kaggle settings, then rerun.")

command = [sys.executable, "-u", str(project / "experiments" / script_name), "--repeats", "3"]
for previous in (project / "data",
                 Path("/kaggle/working/nfcorpus_train_dev_v1/data"),
                 Path("/kaggle/working/nfcorpus_train_diagnostics_v1/data")):
    if (previous / "raw" / "nfcorpus" / "corpus.jsonl").is_file():
        command += ["--data-dir", str(previous)]
        print("Reusing corpus/index directory:", previous)
        break

subprocess.run(command, cwd=project, check=True)
output = project / "results" / "nfcorpus_retention_benchmark_outputs.zip"
print("DOWNLOAD AND RETURN:", output)
from IPython.display import FileLink, display
os.chdir("/kaggle/working")
display(FileLink(str(output.relative_to(Path("/kaggle/working")))))
