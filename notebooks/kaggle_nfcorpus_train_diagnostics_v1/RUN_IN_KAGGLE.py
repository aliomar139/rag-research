# Paste this entire file into ONE Kaggle notebook cell.
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

slug = "nfcorpus-train-router-diagnostics-20261020-v1"
root = Path("/kaggle/input")
candidates = [root / "datasets" / "kira63004" / slug, root / slug]
dataset = next((path for path in candidates if path.is_dir()), None)
if dataset is None:
    dataset = next((path for path in root.rglob(slug) if path.is_dir()), None)
if dataset is None:
    raise FileNotFoundError(f"Use Add Input to attach '{slug}', then rerun this cell.")

script_name = "13_diagnose_nfcorpus_train_router.py"
bundle_name = "nfcorpus_train_diagnostics_source_v1.zip"
project = Path("/kaggle/working/nfcorpus_train_diagnostics_v1")
project.mkdir(parents=True, exist_ok=True)
sources = list(dataset.rglob(script_name))
bundles = list(dataset.rglob(bundle_name))
if sources:
    shutil.copytree(sources[0].parent.parent, project, dirs_exist_ok=True)
elif bundles:
    with zipfile.ZipFile(bundles[0]) as archive:
        archive.extractall(project)
else:
    print("Mounted files:", *sorted(dataset.rglob("*")), sep="\n")
    raise FileNotFoundError(f"Upload {bundle_name} to the diagnostic dataset.")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", str(project / "requirements.txt")],
    check=True,
)
command = [sys.executable, "-u", str(project / "experiments" / script_name)]
# Reuse full-corpus indexes if the earlier Kaggle session is still running.
previous_data = Path("/kaggle/working/nfcorpus_train_dev_v1/data")
if previous_data.is_dir():
    command += ["--data-dir", str(previous_data)]
subprocess.run(command, cwd=project, check=True)
print("DOWNLOAD THIS FILE:", project / "results" / "nfcorpus_train_diagnostics_outputs.zip")
