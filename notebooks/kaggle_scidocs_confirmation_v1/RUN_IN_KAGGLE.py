# Paste into one Kaggle cell. Runs the frozen confirmation without retraining.
from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile

root = Path("/kaggle/input")
script_name = "18_confirm_scidocs_frozen.py"
bundle_name = "scidocs_frozen_confirmation_source_v1.zip"
project = Path("/kaggle/working/scidocs_frozen_confirmation_v1")
project.mkdir(parents=True, exist_ok=True)
scripts = list(root.rglob(script_name))
bundles = list(root.rglob(bundle_name))
if scripts:
    shutil.copytree(scripts[0].parent.parent, project, dirs_exist_ok=True)
elif bundles:
    with zipfile.ZipFile(bundles[0]) as archive:
        archive.extractall(project)
else:
    raise FileNotFoundError(
        "Attach scidocs-frozen-cascade-confirmation-20261020-v1 using Add Input."
    )

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(project / "requirements.txt")],
    check=True,
)
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Enable GPU in Kaggle settings, then rerun.")

subprocess.run(
    [sys.executable, "-u", str(project / "experiments" / script_name), "--repeats", "3"],
    cwd=project,
    check=True,
)
output = project / "results" / "scidocs_confirmation_outputs.zip"
print("DOWNLOAD AND RETURN:", output)
from IPython.display import FileLink, display
os.chdir("/kaggle/working")
display(FileLink(str(output.relative_to(Path("/kaggle/working")))))
