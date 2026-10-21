# NFCorpus train → dev run

## Upload this file

Upload `cascade_source_bundle_nfcorpus_train_dev_v1.zip` from this folder. Do
not upload the older `cascade_source_bundle_nfcorpus_ridge_v1.zip`.

Use this Kaggle dataset title/slug:

`nfcorpus-train-calibrated-dev-check-20261019-v2`

If that dataset already exists, add the corrected ZIP and create a new version.
Keep `cascade-source-bundle-nfcorpus-ridge-v1` unchanged.

## Start fresh in Kaggle

1. Create or open a Kaggle notebook and attach the v2 dataset above.
2. Turn on Internet. Select a GPU accelerator if available.
3. Run the cell below. It accepts either the ZIP file or Kaggle's extracted
   files, checks that the new script is present, copies files to writable
   `/kaggle/working`, installs requirements, and runs the evaluator.

```python
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

dataset = Path(
    "/kaggle/input/datasets/kira63004/"
    "nfcorpus-train-calibrated-dev-check-20261019-v2"
)
script_name = "12_calibrate_nfcorpus_train_check_dev.py"
bundle_name = "cascade_source_bundle_nfcorpus_train_dev_v1.zip"
project = Path("/kaggle/working/nfcorpus_train_dev_v1")
project.mkdir(parents=True, exist_ok=True)

scripts = list(dataset.rglob(script_name))
bundles = list(dataset.rglob(bundle_name))

if scripts:
    source_project = scripts[0].parent.parent
    shutil.copytree(source_project, project, dirs_exist_ok=True)
elif bundles:
    with zipfile.ZipFile(bundles[0]) as archive:
        bad_names = [name for name in archive.namelist() if "\\" in name]
        if bad_names:
            raise ValueError(f"ZIP paths must use forward slashes: {bad_names}")
        archive.extractall(project)
else:
    print("Files found in the attached dataset:")
    for path in sorted(dataset.rglob("*")):
        print(path)
    raise FileNotFoundError(
        f"Missing {script_name} and {bundle_name}. "
        "The attached dataset version needs the corrected source ZIP."
    )

script = project / "experiments" / script_name
requirements = project / "requirements_nfcorpus_train_dev.txt"
if not script.is_file() or not requirements.is_file():
    raise FileNotFoundError(
        f"Expected the new script and requirements in {project}. "
        "Check that the v2 dataset contains the corrected ZIP."
    )

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
    check=True,
)
subprocess.run([sys.executable, str(script)], cwd=project, check=True)

result = project / "results" / "cascaded-blended-rag/results/nfcorpus_train_calibrated_dev.json"
print(f"Result file: {result}")
```

The result is saved at
`/kaggle/working/nfcorpus_train_dev_v1/results/cascaded-blended-rag/results/nfcorpus_train_calibrated_dev.json`.
The script calibrates on NFCorpus `train`, checks on `dev`, and does not read
`test.tsv`.
