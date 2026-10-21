# Run only after the current original SQuAD job has finished and been downloaded.
DATASET = globals().get("DATASET", "squad")
QUERY_LIMIT = globals().get("QUERY_LIMIT", 256)

from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile

original = Path("/kaggle/working/paper_datasets_frozen_v1")
previous = original / "results" / ("paper_"+DATASET)
previous_zip = previous / (DATASET.replace("-", "_")+"_paper_benchmark_outputs.zip")
cache = original / "data" / ("processed_paper_"+DATASET)
required = [previous_zip, previous / "retrieval_first_pass.jsonl",
            cache / "dense_doc_embeddings.npy", cache / "scifact_splade_index.npz",
            cache / "dense_doc_embeddings.npy.json", cache / "scifact_splade_index.npz.json"]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(
        "This short follow-up needs the completed original run and its corpus indexes in the SAME Kaggle session. "
        "Finish and download that result first. Missing: "+", ".join(missing)
    )
project = Path("/kaggle/working/indexed_cascade_followup_v1")
project.mkdir(parents=True, exist_ok=True)
root = Path("/kaggle/input")
script_name = "25_benchmark_indexed_cascade.py"
bundle_name = "indexed_cascade_source_20261021_b8e6_v1.zip"
scripts, bundles = sorted(root.rglob(script_name)), sorted(root.rglob(bundle_name))
if len(scripts) == 1:
    shutil.copytree(scripts[0].parent.parent, project, dirs_exist_ok=True)
elif not scripts and len(bundles) == 1:
    with zipfile.ZipFile(bundles[0]) as archive:
        for name in archive.namelist():
            if "\\" in name or name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("Unsafe archive entry")
        archive.extractall(project)
else:
    raise FileNotFoundError("Attach one copy of indexed-cascade-followup-20261021-b8e6-v1 using Add Input.")
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Keep the GPU enabled.")
output_dir = project / "results" / f"indexed_{DATASET}_q{QUERY_LIMIT}"
subprocess.run([
    sys.executable, "-u", str(project / "experiments" / script_name),
    "--dataset", DATASET, "--query-limit", str(QUERY_LIMIT), "--repeats", "3",
    "--data-dir", str(original / "data"), "--output-dir", str(output_dir),
    "--reference-first-pass", str(previous / "retrieval_first_pass.jsonl"),
    "--require-existing-neural-cache",
], cwd=project, check=True)
output = output_dir / (DATASET.replace("-", "_")+"_indexed_cascade_outputs.zip")
print("DOWNLOAD AND RETURN:", output)
from IPython.display import FileLink, display
os.chdir("/kaggle/working")
display(FileLink(str(output.relative_to(Path("/kaggle/working")))))
