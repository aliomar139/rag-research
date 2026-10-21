DATASET = "trec-covid"  # First run. Later change to "squad" for a separate run.
from pathlib import Path
from zipfile import ZipFile

root = Path("/kaggle/input")
name = "RUN_PAPER_DATASETS_IN_KAGGLE.py"
scripts = list(root.rglob(name))
bundles = list(root.rglob("paper_datasets_frozen_source_20261020_a7c4_v1.zip"))
if len(scripts) == 1:
    code = scripts[0].read_text(encoding="utf-8")
elif not scripts and len(bundles) == 1:
    with ZipFile(bundles[0]) as archive:
        code = archive.read(name).decode("utf-8")
else:
    raise FileNotFoundError("Attach exactly one copy of paper-datasets-frozen-r90-20261020-a7c4-v1 using Add Input.")
exec(compile(code, name, "exec"))
