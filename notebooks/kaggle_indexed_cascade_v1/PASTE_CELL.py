DATASET = "squad"
QUERY_LIMIT = 256
from pathlib import Path
from zipfile import ZipFile

root = Path("/kaggle/input")
name = "RUN_INDEXED_CASCADE_IN_KAGGLE.py"
scripts = list(root.rglob(name))
bundles = list(root.rglob("indexed_cascade_source_20261021_b8e6_v1.zip"))
if len(scripts) == 1:
    code = scripts[0].read_text(encoding="utf-8")
elif not scripts and len(bundles) == 1:
    with ZipFile(bundles[0]) as archive:
        code = archive.read(name).decode("utf-8")
else:
    raise FileNotFoundError("Attach one copy of indexed-cascade-followup-20261021-b8e6-v1 using Add Input.")
exec(compile(code, name, "exec"))
