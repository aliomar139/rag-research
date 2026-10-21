ACTION = "finalize_original"  # After downloading its result: "indexed_followup"
from pathlib import Path
from zipfile import ZipFile

root = Path("/kaggle/input")
name = "RUN_FAST_FINISH_IN_KAGGLE.py"
scripts = list(root.rglob(name))
bundles = list(root.rglob("fast_finish_indexed_20261021_d4a9_v1.zip"))
if len(scripts) == 1:
    PACKAGE_ROOT = scripts[0].parent
elif not scripts and len(bundles) == 1:
    PACKAGE_ROOT = Path("/kaggle/working/fast_finish_indexed_v1")
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    with ZipFile(bundles[0]) as archive:
        for entry in archive.namelist():
            if "\\" in entry or entry.startswith("/") or ".." in Path(entry).parts:
                raise ValueError("Unexpected archive path")
        archive.extractall(PACKAGE_ROOT)
else:
    raise FileNotFoundError("Attach fast-finish-indexed-20261021-d4a9-v1 using Add Input.")
exec(compile((PACKAGE_ROOT / name).read_text(encoding="utf-8"), name, "exec"))
