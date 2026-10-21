"""Small standard-library helpers for shareable-file selection."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git_files():
    if not (ROOT / ".git").exists():
        manifest = ROOT / "SUBMISSION_MANIFEST.json"
        if not manifest.is_file():
            raise RuntimeError("Use a Git checkout or an extracted submission ZIP with its manifest.")
        names = [entry["path"] for entry in json.loads(manifest.read_text(encoding="utf-8"))["files"]]
        if len(names) != len(set(names)):
            raise RuntimeError("Submission manifest contains duplicate paths.")
        for name in names:
            relative = PurePosixPath(name)
            if (not name or "\\" in name or ":" in name or relative.is_absolute()
                    or ".." in relative.parts or relative.as_posix() != name
                    or not (ROOT / name).resolve().is_relative_to(ROOT)):
                raise RuntimeError("Invalid submission manifest path.")
            if not (ROOT / name).is_file():
                raise RuntimeError(f"Submission file is missing: {name}")
        return sorted(names)
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    names = sorted({name for name in raw.decode("utf-8").split("\0") if name and (ROOT / name).is_file()})
    if not names:
        return []
    result = subprocess.run(["git", "check-ignore", "--no-index", "--stdin", "-z"],
                            cwd=ROOT, input=("\0".join(names)+"\0").encode(), capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.decode())
    ignored = set(result.stdout.decode("utf-8").split("\0"))
    return [name for name in names if name not in ignored]


def ignored_tracked_files():
    if not (ROOT / ".git").exists():
        return []
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    result = subprocess.run(["git", "check-ignore", "--no-index", "--stdin", "-z"],
                            cwd=ROOT, input=tracked, capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.decode())
    return sorted(name for name in result.stdout.decode().split("\0") if name)
