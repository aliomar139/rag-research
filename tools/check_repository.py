"""Check sharing hygiene, navigation and frozen bytes. Does not run model tests."""
import argparse
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote

from _repository import ROOT, git_files, ignored_tracked_files, sha256

MAX_FILE_BYTES = 10 * 1024 * 1024
LINK = re.compile(r"!?\[[^\]\n]*\]\((<[^>\n]+>|[^\s)]+)(?:\s+\"[^\"\n]*\")?\)")
SECRETS = {
    "credential_token": re.compile(r"(?:hf_|ghp_|github_pat_|sk-proj-)[A-Za-z0-9_-]{24,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}


def audit():
    names = git_files()
    shared = set(names)
    errors, notices = [], []
    protected = {}
    results = ROOT / "cascaded-blended-rag/results"
    for filename in ("paper_datasets_protocol.json", "indexed_cascade_protocol.json"):
        protocol = json.loads((results / filename).read_text(encoding="utf-8"))
        for relative, expected in protocol["source_sha256"].items():
            source = ROOT / "cascaded-blended-rag" / relative
            actual = sha256(source)
            protected["cascaded-blended-rag/"+relative] = actual
            if actual != expected:
                errors.append({"type": "frozen_source_changed", "path": relative})
        weights = results / "nfcorpus_frozen_retention_routers.json"
        if sha256(weights) != protocol["frozen_routers_sha256"]:
            errors.append({"type": "frozen_weights_changed", "path": str(weights.relative_to(ROOT))})
    scidocs = json.loads((results / "scidocs_confirmation_protocol.json").read_text())
    source = ROOT / "cascaded-blended-rag/experiments/18_confirm_scidocs_frozen.py"
    if sha256(source) != scidocs["benchmark_source_sha256"]:
        errors.append({"type": "frozen_source_changed", "path": str(source.relative_to(ROOT))})
    protected[source.relative_to(ROOT).as_posix()] = sha256(source)
    for name in names:
        path = ROOT / name
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append({"type": "file_over_10_MiB", "path": name, "bytes": path.stat().st_size})
        if path.suffix.lower() not in {".py", ".md", ".json", ".jsonl", ".ipynb", ".txt", ".csv", ".tsv", ".yml", ".yaml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            errors.append({"type": "invalid_text_encoding", "path": name})
            continue
        for rule, pattern in SECRETS.items():
            if pattern.search(text):
                errors.append({"type": "possible_secret", "path": name, "rule": rule})
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=name)
            except SyntaxError as exc:
                errors.append({"type": "syntax_error", "path": name, "line": exc.lineno})
        if path.suffix != ".md":
            continue
        historical = name.startswith("docs/history/") or name.startswith("notebooks/legacy/")
        for match in LINK.finditer(text):
            target = match.group(1).strip("<>")
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
                if re.match(r"^[A-Za-z]:[/\\]", target):
                    (notices if historical else errors).append({"type": "absolute_local_link", "path": name, "target": target})
                continue
            relative = unquote(target.split("#", 1)[0])
            if not relative:
                continue
            resolved = (path.parent / relative).resolve()
            try:
                linked = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                (notices if historical else errors).append({"type": "link_outside_repository", "path": name, "target": target})
                continue
            if resolved.is_dir() and any(item.startswith(linked.rstrip("/")+"/") for item in names):
                continue
            if linked not in shared:
                issue = {"type": "link_not_in_submission", "path": name, "target": target}
                (notices if historical else errors).append(issue)
    tracked = ignored_tracked_files()
    if tracked:
        errors.append({"type": "ignored_files_still_tracked", "paths": tracked})
    return {"shared_files": len(names), "shared_bytes": sum((ROOT / n).stat().st_size for n in names),
            "max_file_bytes": MAX_FILE_BYTES, "frozen_source_sha256": protected,
            "errors": errors, "historical_navigation_notices": notices,
            "scope": "File/secret-pattern/navigation/AST/fingerprint checks only. No unit tests, retrieval or timing experiments run."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/submission/repository_check.json")
    args = parser.parse_args()
    result = audit()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(f"Shareable files: {result['shared_files']}; bytes: {result['shared_bytes']:,}")
    print(f"Frozen source files checked: {len(result['frozen_source_sha256'])}")
    print(f"Errors: {len(result['errors'])}; historical notices: {len(result['historical_navigation_notices'])}")
    for issue in result["errors"][:35]:
        print(json.dumps(issue))
    print("Report:", args.report)
    raise SystemExit(bool(result["errors"]))


if __name__ == "__main__":
    main()
