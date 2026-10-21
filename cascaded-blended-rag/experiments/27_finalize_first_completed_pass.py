"""Finalize the original SQuAD checkpoint after its first full timing pass.

No retrieval or document encoding. Reuses the checksum-verified original
runner's scoring/export code, explicitly records the shortened protocol and
preserves later partial repetitions as supplemental evidence.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stop_matching_runner(script):
    """Stop only a lingering subprocess executing this exact original script."""
    stopped = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            if str(script).encode() in args:
                os.kill(int(entry.name), signal.SIGTERM)
                stopped.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    for _ in range(50):
        if not any((Path("/proc") / str(pid)).exists() for pid in stopped):
            break
        time.sleep(.1)
    for pid in stopped:
        try:
            state = (Path("/proc") / str(pid) / "stat").read_text().split(")", 1)[1].split()[0]
            if state != "Z":
                raise RuntimeError("The original subprocess has not stopped yet. Wait a few seconds and rerun.")
        except FileNotFoundError:
            pass
    return stopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/kaggle/working/paper_datasets_frozen_v1"))
    args = parser.parse_args()
    project = args.project
    old = project / "results/paper_squad"
    full_zip = old / "squad_paper_benchmark_outputs.zip"
    if full_zip.is_file():
        try:
            with zipfile.ZipFile(full_zip) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Existing full-run archive failed CRC validation")
        except zipfile.BadZipFile as error:
            raise RuntimeError("Original runner is writing its result archive. Wait a minute and rerun.") from error
        print("The original full run already finished. Download:", full_zip, flush=True)
        return
    script = project / "experiments/20_benchmark_paper_datasets.py"
    protocol_path = project / "artifacts/paper_datasets_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha(script) != protocol["source_sha256"]["experiments/20_benchmark_paper_datasets.py"]:
        raise ValueError("Original runner fingerprint differs from its protocol")
    stopped = stop_matching_runner(script)
    if stopped:
        print("Stopped lingering original runner processes:", stopped, flush=True)
    # Imports define functions/classes only. No retriever constructor is called.
    spec = importlib.util.spec_from_file_location("original_checkpoint_export", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch.set_num_threads(1)
    base = module.base_module()
    identity = json.loads((old / "run_identity.json").read_text(encoding="utf-8"))
    if identity["runner_sha256"] != sha(script) or identity["protocol_sha256"] != sha(protocol_path):
        raise ValueError("Checkpoint identity mismatch")
    dataset_dir = project / "data/raw/squad"
    corpus_path = dataset_dir / "corpus.jsonl"
    if identity["corpus_sha256"] != sha(corpus_path):
        raise ValueError("Corpus differs from the timed run")
    corpus = base.load_corpus(dataset_dir)
    queries, qrels_path = base.load_split(dataset_dir, "test")  # Query IDs, no grades.
    if len(queries) != 10570 or sha(dataset_dir / "queries.jsonl") != identity["query_file_sha256"]:
        raise ValueError("Expected all 10,570 original SQuAD queries")
    frozen_path = project / "artifacts/nfcorpus_frozen_retention_routers.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if sha(frozen_path) != identity["weights_sha256"]:
        raise ValueError("Frozen gate differs from the timed run")
    engines = dict.fromkeys(("static_rrf", "bm25", "dense", "splade", "ridge11_r90", "ridge11_r92", "ridge11_r95", "ridge11_r99"))
    rankings = {name: {} for name in engines}
    decisions = {name: {} for name in engines}
    timings = {name: defaultdict(list) for name in engines}
    counts = {name: {"dense": 0, "splade": 0} for name in engines}
    changed = dict.fromkeys(engines, 0)
    later_changes = dict.fromkeys(engines, 0)
    seen = set()
    repetition_counts = defaultdict(int)
    output = project / "results/paper_squad_one_pass"
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "additional_repetitions.jsonl"
    checkpoint = old / "timed_queries.jsonl"
    digest = hashlib.sha256()
    ignored_incomplete_tail = False
    with checkpoint.open("rb") as handle, partial_path.open("wb") as extra:
        for line in handle:
            digest.update(line)
            if not line.endswith(b"\n"):
                ignored_incomplete_tail = True
                continue
            row = json.loads(line)
            repeat, qid = row["repeat"], row["query_id"]
            if qid not in queries or (repeat, qid) in seen or set(row["methods"]) != set(engines):
                raise ValueError("Invalid checkpoint query or method coverage")
            seen.add((repeat, qid))
            repetition_counts[repeat] += 1
            if repeat != 0:
                extra.write(line)
                for name, record in row["methods"].items():
                    if qid not in rankings[name]:
                        raise ValueError("Later repeat occurs before the first-pass ranking")
                    if [doc for doc, _ in record["ranking"]] != [doc for doc, _ in rankings[name][qid]]:
                        later_changes[name] += 1
                continue
            for name, record in row["methods"].items():
                rankings[name][qid] = record["ranking"]
                decisions[name][qid] = record["decision"]
                timings[name][qid].append(record["elapsed_ms"])
                for branch, count in record["decision"]["calls"].items():
                    counts[name][branch] += count
            if repetition_counts[0] % 2000 == 0:
                print(f"Loaded completed first pass: {repetition_counts[0]}/10570", flush=True)
    if repetition_counts[0] != len(queries) or any(set(rows) != set(queries) for rows in rankings.values()):
        raise ValueError("The first pass is incomplete; no full-query result will be created")
    amendment = {
        "status": "shortened_for_runtime_before_quality_scoring",
        "planned_timing_repeats": protocol["timing_repeats"], "scored_timing_repeats": 1,
        "completed_query_counts_by_repeat": dict(repetition_counts),
        "reason": "Preserve the completed full-query quality pass and move to indexed execution without spending additional hours on slow repeated timings. Selected before inspecting quality outcomes.",
        "additional_repetitions_preserved": True,
        "rankings_changed_in_observed_additional_repetitions": later_changes,
        "ignored_incomplete_last_checkpoint_line": ignored_incomplete_tail,
        "original_checkpoint_sha256": digest.hexdigest(),
        "original_runner_sha256": sha(script), "finalizer_sha256": sha(Path(__file__)),
        "scope": "All questions scored from first-pass rankings. Latency has one complete pass; bootstrap intervals reflect queries, not repeated-run stability. Later partial timings are supplemental, not pooled into headline means.",
    }
    # Write the runtime amendment before the original export code reads grades.
    module.write_json(output / "execution_amendment.json", amendment)
    for name in ("run_identity.json", "timing_sessions.jsonl"):
        shutil.copyfile(old / name, output / name)
    policy_details = {}
    router = frozen["routers"]["ridge11"]
    for key, policy in router["policies"].items():
        policy_details["ridge11_r"+key] = {"family": "ridge11", "target": policy["target"], "threshold": policy["threshold"]}
    env = dict(module.__dict__)
    env.update({
        "args": SimpleNamespace(dataset="squad", repeats=1, output_dir=output), "base": base,
        "frozen_path": frozen_path, "frozen": frozen, "protocol_path": protocol_path, "protocol": protocol,
        "dataset_info": {"dataset": "SQuAD 1.1 paragraph retrieval (one timing pass)", "split": "all official dev questions",
                         "expected_queries": len(queries), "expected_documents": len(corpus),
                         "corpus_scope": "Full original train+dev paragraph corpus; unchanged from timed run"},
        "corpus_path": corpus_path, "corpus": corpus, "queries": queries, "qrels_path": qrels_path,
        "preparation_seconds": None, "engines": engines, "policy_details": policy_details,
        "rankings": rankings, "decisions": decisions, "timings": timings,
        "counts": counts, "changed_rankings": changed, "sessions_path": output / "timing_sessions.jsonl",
        "dense": SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_commit_hash=identity["dense_revision"]))),
        "splade": SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_commit_hash=identity["splade_revision"]))),
    })
    tree = ast.parse(script.read_text(encoding="utf-8"))
    benchmark = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "benchmark")
    start = next(i for i, node in enumerate(benchmark.body)
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "qrels" for target in node.targets))
    export = ast.Module(body=benchmark.body[start:], type_ignores=[])
    ast.fix_missing_locations(export)
    print("Scoring the completed first pass. No neural inference or document encoding.", flush=True)
    with module.threadpool_limits(limits=1):
        exec(compile(export, str(script)+":checkpoint_export", "exec"), env)
    report_path = output / "paper_dataset_benchmark.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["execution_amendment"] = amendment
    report["timing_scope"] = "One completed pass; planned three-pass protocol shortened before quality scoring"
    report["index_preparation_seconds"] = None
    report["protocol"].append(amendment["scope"])
    module.write_json(report_path, report)
    destination = output / "squad_one_pass_outputs.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("paper_dataset_benchmark.json", "quality_speed.csv", "retrieval_first_pass.jsonl",
                     "run_identity.json", "timing_sessions.jsonl", "execution_amendment.json",
                     "additional_repetitions.jsonl", "quality_speed.png", "quality_speed.svg"):
            archive.write(output / name, arcname=name)
        archive.write(frozen_path, arcname=frozen_path.name)
        archive.write(protocol_path, arcname=protocol_path.name)
    # Remove only the provisional, unamended ZIP created in this new folder.
    # The original checkpoint/result folder is not modified.
    provisional = output / "squad_paper_benchmark_outputs.zip"
    if provisional.is_file():
        provisional.unlink()
    print("DOWNLOAD THIS AMENDED RESULT:", destination, flush=True)


if __name__ == "__main__":
    main()
