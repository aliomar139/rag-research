"""Build the upload bundle and frozen extension protocol, without running retrieval."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
DESTINATION = ROOT / "notebooks/kaggle_paper_datasets_v1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    paths = ["experiments/12_calibrate_nfcorpus_train_check_dev.py",
             "experiments/20_benchmark_paper_datasets.py"]
    paths += ["src/"+name for name in (
        "__init__.py", "bm25_retriever.py", "dense_retriever.py", "splade_retriever.py",
        "cache_utils.py", "bm25_coverage_features.py", "retention_tradeoff.py",
        "tradeoff_analysis.py", "router_diagnostics.py", "query_self_exclusion.py", "paper_datasets.py")]
    entries = {relative: (PROJECT / relative).read_bytes() for relative in paths}
    for relative, raw in entries.items():
        ast.parse(raw.decode("utf-8"), filename=relative)
    launcher = (DESTINATION / "RUN_IN_KAGGLE.py").read_bytes()
    ast.parse(launcher.decode("utf-8"))
    weights = (PROJECT / "results/nfcorpus_frozen_retention_routers.json").read_bytes()
    frozen = json.loads(weights)
    for name, expected in frozen["provenance"]["retrieval_manifest"]["retrieval_code_sha256"].items():
        if "src/"+name in entries and sha(entries["src/"+name]) != expected:
            raise ValueError("Frozen retrieval source mismatch: "+name)
    protocol = {
        "schema": 1, "prepared_date_client": "2026-10-20",
        "experiment": "Original-paper dataset extension after NFCorpus/SCIDOCS analysis",
        "paper": "https://arxiv.org/abs/2404.07220",
        "datasets": ["trec-covid", "squad", "nq", "hotpotqa"],
        "default_run": "trec-covid", "training_dataset": "BEIR NFCorpus train",
        "families": ["ridge11"], "targets": frozen["targets"],
        "primary_candidate_target": .90, "timing_repeats": 3,
        "frozen_routers_sha256": sha(weights),
        "source_sha256": {name: sha(raw) for name, raw in entries.items()},
        "model_revisions": {"dense": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                            "splade": "49cf4c7b0db5b870a401ddf5e2669993ef3699c7"},
        "evaluation_scope": "All evaluation queries and full declared corpus per dataset; no qrel-filtered or reduced corpus",
        "squad_scope": "Official 1.1 train+dev paragraphs, all dev questions; parent-paragraph relevance; retrieval only",
        "selection_reason": "User requests evaluation on original-paper datasets at a 90% floor; primary chosen before these new outcomes",
        "prior_inspection": "Earlier exploratory NQ/static TREC-COVID/SQuAD results exist; this is a new frozen controlled evaluation, not a claim that all dataset families were never inspected",
        "outcome_rules": [
            "No training or threshold adjustment on evaluation grades.",
            "Report all four thresholds and every dataset attempted, including failures and resource limits.",
            "Assess the primary nominal 90% policy separately from secondary policies; do not pick a new primary after outcomes.",
            "Report recall/nDCG/MAP, top-k hit metrics, real repeated latency, neural branch counts, bootstrap intervals and standalone baselines.",
            "A point retention of 90% does not imply a confidence bound or per-query guarantee.",
            "Savings against this full-fusion implementation do not establish a major contribution, original-paper speedup or superiority over single retrievers.",
            "Keep earlier unfavorable NFCorpus/SCIDOCS findings visible; exact ELSER reproduction and answer quality are not evaluated.",
        ],
    }
    protocol_raw = (json.dumps(protocol, indent=2, allow_nan=False)+"\n").encode("utf-8")
    (PROJECT / "results/paper_datasets_protocol.json").write_bytes(protocol_raw)
    entries.update({"artifacts/nfcorpus_frozen_retention_routers.json": weights,
                    "artifacts/paper_datasets_protocol.json": protocol_raw,
                    "RUN_PAPER_DATASETS_IN_KAGGLE.py": launcher,
                    "README.md": (DESTINATION / "START_HERE.md").read_bytes()})
    requirements = (ROOT / "notebooks/kaggle_scidocs_confirmation_v1/requirements.txt").read_text(encoding="utf-8")
    if "threadpoolctl" not in requirements:
        requirements = requirements.rstrip()+"\nthreadpoolctl\n"
    (DESTINATION / "requirements.txt").write_text(requirements, encoding="utf-8")
    entries["requirements.txt"] = requirements.encode("utf-8")
    manifest = {name: sha(raw) for name, raw in entries.items()}
    entries["package_manifest.json"] = (json.dumps(manifest, indent=2)+"\n").encode("utf-8")
    target = DESTINATION / "paper_datasets_frozen_source_20261020_a7c4_v1.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in entries.items():
            if "\\" in name:
                raise ValueError("Windows path separator in archive name")
            archive.writestr(name, raw)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC failure")
        for name, expected in manifest.items():
            if sha(archive.read(name)) != expected:
                raise ValueError("Package manifest mismatch: "+name)
    print(f"UPLOAD: {target}\nBYTES: {target.stat().st_size}\nSHA256: {sha(target.read_bytes())}")
    print("KAGGLE TITLE: paper-datasets-frozen-r90-20261020-a7c4-v1")
    print("Syntax and package checks completed; no datasets downloaded or retrieval experiments run.")


if __name__ == "__main__":
    main()
