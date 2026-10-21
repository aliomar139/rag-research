"""Package the indexed execution follow-up; no retrieval or neural model runs."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
DESTINATION = ROOT / "notebooks/kaggle_indexed_cascade_v1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    paths = ["experiments/12_calibrate_nfcorpus_train_check_dev.py", "experiments/25_benchmark_indexed_cascade.py"]
    paths += ["src/"+name for name in (
        "__init__.py", "bm25_retriever.py", "dense_retriever.py", "splade_retriever.py",
        "cache_utils.py", "bm25_coverage_features.py", "retention_tradeoff.py",
        "tradeoff_analysis.py", "router_diagnostics.py", "query_self_exclusion.py",
        "paper_datasets.py", "indexed_bm25.py")]
    entries = {name: (PROJECT / name).read_bytes() for name in paths}
    for name, raw in entries.items():
        ast.parse(raw.decode("utf-8"), filename=name)
    for name in ("RUN_IN_KAGGLE.py", "PASTE_CELL.py"):
        ast.parse((DESTINATION / name).read_text(encoding="utf-8"), filename=name)
    weights = (PROJECT / "results/nfcorpus_frozen_retention_routers.json").read_bytes()
    frozen = json.loads(weights)
    for name, expected in frozen["provenance"]["retrieval_manifest"]["retrieval_code_sha256"].items():
        if "src/"+name in entries and sha(entries["src/"+name]) != expected:
            raise ValueError("Original retrieval source changed: "+name)
    protocol = {
        "schema": 1, "prepared_date_client": "2026-10-21",
        "experiment": "Indexed-BM25 execution follow-up on inspected datasets",
        "datasets": ["trec-covid", "squad"], "default_run": "squad",
        "timing_repeats": 3, "default_query_limit": 256,
        "query_selection": "Sort query IDs by SHA256(dataset + NUL + query_id), take first limit. 0 means all queries. Full document corpus retained.",
        "training_dataset": "BEIR NFCorpus train", "families": ["ridge11"],
        "targets": frozen["targets"], "primary_candidate_target": .99,
        "primary_policy": "ridge11_r99", "required_retention": .90,
        "primary_selection_reason": "Existing conservative threshold chosen using inspected NFCorpus/SCIDOCS/TREC-COVID evidence before the indexed follow-up or current full SQuAD outcomes are inspected. Calibration label 99% is distinct from the required 90% floor. Earlier experiments retain their original primary policies.",
        "frozen_routers_sha256": sha(weights),
        "source_sha256": {name: sha(raw) for name, raw in entries.items()},
        "model_revisions": {"dense": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                            "splade": "49cf4c7b0db5b870a401ddf5e2669993ef3699c7"},
        "change": "Precompute the existing BM25 field term contributions into postings. No scorer formula, gate, neural model or threshold changes.",
        "reproduction_scope": "Our MiniLM/SPLADE/RRF adaptation, not exact original-paper Elasticsearch/ELSER reproduction",
        "outcome_rules": [
            "Measure optimized static fusion, optimized standalone BM25 and every cascade with the same scorer, plus Dense and SPLADE.",
            "Report actual sampled query count and all available query count. Do not label sampled quality as full-query quality.",
            "No threshold tuning on the sample. Primary frozen r99 is assessed against a 90% floor; keep its original 99% calibration-label verdict too. Report every secondary policy.",
            "Compare prior BM25 top100 IDs and scores after timing; do not serve old rankings as timed inference results.",
            "Report quality, latency, branch counts and uncertainty. No end-to-end speedup is inferred from local BM25 microbenchmarks.",
            "Retain all earlier unfavorable results and standalone baselines. Engineering optimization does not establish new-method priority.",
        ],
    }
    protocol_raw = (json.dumps(protocol, indent=2)+"\n").encode("utf-8")
    (PROJECT / "results/indexed_cascade_protocol.json").write_bytes(protocol_raw)
    entries.update({"artifacts/indexed_cascade_protocol.json": protocol_raw,
                    "artifacts/nfcorpus_frozen_retention_routers.json": weights,
                    "RUN_INDEXED_CASCADE_IN_KAGGLE.py": (DESTINATION / "RUN_IN_KAGGLE.py").read_bytes(),
                    "README.md": (DESTINATION / "START_HERE.md").read_bytes()})
    for dataset in ("trec-covid", "nfcorpus"):
        path = PROJECT / "results" / ("indexed_bm25_cpu_"+dataset+".json")
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            summary = {key: value for key, value in report.items() if key != "timings_per_query_ms"}
            entries["artifacts/"+path.name] = (json.dumps(summary, indent=2)+"\n").encode("utf-8")
    entries["requirements.txt"] = (ROOT / "notebooks/kaggle_paper_datasets_v1/requirements.txt").read_bytes()
    manifest = {name: sha(raw) for name, raw in entries.items()}
    entries["package_manifest.json"] = (json.dumps(manifest, indent=2)+"\n").encode("utf-8")
    output = DESTINATION / "indexed_cascade_source_20261021_b8e6_v1.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in entries.items():
            if "\\" in name:
                raise ValueError("Invalid archive path")
            archive.writestr(name, raw)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        for name, expected in manifest.items():
            if sha(archive.read(name)) != expected:
                raise ValueError("Manifest mismatch: "+name)
    print("UPLOAD:", output)
    print("KAGGLE TITLE: indexed-cascade-followup-20261021-b8e6-v1")
    print("BYTES:", output.stat().st_size)
    print("SHA256:", sha(output.read_bytes()))
    print("Package/syntax checks completed; end-to-end GPU follow-up not executed.")


if __name__ == "__main__":
    main()
