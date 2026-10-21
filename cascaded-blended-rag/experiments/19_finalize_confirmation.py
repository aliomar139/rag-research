"""Validate the downloaded confirmation and consolidate existing results. No retrieval runs."""
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    downloads = ROOT / "artifacts/downloads"
    archive = downloads / "scidocs_confirmation_outputs.zip"
    if not archive.is_file():
        archive = ROOT / archive.name
    output = PROJECT / "results/scidocs_confirmation"
    with zipfile.ZipFile(archive) as bundle:
        require(bundle.testzip() is None, "ZIP CRC failure")
        for name in bundle.namelist():
            require(Path(name).name == name and "\\" not in name,
                    "Unexpected archive path: " + name)
        report = json.loads(bundle.read("scidocs_confirmation_benchmark.json"))
        protocol = json.loads(bundle.read("scidocs_confirmation_protocol.json"))
        for filename, field in [
            ("nfcorpus_frozen_retention_routers.json", "frozen_router_sha256"),
            ("scidocs_confirmation_protocol.json", "confirmation_protocol_sha256"),
        ]:
            raw = bundle.read(filename)
            require(hashlib.sha256(raw).hexdigest() == report[field], filename + " report hash mismatch")
            require(raw == (PROJECT / "results" / filename).read_bytes(), filename + " differs from prepared asset")
        require(digest(PROJECT / "experiments/18_confirm_scidocs_frozen.py") == report["benchmark_source_sha256"],
                "Benchmark source differs from frozen run")
        require(report["benchmark_source_sha256"] == protocol["benchmark_source_sha256"], "Protocol source mismatch")
        require(report["frozen_router_sha256"] == protocol["frozen_routers_sha256"], "Protocol weights mismatch")
        development_zip = downloads / "nfcorpus_retention_benchmark_outputs.zip"
        if not development_zip.is_file():
            development_zip = ROOT / development_zip.name
        require(digest(development_zip) == protocol["development_result_zip_sha256"],
                "Development ZIP differs from prepared protocol")
        require(report["confirmation_protocol"] == protocol, "Embedded protocol mismatch")
        require(report["queries"] == 1000 and report["corpus_documents"] == 25657 and report["repeats"] == 3,
                "Unexpected evaluation dimensions")
        require(not report["nfcorpus_test_qrels_opened"] and not report["evaluation_grades_used_to_select_policies"],
                "Unexpected label scope")
        require(report["evaluation_relevance_grades_scored_only_after_timing"] and
                report["query_document_self_matches_excluded"], "Unexpected scoring protocol")
        methods = report["methods"]
        require(set(methods) == {"static_rrf", "bm25", "dense", "splade", *[f"ridge11_r{t}" for t in (90,92,95,99)]},
                "Unexpected method set")
        rows = defaultdict(dict)
        calls = defaultdict(lambda: defaultdict(int))
        exits = defaultdict(int)
        for line in bundle.open("retrieval_first_pass.jsonl"):
            row = json.loads(line)
            name, qid = row["method"], row["query_id"]
            require(name in methods and qid not in rows[name], "Unknown method or duplicate query")
            ranking = row["ranking"]
            ids = [doc for doc, score in ranking]
            require(len(ids) == 100 and len(set(ids)) == 100 and qid not in ids, "Invalid ranking IDs")
            require(all(math.isfinite(score) for doc, score in ranking), "Nonfinite score")
            require(0 <= row["recall@100"] <= 1, "Invalid saved recall")
            decision = row["decision"]
            early = decision["route"] == "EARLY_EXIT"
            expected = ({"dense": int(not early), "splade": int(not early)} if name.startswith("ridge")
                        else {"dense": int(name in ("dense", "static_rrf")),
                              "splade": int(name in ("splade", "static_rrf"))})
            require(decision["calls"] == expected, "Branch decision mismatch")
            exits[name] += early
            for branch, count in expected.items():
                calls[name][branch] += count * report["repeats"]
            rows[name][qid] = row["recall@100"]
        all_ids = set(rows["static_rrf"])
        baseline = methods["static_rrf"]
        for name, method in methods.items():
            require(len(rows[name]) == 1000 and set(rows[name]) == all_ids, "Query coverage mismatch")
            require(set(method["timings_per_query_ms"]) == all_ids, "Timing IDs mismatch")
            times = [v for values in method["timings_per_query_ms"].values() for v in values]
            require(all(len(v) == 3 for v in method["timings_per_query_ms"].values()), "Missing timing repetition")
            require(all(math.isfinite(t) and t > 0 for t in times), "Invalid timing")
            require(math.isclose(sum(times)/len(times), method["latency_ms"]["mean"], abs_tol=1e-8), "Mean timing mismatch")
            require(math.isclose(sum(rows[name].values())/1000, method["metrics"]["recall@100"], abs_tol=1e-12), "Recall mismatch")
            require(dict(calls[name]) == method["neural_calls_across_timed_repeats"], "Timed neural call mismatch")
            require(exits[name] == method["early_exit_count"], "Exit count mismatch")
            require(math.isclose(exits[name]/1000, method["early_exit_rate"]), "Exit rate mismatch")
            require(method["rankings_changed_on_later_repeats"] == 0, "Ranking instability reported")
            require(math.isclose(method["metrics"]["recall@100"]/baseline["metrics"]["recall@100"], method["recall_retention"]), "Retention mismatch")
            if "target" in method:
                require(method["observed_target_met"] == (method["recall_retention"] >= method["target"]), "Target verdict mismatch")
        output.mkdir(parents=True, exist_ok=True)
        for name in bundle.namelist():
            (output / name).write_bytes(bundle.read(name))

    verification = {
        "archive_sha256": digest(archive), "first_pass_records": sum(map(len, rows.values())),
        "query_timings": 24000, "frozen_assets_and_source_match": True,
        "saved_recalls_reaggregate": True, "timings_and_branch_counts_match": True,
        "rankings_stable_as_reported": True,
        "scope": "Artifact consistency checks; raw relevance grades and timing instrumentation are not independently rerun.",
    }
    (output / "local_verification.json").write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("combined_results", PROJECT / "experiments/22_import_paper_benchmark.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    destination = helper.combine()
    print(f"Validated 8,000 first-pass records and 24,000 timings. Extracted: {output}")
    print(f"Combined results: {destination}")


if __name__ == "__main__":
    main()
