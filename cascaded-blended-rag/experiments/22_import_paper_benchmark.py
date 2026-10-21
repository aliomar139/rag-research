"""Validate an original/indexed paper-dataset result and combine metrics.

No retrieval, training, dataset download or timing experiment is performed.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def close(first, second):
    return math.isclose(first, second, abs_tol=1e-10)


def import_result(archive_path):
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        archived = ROOT / "artifacts/downloads" / archive_path.name
        if not archived.is_file():
            raise FileNotFoundError(f"Result ZIP missing: {archive_path}. Place the downloaded export in {archived.parent}.")
        archive_path = archived
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, "ZIP CRC failure")
        for name in archive.namelist():
            require(Path(name).name == name and "\\" not in name, "Unexpected archive entry")
        indexed = "indexed_cascade_benchmark.json" in archive.namelist()
        report_name = "indexed_cascade_benchmark.json" if indexed else "paper_dataset_benchmark.json"
        protocol_name = "indexed_cascade_protocol.json" if indexed else "paper_datasets_protocol.json"
        runner = "25_benchmark_indexed_cascade.py" if indexed else "20_benchmark_paper_datasets.py"
        report = json.loads(archive.read(report_name))
        protocol_raw = archive.read(protocol_name)
        protocol = json.loads(protocol_raw)
        require(protocol_raw == (PROJECT / "results" / protocol_name).read_bytes(), "Prepared protocol changed")
        require(sha(protocol_raw) == report["extension_protocol_sha256"], "Protocol hash mismatch")
        require(report["extension_protocol"] == protocol, "Embedded protocol mismatch")
        weights = archive.read("nfcorpus_frozen_retention_routers.json")
        require(weights == (PROJECT / "results/nfcorpus_frozen_retention_routers.json").read_bytes(), "Frozen weights changed")
        require(sha(weights) == report["frozen_router_sha256"] == protocol["frozen_routers_sha256"], "Weights hash mismatch")
        for relative, expected in protocol["source_sha256"].items():
            require(sha((PROJECT / relative).read_bytes()) == expected, "Source changed: "+relative)
        require(report["benchmark_source_sha256"] == protocol["source_sha256"]["experiments/"+runner], "Runner fingerprint mismatch")
        require(report["dataset_key"] in protocol["datasets"], "Unexpected dataset")
        repeats = report["repeats"]
        amendment = report.get("execution_amendment")
        if amendment:
            require(amendment == json.loads(archive.read("execution_amendment.json")), "Runtime amendment mismatch")
            require(amendment["status"] == "shortened_for_runtime_before_quality_scoring"
                    and amendment["planned_timing_repeats"] == protocol["timing_repeats"] == 3
                    and amendment["scored_timing_repeats"] == repeats == 1, "Unexpected shortened-run scope")
            require(amendment["finalizer_sha256"] == sha((PROJECT / "experiments/27_finalize_first_completed_pass.py").read_bytes()),
                    "Finalizer source fingerprint mismatch")
        else:
            require(repeats == protocol["timing_repeats"] == 3, "Unexpected repetitions")
        info = report["dataset_preparation"]
        n = report["queries"]
        if indexed:
            total, limit = report["all_available_queries"], report["query_limit"]
            require(total == info["expected_queries"] and limit >= 0
                    and n == (min(limit, total) if limit else total), "Query sample size mismatch")
            require(len(report["selected_query_ids"]) == len(set(report["selected_query_ids"])) == n,
                    "Selected query IDs invalid")
            require(report["primary_outcome"]["policy"] == protocol["primary_policy"]
                    and report["primary_outcome"]["required_retention"] == protocol["required_retention"],
                    "Prospective primary changed")
        else:
            require(n == info["expected_queries"], "Query count mismatch")
        require(report["corpus_documents"] == info["expected_documents"], "Corpus size mismatch")
        require(not report["evaluation_grades_used_to_select_policies"] and not report["nfcorpus_test_qrels_opened"], "Unexpected label use")
        require(report["evaluation_relevance_grades_scored_only_after_timing"], "Unexpected scoring protocol")
        identity = json.loads(archive.read("run_identity.json"))
        require(identity["corpus_sha256"] == report["corpus_sha256"] and identity["weights_sha256"] == sha(weights)
                and identity["protocol_sha256"] == sha(protocol_raw)
                and identity["runner_sha256"] == report["benchmark_source_sha256"], "Run identity mismatch")
        require(identity["repeats"] == repeats, "Run identity repetition mismatch")
        if indexed:
            require(identity["query_ids"] == report["selected_query_ids"]
                    and identity["query_limit"] == report["query_limit"], "Run identity sample mismatch")
        for model, revision in protocol["model_revisions"].items():
            require(report["hardware"][model+"_model_revision"] == revision, "Model revision changed")
        sessions = [json.loads(line) for line in archive.read("timing_sessions.jsonl").splitlines()]
        require(sessions == report["timing_sessions"], "Timing session mismatch")
        methods = report["methods"]
        require(set(methods) == {"static_rrf", "bm25", "dense", "splade", *[f"ridge11_r{t}" for t in (90,92,95,99)]}, "Method set mismatch")
        rows = defaultdict(dict)
        exits = defaultdict(int)
        calls = defaultdict(lambda: {"dense": 0, "splade": 0})
        for line in archive.open("retrieval_first_pass.jsonl"):
            row = json.loads(line)
            name, qid = row["method"], row["query_id"]
            require(name in methods and qid not in rows[name], "Unknown method or duplicate query")
            ids = [doc for doc, score in row["ranking"]]
            require(len(ids) == 100 and len(set(ids)) == 100 and qid not in ids, "Ranking IDs invalid")
            require(all(math.isfinite(score) for doc, score in row["ranking"]), "Nonfinite score")
            require(0 <= row["recall@100"] <= 1, "Saved recall invalid")
            early = row["decision"]["route"] == "EARLY_EXIT"
            expected = ({"dense": int(not early), "splade": int(not early)} if name.startswith("ridge")
                        else {"dense": int(name in ("static_rrf", "dense")), "splade": int(name in ("static_rrf", "splade"))})
            require(row["decision"]["calls"] == expected, "Branch decision mismatch")
            exits[name] += early
            for branch, count in expected.items():
                calls[name][branch] += count * report["repeats"]
            # Keep compact digests while validating large full-query artifacts.
            # Full rankings remain preserved in the original ZIP and extraction.
            rows[name][qid] = {
                "recall@100": row["recall@100"], "early_exit": early,
                "ranking_sha256": sha(json.dumps(row["ranking"], separators=(",", ":")).encode("utf-8")),
                "ids_sha256": sha(json.dumps(ids, separators=(",", ":")).encode("utf-8")),
            }
        qids = set(rows["static_rrf"])
        if indexed:
            require(qids == set(report["selected_query_ids"]), "Selected sample coverage mismatch")
        for name, method in methods.items():
            require(len(rows[name]) == n and set(rows[name]) == qids, "Query coverage mismatch")
            require(set(method["timings_per_query_ms"]) == qids, "Timing query mismatch")
            require(all(len(values) == repeats for values in method["timings_per_query_ms"].values()), "Timing repetition missing")
            times = [value for values in method["timings_per_query_ms"].values() for value in values]
            require(all(math.isfinite(value) and value > 0 for value in times), "Invalid timing")
            require(close(sum(times)/len(times), method["latency_ms"]["mean"]), "Mean latency mismatch")
            require(close(sum(row["recall@100"] for row in rows[name].values())/n, method["metrics"]["recall@100"]), "Recall aggregation mismatch")
            require(exits[name] == method["early_exit_count"] and close(exits[name]/n, method["early_exit_rate"]), "Exit count mismatch")
            require(calls[name] == method["neural_calls_across_timed_repeats"], "Timed call count mismatch")
            require(method["rankings_changed_on_later_repeats"] == 0, "Ranking instability reported")
            static = methods["static_rrf"]
            require(close(method["recall_retention"], method["metrics"]["recall@100"]/static["metrics"]["recall@100"]), "Retention mismatch")
            require(close(method["mean_latency_reduction"], 1-method["latency_ms"]["mean"]/static["latency_ms"]["mean"]), "Time reduction mismatch")
            if name.startswith("ridge"):
                router = json.loads(weights)["routers"]["ridge11"]
                policy = router["policies"][name.split("_r")[1]]
                require(method["threshold"] == policy["threshold"] and method["target"] == policy["target"], "Policy mismatch")
                require(method["observed_target_met"] == (method["recall_retention"] >= method["target"]), "Target verdict mismatch")
                for qid, row in rows[name].items():
                    early = row["early_exit"]
                    reference = rows["bm25" if early else "static_rrf"][qid]
                    require(row["ranking_sha256"] == reference["ranking_sha256"] and row["recall@100"] == reference["recall@100"], "Conditional result differs from baseline branch")
        if amendment:
            observed_counts = {0: n}
            observed_extra_changes = dict.fromkeys(methods, 0)
            observed_extra_keys = set()
            for line in archive.open("additional_repetitions.jsonl"):
                extra = json.loads(line)
                repeat, qid = extra["repeat"], extra["query_id"]
                require(repeat > 0 and qid in qids and (repeat, qid) not in observed_extra_keys
                        and set(extra["methods"]) == set(methods), "Invalid additional repetition")
                observed_extra_keys.add((repeat, qid))
                observed_counts[repeat] = observed_counts.get(repeat, 0)+1
                for name, record in extra["methods"].items():
                    ids_hash = sha(json.dumps([doc for doc, _ in record["ranking"]], separators=(",", ":")).encode("utf-8"))
                    if ids_hash != rows[name][qid]["ids_sha256"]:
                        observed_extra_changes[name] += 1
            require(observed_counts == {int(key): value for key, value in amendment["completed_query_counts_by_repeat"].items()},
                    "Additional repetition counts differ from runtime amendment")
            require(observed_extra_changes == amendment["rankings_changed_in_observed_additional_repetitions"],
                    "Additional ranking-change counts differ from runtime amendment")
        suffix = "_one_pass" if amendment else ""
        folder_name = (f"indexed_{report['dataset_key']}_q{report['query_limit']}" if indexed
                       else "paper_"+report["dataset_key"]+suffix)
        destination = PROJECT / "results" / folder_name
        destination.mkdir(parents=True, exist_ok=True)
        for name in archive.namelist():
            with archive.open(name) as source, (destination / name).open("wb") as target:
                shutil.copyfileobj(source, target, length=1024*1024)
        verification = {"archive_sha256": sha(archive_path.read_bytes()), "dataset": report["dataset_key"],
                        "records": n*len(methods), "query_timings": n*len(methods)*repeats,
                        "scored_timing_repeats": repeats, "runtime_amendment": bool(amendment),
                        "indexed_followup": indexed, "evaluated_queries": n,
                        "frozen_assets_sources_and_protocol_match": True,
                        "saved_recall_timings_calls_and_conditional_rankings_match": True,
                        "timing_sessions": len(sessions),
                        "scope": "Artifact consistency checks. Raw relevance grades are absent; nDCG/MAP and timing instrumentation are not independently rerun."}
        (destination / "local_verification.json").write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    return destination, verification


def combine():
    sources = [("NFCorpus dev (development)", PROJECT / "results/retention_tradeoff_dev/nfcorpus_retention_benchmark.json"),
               ("SCIDOCS test (frozen confirmation)", PROJECT / "results/scidocs_confirmation/scidocs_confirmation_benchmark.json")]
    sources += [(path.parent.name, path) for path in sorted((PROJECT / "results").glob("paper_*/paper_dataset_benchmark.json"))
                if (path.parent / "local_verification.json").is_file()]
    for path in sorted((PROJECT / "results").glob("indexed_*/indexed_cascade_benchmark.json")):
        if (path.parent / "local_verification.json").is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            scope = ("all queries" if report["queries"] == report["all_available_queries"] else "sample")
            sources.append((f"{path.parent.name} (indexed {scope}, n={report['queries']})", path))
    output = PROJECT / "results/final_quality_speed.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset", "method", "target", "recall@100", "retention", "ndcg@10", "map@10", "mean_ms", "speedup", "exit_rate"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, path in sources:
            if not path.is_file():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            for name, method in report["methods"].items():
                writer.writerow({"dataset": label, "method": name, "target": method.get("target", ""),
                                 **method["metrics"], "retention": method["recall_retention"],
                                 "mean_ms": method["latency_ms"]["mean"], "speedup": method["mean_speedup_factor"],
                                 "exit_rate": method["early_exit_rate"]})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?", default=ROOT / "artifacts/downloads/trec_covid_paper_benchmark_outputs.zip")
    args = parser.parse_args()
    destination, verification = import_result(args.archive)
    print(json.dumps(verification, indent=2))
    print("Extracted:", destination)
    print("Combined table:", combine())


if __name__ == "__main__":
    main()
