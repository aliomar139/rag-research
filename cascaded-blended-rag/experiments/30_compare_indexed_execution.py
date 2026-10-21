"""Compare saved original and indexed rankings on exactly the same queries.

Only reads completed artifacts. No retrieval, downloads or neural inference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, default=PROJECT / "results/paper_squad_one_pass")
    parser.add_argument("--indexed-dir", type=Path, default=PROJECT / "results/indexed_squad_q256")
    args = parser.parse_args()
    old = json.loads((args.old_dir / "paper_dataset_benchmark.json").read_text(encoding="utf-8"))
    new = json.loads((args.indexed_dir / "indexed_cascade_benchmark.json").read_text(encoding="utf-8"))
    require(old["dataset_key"] == new["dataset_key"], "Dataset differs")
    for key in ("corpus_sha256", "evaluation_qrels_sha256", "frozen_router_sha256", "corpus_documents"):
        require(old[key] == new[key], "Compared assets differ: "+key)
    old_ids = list(old["methods"]["static_rrf"]["timings_per_query_ms"])
    selected = new["selected_query_ids"]
    require(len(old_ids) == new["all_available_queries"], "Available query count differs")
    if new["query_limit"]:
        expected = sorted(old_ids, key=lambda q: hashlib.sha256((new["dataset_key"]+"\0"+q).encode()).hexdigest())[:new["query_limit"]]
        require(selected == expected, "Query sample differs from declared hash selection")
    else:
        require(set(selected) == set(old_ids), "All-query selection differs")
    selected_set = set(selected)
    saved = {}
    for line in (args.indexed_dir / "retrieval_first_pass.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        saved[row["method"], row["query_id"]] = row
    comparisons = {name: {"compared_queries": 0, "ordered_ids_changed": 0,
                          "ranking_scores_not_exact": 0, "max_absolute_score_difference": 0.0,
                          "routes_or_calls_changed": 0, "per_query_recall_changed": 0}
                   for name in new["methods"]}
    previous_hash = hashlib.sha256()
    with (args.old_dir / "retrieval_first_pass.jsonl").open("rb") as handle:
        for line in handle:
            previous_hash.update(line)
            row = json.loads(line)
            if row["query_id"] not in selected_set:
                continue
            other = saved[row["method"], row["query_id"]]
            counts = comparisons[row["method"]]
            counts["compared_queries"] += 1
            same_ids = [d for d, _ in row["ranking"]] == [d for d, _ in other["ranking"]]
            counts["ordered_ids_changed"] += not same_ids
            counts["ranking_scores_not_exact"] += row["ranking"] != other["ranking"]
            if same_ids:
                counts["max_absolute_score_difference"] = max(counts["max_absolute_score_difference"],
                    max(abs(a[1]-b[1]) for a, b in zip(row["ranking"], other["ranking"])))
            counts["routes_or_calls_changed"] += any(row["decision"][k] != other["decision"][k] for k in ("route", "calls"))
            counts["per_query_recall_changed"] += row["recall@100"] != other["recall@100"]
    reference = new.get("bm25_previous_run_agreement")
    require(reference is not None and previous_hash.hexdigest() == reference["reference_sha256"],
            "Earlier first-pass reference fingerprint differs")
    for name, counts in comparisons.items():
        require(counts["compared_queries"] == len(selected), "Missing paired rankings: "+name)
    paired_times = {}
    for name, method in new["methods"].items():
        old_mean = math.fsum(math.fsum(old["methods"][name]["timings_per_query_ms"][q]) /
                             len(old["methods"][name]["timings_per_query_ms"][q]) for q in selected)/len(selected)
        new_mean = math.fsum(math.fsum(method["timings_per_query_ms"][q]) /
                             len(method["timings_per_query_ms"][q]) for q in selected)/len(selected)
        paired_times[name] = {"old_same_queries_mean_ms": old_mean, "indexed_same_queries_mean_ms": new_mean,
                              "observed_old_to_indexed_ratio": old_mean/new_mean}
    output = {
        "dataset": new["dataset_key"], "evaluated_queries": len(selected),
        "all_available_queries": len(old_ids), "declared_query_selection_matches": True,
        "query_scope": "all queries" if len(selected) == len(old_ids) else "fixed-hash sample",
        "corpus_qrels_weights_match": True, "previous_first_pass_sha256": previous_hash.hexdigest(),
        "all_methods_ordered_ids_routes_and_recall_identical": all(
            not c["ordered_ids_changed"] and not c["routes_or_calls_changed"] and not c["per_query_recall_changed"]
            for c in comparisons.values()),
        "all_methods_ranking_scores_exactly_identical": all(not c["ranking_scores_not_exact"] for c in comparisons.values()),
        "rank_comparisons": comparisons, "same_query_timing_comparison": paired_times,
        "old_repeats": old["repeats"], "indexed_repeats": new["repeats"],
        "old_hardware": old["hardware"], "indexed_hardware": new["hardware"],
        "scope": "Same query IDs across separate timing runs, not an interleaved randomized backend experiment. "
                 f"Old run has {old['repeats']} complete pass(es); indexed run has {new['repeats']}. "
                 "Timing ratios include run variability. "
                 "Indexed cascade-versus-indexed fusion comparisons come from the new matched experiment. "
                 "Rank agreement applies only to evaluated query IDs; timing is specific to the measured setting.",
    }
    destination = args.indexed_dir / "paired_backend_comparison.json"
    destination.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k not in ("old_hardware", "indexed_hardware")}, indent=2))
    print("SAVED:", destination)


if __name__ == "__main__":
    main()
