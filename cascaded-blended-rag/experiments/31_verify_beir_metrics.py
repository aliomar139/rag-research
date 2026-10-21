"""Independently score completed BEIR rankings against exact local raw qrels.

No retriever, metric library, download or model inference is used.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=PROJECT / "results/indexed_trec-covid_q0")
    parser.add_argument("--qrels", type=Path, default=PROJECT / "data/raw/trec-covid/qrels/test.tsv")
    args = parser.parse_args()
    folder = args.result_dir
    report_name = ("indexed_cascade_benchmark.json" if (folder / "indexed_cascade_benchmark.json").is_file()
                   else "paper_dataset_benchmark.json")
    report = json.loads((folder / report_name).read_text(encoding="utf-8"))
    raw_hash = hashlib.sha256(args.qrels.read_bytes()).hexdigest()
    if raw_hash != report["evaluation_qrels_sha256"]:
        raise ValueError("Local raw qrels do not match the exact evaluation fingerprint")
    labels = defaultdict(dict)
    grade_counts = Counter()
    with args.qrels.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            grade = int(row["score"])
            grade_counts[grade] += 1
            if grade > 0:
                labels[row["query-id"]][row["corpus-id"]] = grade
    expected_ids = set(report.get("selected_query_ids", report["methods"]["static_rrf"]["timings_per_query_ms"]))
    if len(expected_ids) != report["queries"] or not expected_ids.issubset(labels):
        raise ValueError("Qrels do not cover the declared query scope")
    observed = {name: set() for name in report["methods"]}
    values = {name: defaultdict(list) for name in observed}
    for line in (folder / "retrieval_first_pass.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        name, qid = row["method"], row["query_id"]
        if name not in observed or qid not in expected_ids or qid in observed[name]:
            raise ValueError("Unknown method/query or duplicate retrieval record")
        observed[name].add(qid)
        grades = labels[qid]
        ids = [doc for doc, _ in row["ranking"]]
        relevant_count = len(grades)
        recall = sum(doc in grades for doc in ids[:100])/relevant_count
        ideal = sorted(grades.values(), reverse=True)[:10]
        idcg = math.fsum(grade/math.log2(rank+1) for rank, grade in enumerate(ideal, 1))
        dcg = math.fsum(grades.get(doc, 0)/math.log2(rank+1) for rank, doc in enumerate(ids[:10], 1))
        hits, precision_sum = 0, 0.0
        for rank, doc in enumerate(ids[:10], 1):
            if doc in grades:
                hits += 1
                precision_sum += hits/rank
        actual = {"recall@100": recall, "ndcg@10": dcg/idcg if idcg else 0.0,
                  "map@10": precision_sum/relevant_count}
        if abs(recall-row["recall@100"]) > 1e-12:
            raise ValueError("Saved per-query recall differs from raw qrels")
        for metric, value in actual.items():
            values[name][metric].append(value)
    output = {
        "dataset": report["dataset_key"], "raw_qrels_sha256": raw_hash,
        "qrels_match_exactly": True, "evaluated_queries": len(expected_ids),
        "qrels_grade_counts": dict(sorted(grade_counts.items())),
        "definition": "Positive grades are relevant; nDCG uses linear graded gains and log2(rank+1). "
                      "AP@10 divides the sum of precision at relevant top10 ranks by all judged relevant documents.",
        "scope": "Independent scoring in saved ranking order, not inference or clock verification.",
        "methods": {},
    }
    for name, metrics in values.items():
        if observed[name] != expected_ids:
            raise ValueError("Incomplete query coverage: "+name)
        actual = {metric: math.fsum(scores)/len(scores) for metric, scores in metrics.items()}
        delta = {metric: actual[metric]-report["methods"][name]["metrics"][metric] for metric in actual}
        output["methods"][name] = {
            "independent_metrics": actual, "difference_from_report": delta,
            "matches_within_1e_10": all(abs(v) < 1e-10 for v in delta.values())}
    destination = folder / "independent_metric_verification.json"
    destination.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print("SAVED:", destination)
    if not all(m["matches_within_1e_10"] for m in output["methods"].values()):
        raise ValueError("Independent metrics differ in saved rank order; inspect score ties and metric definitions")


if __name__ == "__main__":
    main()
