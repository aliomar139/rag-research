"""Independently rebuild SQuAD parent qrels and score saved first-pass ranks."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=PROJECT / "results/paper_squad_one_pass")
    args = parser.parse_args()
    folder = args.result_dir
    report_name = ("indexed_cascade_benchmark.json" if (folder / "indexed_cascade_benchmark.json").is_file()
                   else "paper_dataset_benchmark.json")
    report = json.loads((folder / report_name).read_text(encoding="utf-8"))
    if report["dataset_key"] != "squad":
        raise ValueError("This scorer only supports SQuAD parent-paragraph relevance")
    source = PROJECT / "data/raw/squad_dev_v1.1.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    labels = {}
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t")
    writer.writerow(["query-id", "corpus-id", "score"])
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            identity = article["title"]+"\0"+paragraph["context"]
            did = "squad-paragraph-"+hashlib.sha256(identity.encode("utf-8")).hexdigest()
            for question in paragraph["qas"]:
                qid = "squad-question-"+question["id"]
                labels[qid] = did
                writer.writerow([qid, did, 1])
    rebuilt = hashlib.sha256(stream.getvalue().encode("utf-8")).hexdigest()
    if len(labels) != 10570 or rebuilt != report["evaluation_qrels_sha256"]:
        raise ValueError("Local raw SQuAD data do not reconstruct the exact Kaggle qrels")
    expected_ids = set(report.get("selected_query_ids", labels))
    if len(expected_ids) != report["queries"] or not expected_ids.issubset(labels):
        raise ValueError("Invalid selected query scope")
    sums = {name: {metric: [] for metric in ("recall@100", "ndcg@10", "map@10")} for name in report["methods"]}
    observed = {name: set() for name in sums}
    with (folder / "retrieval_first_pass.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            name, qid = row["method"], row["query_id"]
            if qid not in expected_ids or qid in observed[name]:
                raise ValueError("Unknown or duplicate query")
            observed[name].add(qid)
            rank = next((i for i, (doc, _) in enumerate(row["ranking"], 1) if doc == labels[qid]), None)
            values = {"recall@100": float(rank is not None and rank <= 100),
                      "ndcg@10": 1/math.log2(rank+1) if rank and rank <= 10 else 0.0,
                      "map@10": 1/rank if rank and rank <= 10 else 0.0}
            if values["recall@100"] != row["recall@100"]:
                raise ValueError("Saved per-query recall differs from independently reconstructed parent label")
            for metric, value in values.items():
                sums[name][metric].append(value)
    output = {"raw_dev_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "rebuilt_qrels_sha256": rebuilt, "qrels_match_exactly": True,
              "all_labeled_queries": len(labels), "evaluated_queries": len(expected_ids),
              "scope": "Independent parent-label scoring in stored ranking order. Score-tie order can differ from the original evaluator.",
              "methods": {}}
    for name, metrics in sums.items():
        if expected_ids != observed[name]:
            raise ValueError("Missing queries")
        actual = {metric: math.fsum(values)/len(values) for metric, values in metrics.items()}
        delta = {metric: actual[metric]-report["methods"][name]["metrics"][metric] for metric in actual}
        output["methods"][name] = {"independent_metrics": actual, "difference_from_report": delta,
                                   "matches_within_1e_10": all(abs(value) < 1e-10 for value in delta.values())}
        if not output["methods"][name]["matches_within_1e_10"]:
            raise ValueError("Independent aggregate metrics differ: "+name)
    destination = folder / "independent_metric_verification.json"
    destination.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print("SAVED:", destination)


if __name__ == "__main__":
    main()
