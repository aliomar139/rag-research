"""One real online NFCorpus DEV benchmark of frozen 90/92/95/99% policies.

Dev has already been inspected: this is development evidence. No test qrels
are read, and no thresholds are selected or changed from dev outcomes.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import random
import sys
import time
import zipfile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.bm25_coverage_features import BM25CoverageFeatures
from src.retention_tradeoff import ConditionalCascade
from src.tradeoff_analysis import latency_summary, paired_intervals, random_routing_reference


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def model_revision(retriever):
    model = getattr(retriever, "model", None)
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "_first_module"):
        config = getattr(getattr(model._first_module(), "auto_model", None), "config", None)
    return getattr(config, "_commit_hash", None)


def base_module():
    path = PROJECT / "experiments" / "12_calibrate_nfcorpus_train_check_dev.py"
    spec = importlib.util.spec_from_file_location("frozen_nfcorpus_retrieval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_results(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(8, 5))
    for family, color in (("ridge11", "#1769aa"), ("ridge69", "#ca6b20")):
        group = [row for row in results.values() if row.get("family") == family]
        group.sort(key=lambda row: row["target"])
        ax.plot([row["mean_speedup_factor"] for row in group],
                [row["recall_retention"] for row in group], "o-", color=color, label=family)
        for row in group:
            ax.annotate(f"{row['target']:.0%}", (row["mean_speedup_factor"], row["recall_retention"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    for name, marker in (("static_rrf", "s"), ("bm25", "^"), ("dense", "D"), ("splade", "P")):
        row = results[name]
        ax.scatter(row["mean_speedup_factor"], row["recall_retention"], marker=marker, s=65, label=name)
    for value in (.90, .92, .95, .99):
        ax.axhline(value, color="#cccccc", lw=.6, zorder=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Measured mean speedup relative to static RRF (higher is faster)",
           ylabel="Recall@100 retained relative to static RRF",
           title="NFCorpus dev: measured quality / latency trade-off")
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "nfcorpus_quality_speed.png", dpi=180)
    fig.savefig(output / "nfcorpus_quality_speed.svg")
    plt.close(fig)


def benchmark(args):
    base = base_module()
    frozen_path = PROJECT / "artifacts" / "nfcorpus_frozen_retention_routers.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["targets"] != [.90, .92, .95, .99] or frozen["dataset"] != "BEIR NFCorpus":
        raise ValueError("Unexpected frozen policy configuration")
    # Match the exact retrieval implementations used for the saved train labels.
    for name in ("bm25_retriever.py", "dense_retriever.py", "splade_retriever.py", "cache_utils.py"):
        expected = frozen["provenance"]["retrieval_manifest"]["retrieval_code_sha256"][name]
        if hashlib.sha256((PROJECT / "src" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Retrieval source changed: {name}")
    expected_feature_code = frozen["extra_feature_provenance"]["feature_code_sha256"]
    if hashlib.sha256((PROJECT / "src" / "bm25_coverage_features.py").read_bytes()).hexdigest() != expected_feature_code:
        raise ValueError("Coverage feature source differs from frozen policy")
    dataset_dir = base.ensure_dataset(args.data_dir / "raw")
    corpus_path = dataset_dir / "corpus.jsonl"
    if hashlib.sha256(corpus_path.read_bytes()).hexdigest() != frozen["provenance"]["retrieval_manifest"]["corpus_sha256"]:
        raise ValueError("Corpus does not match frozen training artifact")
    corpus = base.load_corpus(dataset_dir)
    queries, qrels_path = base.load_split(dataset_dir, "dev")  # query IDs only
    if set(queries) & set(frozen["train_query_ids"]):
        raise ValueError("Train/dev query ID overlap")
    print(f"Full NFCorpus: {len(corpus)} documents; {len(queries)} dev queries; {args.repeats} repeats", flush=True)
    stage = BM25CoverageFeatures(corpus)
    cache_dir = str(args.data_dir / "processed_nfcorpus_router")
    dense = base.DenseRetriever(corpus, cache_dir=cache_dir)
    splade = base.SPLADERetriever(corpus, cache_dir=cache_dir)
    feature_fn = lambda query, ranking: base.features(query, ranking, stage.dfs, len(corpus))
    common = (stage, dense, splade, feature_fn, base.fuse)
    engines = {name: (ConditionalCascade(*common), name)
               for name in ("static_rrf", "bm25", "dense", "splade")}
    policy_details = {}
    for family, router in frozen["routers"].items():
        for key, policy in router["policies"].items():
            name = f"{family}_r{key}"
            engines[name] = (ConditionalCascade(*common, router, policy["threshold"]), "cascade")
            policy_details[name] = {"family": family, "target": policy["target"], "threshold": policy["threshold"]}
    warm_queries = list(queries.values())[:3]
    for query in warm_queries:
        for engine, method in engines.values():
            engine.run(query, method)
    base.sync_cuda()
    rankings, decisions = {name: {} for name in engines}, {name: {} for name in engines}
    timings = {name: defaultdict(list) for name in engines}
    counts = {name: {"dense": 0, "splade": 0} for name in engines}
    changed_rankings = dict.fromkeys(engines, 0)
    rng = random.Random(20261020)
    for repeat in range(args.repeats):
        query_order = list(queries)
        rng.shuffle(query_order)
        for i, qid in enumerate(query_order, 1):
            order = list(engines)
            rng.shuffle(order)
            for name in order:
                engine, method = engines[name]
                base.sync_cuda()
                start = time.perf_counter()
                ranking, decision = engine.run(queries[qid], method)
                base.sync_cuda()
                elapsed = (time.perf_counter()-start)*1000
                timings[name][qid].append(elapsed)
                for branch in counts[name]:
                    counts[name][branch] += decision["calls"][branch]
                if method == "cascade":
                    expected = 0 if decision["route"] == "EARLY_EXIT" else 1
                    if decision["calls"] != {"dense": expected, "splade": expected}:
                        raise RuntimeError("Neural branch counts do not match the route")
                if repeat == 0:
                    rankings[name][qid], decisions[name][qid] = ranking, decision
                else:
                    if [doc for doc, _ in ranking] != [doc for doc, _ in rankings[name][qid]]:
                        changed_rankings[name] += 1
                    if decision["route"] != decisions[name][qid]["route"]:
                        raise RuntimeError("Frozen route changed between repetitions")
            if i % 50 == 0 or i == len(queries):
                print(f"Timed repeat {repeat+1}/{args.repeats}: {i}/{len(queries)} queries, all {len(engines)} methods", flush=True)
        write_json(args.output_dir / "benchmark_progress.json", {
            "completed_repeats": repeat+1, "repeats_expected": args.repeats,
            "timings_ms": timings, "neural_calls_across_timed_repeats": counts})

    # Frozen routes and ALL timed rankings are complete before consuming grades.
    qrels = base.load_qrels(qrels_path)
    if set(qrels) != set(queries):
        raise ValueError("Positive dev qrels and query IDs must match")
    ids = list(queries)
    recall = {name: np.asarray([base.recall(qrels[qid], rankings[name][qid]) for qid in ids]) for name in engines}
    means = {name: np.asarray([np.mean(timings[name][qid]) for qid in ids]) for name in engines}
    static_mean = float(means["static_rrf"].mean())
    static_recall = float(recall["static_rrf"].mean())
    results = {}
    for name in engines:
        exits = sum(row["route"] == "EARLY_EXIT" for row in decisions[name].values())
        latency = latency_summary([value for qid in ids for value in timings[name][qid]])
        metrics = base.score_run(qrels, rankings[name])
        if abs(metrics["recall@100"]-float(recall[name].mean())) > 1e-8:
            raise ValueError("Custom per-query recall differs from ranx aggregate")
        results[name] = {**policy_details.get(name, {}), "metrics": metrics,
            "recall_retention": float(recall[name].mean()/static_recall),
            "latency_ms": latency, "mean_speedup_factor": static_mean/latency["mean"],
            "mean_latency_reduction": 1-latency["mean"]/static_mean,
            "early_exit_count": exits, "early_exit_rate": exits/len(ids),
            "neural_calls_across_timed_repeats": counts[name],
            "rankings_changed_on_later_repeats": changed_rankings[name],
            "timings_per_query_ms": dict(timings[name]),
            **paired_intervals(recall["static_rrf"], recall[name], means["static_rrf"], means[name])}
        if name in policy_details:
            results[name]["observed_target_met"] = results[name]["recall_retention"] >= policy_details[name]["target"]
            results[name]["random_exit_reference"] = random_routing_reference(recall["bm25"], recall["static_rrf"], exits)
    for name, row in results.items():
        row["dominated_in_mean_latency_and_recall_by"] = [other for other, value in results.items()
            if other != name and value["latency_ms"]["mean"] <= row["latency_ms"]["mean"]
            and value["metrics"]["recall@100"] >= row["metrics"]["recall@100"]
            and (value["latency_ms"]["mean"] < row["latency_ms"]["mean"]
                 or value["metrics"]["recall@100"] > row["metrics"]["recall@100"])]
    report = {"dataset": "BEIR NFCorpus", "evaluation_split": "official dev (previously inspected)",
        "corpus_documents": len(corpus), "queries": len(ids), "test_qrels_opened": False,
        "dev_grades_used_to_select_policies": False, "repeats": args.repeats,
        "targets": frozen["targets"], "methods": results,
        "frozen_router_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hardware": {"cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "torch_cpu_threads": torch.get_num_threads(),
            "dense_model_revision": model_revision(dense), "splade_model_revision": model_revision(splade)},
        "protocol": [
            "Fixed train-only thresholds for both 11-feature and 69-feature Ridge gates; no dev threshold adjustment.",
            "Identical stable vectorized BM25 ranking implementation for ALL methods; rankings matched original saved train rankings.",
            "Query and method order randomized in each repetition; synchronization before and after every timed query. Warmups excluded.",
            "No cached query results or shared per-query features; fallback neural retrieval executes only on escalation. Corpus indexes may be reused.",
            "All methods scored on identical queries/corpus. Recall/nDCG/MAP metrics use first timed repeat rankings.",
            "Bootstrap resamples queries, pairing recall and repeat-averaged latency; intervals do not capture all hardware/session variability.",
            "Random routing is an offline, count-matched recall reference; its latency was not measured.",
            "Development curve only. Independent confirmation is needed for claims about untouched data; RAG answer quality is not measured.",
        ]}
    write_json(args.output_dir / "nfcorpus_retention_benchmark.json", report)
    with (args.output_dir / "quality_speed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "target", "recall@100", "retention", "ndcg@10", "map@10", "mean_ms", "p95_ms", "p99_ms", "speedup", "exit_rate"])
        writer.writeheader()
        for name, row in results.items():
            writer.writerow({"method": name, "target": row.get("target", ""),
                **row["metrics"], "retention": row["recall_retention"],
                "mean_ms": row["latency_ms"]["mean"], "p95_ms": row["latency_ms"]["p95"],
                "p99_ms": row["latency_ms"]["p99"], "speedup": row["mean_speedup_factor"], "exit_rate": row["early_exit_rate"]})
    with (args.output_dir / "retrieval_first_pass.jsonl").open("w", encoding="utf-8") as handle:
        for name in engines:
            for i, qid in enumerate(ids):
                handle.write(json.dumps({"method": name, "query_id": qid,
                    "ranking": rankings[name][qid], "decision": decisions[name][qid],
                    "recall@100": float(recall[name][i])}, allow_nan=False)+"\n")
    plot_results(results, args.output_dir)
    output_zip = args.output_dir / "nfcorpus_retention_benchmark_outputs.zip"
    filenames = ["nfcorpus_retention_benchmark.json", "quality_speed.csv", "retrieval_first_pass.jsonl",
                 "nfcorpus_quality_speed.png", "nfcorpus_quality_speed.svg"]
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in filenames:
            archive.write(args.output_dir / filename, arcname=filename)
        archive.write(frozen_path, arcname=frozen_path.name)
    print(f"DOWNLOAD: {output_zip}", flush=True)
    for name, row in results.items():
        print(f"{name}: retention {row['recall_retention']:.2%}, mean {row['latency_ms']['mean']:.2f} ms, "
              f"speedup {row['mean_speedup_factor']:.2f}x, exits {row['early_exit_rate']:.2%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Use at least two timing repeats")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        benchmark(args)


if __name__ == "__main__":
    main()
