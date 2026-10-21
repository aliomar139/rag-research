"""Frozen cascade evaluation on datasets used in Blended RAG.

Same NFCorpus-trained gate and four thresholds; full corpora, real conditional
execution, strong baselines and three timing repeats. SQuAD is a declared
paragraph-retrieval adaptation. This is not an exact reproduction of ELSER.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
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
from src.paper_datasets import NAMES, prepare_dataset, file_hash
from src.query_self_exclusion import QueryContext, ExcludingBM25, ExcludingNeural



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


def plot_results(results, output, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(8, 5))
    for family, color in (("ridge11", "#1769aa"),):
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
           title=f"{dataset}: frozen NFCorpus gate")
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "quality_speed.png", dpi=180)
    fig.savefig(output / "quality_speed.svg")
    plt.close(fig)


def benchmark(args):
    base = base_module()
    frozen_path = PROJECT / "artifacts" / "nfcorpus_frozen_retention_routers.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    protocol_path = PROJECT / "artifacts" / "paper_datasets_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if args.dataset not in protocol["datasets"] or protocol["primary_candidate_target"] != .90:
        raise ValueError("Dataset/primary target differs from the prepared extension protocol")
    for relative, expected in protocol["source_sha256"].items():
        if file_hash(PROJECT / relative) != expected:
            raise ValueError(f"Source differs from packaged protocol: {relative}")
    if (protocol["frozen_routers_sha256"] != hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        or protocol["timing_repeats"] != args.repeats
        or protocol["targets"] != frozen["targets"]
        or protocol["families"] != ["ridge11"]):
        raise ValueError("Confirmation configuration differs from its frozen protocol")
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
    if args.dataset in ("nq", "hotpotqa") and not args.allow_large_corpus:
        raise ValueError("NQ/HotpotQA use millions of documents. Run on a suitably provisioned machine "
                         "with --allow-large-corpus; the default Kaggle run is TREC-COVID. "
                         "No smaller relevance-filtered corpus will be substituted.")
    dataset_dir, dataset_info = prepare_dataset(args.dataset, args.data_dir / "raw")
    corpus_path = dataset_dir / "corpus.jsonl"
    corpus = base.load_corpus(dataset_dir)
    queries, qrels_path = base.load_split(dataset_dir, "test")  # IDs only, no grades
    if len(queries) != dataset_info["expected_queries"] or len(corpus) != dataset_info["expected_documents"]:
        raise ValueError("Dataset dimensions differ from the declared full-corpus protocol")
    # IDs are local to a dataset: matching numeric strings across collections
    # are not evidence of training/evaluation overlap.
    print(f"{dataset_info['dataset']}: {len(corpus)} documents; {len(queries)} queries; {args.repeats} repeats", flush=True)
    preparation_seconds = time.perf_counter()
    context = QueryContext()
    raw_stage = BM25CoverageFeatures(corpus)
    # This extension uses only ridge11. These coverage-ablation collections
    # are unused by retrieve_only and the original features; release their RAM.
    for attribute in ("titles", "abstracts", "combined", "doc_lengths", "idfs"):
        delattr(raw_stage, attribute)
    stage = ExcludingBM25(raw_stage, context)
    cache_dir = str(args.data_dir / ("processed_paper_"+args.dataset))
    dense = ExcludingNeural(base.DenseRetriever(corpus, cache_dir=cache_dir), context)
    splade = ExcludingNeural(base.SPLADERetriever(corpus, cache_dir=cache_dir), context)
    for name, retriever in (("dense", dense), ("splade", splade)):
        expected = protocol["model_revisions"][name]
        if model_revision(retriever) != expected:
            raise ValueError(f"{name} model revision differs from the prepared protocol: "
                             f"{model_revision(retriever)} != {expected}")
    preparation_seconds = time.perf_counter()-preparation_seconds
    print(f"Indexes ready in {preparation_seconds/60:.1f} minutes; starting warmup/timing", flush=True)
    feature_fn = lambda query, ranking: base.features(query, ranking, stage.dfs, len(corpus))
    common = (stage, dense, splade, feature_fn, base.fuse)
    engines = {name: (ConditionalCascade(*common), name)
               for name in ("static_rrf", "bm25", "dense", "splade")}
    policy_details = {}
    for family, router in frozen["routers"].items():
        if family != "ridge11":
            continue  # Original compact gate is predeclared as the confirmation method.
        for key, policy in router["policies"].items():
            name = f"{family}_r{key}"
            engines[name] = (ConditionalCascade(*common, router, policy["threshold"]), "cascade")
            policy_details[name] = {"family": family, "target": policy["target"], "threshold": policy["threshold"]}
    warm_queries = list(queries.items())[:3]
    for qid, query in warm_queries:
        context.document_id = qid
        for engine, method in engines.values():
            engine.run(query, method)
    base.sync_cuda()
    rankings, decisions = {name: {} for name in engines}, {name: {} for name in engines}
    timings = {name: defaultdict(list) for name in engines}
    counts = {name: {"dense": 0, "splade": 0} for name in engines}
    changed_rankings = dict.fromkeys(engines, 0)
    # Persist each completed query so a restarted notebook can finish the same
    # run without reusing any query ranking as an inference cache.
    checkpoint = args.output_dir / "timed_queries.jsonl"
    run_identity = {"dataset": args.dataset, "corpus_sha256": file_hash(corpus_path),
                    "query_file_sha256": file_hash(dataset_dir / "queries.jsonl"),
                    "weights_sha256": file_hash(frozen_path), "protocol_sha256": file_hash(protocol_path),
                    "runner_sha256": file_hash(Path(__file__)), "repeats": args.repeats,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": np.__version__, "dense_revision": model_revision(dense),
                    "splade_revision": model_revision(splade)}
    identity_path = args.output_dir / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != run_identity:
            raise ValueError("Checkpoint belongs to different data, code, models or hardware; use a new output directory")
    else:
        write_json(identity_path, run_identity)
    completed = set()

    def consume(repeat, qid, records):
        if (repeat, qid) in completed or set(records) != set(engines):
            raise ValueError("Invalid or duplicate checkpoint record")
        for name, record in records.items():
            ranking, decision, elapsed = record["ranking"], record["decision"], record["elapsed_ms"]
            timings[name][qid].append(elapsed)
            for branch in counts[name]:
                counts[name][branch] += decision["calls"][branch]
            if repeat == 0:
                rankings[name][qid], decisions[name][qid] = ranking, decision
            else:
                if [doc for doc, _ in ranking] != [doc for doc, _ in rankings[name][qid]]:
                    changed_rankings[name] += 1
                if decision["route"] != decisions[name][qid]["route"]:
                    raise RuntimeError("Frozen route changed between repetitions")
        completed.add((repeat, qid))

    if checkpoint.exists():
        # A process interruption can leave only the final line incomplete.
        valid_end = 0
        with checkpoint.open("rb") as handle:
            while line := handle.readline():
                if not line.endswith(b"\n"):
                    break
                record = json.loads(line)
                consume(record["repeat"], record["query_id"], record["methods"])
                valid_end = handle.tell()
        with checkpoint.open("r+b") as handle:
            handle.truncate(valid_end)
        print(f"Resuming {len(completed)} completed query repetitions. Session changes will be recorded.", flush=True)
    sessions_path = args.output_dir / "timing_sessions.jsonl"
    with sessions_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(),
                                 "completed_before_session": len(completed), "hardware": run_identity})+"\n")
    rng = random.Random(20261020)
    for repeat in range(args.repeats):
        query_order = list(queries)
        rng.shuffle(query_order)
        for i, qid in enumerate(query_order, 1):
            context.document_id = qid
            order = list(engines)
            rng.shuffle(order)
            # Shuffle even when resuming, so unfinished query orders are identical.
            if (repeat, qid) in completed:
                continue
            records = {}
            for name in order:
                engine, method = engines[name]
                base.sync_cuda()
                start = time.perf_counter()
                ranking, decision = engine.run(queries[qid], method)
                base.sync_cuda()
                elapsed = (time.perf_counter()-start)*1000
                if any(doc == qid for doc, _ in ranking) or len(ranking) != 100:
                    raise RuntimeError("Expected 100 non-self documents")
                if method == "cascade":
                    expected = 0 if decision["route"] == "EARLY_EXIT" else 1
                    if decision["calls"] != {"dense": expected, "splade": expected}:
                        raise RuntimeError("Neural branch counts do not match the route")
                records[name] = {"ranking": ranking, "decision": decision, "elapsed_ms": elapsed}
            # Disk writes happen outside all measured method calls.
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"repeat": repeat, "query_id": qid, "methods": records}, allow_nan=False)+"\n")
                handle.flush()
            consume(repeat, qid, records)
            if i % 50 == 0 or i == len(queries):
                print(f"Timed repeat {repeat+1}/{args.repeats}: {i}/{len(queries)} queries, all {len(engines)} methods", flush=True)
        write_json(args.output_dir / "benchmark_progress.json", {
            "completed_repeats": repeat+1, "repeats_expected": args.repeats,
            "timings_ms": timings, "neural_calls_across_timed_repeats": counts})

    # All timed rankings are fixed before evaluation grades are used to score.
    qrels = base.load_qrels(qrels_path)
    if set(qrels) != set(queries):
        raise ValueError("Positive evaluation qrels and query IDs must match")
    ids = list(queries)
    recall = {name: np.asarray([base.recall(qrels[qid], rankings[name][qid]) for qid in ids]) for name in engines}
    means = {name: np.asarray([np.mean(timings[name][qid]) for qid in ids]) for name in engines}
    static_mean = float(means["static_rrf"].mean())
    static_recall = float(recall["static_rrf"].mean())
    if static_recall <= 0 or static_mean <= 0:
        raise ValueError("Static reference must have positive mean recall and latency")
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
            "paper_retrieval_metrics": {
                "hit@10": float(np.mean([any(doc in qrels[qid] for doc, _ in rankings[name][qid][:10]) for qid in ids])),
                "hit@20": float(np.mean([any(doc in qrels[qid] for doc, _ in rankings[name][qid][:20]) for qid in ids])),
                "recall@10": float(np.mean([len(set(qrels[qid]) & {doc for doc, _ in rankings[name][qid][:10]})/len(qrels[qid]) for qid in ids])),
            },
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
    for row in results.values():
        row["dominators_all_quality_metrics_and_mean_latency"] = [
            other for other, candidate in results.items() if candidate is not row
            and candidate["latency_ms"]["mean"] <= row["latency_ms"]["mean"]
            and all(candidate["metrics"][metric] >= row["metrics"][metric] for metric in row["metrics"])
            and (candidate["latency_ms"]["mean"] < row["latency_ms"]["mean"]
                 or any(candidate["metrics"][metric] > row["metrics"][metric] for metric in row["metrics"]))]
    primary = results["ridge11_r90"]
    primary_outcome = {
        "policy": "ridge11_r90", "required_retention": 0.90,
        "point_retention_at_least_90": primary["recall_retention"] >= .90,
        "retention_interval_lower_at_least_90": bool(primary["recall_retention_95_interval"]
                                                   and primary["recall_retention_95_interval"][0] >= .90),
        "positive_mean_latency_saving": primary["mean_latency_reduction"] > 0,
        "latency_reduction_interval_lower_positive": primary["mean_latency_reduction_95_interval"][0] > 0,
        "interpretation": "These flags describe measured evidence, not novelty, answer quality or a major-improvement verdict."}
    report = {"dataset": dataset_info["dataset"], "dataset_key": args.dataset,
        "dataset_preparation": dataset_info, "evaluation_split": dataset_info["split"],
        "original_paper_exact_reproduction": False, "primary_outcome": primary_outcome,
        "index_preparation_seconds": preparation_seconds,
        "timing_sessions": [json.loads(line) for line in sessions_path.read_text(encoding="utf-8").splitlines()],
        "corpus_documents": len(corpus), "queries": len(ids), "nfcorpus_test_qrels_opened": False,
        "evaluation_qrels_opened_for_scoring": True,
        "evaluation_relevance_grades_scored_only_after_timing": True,
        "evaluation_grades_used_to_select_policies": False, "repeats": args.repeats,
        "training_dataset": "BEIR NFCorpus train",
        "corpus_sha256": file_hash(corpus_path),
        "evaluation_qrels_sha256": file_hash(qrels_path),
        "query_document_self_matches_excluded": True,
        "query_ids_also_present_in_corpus": len(set(queries) & set(corpus)),
        "targets": frozen["targets"], "methods": results,
        "frozen_router_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "extension_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "extension_protocol": protocol,
        "hardware": {"cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "torch_cpu_threads": torch.get_num_threads(),
            "dense_model_revision": model_revision(dense), "splade_model_revision": model_revision(splade)},
        "protocol": [
            "All four original-feature Ridge gates are frozen from NFCorpus train. No refit or threshold adjustment on evaluation datasets.",
            "Each retriever requests 101 candidates and removes the identical query/document ID before keeping 100, feature extraction, and fusion. This follows BEIR self-match exclusion.",
            "Identical stable vectorized BM25 ranking implementation for ALL methods. Implementation previously checked against saved original NF train rankings.",
            "Unused coverage-ablation metadata is released before timing; original ridge11 features and BM25 scoring are unchanged.",
            "Query and method order randomized in each repetition; synchronization before and after every timed query. Warmups excluded.",
            "No cached query results or shared per-query features; fallback neural retrieval executes only on escalation. Corpus indexes may be reused.",
            "All methods scored on identical queries/corpus. Recall/nDCG/MAP metrics use first timed repeat rankings.",
            "Bootstrap resamples queries, pairing recall and repeat-averaged latency; intervals do not capture all hardware/session variability.",
            "Random routing is an offline, count-matched recall reference; its latency was not measured.",
            "Original-paper dataset extension, after inspected NFCorpus/SCIDOCS results. Outcomes and target misses are reported without retuning. RAG answer quality is not measured.",
        ]}
    write_json(args.output_dir / "paper_dataset_benchmark.json", report)
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
    plot_results(results, args.output_dir, dataset_info["dataset"])
    output_zip = args.output_dir / (args.dataset.replace("-", "_")+"_paper_benchmark_outputs.zip")
    filenames = ["paper_dataset_benchmark.json", "quality_speed.csv", "retrieval_first_pass.jsonl", "run_identity.json", "timing_sessions.jsonl",
                 "quality_speed.png", "quality_speed.svg"]
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in filenames:
            archive.write(args.output_dir / filename, arcname=filename)
        archive.write(frozen_path, arcname=frozen_path.name)
        archive.write(protocol_path, arcname=protocol_path.name)
    print(f"DOWNLOAD: {output_zip}", flush=True)
    for name, row in results.items():
        print(f"{name}: retention {row['recall_retention']:.2%}, mean {row['latency_ms']['mean']:.2f} ms, "
              f"speedup {row['mean_speedup_factor']:.2f}x, exits {row['early_exit_rate']:.2%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    parser.add_argument("--dataset", choices=NAMES, default="trec-covid")
    parser.add_argument("--allow-large-corpus", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Use at least two timing repeats")
    if args.output_dir is None:
        args.output_dir = PROJECT / "results" / ("paper_"+args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        benchmark(args)


if __name__ == "__main__":
    main()
