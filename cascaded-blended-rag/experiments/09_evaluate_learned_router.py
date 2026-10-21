"""Train and evaluate a query-level BM25 escalation-risk router on SciFact.

Run from cascaded-blended-rag/. Training examples must be in
results/router_training_scifact.jsonl. Threshold calibration uses out-of-fold
predictions on the training examples; the test qrels are used only for final
evaluation. The test path conditionally invokes Dense and SPLADE.
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import joblib
import numpy as np
import torch
from ranx import Qrels, Run, evaluate
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import download_and_load_scifact
from src.bm25_retriever import BM25Retriever, simple_tokenize
from src.dense_retriever import DenseRetriever
from src.splade_retriever import SPLADERetriever


TOP_K = 100
RRF_K = 60
RECALL_FRACTION = 0.98
FEATURE_NAMES = [
    "query_token_count", "query_unique_token_count", "bm25_top1", "bm25_top2",
    "bm25_gap12", "bm25_top10_mean", "bm25_top10_std",
    "bm25_top1_per_query_token", "max_idf", "mean_idf", "idf_std",
]


def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn, *args, **kwargs):
    synchronize_cuda()
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    synchronize_cuda()
    return value, (time.perf_counter() - start) * 1000.0


def router_features(query, bm25_ranking, doc_freqs, total_docs):
    tokens = simple_tokenize(query)
    unique_tokens = list(dict.fromkeys(tokens))
    scores = [float(score) for _, score in bm25_ranking[:10]]
    s1 = scores[0] if scores else 0.0
    s2 = scores[1] if len(scores) > 1 else 0.0
    idfs = []
    for token in unique_tokens:
        df = doc_freqs.get(token, 0)
        idfs.append(math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0))
    idf_mean = mean(idfs) if idfs else 0.0
    score_mean = mean(scores) if scores else 0.0
    return {
        "query_token_count": len(tokens),
        "query_unique_token_count": len(unique_tokens),
        "bm25_top1": s1,
        "bm25_top2": s2,
        "bm25_gap12": s1 - s2,
        "bm25_top10_mean": score_mean,
        "bm25_top10_std": (sum((x - score_mean) ** 2 for x in scores) / len(scores)) ** 0.5 if scores else 0.0,
        "bm25_top1_per_query_token": s1 / max(1, len(tokens)),
        "max_idf": max(idfs, default=0.0),
        "mean_idf": idf_mean,
        "idf_std": (sum((x - idf_mean) ** 2 for x in idfs) / len(idfs)) ** 0.5 if idfs else 0.0,
    }


def make_model(seed):
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=20,
        max_features=0.8,
        random_state=seed,
        n_jobs=1,
    )


def choose_threshold(oof_predictions, gains, static_recall):
    """Maximize OOF exits within the aggregate 2%-of-static recall-loss budget."""
    budget = (1.0 - RECALL_FRACTION) * static_recall
    candidates = np.r_[np.unique(oof_predictions), np.inf]
    feasible = []
    for threshold in candidates:
        exits = oof_predictions <= threshold
        aggregate_loss = float(np.where(exits, gains, 0.0).mean())
        if aggregate_loss <= budget:
            feasible.append((int(exits.sum()), -aggregate_loss, float(threshold), aggregate_loss))
    # Zero exits always satisfies the constraint; this is a defensive fallback.
    if not feasible:
        return float(np.nextafter(oof_predictions.min(), -np.inf)), budget, 0.0
    exits_count, _, threshold, aggregate_loss = max(feasible)
    return threshold, budget, aggregate_loss


def fuse_rankings(rankings):
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (RRF_K + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:TOP_K]


def as_run(rankings):
    return {qid: {doc_id: score for doc_id, score in ranking} for qid, ranking in rankings.items()}


def score_run(qrels, rankings):
    values = evaluate(
        Qrels(qrels), Run(as_run(rankings)),
        metrics=["recall@100", "ndcg@10", "map@10"],
    )
    return {key: float(value) for key, value in values.items()}


def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    point = (len(ordered) - 1) * pct
    lower, upper = math.floor(point), math.ceil(point)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - point) + ordered[upper] * (point - lower))


def main():
    project_dir = Path(__file__).resolve().parents[1]
    data_path = project_dir / "results" / "router_training_scifact.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Training examples missing: {data_path}")

    examples = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    X = np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in examples], dtype=float)
    gains = np.asarray([row["recall_gain_from_escalation"] for row in examples], dtype=float)
    bm25_train = np.asarray([row["bm25_recall@100"] for row in examples], dtype=float)
    static_train = np.asarray([row["static_recall@100"] for row in examples], dtype=float)

    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_gain = np.maximum(cross_val_predict(make_model(42), X, gains, cv=folds, n_jobs=1), 0.0)
    static_train_recall = float(static_train.mean())
    threshold, allowed_loss, oof_loss = choose_threshold(oof_gain, gains, static_train_recall)
    oof_exit = oof_gain <= threshold
    oof_policy = np.where(oof_exit, bm25_train, static_train)

    model = make_model(42).fit(X, gains)
    model_path = project_dir / "results" / "learned_router_scifact.joblib"
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "threshold": threshold}, model_path)

    corpus, _, _ = download_and_load_scifact(split="train")
    _, test_queries, test_qrels = download_and_load_scifact(split="test")
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, cache_dir="data/processed_scifact")
    splade = SPLADERetriever(corpus, cache_dir="data/processed_scifact")
    doc_freqs = defaultdict(int)
    for doc in corpus.values():
        for token in set(simple_tokenize(f"{doc['title']} {doc['abstract']}")):
            doc_freqs[token] += 1

    # Warm up before timing.
    for query in list(test_queries.values())[:5]:
        bm25.retrieve_best_fields(query, top_k=TOP_K)
        dense.retrieve(query, top_k=TOP_K)
        splade.retrieve(query, top_k=TOP_K)

    static_rankings, static_components, static_times = {}, {"bm25": {}, "dense": {}, "splade": {}}, []
    for i, (qid, query) in enumerate(test_queries.items(), start=1):
        start = time.perf_counter()
        b, b_ms = timed_call(bm25.retrieve_best_fields, query, top_k=TOP_K)
        d, d_ms = timed_call(dense.retrieve, query, top_k=TOP_K)
        s, s_ms = timed_call(splade.retrieve, query, top_k=TOP_K)
        static_components["bm25"][qid] = b
        static_components["dense"][qid] = d
        static_components["splade"][qid] = s
        static_rankings[qid] = fuse_rankings([b, d, s])
        synchronize_cuda()
        static_times.append((time.perf_counter() - start) * 1000.0)
        if i % 25 == 0 or i == len(test_queries):
            print(f"Static test retrievals: {i}/{len(test_queries)}", flush=True)

    cascade_rankings, per_query = {}, {}
    for i, (qid, query) in enumerate(test_queries.items(), start=1):
        start = time.perf_counter()
        b, b_ms = timed_call(bm25.retrieve_best_fields, query, top_k=TOP_K)
        route_start = time.perf_counter()
        features = router_features(query, b, doc_freqs, len(corpus))
        vector = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=float)
        predicted_gain = max(0.0, float(model.predict(vector)[0]))
        exit_early = predicted_gain <= threshold
        route_ms = (time.perf_counter() - route_start) * 1000.0
        timings = {"bm25_ms": b_ms, "router_ms": route_ms, "dense_ms": 0.0, "splade_ms": 0.0}
        if exit_early:
            ranking = b
        else:
            d, timings["dense_ms"] = timed_call(dense.retrieve, query, top_k=TOP_K)
            s, timings["splade_ms"] = timed_call(splade.retrieve, query, top_k=TOP_K)
            fusion_start = time.perf_counter()
            ranking = fuse_rankings([b, d, s])
            timings["fusion_ms"] = (time.perf_counter() - fusion_start) * 1000.0
        timings.setdefault("fusion_ms", 0.0)
        synchronize_cuda()
        timings["total_ms"] = (time.perf_counter() - start) * 1000.0
        cascade_rankings[qid] = ranking
        per_query[qid] = {
            "route": "EARLY_EXIT" if exit_early else "ESCALATE",
            "predicted_recall_gain": predicted_gain,
            "threshold": threshold,
            "timings": timings,
        }
        if i % 25 == 0 or i == len(test_queries):
            print(f"Learned cascade test queries: {i}/{len(test_queries)}", flush=True)

    exits = sum(row["route"] == "EARLY_EXIT" for row in per_query.values())
    cascade_times = [row["timings"]["total_ms"] for row in per_query.values()]
    static_metrics = score_run(test_qrels, static_rankings)
    cascade_metrics = score_run(test_qrels, cascade_rankings)
    branch_metrics = {name: score_run(test_qrels, values) for name, values in static_components.items()}
    result = {
        "dataset": "BEIR SciFact",
        "train_queries": len(examples),
        "test_queries": len(test_queries),
        "router": "RandomForestRegressor predicts nonnegative Recall@100 gain from escalation",
        "feature_names": FEATURE_NAMES,
        "threshold": threshold,
        "train_oof_calibration": {
            "folds": 5,
            "static_recall@100": static_train_recall,
            "target_recall@100": RECALL_FRACTION * static_train_recall,
            "allowed_aggregate_recall_loss": allowed_loss,
            "observed_oof_aggregate_loss_at_threshold": oof_loss,
            "oof_exit_rate": float(oof_exit.mean()),
            "oof_policy_recall@100": float(oof_policy.mean()),
            "oof_policy_recall_ratio": float(oof_policy.mean() / static_train_recall),
        },
        "test_static": {
            "metrics": static_metrics,
            "latency_ms": {"mean": mean(static_times), "p50": percentile(static_times, .5),
                           "p95": percentile(static_times, .95), "p99": percentile(static_times, .99)},
        },
        "test_single_retrievers": branch_metrics,
        "test_cascade": {
            "metrics": cascade_metrics,
            "early_exit_rate": exits / len(test_queries),
            "queries_with_neural_branches_skipped": exits,
            "dense_query_invocations_skipped": exits,
            "splade_query_invocations_skipped": exits,
            "latency_ms": {"mean": mean(cascade_times), "p50": percentile(cascade_times, .5),
                           "p95": percentile(cascade_times, .95), "p99": percentile(cascade_times, .99)},
            "per_query": per_query,
        },
        "model_artifact": str(model_path),
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }
    out_path = project_dir / "results" / "learned_cascade_heldout_scifact.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out_path),
        "model": str(model_path),
        "train_oof_exit_rate": result["train_oof_calibration"]["oof_exit_rate"],
        "train_oof_recall_ratio": result["train_oof_calibration"]["oof_policy_recall_ratio"],
        "test_static": static_metrics,
        "test_cascade": cascade_metrics,
        "test_exit_rate": exits / len(test_queries),
        "static_mean_ms": mean(static_times),
        "cascade_mean_ms": mean(cascade_times),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
