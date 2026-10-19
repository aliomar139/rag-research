"""Train a BM25 confidence gate on SciFact train queries and evaluate it on test.

Unlike the earlier cache simulation, the held-out cascade below conditionally
calls the dense and SPLADE retrievers. Run from `cascaded-blended-rag/`.
"""

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from statistics import mean

import torch
from ranx import Qrels, Run, evaluate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import download_and_load_scifact
from src.bm25_retriever import BM25Retriever, simple_tokenize
from src.cascaded_router import CascadedRouter
from src.dense_retriever import DenseRetriever
from src.splade_retriever import SPLADERetriever


TAU_HIGH = [18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
TAU_GAP = [2.0, 3.0, 4.0, 5.0, 6.0]
TAU_IDF = [2.5, 3.5, 4.5]
TOP_K = 100
TRAIN_RECALL_FRACTION = 0.98
RRF_K = 60


def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn, *args, **kwargs):
    synchronize_cuda()
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    synchronize_cuda()
    return value, (time.perf_counter() - start) * 1000.0


def make_doc_freqs(corpus):
    freqs = defaultdict(int)
    for doc in corpus.values():
        for token in set(simple_tokenize(f"{doc['title']} {doc['abstract']}")):
            freqs[token] += 1
    return freqs


def fuse_rankings(rankings, top_k=TOP_K):
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (RRF_K + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]


def as_run(rankings):
    return {qid: {doc_id: score for doc_id, score in ranking} for qid, ranking in rankings.items()}


def score_run(qrels, rankings):
    metrics = evaluate(
        Qrels(qrels),
        Run(as_run(rankings)),
        metrics=["recall@100", "ndcg@10", "map@10"],
    )
    return {key: float(value) for key, value in metrics.items()}


def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - index) + ordered[upper] * (index - lower))


def main():
    corpus, train_queries, train_qrels = download_and_load_scifact(split="train")
    _, test_queries, test_qrels = download_and_load_scifact(split="test")
    print(f"Corpus: {len(corpus)} | train queries: {len(train_queries)} | test queries: {len(test_queries)}")

    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, cache_dir="data/processed_scifact")
    splade = SPLADERetriever(corpus, cache_dir="data/processed_scifact")
    doc_freqs = make_doc_freqs(corpus)
    router = CascadedRouter(doc_freqs, len(corpus))

    # Warm each component before collecting measurements.
    warmup_queries = list(train_queries.values())[:5]
    for query in warmup_queries:
        bm25.retrieve_best_fields(query, top_k=TOP_K)
        dense.retrieve(query, top_k=TOP_K)
        splade.retrieve(query, top_k=TOP_K)

    # Calibrate only on the designated train qrels. All branches run here to
    # build the oracle-free training table; no test qrels influence selection.
    train_rows = {}
    train_static = {}
    for qid, query in train_queries.items():
        b, b_ms = timed_call(bm25.retrieve_best_fields, query, top_k=TOP_K)
        d, d_ms = timed_call(dense.retrieve, query, top_k=TOP_K)
        s, s_ms = timed_call(splade.retrieve, query, top_k=TOP_K)
        train_rows[qid] = {
            "query": query,
            "bm25": b,
            "dense": d,
            "splade": s,
            "latencies": {"bm25_ms": b_ms, "dense_ms": d_ms, "splade_ms": s_ms},
        }
        fuse_start = time.perf_counter()
        train_static[qid] = fuse_rankings([b, d, s])
        train_rows[qid]["latencies"]["fusion_ms"] = (time.perf_counter() - fuse_start) * 1000.0

    train_baseline = score_run(train_qrels, train_static)["recall@100"]
    train_target = TRAIN_RECALL_FRACTION * train_baseline
    best = None
    for tau_high in TAU_HIGH:
        for tau_gap in TAU_GAP:
            for tau_idf in TAU_IDF:
                candidate_router = CascadedRouter(
                    doc_freqs, len(corpus), tau_high=tau_high,
                    tau_gap=tau_gap, tau_idf=tau_idf,
                )
                candidate_run = {}
                candidate_latency = []
                for qid, row in train_rows.items():
                    route_start = time.perf_counter()
                    decision, _ = candidate_router.route_query(
                        simple_tokenize(row["query"]), row["bm25"]
                    )
                    route_ms = (time.perf_counter() - route_start) * 1000.0
                    if decision == "EARLY_EXIT":
                        candidate_run[qid] = row["bm25"]
                        candidate_latency.append(row["latencies"]["bm25_ms"] + route_ms)
                    else:
                        candidate_run[qid] = train_static[qid]
                        candidate_latency.append(
                            row["latencies"]["bm25_ms"]
                            + route_ms
                            + row["latencies"]["dense_ms"]
                            + row["latencies"]["splade_ms"]
                            + row["latencies"]["fusion_ms"]
                        )

                recall = score_run(train_qrels, candidate_run)["recall@100"]
                avg_latency = mean(candidate_latency)
                if recall >= train_target and (best is None or avg_latency < best["train_estimated_latency_ms"]):
                    best = {
                        "thresholds": {
                            "tau_high": tau_high,
                            "tau_gap": tau_gap,
                            "tau_idf": tau_idf,
                        },
                        "train_recall@100": recall,
                        "train_static_recall@100": train_baseline,
                        "train_estimated_latency_ms": avg_latency,
                    }

    if best is None:
        # No tested gate met the predeclared training quality constraint.
        best = {
            "thresholds": None,
            "train_recall@100": train_baseline,
            "train_static_recall@100": train_baseline,
            "train_estimated_latency_ms": None,
            "selection": "static_fallback_no_threshold_met_constraint",
        }
        selected = None
    else:
        selected = CascadedRouter(
            doc_freqs,
            len(corpus),
            tau_high=best["thresholds"]["tau_high"],
            tau_gap=best["thresholds"]["tau_gap"],
            tau_idf=best["thresholds"]["tau_idf"],
        )

    # Measure both systems on the exact same held-out queries. The static path
    # runs all branches. The cascade path really skips neural calls on exits.
    static_rankings = {}
    test_branch_rankings = {"bm25_best_fields": {}, "dense": {}, "splade": {}}
    static_ms = []
    for qid, query in test_queries.items():
        query_start = time.perf_counter()
        b, b_ms = timed_call(bm25.retrieve_best_fields, query, top_k=TOP_K)
        d, d_ms = timed_call(dense.retrieve, query, top_k=TOP_K)
        s, s_ms = timed_call(splade.retrieve, query, top_k=TOP_K)
        test_branch_rankings["bm25_best_fields"][qid] = b
        test_branch_rankings["dense"][qid] = d
        test_branch_rankings["splade"][qid] = s
        static_rankings[qid] = fuse_rankings([b, d, s])
        synchronize_cuda()
        static_ms.append({
            "bm25_ms": b_ms,
            "dense_ms": d_ms,
            "splade_ms": s_ms,
            "total_ms": (time.perf_counter() - query_start) * 1000.0,
        })

    cascade_rankings = {}
    cascade_rows = {}
    for qid, query in test_queries.items():
        query_start = time.perf_counter()
        b, b_ms = timed_call(bm25.retrieve_best_fields, query, top_k=TOP_K)
        route_start = time.perf_counter()
        if selected is None:
            decision, telemetry = "ESCALATE", {"decision": "static_fallback"}
        else:
            decision, telemetry = selected.route_query(simple_tokenize(query), b)
        route_ms = (time.perf_counter() - route_start) * 1000.0

        timings = {"bm25_ms": b_ms, "router_ms": route_ms, "dense_ms": 0.0, "splade_ms": 0.0}
        if decision == "EARLY_EXIT":
            ranking = b
        else:
            d, timings["dense_ms"] = timed_call(dense.retrieve, query, top_k=TOP_K)
            s, timings["splade_ms"] = timed_call(splade.retrieve, query, top_k=TOP_K)
            fusion_start = time.perf_counter()
            ranking = fuse_rankings([b, d, s])
            timings["fusion_ms"] = (time.perf_counter() - fusion_start) * 1000.0
        timings.setdefault("fusion_ms", 0.0)
        synchronize_cuda()
        timings["total_ms"] = (time.perf_counter() - query_start) * 1000.0
        cascade_rankings[qid] = ranking
        cascade_rows[qid] = {
            "route": decision,
            "telemetry": telemetry,
            "timings": timings,
        }

    static_metrics = score_run(test_qrels, static_rankings)
    branch_metrics = {
        name: score_run(test_qrels, rankings)
        for name, rankings in test_branch_rankings.items()
    }
    cascade_metrics = score_run(test_qrels, cascade_rankings)
    static_total = [row["total_ms"] for row in static_ms]
    cascade_total = [row["timings"]["total_ms"] for row in cascade_rows.values()]
    early_exits = sum(row["route"] == "EARLY_EXIT" for row in cascade_rows.values())
    n_test = len(test_queries)
    escalations = n_test - early_exits

    results = {
        "dataset": "BEIR SciFact",
        "corpus_documents": len(corpus),
        "train_queries": len(train_queries),
        "test_queries": n_test,
        "bm25_variant": "Best Fields: max(title BM25, abstract BM25)",
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "sparse_model": "naver/splade-cocondenser-ensembledistil",
        "rrf_k": RRF_K,
        "train_recall_constraint_fraction": TRAIN_RECALL_FRACTION,
        "selection": best,
        "test_static": {
            "metrics": static_metrics,
            "latency_ms": {
                "mean": mean(static_total),
                "p50": percentile(static_total, 0.50),
                "p95": percentile(static_total, 0.95),
                "p99": percentile(static_total, 0.99),
            },
        },
        "test_single_retrievers": branch_metrics,
        "test_cascade": {
            "metrics": cascade_metrics,
            "early_exit_rate": early_exits / n_test,
            "queries_with_neural_branches_skipped": early_exits,
            "dense_query_invocations_skipped": early_exits,
            "splade_query_invocations_skipped": early_exits,
            "neural_model_invocations_skipped": 2 * early_exits,
            "dense_query_invocations": escalations,
            "splade_query_invocations": escalations,
            "latency_ms": {
                "mean": mean(cascade_total),
                "p50": percentile(cascade_total, 0.50),
                "p95": percentile(cascade_total, 0.95),
                "p99": percentile(cascade_total, 0.99),
            },
            "per_query": cascade_rows,
        },
        "test_static_per_query_latency_ms": static_ms,
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/cascade_heldout_scifact.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({
        "output": out_path,
        "thresholds": best["thresholds"],
        "static_test_metrics": static_metrics,
        "cascade_test_metrics": cascade_metrics,
        "early_exit_rate": early_exits / n_test,
        "static_mean_ms": mean(static_total),
        "cascade_mean_ms": mean(cascade_total),
        "device": results["hardware"]["cuda_device"],
    }, indent=2))


if __name__ == "__main__":
    main()
