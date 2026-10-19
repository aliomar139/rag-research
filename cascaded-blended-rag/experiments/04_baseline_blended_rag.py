import os
import sys
import json
from collections import defaultdict
from ranx import Qrels, Run, evaluate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import download_and_load_scifact

def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

def run_blended_rag_experiment():
    print("Loading SciFact dataset...")
    corpus, queries, qrels = download_and_load_scifact()

    # Load caches from disk
    print("Loading retriever caches from results/...")
    with open("results/bm25_single_cache.json", "r") as f:
        bm25_cache = json.load(f)
    with open("results/dense_cache.json", "r") as f:
        dense_cache = json.load(f)
    with open("results/splade_cache.json", "r") as f:
        splade_cache = json.load(f)

    blended_run_dict = {}
    latencies = []

    for qid in queries:
        bm25_entry = bm25_cache[qid]
        dense_entry = dense_cache[qid]
        splade_entry = splade_cache[qid]

        # Static Blended RAG latency: all 3 retrievers executed for every query
        # In sequential execution: BM25 + Dense + SPLADE
        query_lat = bm25_entry["latency_ms"] + dense_entry["latency_ms"] + splade_entry["latency_ms"]
        latencies.append(query_lat)

        # 3-Way Reciprocal Rank Fusion (k=60 as standard / paper default)
        fused_scores = defaultdict(float)

        for rank, (did, _) in enumerate(bm25_entry["top_docs"], start=1):
            fused_scores[did] += rrf_score(rank, k=60)

        for rank, (did, _) in enumerate(dense_entry["top_docs"], start=1):
            fused_scores[did] += rrf_score(rank, k=60)

        for rank, (did, _) in enumerate(splade_entry["top_docs"], start=1):
            fused_scores[did] += rrf_score(rank, k=60)

        # Sort top-100 by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:100]
        blended_run_dict[qid] = {did: score for did, score in sorted_docs}

    ranx_qrels = Qrels(qrels)
    metrics = evaluate(ranx_qrels, Run(blended_run_dict), metrics=["ndcg@10", "recall@100", "map@10"])
    avg_lat = sum(latencies) / len(latencies)

    print("\n=======================================================")
    print(" BASE PAPER REPRODUCTION: BLENDED RAG (Sawarkar et al.)")
    print("=======================================================")
    print(f"NDCG@10   : {metrics['ndcg@10']:.4f}")
    print(f"Recall@100: {metrics['recall@100']:.4f}")
    print(f"MAP@10    : {metrics['map@10']:.4f}")
    print(f"Avg Latency (Sequential): {avg_lat:.2f} ms")
    print("=======================================================\n")

    with open("results/blended_rag_results.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "avg_latency_ms": avg_lat,
            "run": blended_run_dict
        }, f, indent=2)

if __name__ == "__main__":
    run_blended_rag_experiment()

