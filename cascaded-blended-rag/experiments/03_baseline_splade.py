import os
import sys
import json
import time
from ranx import Qrels, Run, evaluate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import download_and_load_scifact
from src.splade_retriever import SPLADERetriever

def run_splade_experiment():
    print("Loading SciFact dataset...")
    corpus, queries, qrels = download_and_load_scifact()

    print("Initializing SPLADE retriever...")
    retriever = SPLADERetriever(corpus, batch_size=16)

    # 5 warm-ups
    sample_queries = list(queries.values())[:5]
    for q in sample_queries:
        _ = retriever.retrieve(q, top_k=10)

    print("Evaluating SPLADE on 300 test queries (batch_size=1 timing)...")
    splade_run_dict = {}
    splade_per_query_cache = {}
    splade_latencies = []

    for qid, query_text in queries.items():
        t0 = time.perf_counter()
        results = retriever.retrieve(query_text, top_k=100)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        splade_latencies.append(latency_ms)

        splade_run_dict[qid] = {did: score for did, score in results}
        splade_per_query_cache[qid] = {
            "latency_ms": latency_ms,
            "top_docs": results
        }

    ranx_qrels = Qrels(qrels)
    metrics = evaluate(ranx_qrels, Run(splade_run_dict), metrics=["ndcg@10", "recall@100", "map@10"])
    avg_latency = sum(splade_latencies) / len(splade_latencies)

    print("\n--- SPLADE Baseline Results (300 Test Queries) ---")
    print(f"NDCG@10   : {metrics['ndcg@10']:.4f}")
    print(f"Recall@100: {metrics['recall@100']:.4f}")
    print(f"MAP@10    : {metrics['map@10']:.4f}")
    print(f"Avg Latency (batch_size=1): {avg_latency:.2f} ms")

    os.makedirs("results", exist_ok=True)
    with open("results/splade_cache.json", "w") as f:
        json.dump(splade_per_query_cache, f, indent=2)
    print("Saved cache to results/splade_cache.json")

if __name__ == "__main__":
    run_splade_experiment()

