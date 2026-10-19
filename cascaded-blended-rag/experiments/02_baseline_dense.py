import os
import sys
import json
import time
from typing import Dict, List
from ranx import Qrels, Run, evaluate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import download_and_load_scifact
from src.dense_retriever import DenseRetriever

def run_dense_experiment():
    print("Loading SciFact dataset...")
    corpus, queries, qrels = download_and_load_scifact()
    
    print("Initializing Dense Retriever (MiniLM + FAISS)...")
    retriever = DenseRetriever(corpus)
    
    # 5 warm-up queries
    sample_queries = list(queries.values())[:5]
    for q in sample_queries:
        _ = retriever.retrieve(q, top_k=10)
        
    print("Evaluating Dense Retriever on 300 test queries (batch_size=1 timing)...")
    dense_run_dict = {}
    dense_per_query_cache = {}
    dense_latencies = []
    
    for qid, query_text in queries.items():
        t0 = time.perf_counter()
        results = retriever.retrieve(query_text, top_k=100)
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000.0
        dense_latencies.append(latency_ms)
        
        dense_run_dict[qid] = {did: score for did, score in results}
        dense_per_query_cache[qid] = {
            "latency_ms": latency_ms,
            "top_docs": results
        }
        
    ranx_qrels = Qrels(qrels)
    metrics = evaluate(ranx_qrels, Run(dense_run_dict), metrics=["ndcg@10", "recall@100", "map@10"])
    avg_latency = sum(dense_latencies) / len(dense_latencies)
    
    print("\n--- Dense MiniLM Baseline Results (300 Test Queries) ---")
    print(f"NDCG@10   : {metrics['ndcg@10']:.4f}")
    print(f"Recall@100: {metrics['recall@100']:.4f}")
    print(f"MAP@10    : {metrics['map@10']:.4f}")
    print(f"Avg Latency (batch_size=1): {avg_latency:.2f} ms")
    
    os.makedirs("results", exist_ok=True)
    with open("results/dense_cache.json", "w") as f:
        json.dump(dense_per_query_cache, f, indent=2)
    print("Saved cache to results/dense_cache.json")

if __name__ == "__main__":
    run_dense_experiment()

