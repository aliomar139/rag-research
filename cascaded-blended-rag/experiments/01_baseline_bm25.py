import os
import sys
import json
import time
from typing import Dict, List
from ranx import Qrels, Run, evaluate

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import download_and_load_scifact
from src.bm25_retriever import BM25Retriever

def run_bm25_experiment():
    print("Loading SciFact dataset...")
    corpus, queries, qrels = download_and_load_scifact()
    
    print("Initializing BM25 indexes (Single & Best Fields)...")
    retriever = BM25Retriever(corpus)
    
    # Warm-up (5 queries) to avoid initial cold-start skew
    sample_queries = list(queries.values())[:5]
    for q in sample_queries:
        _ = retriever.retrieve_single(q, top_k=10)
        _ = retriever.retrieve_best_fields(q, top_k=10)
        
    print("Evaluating BM25 Single-Field (Title + Abstract)...")
    single_run_dict = {}
    single_per_query_cache = {}
    single_latencies = []
    
    for qid, query_text in queries.items():
        t0 = time.perf_counter()
        results = retriever.retrieve_single(query_text, top_k=100)
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000.0
        single_latencies.append(latency_ms)
        
        single_run_dict[qid] = {did: score for did, score in results}
        single_per_query_cache[qid] = {
            "latency_ms": latency_ms,
            "top_docs": results  # [(doc_id, score), ...]
        }
        
    print("Evaluating BM25 Best Fields...")
    best_run_dict = {}
    best_per_query_cache = {}
    best_latencies = []
    
    for qid, query_text in queries.items():
        t0 = time.perf_counter()
        results = retriever.retrieve_best_fields(query_text, top_k=100)
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000.0
        best_latencies.append(latency_ms)
        
        best_run_dict[qid] = {did: score for did, score in results}
        best_per_query_cache[qid] = {
            "latency_ms": latency_ms,
            "top_docs": results
        }
        
    # ranx evaluation
    ranx_qrels = Qrels(qrels)
    single_metrics = evaluate(ranx_qrels, Run(single_run_dict), metrics=["ndcg@10", "recall@100", "map@10"])
    best_metrics = evaluate(ranx_qrels, Run(best_run_dict), metrics=["ndcg@10", "recall@100", "map@10"])
    
    avg_single_lat = sum(single_latencies) / len(single_latencies)
    avg_best_lat = sum(best_latencies) / len(best_latencies)
    
    print("\n--- BM25 Results (300 Test Queries) ---")
    print(f"Single-Field: NDCG@10 = {single_metrics['ndcg@10']:.4f} | Recall@100 = {single_metrics['recall@100']:.4f} | Latency = {avg_single_lat:.2f} ms")
    print(f"Best Fields : NDCG@10 = {best_metrics['ndcg@10']:.4f} | Recall@100 = {best_metrics['recall@100']:.4f} | Latency = {avg_best_lat:.2f} ms")
    
    # Save cache for Phase 4 cascade tuning
    os.makedirs("results", exist_ok=True)
    with open("results/bm25_single_cache.json", "w") as f:
        json.dump(single_per_query_cache, f, indent=2)
    with open("results/bm25_best_cache.json", "w") as f:
        json.dump(best_per_query_cache, f, indent=2)
    print("Cached top-100 results saved to results/bm25_*_cache.json")

if __name__ == "__main__":
    run_bm25_experiment()

