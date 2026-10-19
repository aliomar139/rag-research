import os
import sys
import json
import time
from collections import defaultdict
from ranx import Qrels, Run, evaluate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trec_covid_fast_loader import load_trec_covid_fast
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.splade_retriever import SPLADERetriever

def rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

def run_trec_covid_replication():
    corpus, queries, qrels = load_trec_covid_fast(max_docs=1500)

    # 1. BM25 Best Fields
    print("\n--- 1. BM25 (Best Fields) ---")
    bm25 = BM25Retriever(corpus)
    bm25_runs = {}
    bm25_lats = []
    for qid, q in queries.items():
        t0 = time.perf_counter()
        res = bm25.retrieve_best_fields(q, top_k=50)
        bm25_lats.append((time.perf_counter() - t0) * 1000)
        bm25_runs[qid] = {d: s for d, s in res}

    # 2. Dense MiniLM
    print("\n--- 2. Dense MiniLM + FAISS ---")
    dense = DenseRetriever(corpus, cache_dir="data/processed_trec_covid")
    dense_runs = {}
    dense_lats = []
    for qid, q in queries.items():
        t0 = time.perf_counter()
        res = dense.retrieve(q, top_k=50)
        dense_lats.append((time.perf_counter() - t0) * 1000)
        dense_runs[qid] = {d: s for d, s in res}

    # 3. SPLADE
    print("\n--- 3. SPLADE Sparse ---")
    splade = SPLADERetriever(corpus, cache_dir="data/processed_trec_covid", batch_size=32)
    splade_runs = {}
    splade_lats = []
    for qid, q in queries.items():
        t0 = time.perf_counter()
        res = splade.retrieve(q, top_k=50)
        splade_lats.append((time.perf_counter() - t0) * 1000)
        splade_runs[qid] = {d: s for d, s in res}

    # 4. Blended RAG (3-Way RRF)
    print("\n--- 4. Blended RAG (3-Way RRF) ---")
    blended_runs = {}
    for qid in queries:
        fused = defaultdict(float)
        for rank, (d, _) in enumerate(bm25_runs[qid].items(), start=1):
            fused[d] += rrf(rank)
        for rank, (d, _) in enumerate(dense_runs[qid].items(), start=1):
            fused[d] += rrf(rank)
        for rank, (d, _) in enumerate(splade_runs[qid].items(), start=1):
            fused[d] += rrf(rank)
        sorted_docs = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:50]
        blended_runs[qid] = {d: s for d, s in sorted_docs}

    # Evaluation: NDCG@10 (Paper Table II metric) and Recall@50
    ranx_qrels = Qrels(qrels)
    metrics_list = ["ndcg@10", "map@10", "recall@50", "hit_rate@10"]

    m_bm25 = evaluate(ranx_qrels, Run(bm25_runs), metrics=metrics_list)
    m_dense = evaluate(ranx_qrels, Run(dense_runs), metrics=metrics_list)
    m_splade = evaluate(ranx_qrels, Run(splade_runs), metrics=metrics_list)
    m_blended = evaluate(ranx_qrels, Run(blended_runs), metrics=metrics_list)

    print("\n" + "="*70)
    print(f" TREC-COVID REPLICATION RESULTS ({len(queries)} queries, {len(corpus)} docs)")
    print("="*70)
    print(f"{'Method':<20} | {'NDCG@10':<10} | {'Recall@50':<10} | {'Latency':<10}")
    print("-" * 70)
    print(f"{'BM25 (Best Fields)':<20} | {m_bm25['ndcg@10']:>10.4f} | {m_bm25['recall@50']:>10.4f} | {sum(bm25_lats)/len(bm25_lats):>7.2f} ms")
    print(f"{'Dense (MiniLM)':<20} | {m_dense['ndcg@10']:>10.4f} | {m_dense['recall@50']:>10.4f} | {sum(dense_lats)/len(dense_lats):>7.2f} ms")
    print(f"{'SPLADE (Sparse)':<20} | {m_splade['ndcg@10']:>10.4f} | {m_splade['recall@50']:>10.4f} | {sum(splade_lats)/len(splade_lats):>7.2f} ms")
    total_lat = sum(bm25_lats)/len(bm25_lats) + sum(dense_lats)/len(dense_lats) + sum(splade_lats)/len(splade_lats)
    print(f"{'Blended RAG (Paper)':<20} | {m_blended['ndcg@10']:>10.4f} | {m_blended['recall@50']:>10.4f} | {total_lat:>7.2f} ms")
    print("="*70)

    os.makedirs("results", exist_ok=True)
    with open("results/trec_covid_replication_results.json", "w") as f:
        json.dump({
            "bm25": m_bm25,
            "dense": m_dense,
            "splade": m_splade,
            "blended": m_blended
        }, f, indent=2)

if __name__ == "__main__":
    run_trec_covid_replication()

