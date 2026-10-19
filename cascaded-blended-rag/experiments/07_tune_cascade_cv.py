"""Legacy exploratory analysis on cached SciFact test-query rankings.

This script uses the test query set for both threshold selection and reported
fold scores, reads the single-field BM25 cache, and reconstructs latency from
cached component timings. Do not treat its outputs as the held-out evaluation.
Use 08_evaluate_cascade_heldout.py for train-qrels calibration and conditional
test-time execution.
"""

import os
import sys
import json
import re
from collections import defaultdict
import numpy as np
from ranx import Qrels, Run, evaluate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import download_and_load_scifact
from src.cascaded_router import CascadedRouter

def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

def rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

def run_5fold_cross_validation():
    print("Loading SciFact dataset & cached retriever results...")
    corpus, queries, qrels = download_and_load_scifact()

    # Build document frequencies for IDF calculation
    doc_freqs = defaultdict(int)
    for did in corpus:
        tokens = set(tokenize(f"{corpus[did]['title']} {corpus[did]['abstract']}"))
        for tok in tokens:
            doc_freqs[tok] += 1
    total_docs = len(corpus)

    with open("results/bm25_single_cache.json", "r") as f:
        bm25_cache = json.load(f)
    with open("results/dense_cache.json", "r") as f:
        dense_cache = json.load(f)
    with open("results/splade_cache.json", "r") as f:
        splade_cache = json.load(f)

    qids = list(queries.keys())
    np.random.seed(42)
    shuffled_qids = np.random.permutation(qids)
    folds = np.array_split(shuffled_qids, 5)

    # Grid search candidate parameters
    tau_high_candidates = [18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    tau_gap_candidates = [2.0, 3.0, 4.0, 5.0, 6.0]
    tau_idf_candidates = [2.5, 3.5, 4.5]

    ranx_qrels = Qrels(qrels)

    # Precompute blended run results for all queries
    blended_cache = {}
    for qid in qids:
        fused = defaultdict(float)
        for r, (d, _) in enumerate(bm25_cache[qid]["top_docs"][:100], start=1):
            fused[d] += rrf(r)
        for r, (d, _) in enumerate(dense_cache[qid]["top_docs"][:100], start=1):
            fused[d] += rrf(r)
        for r, (d, _) in enumerate(splade_cache[qid]["top_docs"][:100], start=1):
            fused[d] += rrf(r)
        blended_cache[qid] = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:100]

    def evaluate_thresholds(eval_ids, t_high, t_gap, t_idf):
        router = CascadedRouter(doc_freqs, total_docs, tau_high=t_high, tau_gap=t_gap, tau_idf=t_idf)
        run_dict = {}
        total_lat = 0.0
        early_exits = 0

        for qid in eval_ids:
            q_toks = tokenize(queries[qid])
            decision, telemetry = router.route_query(q_toks, bm25_cache[qid]["top_docs"])

            bm25_lat = bm25_cache[qid]["latency_ms"]
            dense_lat = dense_cache[qid]["latency_ms"]
            splade_lat = splade_cache[qid]["latency_ms"]

            if decision == "EARLY_EXIT":
                early_exits += 1
                total_lat += bm25_lat
                run_dict[qid] = {d: s for d, s in bm25_cache[qid]["top_docs"][:100]}
            else:
                total_lat += (bm25_lat + dense_lat + splade_lat)
                run_dict[qid] = {d: s for d, s in blended_cache[qid]}

        sub_qrels = Qrels({qid: qrels[qid] for qid in eval_ids})
        m = evaluate(sub_qrels, Run(run_dict), metrics=["ndcg@10", "recall@100"])
        # ranx returns float when single metric, dict when list
        rec_val = m["recall@100"] if isinstance(m, dict) else m
        ndcg_val = m["ndcg@10"] if isinstance(m, dict) else m
        avg_lat = total_lat / len(eval_ids)
        exit_pct = (early_exits / len(eval_ids)) * 100
        return float(rec_val), float(ndcg_val), avg_lat, exit_pct

    print("\n--- Running 5-Fold Cross-Validation on SciFact ---")
    cv_test_recalls = []
    cv_test_ndcgs = []
    cv_test_latencies = []
    cv_test_exit_pcts = []
    best_params_per_fold = []

    for fold_idx in range(5):
        test_ids = list(folds[fold_idx])
        train_ids = [qid for i, f in enumerate(folds) if i != fold_idx for qid in f]

        # Calculate Blended baseline recall on train set
        baseline_train_run = {qid: {d: s for d, s in blended_cache[qid]} for qid in train_ids}
        train_qrels = Qrels({qid: qrels[qid] for qid in train_ids})
        m_base_train = evaluate(train_qrels, Run(baseline_train_run), metrics=["recall@100"])
        base_train_rec = m_base_train["recall@100"] if isinstance(m_base_train, dict) else float(m_base_train)
        target_recall = base_train_rec * 0.98  # Constrain recall loss to < 2%


        # Grid search on train_ids
        best_cfg = None
        min_train_latency = float("inf")

        for th in tau_high_candidates:
            for tg in tau_gap_candidates:
                for ti in tau_idf_candidates:
                    rec, ndcg, lat, exit_rate = evaluate_thresholds(train_ids, th, tg, ti)
                    if rec >= target_recall: # Meets recall constraint
                        if lat < min_train_latency:
                            min_train_latency = lat
                            best_cfg = (th, tg, ti)

        if best_cfg is None:
            best_cfg = (26.0, 4.0, 3.5) # Safe fallback

        best_params_per_fold.append(best_cfg)

        # Evaluate out-of-sample on test_ids
        t_rec, t_ndcg, t_lat, t_exit = evaluate_thresholds(test_ids, *best_cfg)
        cv_test_recalls.append(t_rec)
        cv_test_ndcgs.append(t_ndcg)
        cv_test_latencies.append(t_lat)
        cv_test_exit_pcts.append(t_exit)

        print(f"Fold {fold_idx + 1}/5: Tuned {best_cfg} -> Test Recall@100: {t_rec:.4f} | Latency: {t_lat:.2f} ms | Early Exits: {t_exit:.1f}%")

    mean_rec = np.mean(cv_test_recalls)
    mean_ndcg = np.mean(cv_test_ndcgs)
    mean_lat = np.mean(cv_test_latencies)
    mean_exits = np.mean(cv_test_exit_pcts)

    # Base Blended RAG baseline for comparison
    full_base_run = {qid: {d: s for d, s in blended_cache[qid]} for qid in qids}
    full_base_m = evaluate(ranx_qrels, Run(full_base_run), metrics=["recall@100", "ndcg@10"])
    base_lat = (20.12 + 14.93 + 55.82) # Sequential sum = 90.87 ms

    print("\n" + "="*75)
    print(" 5-FOLD CROSS-VALIDATION SUMMARY: CASCADED BLENDED RAG")
    print("="*75)
    print(f"Static Blended RAG (Base Paper): Recall@100 = {full_base_m['recall@100']:.4f} | NDCG@10 = {full_base_m['ndcg@10']:.4f} | Latency = {base_lat:.2f} ms")
    print(f"Cascaded Blended RAG (Our System): Recall@100 = {mean_rec:.4f} | NDCG@10 = {mean_ndcg:.4f} | Latency = {mean_lat:.2f} ms")
    print("-" * 75)
    lat_savings = ((base_lat - mean_lat) / base_lat) * 100
    print(f"Neural Calls Avoided (Early Exit Rate): {mean_exits:.1f}%")
    print(f"End-to-End Latency Reduction          : {lat_savings:.1f}% FASTER")
    print(f"Recall Preserved                      : {(mean_rec / full_base_m['recall@100']) * 100:.2f}%")
    print("="*75)

    os.makedirs("results", exist_ok=True)
    with open("results/cv_cascaded_results.json", "w") as f:
        json.dump({
            "mean_test_recall": mean_rec,
            "mean_test_ndcg": mean_ndcg,
            "mean_test_latency_ms": mean_lat,
            "mean_neural_avoided_pct": mean_exits,
            "latency_reduction_pct": lat_savings,
            "best_params_per_fold": best_params_per_fold
        }, f, indent=2)

if __name__ == "__main__":
    run_5fold_cross_validation()
