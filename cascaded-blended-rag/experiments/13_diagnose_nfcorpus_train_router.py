"""Diagnose Ridge versus label-informed oracles on NFCorpus TRAIN only.

Run once with a GPU to save train retrieval labels. Subsequent runs reuse the
validated JSONL examples and require no neural query retrieval. Dev/test qrels
are never opened. Oracle policies use labels and cannot be deployed as routers.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.router_diagnostics import oracle_mask, policy_summary, select_threshold

TARGET = 0.99
FOLDS, SEED = 5, 42


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_base():
    path = PROJECT / "experiments" / "12_calibrate_nfcorpus_train_check_dev.py"
    spec = importlib.util.spec_from_file_location("nfcorpus_frozen_train_dev", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_examples(base, data_dir, output_dir):
    dataset_dir = base.ensure_dataset(data_dir / "raw")
    corpus = base.load_corpus(dataset_dir)
    queries, qrels_path = base.load_split(dataset_dir, "train")
    qrels = base.load_qrels(qrels_path)
    if not corpus or not queries or any(qid not in qrels for qid in queries):
        raise ValueError("Train queries must have positive qrels and a nonempty corpus")
    code_paths = [PROJECT / "experiments" / "12_calibrate_nfcorpus_train_check_dev.py",
                  Path(__file__).resolve()]
    code_paths += [PROJECT / "src" / name for name in (
        "bm25_retriever.py", "dense_retriever.py", "splade_retriever.py", "cache_utils.py")]
    provenance = {
        "schema": 1, "dataset": "BEIR NFCorpus", "split": "train",
        "corpus_sha256": sha256(dataset_dir / "corpus.jsonl"),
        "queries_sha256": sha256(dataset_dir / "queries.jsonl"),
        "train_qrels_sha256": sha256(qrels_path),
        "retrieval_code_sha256": {p.name: sha256(p) for p in code_paths},
        "feature_names": base.FEATURE_NAMES, "top_k": base.TOP_K,
        "rrf_k": base.RRF_K, "corpus_documents": len(corpus), "queries": len(queries),
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "splade_model": "naver/splade-cocondenser-ensembledistil",
    }
    manifest = output_dir / "nfcorpus_train_examples_manifest.json"
    examples_path = output_dir / "nfcorpus_train_router_examples.jsonl"
    rows = {}
    if examples_path.exists():
        if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")) != provenance:
            raise ValueError("Training example cache provenance differs; use a fresh --output-dir")
        lines = examples_path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines)-1:
                    print("Discarding incomplete final checkpoint line", flush=True)
                    continue
                raise
            qid = row["query_id"]
            if qid not in queries or qid in rows:
                raise ValueError("Unexpected or duplicate query ID in training checkpoint")
            rows[qid] = row
        # Rewrite clean complete rows before resuming an interrupted append.
        examples_path.write_text("".join(json.dumps(row)+"\n" for row in rows.values()), encoding="utf-8")
    else:
        write_json(manifest, provenance)
    missing = [(qid, text) for qid, text in queries.items() if qid not in rows]
    print(f"Train labels: {len(rows)} cached; {len(missing)} remaining; full corpus {len(corpus)}", flush=True)
    if missing:
        cache = str(data_dir / "processed_nfcorpus_router")
        bm25 = base.BM25Retriever(corpus)
        dense = base.DenseRetriever(corpus, cache_dir=cache)
        splade = base.SPLADERetriever(corpus, cache_dir=cache)
        dfs = base.doc_freqs(corpus)
        with examples_path.open("a", encoding="utf-8") as handle:
            for i, (qid, query) in enumerate(missing, 1):
                b = bm25.retrieve_best_fields(query, base.TOP_K)
                d = dense.retrieve(query, base.TOP_K)
                s = splade.retrieve(query, base.TOP_K)
                fused = base.fuse([b, d, s])
                rb, rs = base.recall(qrels[qid], b), base.recall(qrels[qid], fused)
                row = {"query_id": qid, "query": query,
                       "features": base.features(query, b, dfs, len(corpus)),
                       "bm25_recall": rb, "static_recall": rs,
                       "gain": max(0.0, rs-rb), "signed_recall_loss": rs-rb,
                       "bm25_top100_doc_ids": [doc for doc, _ in b],
                       "static_top100_doc_ids": [doc for doc, _ in fused]}
                handle.write(json.dumps(row, allow_nan=False)+"\n")
                handle.flush()
                rows[qid] = row
                if i % 50 == 0 or i == len(missing):
                    print(f"Train retrieval: {i}/{len(missing)} new queries", flush=True)
    return [rows[qid] for qid in queries], provenance


def diagnose(rows, feature_names, output_dir):
    X = np.asarray([[row["features"][name] for name in feature_names] for row in rows])
    bm25 = [row["bm25_recall"] for row in rows]
    static = [row["static_recall"] for row in rows]
    signed = [s-b for b, s in zip(bm25, static)]
    clipped = [max(0.0, cost) for cost in signed]
    if len(rows) < FOLDS or not np.isfinite(X).all():
        raise ValueError("Need at least five training rows with finite features")
    baseline = math.fsum(static) / len(rows)
    if baseline <= 0:
        raise ValueError("Static train recall must be positive")
    folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(X))
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    predicted = np.maximum(cross_val_predict(model, X, np.asarray(clipped), cv=folds, n_jobs=1), 0.0).tolist()
    budget = (1-TARGET) * baseline * len(rows)
    threshold, ridge_exits, frontier = select_threshold(predicted, clipped, budget)
    conservative = oracle_mask(clipped, budget)
    net = oracle_mask(signed, budget)
    fold_ids = [None] * len(rows)
    for fold_id, (_, validation) in enumerate(folds):
        for idx in validation:
            fold_ids[int(idx)] = fold_id
    decisions = output_dir / "nfcorpus_train_router_decisions.jsonl"
    with decisions.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({"query_id": row["query_id"], "features": row["features"],
                "bm25_recall": bm25[i], "static_recall": static[i],
                "signed_recall_loss": signed[i], "clipped_recall_loss": clipped[i],
                "oof_fold": fold_ids[i], "oof_predicted_gain": predicted[i],
                "ridge_oof_route": "EARLY_EXIT" if ridge_exits[i] else "ESCALATE",
                "oracle_conservative_route": "EARLY_EXIT" if conservative[i] else "ESCALATE",
                "oracle_net_recall_route": "EARLY_EXIT" if net[i] else "ESCALATE"}, allow_nan=False)+"\n")
    for point in frontier:
        point["exit_rate"] = point["exits"] / len(rows)
        point["conservative_recall_lower_bound"] = baseline-point["total_clipped_loss"]/len(rows)
    frontier_path = output_dir / "nfcorpus_train_threshold_frontier.json"
    write_json(frontier_path, frontier)
    ridge = policy_summary(bm25, static, ridge_exits)
    oracle = policy_summary(bm25, static, conservative)
    report = {
        "dataset": "BEIR NFCorpus", "split": "train", "queries": len(rows),
        "dev_qrels_opened": False, "test_qrels_opened": False,
        "target_recall_fraction": TARGET, "static_recall@100": baseline,
        "bm25_recall@100": math.fsum(bm25)/len(rows),
        "total_clipped_loss_budget": budget,
        "ridge_oof": {**ridge, "threshold": threshold, "folds": FOLDS, "seed": SEED,
                      "alpha": 10.0},
        "oracle_same_clipped_budget": oracle,
        "oracle_net_recall_budget": policy_summary(bm25, static, net),
        "opportunities": {
            "bm25_not_worse_than_static_queries": sum(cost <= 0 for cost in signed),
            "non_damaging_queries_escalated_by_ridge": sum(cost <= 0 and not exit_now for cost, exit_now in zip(signed, ridge_exits)),
            "recall_harming_ridge_exits": sum(cost > 0 and exit_now for cost, exit_now in zip(signed, ridge_exits)),
            "oracle_minus_ridge_exit_rate": oracle["exit_rate"]-ridge["exit_rate"],
        },
        "prediction_quantiles": {str(q): float(np.quantile(predicted, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
        "limits": [
            "Oracles use train relevance labels and are retrospective equal-query-weight upper bounds, not deployable routers.",
            "Primary oracle uses the same clipped loss budget as the previous Ridge gate. Secondary net-recall oracle credits BM25 wins.",
            "Ridge predictions are out-of-fold; its threshold is selected on all train labels. This is a development diagnostic, not an independent held-out estimate.",
            "No end-to-end latency or attainable speedup is inferred from oracle exit rates.",
        ],
    }
    write_json(output_dir / "nfcorpus_train_router_diagnostics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_base()
    rows, provenance = train_examples(base, args.data_dir, args.output_dir)
    report = diagnose(rows, provenance["feature_names"], args.output_dir)
    report["provenance"] = provenance
    write_json(args.output_dir / "nfcorpus_train_router_diagnostics.json", report)
    files = [
        "nfcorpus_train_router_diagnostics.json", "nfcorpus_train_router_examples.jsonl",
        "nfcorpus_train_router_decisions.jsonl", "nfcorpus_train_threshold_frontier.json",
        "nfcorpus_train_examples_manifest.json",
    ]
    archive_path = args.output_dir / "nfcorpus_train_diagnostics_outputs.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(args.output_dir / name, arcname=name)
    print(json.dumps({"ridge_oof": report["ridge_oof"],
                      "oracle_same_clipped_budget": report["oracle_same_clipped_budget"],
                      "oracle_net_recall_budget": report["oracle_net_recall_budget"],
                      "opportunities": report["opportunities"],
                      "download": str(archive_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
