"""Add label-free BM25 coverage features and run matched train-only nested CV.

Uses the uploaded corpus plus saved train labels; no neural model is loaded.
Original BM25 rankings must reproduce exactly before any new feature is used.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.bm25_coverage_features import BM25CoverageFeatures


def load_comparison():
    source = PROJECT / "experiments" / "14_compare_nfcorpus_train_routers.py"
    spec = importlib.util.spec_from_file_location("nfcorpus_base_comparison", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expanded_features(rows, corpus_path, provenance, output_dir):
    corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    if corpus_hash != provenance["retrieval_manifest"]["corpus_sha256"]:
        raise ValueError("Corpus does not match the corpus used for the saved retrieval labels")
    collector = PROJECT / "src" / "bm25_coverage_features.py"
    metadata = {"corpus_sha256": corpus_hash, "input_zip_sha256": provenance["input_zip_sha256"],
                "feature_code_sha256": hashlib.sha256(collector.read_bytes()).hexdigest(),
                "query_ids": [row["query_id"] for row in rows]}
    cache_path = output_dir / "nfcorpus_train_coverage_features.jsonl"
    manifest_path = output_dir / "nfcorpus_train_coverage_features_manifest.json"
    if cache_path.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) == metadata:
            cached = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
            if [row["query_id"] for row in cached] == metadata["query_ids"]:
                print("Reusing validated coverage feature cache", flush=True)
                return cached, metadata
    corpus = {}
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        corpus[str(row["_id"])] = {"title": row.get("title", "").strip(), "abstract": row.get("text", "").strip()}
    extractor = BM25CoverageFeatures(corpus)
    expanded = []
    for i, row in enumerate(rows, 1):
        ranking, features, timings = extractor.retrieve_with_features(row["query"])
        if [doc for doc, _ in ranking] != row["bm25_top100_doc_ids"]:
            raise ValueError(f"BM25 ranking differs from saved label source: {row['query_id']}")
        expanded.append({"query_id": row["query_id"], "extra_features": features,
                         "local_timings_ms": timings})
        if i % 250 == 0 or i == len(rows):
            print(f"Coverage features and BM25 rank checks: {i}/{len(rows)}", flush=True)
    cache_path.write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in expanded), encoding="utf-8")
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return expanded, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, default=PROJECT.parent / "nfcorpus_train_diagnostics_outputs.zip")
    parser.add_argument("--corpus", type=Path, default=PROJECT.parent / "nfcorpus_corpus_for_features.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = load_comparison()
    rows, X_base, gains, bm25, static, provenance, oracle = previous.read_examples(args.input_zip)
    extras, feature_metadata = expanded_features(rows, args.corpus, provenance, args.output_dir)
    feature_names = list(extras[0]["extra_features"])
    extra_X = np.asarray([[row["extra_features"][name] for name in feature_names] for row in extras], dtype=float)
    X = np.hstack([X_base, extra_X])
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite expanded feature matrix")
    evaluator = previous.load_evaluator()
    output = args.output_dir / "nfcorpus_train_coverage_comparison.json"
    results = []
    for name in previous.candidate_models(evaluator.SEED):
        print(f"Enriched nested train CV: {name}", flush=True)
        with threadpool_limits(limits=1):
            result = evaluator.nested_evaluate(name,
                lambda seed, key=name: previous.candidate_models(seed)[key], X, gains, bm25, static)
        ratios = [fold["outer_recall_ratio"] for fold in result["folds"]]
        result["outer_folds_meeting_target"] = sum(r >= evaluator.RECALL_FRACTION for r in ratios)
        result["minimum_outer_fold_recall_ratio"] = min(ratios)
        result["passes_pooled_target"] = result["nested_oof_recall_ratio"] >= evaluator.RECALL_FRACTION
        results.append(result)
        timings = [row["local_timings_ms"]["extra_feature_ms"] for row in extras]
        report = {"dataset": "BEIR NFCorpus", "split": "train", "queries": len(rows),
            "base_feature_count": X_base.shape[1], "extra_feature_count": extra_X.shape[1],
            "extra_feature_names": feature_names, "models": results,
            "models_expected": len(previous.candidate_models(evaluator.SEED)),
            "quality_target_fraction": evaluator.RECALL_FRACTION,
            "outer_folds": evaluator.OUTER_FOLDS, "inner_folds": evaluator.INNER_FOLDS, "seed": evaluator.SEED,
            "all_bm25_top100_rankings_reproduced": True,
            "mean_local_extra_feature_ms": float(np.mean(timings)),
            "p95_local_extra_feature_ms": float(np.percentile(timings, 95)),
            "provenance": provenance, "feature_provenance": feature_metadata,
            "versions": {"numpy": np.__version__},
            "limits": [
                "Features use corpus/query/BM25 only. Static rankings and recall labels are excluded from feature extraction.",
                "Outer and inner folds match the original-feature comparison. Thresholds use inner labels only.",
                "This is iterative train-only development, not an independent held-out guarantee.",
                "Local feature and prediction timings are not end-to-end retrieval latency or Kaggle speedup measurements.",
            ]}
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"model": name, "exit_rate": result["nested_oof_exit_rate"],
            "recall_ratio": result["nested_oof_recall_ratio"],
            "folds_meeting_target": result["outer_folds_meeting_target"],
            "predict_ms": result["local_single_row_predict_ms"]}), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
