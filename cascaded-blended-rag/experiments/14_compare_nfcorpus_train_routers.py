"""CPU-only nested CV over saved NFCorpus TRAIN retrieval examples.

No retrieval models, downloads, dev qrels, or test qrels are used. Four fixed
model specifications are compared, using the previous nested-CV evaluator.
Outer-fold results are development comparisons, not untouched final tests.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]


def candidate_models(seed):
    return {
        "ridge_alpha10": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "tree_depth4_leaf25": DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=25, random_state=seed),
        "extra_trees64_depth8_leaf10": ExtraTreesRegressor(
            n_estimators=64, max_depth=8, min_samples_leaf=10,
            max_features=0.8, n_jobs=1, random_state=seed),
        "hist_gradient_boosting100_leaf7": HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=7, min_samples_leaf=20,
            learning_rate=0.05, l2_regularization=1.0,
            early_stopping=False, random_state=seed),
    }


def read_examples(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("nfcorpus_train_examples_manifest.json"))
        report = json.loads(archive.read("nfcorpus_train_router_diagnostics.json"))
        raw = archive.read("nfcorpus_train_router_examples.jsonl")
    if manifest.get("split") != "train" or report.get("split") != "train":
        raise ValueError("This comparison only accepts train artifacts")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != manifest["queries"] or len(rows) != report["queries"]:
        raise ValueError("Query counts differ between labels and provenance")
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate training query IDs")
    feature_names = manifest["feature_names"]
    X = np.asarray([[row["features"][name] for name in feature_names] for row in rows], dtype=float)
    bm25 = np.asarray([row["bm25_recall"] for row in rows], dtype=float)
    static = np.asarray([row["static_recall"] for row in rows], dtype=float)
    gains = np.maximum(static-bm25, 0.0)
    if not np.isfinite(X).all() or not np.isfinite(bm25).all() or not np.isfinite(static).all():
        raise ValueError("Nonfinite input data")
    if ((bm25 < 0) | (bm25 > 1) | (static < 0) | (static > 1)).any():
        raise ValueError("Recall labels must be in [0, 1]")
    if not np.allclose(gains, [row["gain"] for row in rows], atol=1e-12, rtol=0):
        raise ValueError("Saved gains differ from per-query recall labels")
    if abs(float(static.mean())-report["static_recall@100"]) > 1e-12:
        raise ValueError("Saved baseline differs from the diagnostic report")
    provenance = {"input_zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "examples_sha256": hashlib.sha256(raw).hexdigest(),
                  "retrieval_manifest": manifest}
    return rows, X, gains, bm25, static, provenance, report


def load_evaluator():
    source = PROJECT / "experiments" / "10_compare_router_models.py"
    spec = importlib.util.spec_from_file_location("previous_nested_router_evaluator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path,
                        default=PROJECT.parent / "nfcorpus_train_diagnostics_outputs.zip")
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "results" / "nfcorpus_train_router_comparison.json")
    args = parser.parse_args()
    rows, X, gains, bm25, static, provenance, oracle = read_examples(args.input_zip)
    evaluator = load_evaluator()
    results = []
    specs = {name: repr(model) for name, model in candidate_models(evaluator.SEED).items()}
    for name in specs:
        print(f"Nested train CV: {name}", flush=True)
        with threadpool_limits(limits=1):
            result = evaluator.nested_evaluate(
                name, lambda seed, key=name: candidate_models(seed)[key], X, gains, bm25, static)
        ratios = [fold["outer_recall_ratio"] for fold in result["folds"]]
        result["outer_folds_meeting_target"] = sum(r >= evaluator.RECALL_FRACTION for r in ratios)
        result["minimum_outer_fold_recall_ratio"] = min(ratios)
        result["passes_pooled_target"] = result["nested_oof_recall_ratio"] >= evaluator.RECALL_FRACTION
        result["specification"] = specs[name]
        results.append(result)
        report = {
            "dataset": "BEIR NFCorpus", "split": "train", "queries": len(rows),
            "features": provenance["retrieval_manifest"]["feature_names"],
            "quality_target_fraction": evaluator.RECALL_FRACTION,
            "outer_folds": evaluator.OUTER_FOLDS, "inner_folds": evaluator.INNER_FOLDS,
            "seed": evaluator.SEED, "models_completed": len(results),
            "local_thread_limit": 1,
            "models_expected": len(specs), "models": results,
            "oracle_same_clipped_budget": oracle["oracle_same_clipped_budget"],
            "provenance": provenance,
            "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
            "limits": [
                "Nested train-only comparison: thresholds use inner OOF labels; outer-fold labels score the fixed policy.",
                "Model selection from these comparisons remains development work, not an independent final test.",
                "Prediction timings are local single-row model inference, not Kaggle retrieval or end-to-end latency.",
                "The oracle uses relevance labels and is a retrospective upper bound.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"model": name,
            "exit_rate": result["nested_oof_exit_rate"],
            "recall_ratio": result["nested_oof_recall_ratio"],
            "folds_meeting_target": result["outer_folds_meeting_target"],
            "predict_ms": result["local_single_row_predict_ms"]}), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
