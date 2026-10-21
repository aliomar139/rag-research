"""Prepare fixed 90/92/95/99% policies from cached train data, on CPU only."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.retention_tradeoff import TARGETS, calibrate_targets, export_ridge
from src.router_diagnostics import policy_summary


def load_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def nested_curve(X, bm25, static):
    gains = np.maximum(static-bm25, 0)
    masks = {str(int(100*t)): np.zeros(len(X), dtype=bool) for t in TARGETS}
    fold_records = {key: [] for key in masks}
    outer_predictions = np.zeros(len(X))
    fold_ids = np.zeros(len(X), dtype=int)
    for fold, (train, validation) in enumerate(KFold(5, shuffle=True, random_state=17).split(X), 1):
        inner = KFold(4, shuffle=True, random_state=17+fold)
        inner_predictions = np.maximum(cross_val_predict(model(), X[train], gains[train], cv=inner, n_jobs=1), 0)
        thresholds = calibrate_targets(inner_predictions.tolist(), bm25[train].tolist(), static[train].tolist())
        estimator = model().fit(X[train], gains[train])
        predictions = np.maximum(estimator.predict(X[validation]), 0)
        outer_predictions[validation], fold_ids[validation] = predictions, fold
        for key, policy in thresholds.items():
            exits = predictions <= policy["threshold"]
            masks[key][validation] = exits
            fold_records[key].append({"fold": fold, "threshold": policy["threshold"],
                **policy_summary(bm25[validation].tolist(), static[validation].tolist(), exits.tolist())})
    result = {}
    for key, mask in masks.items():
        result[key] = {"target": int(key)/100,
            **policy_summary(bm25.tolist(), static.tolist(), mask.tolist()),
            "folds": fold_records[key],
            "folds_meeting_target": sum(row["recall_ratio"] >= int(key)/100 for row in fold_records[key])}
    return result, outer_predictions, fold_ids


def main():
    previous = load_script("14_compare_nfcorpus_train_routers.py", "tradeoff_previous")
    coverage = load_script("15_compare_nfcorpus_coverage_features.py", "tradeoff_coverage")
    results_dir = PROJECT / "results"
    rows, X_base, gains, bm25, static, provenance, _ = previous.read_examples(
        PROJECT.parent / "nfcorpus_train_diagnostics_outputs.zip")
    extras, extra_provenance = coverage.expanded_features(rows,
        PROJECT.parent / "nfcorpus_corpus_for_features.jsonl", provenance, results_dir)
    extra_names = list(extras[0]["extra_features"])
    X_extra = np.asarray([[row["extra_features"][name] for name in extra_names] for row in extras])
    base_names = provenance["retrieval_manifest"]["feature_names"]
    families = {"ridge11": (X_base, base_names),
                "ridge69": (np.hstack([X_base, X_extra]), base_names+extra_names)}
    routers, curves, per_query = {}, {}, []
    for family, (X, names) in families.items():
        print(f"Preparing four targets for {family}", flush=True)
        with threadpool_limits(limits=1):
            curve, predictions, fold_ids = nested_curve(X, bm25, static)
            oof = np.maximum(cross_val_predict(model(), X, gains,
                cv=KFold(5, shuffle=True, random_state=42), n_jobs=1), 0)
            estimator = model().fit(X, gains)
        router = export_ridge(estimator, names)
        router.update(family=family, policies=calibrate_targets(oof.tolist(), bm25.tolist(), static.tolist()))
        routers[family], curves[family] = router, curve
        for i, row in enumerate(rows):
            per_query.append({"query_id": row["query_id"], "family": family,
                "outer_fold": int(fold_ids[i]), "outer_prediction": float(predictions[i]),
                "final_calibration_oof_prediction": float(oof[i])})
        for key, value in curve.items():
            print(f"{family} target {key}%: exits {value['exit_rate']:.2%}, "
                  f"retention {value['recall_ratio']:.2%}, folds {value['folds_meeting_target']}/5", flush=True)
    frozen = {"schema": 1, "dataset": "BEIR NFCorpus", "train_queries": len(rows),
        "train_query_ids": [r["query_id"] for r in rows], "targets": list(TARGETS),
        "routers": routers, "provenance": provenance,
        "extra_feature_provenance": extra_provenance,
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "protocol": "Ridge alpha=10; final thresholds use five-fold train OOF predictions (seed 42), clipped-loss budget. No dev/test labels."}
    frozen_path = results_dir / "nfcorpus_frozen_retention_routers.json"
    frozen_path.write_text(json.dumps(frozen, indent=2, allow_nan=False), encoding="utf-8")
    report = {"dataset": "BEIR NFCorpus", "split": "train", "queries": len(rows),
        "targets": list(TARGETS), "nested_curves": curves, "provenance": provenance,
        "frozen_routers_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "protocol": "Same five outer/four inner folds as prior CPU comparisons (seed 17). Each inner threshold is frozen before outer scoring.",
        "limits": "Iterative train-only development; no unseen-data quality guarantee and no latency estimate."}
    path = results_dir / "nfcorpus_retention_train_curve.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (results_dir / "nfcorpus_retention_train_predictions.jsonl").write_text(
        "".join(json.dumps(row)+"\n" for row in per_query), encoding="utf-8")
    print(f"Saved {path} and {frozen_path}", flush=True)


if __name__ == "__main__":
    main()
