"""Compare compact escalation-risk predictors with nested train-only CV.

This CPU-only analysis uses the saved 809 SciFact training examples. In each
outer fold, inner out-of-fold predictions calibrate the exit threshold; the
outer fold estimates policy quality and exit rate. The held-out test set is
not read by this script.

Run from cascaded-blended-rag/:
    python experiments/10_compare_router_models.py
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


RECALL_FRACTION = 0.99  # add a safety margin over the original 98% target
OUTER_FOLDS = 5
INNER_FOLDS = 4
SEED = 17
FEATURE_NAMES = [
    "query_token_count", "query_unique_token_count", "bm25_top1", "bm25_top2",
    "bm25_gap12", "bm25_top10_mean", "bm25_top10_std",
    "bm25_top1_per_query_token", "max_idf", "mean_idf", "idf_std",
]


def candidate_models(seed):
    """Small models to compare, plus the previous forest as a reference."""
    return {
        "decision_tree_depth_2": DecisionTreeRegressor(
            max_depth=2, min_samples_leaf=25, random_state=seed
        ),
        "decision_tree_depth_3": DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=25, random_state=seed
        ),
        "decision_tree_depth_4": DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=25, random_state=seed
        ),
        "extra_trees_8": ExtraTreesRegressor(
            n_estimators=8, max_depth=4, min_samples_leaf=20,
            max_features=0.8, random_state=seed, n_jobs=1,
        ),
        "extra_trees_32": ExtraTreesRegressor(
            n_estimators=32, max_depth=4, min_samples_leaf=20,
            max_features=0.8, random_state=seed, n_jobs=1,
        ),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest_300_reference": RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=20,
            max_features=0.8, random_state=seed, n_jobs=1,
        ),
    }


def select_threshold(predicted_gain, true_gain, static_recall):
    """Maximize exits while expected aggregate Recall@100 loss stays in budget."""
    allowed_loss = (1.0 - RECALL_FRACTION) * static_recall
    candidates = np.r_[np.nextafter(predicted_gain.min(), -np.inf),
                       np.unique(predicted_gain), np.inf]
    feasible = []
    for threshold in candidates:
        exits = predicted_gain <= threshold
        aggregate_loss = float(np.where(exits, true_gain, 0.0).mean())
        if aggregate_loss <= allowed_loss:
            feasible.append((int(exits.sum()), -aggregate_loss, float(threshold)))
    # The threshold below the minimum prediction guarantees a feasible no-exit policy.
    exits_count, neg_loss, threshold = max(feasible)
    return threshold, allowed_loss, -neg_loss, exits_count / len(predicted_gain)


def nested_evaluate(name, estimator_factory, X, gains, bm25_recall, static_recall):
    decisions = np.zeros(len(gains), dtype=bool)
    fold_records = []
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)

    for fold_index, (train_index, test_index) in enumerate(outer.split(X), start=1):
        inner = KFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold_index
        )
        inner_predictions = np.maximum(
            cross_val_predict(
                estimator_factory(SEED + 100 + fold_index),
                X[train_index], gains[train_index], cv=inner, n_jobs=1,
            ),
            0.0,
        )
        threshold, budget, calibration_loss, calibration_exit_rate = select_threshold(
            inner_predictions, gains[train_index], float(static_recall[train_index].mean())
        )

        estimator = estimator_factory(SEED + 200 + fold_index)
        estimator.fit(X[train_index], gains[train_index])
        test_predictions = np.maximum(estimator.predict(X[test_index]), 0.0)
        exits = test_predictions <= threshold
        decisions[test_index] = exits
        policy_recall = np.where(exits, bm25_recall[test_index], static_recall[test_index])
        fold_records.append({
            "fold": fold_index,
            "queries": int(len(test_index)),
            "threshold": float(threshold),
            "calibration_exit_rate": calibration_exit_rate,
            "calibration_aggregate_loss": calibration_loss,
            "calibration_allowed_loss": budget,
            "outer_exit_rate": float(exits.mean()),
            "outer_policy_recall@100": float(policy_recall.mean()),
            "outer_static_recall@100": float(static_recall[test_index].mean()),
            "outer_recall_ratio": float(policy_recall.mean() / static_recall[test_index].mean()),
        })

    policy_recall = np.where(decisions, bm25_recall, static_recall)
    result = {
        "model": name,
        "nested_oof_exit_rate": float(decisions.mean()),
        "nested_oof_policy_recall@100": float(policy_recall.mean()),
        "nested_oof_static_recall@100": float(static_recall.mean()),
        "nested_oof_recall_ratio": float(policy_recall.mean() / static_recall.mean()),
        "nested_oof_aggregate_recall_loss": float(static_recall.mean() - policy_recall.mean()),
        "quality_target_fraction": RECALL_FRACTION,
        "folds": fold_records,
    }

    final_estimator = estimator_factory(SEED).fit(X, gains)
    sample = X[:1]
    for _ in range(30):
        final_estimator.predict(sample)
    start = time.perf_counter()
    repeats = 1000
    for _ in range(repeats):
        final_estimator.predict(sample)
    result["local_single_row_predict_ms"] = (time.perf_counter() - start) * 1000.0 / repeats
    result["model_nodes_or_estimators"] = (
        int(final_estimator.tree_.node_count)
        if hasattr(final_estimator, "tree_")
        else int(len(final_estimator.estimators_))
        if hasattr(final_estimator, "estimators_")
        else None
    )
    return result


def main():
    project_dir = Path(__file__).resolve().parents[1]
    data_path = project_dir / "results" / "router_training_scifact.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    X = np.asarray(
        [[row["features"][name] for name in FEATURE_NAMES] for row in rows], dtype=float
    )
    gains = np.asarray([row["recall_gain_from_escalation"] for row in rows], dtype=float)
    bm25_recall = np.asarray([row["bm25_recall@100"] for row in rows], dtype=float)
    static_recall = np.asarray([row["static_recall@100"] for row in rows], dtype=float)

    if len(rows) < OUTER_FOLDS * INNER_FOLDS:
        raise ValueError(f"Expected enough training rows for nested CV; found {len(rows)}")

    results = []
    models = candidate_models
    for name in models(SEED):
        print(f"Evaluating {name}...", flush=True)
        result = nested_evaluate(
            name, lambda seed, model_name=name: models(seed)[model_name],
            X, gains, bm25_recall, static_recall,
        )
        results.append(result)
        print(json.dumps({
            "model": name,
            "nested_oof_exit_rate": result["nested_oof_exit_rate"],
            "nested_oof_recall_ratio": result["nested_oof_recall_ratio"],
            "local_single_row_predict_ms": result["local_single_row_predict_ms"],
        }), flush=True)

    results.sort(key=lambda row: (
        row["nested_oof_recall_ratio"] >= RECALL_FRACTION,
        row["nested_oof_exit_rate"],
        -row["local_single_row_predict_ms"],
    ), reverse=True)
    output = {
        "dataset": "BEIR SciFact train split",
        "queries": len(rows),
        "features": FEATURE_NAMES,
        "threshold_calibration": "nested cross-validation using train-only labels",
        "quality_target_fraction": RECALL_FRACTION,
        "note": "Development comparison only; no original held-out test qrels are used.",
        "models": results,
    }
    out_path = project_dir / "results" / "compact_router_cv_scifact.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
