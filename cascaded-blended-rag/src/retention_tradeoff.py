"""Frozen trade-off policies and portable Ridge inference."""

import numpy as np

from src.router_diagnostics import policy_summary, select_threshold

TARGETS = (0.90, 0.92, 0.95, 0.99)


def calibrate_targets(predictions, bm25, static, targets=TARGETS):
    predictions, bm25, static = list(predictions), list(bm25), list(static)
    losses = [max(0.0, s-b) for b, s in zip(bm25, static)]
    policies = {}
    for target in targets:
        if not 0 < target <= 1:
            raise ValueError("Retention targets must be in (0, 1]")
        budget = (1-target)*sum(static)
        threshold, mask, _ = select_threshold(predictions, losses, budget)
        policies[str(int(round(100*target)))] = {
            "target": target, "threshold": threshold,
            "training_total_clipped_loss_budget": budget,
            "calibration": policy_summary(bm25, static, mask),
        }
    return policies


def export_ridge(pipeline, feature_names):
    scaler = pipeline.named_steps["standardscaler"]
    ridge = pipeline.named_steps["ridge"]
    if len(feature_names) != len(ridge.coef_):
        raise ValueError("Feature names do not match fitted coefficients")
    return {"feature_names": list(feature_names), "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(), "coefficients": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_)}


def predict_gain(router, features):
    vector = np.asarray([features[name] for name in router["feature_names"]], dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite query features")
    value = np.dot((vector-np.asarray(router["mean"]))/np.asarray(router["scale"]),
                   np.asarray(router["coefficients"])) + router["intercept"]
    return max(0.0, float(value))


class ConditionalCascade:
    """Query-time path with observed neural branch counts, also used in tests."""

    def __init__(self, stage, dense, splade, base_features, fuse, router=None, threshold=None):
        self.stage, self.dense, self.splade = stage, dense, splade
        self.base_features, self.fuse = base_features, fuse
        self.router, self.threshold = router, threshold

    def run(self, query, method):
        calls = {"dense": 0, "splade": 0}
        if method in ("dense", "splade"):
            retriever = self.dense if method == "dense" else self.splade
            calls[method] += 1
            return retriever.retrieve(query, top_k=100), {"route": "BASELINE", "calls": calls}
        if self.router and self.router["family"] == "ridge69" and method == "cascade":
            bm25, extras, _ = self.stage.retrieve_with_features(query)
        else:
            bm25, extras = self.stage.retrieve_only(query), {}
        if method == "bm25":
            return bm25, {"route": "BASELINE", "calls": calls}
        decision = {"route": "BASELINE", "calls": calls}
        if method == "cascade":
            features = {**self.base_features(query, bm25), **extras}
            prediction = predict_gain(self.router, features)
            decision.update(predicted_gain=prediction)
            if prediction <= self.threshold:
                decision["route"] = "EARLY_EXIT"
                return bm25, decision
            decision["route"] = "ESCALATE"
        elif method != "static_rrf":
            raise ValueError(f"Unknown retrieval method: {method}")
        dense = self.dense.retrieve(query, top_k=100)
        calls["dense"] += 1
        sparse = self.splade.retrieve(query, top_k=100)
        calls["splade"] += 1
        return self.fuse([bm25, dense, sparse]), decision
