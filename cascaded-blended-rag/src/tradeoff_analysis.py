"""Query-paired uncertainty and count-matched random-routing references."""

import numpy as np


def latency_summary(values):
    array = np.asarray(values, dtype=float).ravel()
    return {"mean": float(array.mean()), **{
        f"p{q}": float(np.percentile(array, q)) for q in (50, 95, 99)}}


def paired_intervals(static_recall, recall, static_times, times, seed=42, resamples=1000):
    static_recall, recall = np.asarray(static_recall), np.asarray(recall)
    static_times, times = np.asarray(static_times), np.asarray(times)
    if not (len(static_recall) == len(recall) == len(static_times) == len(times)):
        raise ValueError("Bootstrap arrays must have matching query counts")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(recall), size=(resamples, len(recall)))
    base_r = static_recall[indices].mean(axis=1)
    base_t = static_times[indices].mean(axis=1)
    ratios = recall[indices].mean(axis=1)[base_r > 0]/base_r[base_r > 0]
    reductions = 1-times[indices].mean(axis=1)/base_t
    return {"recall_retention_95_interval": np.quantile(ratios, [.025, .975]).tolist() if len(ratios) else None,
            "mean_latency_reduction_95_interval": np.quantile(reductions, [.025, .975]).tolist(),
            "bootstrap_resamples": resamples}


def random_routing_reference(bm25, static, exits, seed=42, assignments=200):
    bm25, static = np.asarray(bm25), np.asarray(static)
    if not 0 <= exits <= len(static) or len(bm25) != len(static) or static.mean() <= 0:
        raise ValueError("Invalid random-routing input")
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(assignments):
        mask = np.zeros(len(static), dtype=bool)
        mask[rng.choice(len(static), exits, replace=False)] = True
        ratios.append(float(np.where(mask, bm25, static).mean()/static.mean()))
    return {"exits_per_assignment": exits, "assignments": assignments,
            "mean_recall_retention": float(np.mean(ratios)),
            "assignment_retention_p025_p975": np.quantile(ratios, [.025, .975]).tolist(),
            "latency_measured": False,
            "note": "Offline quality reference with identical exit count. Assignment variation is not a confidence interval for latency."}
