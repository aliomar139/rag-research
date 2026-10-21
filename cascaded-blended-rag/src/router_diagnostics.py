"""Train-only policy diagnostics; no retrieval or model dependencies."""

import math


def select_threshold(predictions, clipped_losses, total_budget):
    """Maximize exits from prediction <= threshold, keeping ties together."""
    if not predictions or len(predictions) != len(clipped_losses):
        raise ValueError("Expected equally sized, nonempty predictions and losses")
    if total_budget < 0 or not math.isfinite(total_budget):
        raise ValueError("Budget must be finite and nonnegative")
    if any(not math.isfinite(x) for x in predictions + clipped_losses):
        raise ValueError("Predictions and losses must be finite")
    if any(x < 0 for x in clipped_losses):
        raise ValueError("Threshold selection uses clipped nonnegative losses")
    ordered = sorted(range(len(predictions)), key=lambda i: (predictions[i], i))
    threshold = math.nextafter(min(predictions), -math.inf)
    frontier = [{"threshold": threshold, "exits": 0, "total_clipped_loss": 0.0,
                 "feasible": True}]
    loss, count, pos = 0.0, 0, 0
    while pos < len(ordered):
        value = predictions[ordered[pos]]
        while pos < len(ordered) and predictions[ordered[pos]] == value:
            loss += clipped_losses[ordered[pos]]
            count += 1
            pos += 1
        feasible = loss <= total_budget
        frontier.append({"threshold": value, "exits": count,
                         "total_clipped_loss": loss, "feasible": feasible})
        if feasible:
            threshold = value
    return threshold, [p <= threshold for p in predictions], frontier


def oracle_mask(losses, total_budget):
    """Retrospective max-count exits with equal query weights and known costs.

    Ascending costs minimize total cost for each possible exit count. Signed
    costs may be negative when BM25 outperforms static RRF on a query.
    """
    if total_budget < 0 or not math.isfinite(total_budget):
        raise ValueError("Budget must be finite and nonnegative")
    if any(not math.isfinite(x) for x in losses):
        raise ValueError("Losses must be finite")
    ordered = sorted(range(len(losses)), key=lambda i: (losses[i], i))
    best, total = 0, 0.0
    for count, idx in enumerate(ordered, 1):
        total += losses[idx]
        if total <= total_budget:
            best = count
    selected = set(ordered[:best])
    return [i in selected for i in range(len(losses))]


def policy_summary(bm25, static, exits):
    if not bm25 or not (len(bm25) == len(static) == len(exits)):
        raise ValueError("Expected equally sized, nonempty policy arrays")
    n = len(bm25)
    baseline = math.fsum(static) / n
    actual = math.fsum(b if exit_now else s
                       for b, s, exit_now in zip(bm25, static, exits)) / n
    clipped_loss = math.fsum(max(0.0, s-b) if exit_now else 0.0
                             for b, s, exit_now in zip(bm25, static, exits)) / n
    return {"exits": sum(exits), "exit_rate": sum(exits) / n,
            "recall@100": actual,
            "recall_ratio": actual / baseline if baseline else None,
            "mean_clipped_loss": clipped_loss,
            "mean_signed_loss": baseline - actual}
