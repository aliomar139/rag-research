import itertools
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.router_diagnostics import oracle_mask, policy_summary, select_threshold


class RouterDiagnosticsTests(unittest.TestCase):
    def test_oracle_matches_exhaustive_maximum_for_signed_and_clipped_costs(self):
        rng = random.Random(42)
        for signed in (False, True):
            for _ in range(40):
                losses = [rng.randint(-2 if signed else 0, 5)/8 for _ in range(7)]
                budget = rng.randint(0, 5)/8
                mask = oracle_mask(losses, budget)
                possible = [sum(candidate) for candidate in itertools.product((False, True), repeat=7)
                            if sum(cost for cost, take in zip(losses, candidate) if take) <= budget]
                self.assertEqual(sum(mask), max(possible))
                self.assertLessEqual(sum(c for c, take in zip(losses, mask) if take), budget)

    def test_threshold_never_splits_prediction_ties(self):
        threshold, mask, _ = select_threshold([0.1, 0.1, 0.2], [0, 0.5, 0], 0.25)
        self.assertLess(threshold, 0.1)
        self.assertEqual(mask, [False, False, False])

    def test_threshold_matches_exhaustive_feasible_thresholds(self):
        predictions = [0, .5, .5, .75, 1]
        losses = [0, .125, 0, .25, .125]
        threshold, mask, frontier = select_threshold(predictions, losses, .25)
        self.assertEqual(threshold, .5)
        self.assertEqual(sum(mask), 3)
        self.assertEqual(sum(mask), max(p["exits"] for p in frontier if p["feasible"]))

    def test_signed_oracle_can_credit_bm25_wins(self):
        self.assertEqual(sum(oracle_mask([-.5, .25, .25], 0)), 3)
        self.assertEqual(sum(oracle_mask([0, .25, .25], 0)), 1)

    def test_actual_recall_and_clipped_loss_are_reported_separately(self):
        result = policy_summary([1, 0], [.5, .5], [True, False])
        self.assertEqual(result["recall@100"], .75)
        self.assertEqual(result["mean_clipped_loss"], 0)
        self.assertEqual(result["mean_signed_loss"], -.25)

    def test_zero_budget_exits_all_zero_cost_queries(self):
        self.assertEqual(oracle_mask([.25, 0, 0], 0), [False, True, True])
        _, mask, _ = select_threshold([0, 0], [0, 0], 0)
        self.assertEqual(mask, [True, True])

    def test_nonfinite_values_and_invalid_costs_are_rejected(self):
        with self.assertRaises(ValueError):
            select_threshold([math.nan], [0], 0)
        with self.assertRaises(ValueError):
            select_threshold([0], [-1], 0)
        with self.assertRaises(ValueError):
            oracle_mask([0], -1)


if __name__ == "__main__":
    unittest.main()
