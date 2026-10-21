import json
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retention_tradeoff import ConditionalCascade, calibrate_targets, export_ridge, predict_gain
from src.tradeoff_analysis import paired_intervals, random_routing_reference


class FakeStage:
    def __init__(self):
        self.plain_calls = self.extra_calls = 0

    def retrieve_only(self, query):
        self.plain_calls += 1
        return [("bm25", 1.0)]

    def retrieve_with_features(self, query):
        self.extra_calls += 1
        return [("bm25", 1.0)], {"extra": 2}, {}


class FakeNeural:
    def __init__(self, name):
        self.name, self.calls = name, 0

    def retrieve(self, query, top_k):
        self.calls += 1
        return [(self.name, 1.0)]


class RetentionTests(unittest.TestCase):
    def test_portable_inference_matches_sklearn_after_json_roundtrip(self):
        rng = np.random.default_rng(42)
        X, y = rng.normal(size=(25, 3)), rng.random(25)
        pipeline = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(X, y)
        router = json.loads(json.dumps(export_ridge(pipeline, ["a", "b", "c"])))
        actual = [predict_gain(router, dict(zip(["a", "b", "c"], row))) for row in X]
        np.testing.assert_allclose(actual, np.maximum(pipeline.predict(X), 0), atol=1e-12)

    def test_stricter_targets_cannot_increase_exit_count(self):
        policies = calibrate_targets([0, .1, .2, .3], [.5, .49, .45, .4], [.5]*4)
        self.assertEqual(list(policies), ["90", "92", "95", "99"])
        counts = [p["calibration"]["exits"] for p in policies.values()]
        self.assertEqual(counts, sorted(counts, reverse=True))
        for policy in policies.values():
            self.assertGreaterEqual(policy["calibration"]["recall_ratio"]+1e-12, policy["target"])

    def test_early_exit_never_invokes_neural_retrievers(self):
        stage, dense, splade = FakeStage(), FakeNeural("dense"), FakeNeural("splade")
        router = {"family": "ridge11", "feature_names": ["base"], "mean": [0],
                  "scale": [1], "coefficients": [1], "intercept": 0}
        engine = ConditionalCascade(stage, dense, splade, lambda q, r: {"base": 1}, lambda ranks: ranks[0], router, 1)
        rank, decision = engine.run("query", "cascade")
        self.assertEqual(rank[0][0], "bm25")
        self.assertEqual(decision["route"], "EARLY_EXIT")
        self.assertEqual((dense.calls, splade.calls), (0, 0))
        self.assertEqual(stage.extra_calls, 0)

    def test_escalation_calls_each_branch_once_and_coverage_only_when_used(self):
        stage, dense, splade = FakeStage(), FakeNeural("dense"), FakeNeural("splade")
        router = {"family": "ridge69", "feature_names": ["base", "extra"],
                  "mean": [0, 0], "scale": [1, 1], "coefficients": [1, 1], "intercept": 0}
        engine = ConditionalCascade(stage, dense, splade, lambda q, r: {"base": 1}, lambda ranks: ranks[0], router, 0)
        _, decision = engine.run("query", "cascade")
        self.assertEqual(decision["route"], "ESCALATE")
        self.assertEqual((dense.calls, splade.calls), (1, 1))
        self.assertEqual((stage.extra_calls, stage.plain_calls), (1, 0))
        self.assertEqual(decision["calls"], {"dense": 1, "splade": 1})

    def test_single_neural_baseline_never_calls_bm25_or_other_neural_branch(self):
        stage, dense, splade = FakeStage(), FakeNeural("dense"), FakeNeural("splade")
        engine = ConditionalCascade(stage, dense, splade, lambda q, r: {}, lambda ranks: ranks[0])
        engine.run("query", "dense")
        self.assertEqual((dense.calls, splade.calls, stage.plain_calls), (1, 0, 0))

    def test_identical_query_pairs_produce_unit_retention_zero_latency_change(self):
        intervals = paired_intervals([.1, .2], [.1, .2], [10, 20], [10, 20], resamples=50)
        self.assertEqual(intervals["recall_retention_95_interval"], [1, 1])
        self.assertEqual(intervals["mean_latency_reduction_95_interval"], [0, 0])

    def test_random_reference_respects_zero_and_all_exit_limits(self):
        bm25, static = [.1, .2, .3], [.4, .4, .4]
        zero = random_routing_reference(bm25, static, 0, assignments=10)
        all_exits = random_routing_reference(bm25, static, 3, assignments=10)
        self.assertEqual(zero["mean_recall_retention"], 1)
        self.assertAlmostEqual(all_exits["mean_recall_retention"], .5)
        self.assertFalse(all_exits["latency_measured"])


if __name__ == "__main__":
    unittest.main()
