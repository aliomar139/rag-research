"""Small synthetic report check; no retrieval, model download, or GPU."""

import importlib.util
import json
from pathlib import Path
import unittest
import uuid

PROJECT = Path(__file__).resolve().parents[1]


class TrainReportTests(unittest.TestCase):
    def test_exports_agree_with_report_and_oracle_budgets(self):
        try:
            import numpy
            import sklearn
        except ImportError:
            self.skipTest("Run with the project venv for the report integration check")
        spec = importlib.util.spec_from_file_location(
            "train_diagnostic_under_test", PROJECT / "experiments" / "13_diagnose_nfcorpus_train_router.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bm25 = [.75, .5, .5, .25, .25, .25, .5, .5, .75, .25]
        rows = [{"query_id": f"train-{i}", "features": {"f1": i, "f2": i % 2},
                 "bm25_recall": b, "static_recall": .5} for i, b in enumerate(bm25)]
        output = PROJECT / "tests" / (".tmp-router-" + uuid.uuid4().hex)
        output.mkdir()
        try:
            report = module.diagnose(rows, ["f1", "f2"], output)
            decisions = [json.loads(line) for line in
                         (output / "nfcorpus_train_router_decisions.jsonl").read_text().splitlines()]
            self.assertEqual(report["oracle_same_clipped_budget"]["exits"], 6)
            self.assertEqual(report["oracle_net_recall_budget"]["exits"], 8)
            for name in ("ridge_oof", "oracle_same_clipped_budget", "oracle_net_recall_budget"):
                self.assertGreaterEqual(report[name]["recall_ratio"], .99)
            exits = [d["ridge_oof_route"] == "EARLY_EXIT" for d in decisions]
            actual = sum(b if take else .5 for b, take in zip(bm25, exits)) / len(rows)
            self.assertAlmostEqual(actual, report["ridge_oof"]["recall@100"])
            self.assertEqual(sum(exits), report["ridge_oof"]["exits"])
            self.assertEqual(len(decisions), len(rows))
            self.assertEqual({d["oof_fold"] for d in decisions}, set(range(5)))
            self.assertFalse(report["dev_qrels_opened"])
            self.assertFalse(report["test_qrels_opened"])
            self.assertTrue((output / "nfcorpus_train_threshold_frontier.json").is_file())
        finally:
            for path in output.iterdir():
                path.unlink()
            output.rmdir()


if __name__ == "__main__":
    unittest.main()
