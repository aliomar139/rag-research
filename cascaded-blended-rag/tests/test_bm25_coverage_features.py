import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bm25_coverage_features import BM25CoverageFeatures


class CoverageFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = {
            "1": {"title": "apple carrot trial", "abstract": "apple carrot nutrition report"},
            "2": {"title": "apple trial", "abstract": "apple nutrition report"},
            "3": {"title": "carrot", "abstract": "carrot carrot trial"},
            "4": {"title": "banana", "abstract": "banana diet study"},
            "5": {"title": "pear", "abstract": "pear diet study"},
            "6": {"title": "grape", "abstract": "grape diet study"},
        }
        cls.extractor = BM25CoverageFeatures(cls.corpus)

    def test_ranking_matches_existing_best_fields_including_ties(self):
        for query in ("apple carrot", "apple", "diet", "unknownword", ""):
            ranking, _, _ = self.extractor.retrieve_with_features(query)
            self.assertEqual(ranking, self.extractor.bm25.retrieve_best_fields(query))
            self.assertEqual(ranking, self.extractor.retrieve_only(query))

    def test_full_coverage_and_weighted_coverage(self):
        ranking, features, _ = self.extractor.retrieve_with_features("apple carrot")
        self.assertEqual(ranking[0][0], "1")
        self.assertEqual(features["combined_coverage_top1"], 1)
        self.assertEqual(features["title_weighted_coverage_top1"], 1)
        self.assertAlmostEqual(features["title_full_coverage_fraction_top10"], 1/6)
        self.assertEqual(features["query_oov_fraction"], 0)

    def test_empty_and_oov_queries_are_finite_without_fake_full_coverage(self):
        for query in ("", "unknownword"):
            _, features, timings = self.extractor.retrieve_with_features(query)
            self.assertTrue(all(math.isfinite(value) for value in features.values()))
            self.assertEqual(features["combined_coverage_top1"], 0)
            self.assertEqual(features["combined_full_coverage_fraction_top10"], 0)
            self.assertEqual(features["score_entropy_top100"], 0)
            self.assertEqual(features["query_oov_fraction"], float(bool(query)))
            self.assertGreaterEqual(timings["extra_feature_ms"], 0)

    def test_coverage_and_agreement_features_are_bounded(self):
        _, features, _ = self.extractor.retrieve_with_features("the apple 2020 unknownword")
        for key, value in features.items():
            if "coverage" in key or "fraction" in key or "jaccard" in key or "entropy" in key:
                self.assertGreaterEqual(value, 0, key)
                self.assertLessEqual(value, 1+1e-12, key)


if __name__ == "__main__":
    unittest.main()
