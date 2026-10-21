"""Exercise all benchmark methods/artifact exports with fake neural searches."""

import hashlib
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import zipfile

PROJECT = Path(__file__).resolve().parents[1]


class BenchmarkExportTests(unittest.TestCase):
    def test_full_export_uses_frozen_policies_and_reads_grades_after_timing(self):
        self.run_fixture("17_benchmark_retention_tradeoff.py", False)

    def test_scidocs_confirmation_uses_test_only_for_scoring_and_preserves_weights(self):
        self.run_fixture("18_confirm_scidocs_frozen.py", True)

    def run_fixture(self, script_name, confirmation):
        spec = importlib.util.spec_from_file_location("benchmark_test", PROJECT / "experiments" / script_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        folder = PROJECT / "tests" / (".tmp-benchmark-"+uuid.uuid4().hex)
        folder.mkdir()
        try:
            (folder / "src").mkdir()
            (folder / "artifacts").mkdir()
            output = folder / "results"
            output.mkdir()
            corpus = {str(i): {"title": f"study word{i}", "abstract": f"evidence word{i} nutrition"} for i in range(1, 7)}
            raw = "".join(json.dumps({"_id": qid, "title": row["title"], "text": row["abstract"]})+"\n" for qid, row in corpus.items())
            (folder / "corpus.jsonl").write_text(raw, encoding="utf-8")
            (folder / "dev.tsv").write_text("fixture qrels", encoding="utf-8")
            source_hashes = {}
            for name in ("bm25_retriever.py", "dense_retriever.py", "splade_retriever.py", "cache_utils.py", "bm25_coverage_features.py"):
                (folder / "src" / name).write_text("test source", encoding="utf-8")
                source_hashes[name] = hashlib.sha256(b"test source").hexdigest()
            policies = {str(target): {"target": target/100, "threshold": 10 if target == 90 else 0} for target in (90, 92, 95, 99)}
            routers = {family: {"family": family, "feature_names": ["query_token_count"],
                       "mean": [0], "scale": [1], "coefficients": [1], "intercept": 0,
                       "policies": policies} for family in ("ridge11", "ridge69")}
            frozen = {"dataset": "BEIR NFCorpus", "targets": [.90, .92, .95, .99],
                "train_query_ids": ["train-1"], "routers": routers,
                "provenance": {"retrieval_manifest": {"corpus_sha256": hashlib.sha256((folder / "corpus.jsonl").read_bytes()).hexdigest(),
                    "retrieval_code_sha256": source_hashes}},
                "extra_feature_provenance": {"feature_code_sha256": source_hashes["bm25_coverage_features.py"]}}
            (folder / "artifacts" / "nfcorpus_frozen_retention_routers.json").write_text(json.dumps(frozen), encoding="utf-8")
            if confirmation:
                protocol = {"frozen_routers_sha256": hashlib.sha256((folder / "artifacts" / "nfcorpus_frozen_retention_routers.json").read_bytes()).hexdigest(),
                    "timing_repeats": 2, "targets": frozen["targets"], "families": ["ridge11"]}
                (folder / "artifacts" / "scidocs_confirmation_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            queries = {"dev-1": "word1", "dev-2": "word2", "dev-3": "word3"}
            qrels = {qid: {str(i): 1} for i, qid in enumerate(queries, 1)}
            calls = []
            class Neural:
                def __init__(self, *args, **kwargs):
                    pass
                def retrieve(self, query, top_k=100):
                    calls.append(query)
                    return [(str(i), 1/i) for i in range(1, 7)]
            def load_split(path, split):
                self.assertEqual(split, "test" if confirmation else "dev")
                return queries, folder / "dev.tsv"
            def load_grades(path):
                calls_per_query = 10 if confirmation else 16
                self.assertEqual(len(calls), calls_per_query*3+calls_per_query*3*2)
                return qrels
            def recall(relevant, ranking):
                return len(set(relevant) & {doc for doc, _ in ranking})/len(relevant)
            def score(qrels, rankings):
                value = sum(recall(qrels[qid], rank) for qid, rank in rankings.items())/len(rankings)
                return {"recall@100": value, "ndcg@10": value, "map@10": value}
            base = SimpleNamespace(ensure_dataset=lambda path: folder, load_corpus=lambda path: corpus,
                load_split=load_split, load_qrels=load_grades, DenseRetriever=Neural, SPLADERetriever=Neural,
                features=lambda q, r, dfs, n: {"query_token_count": 1},
                fuse=lambda ranks: ranks[0], sync_cuda=lambda: None, recall=recall, score_run=score)
            def fake_plot(results, out):
                prefix = "scidocs" if confirmation else "nfcorpus"
                (out / (prefix+"_quality_speed.png")).write_bytes(b"test")
                (out / (prefix+"_quality_speed.svg")).write_text("test", encoding="utf-8")
            with ExitStack() as patches:
                patches.enter_context(patch.object(module, "PROJECT", folder))
                patches.enter_context(patch.object(module, "base_module", return_value=base))
                patches.enter_context(patch.object(module, "plot_results", side_effect=fake_plot))
                if confirmation:
                    patches.enter_context(patch.object(module, "ensure_scidocs", return_value=folder))
                    patches.enter_context(patch.object(module, "EXPECTED_QUERY_COUNT", 3))
                module.benchmark(SimpleNamespace(data_dir=folder, output_dir=output, repeats=2))
            report_name = "scidocs_confirmation_benchmark.json" if confirmation else "nfcorpus_retention_benchmark.json"
            report = json.loads((output / report_name).read_text())
            self.assertEqual(len(report["methods"]), 8 if confirmation else 12)
            self.assertFalse(report["nfcorpus_test_qrels_opened"] if confirmation else report["test_qrels_opened"])
            self.assertFalse(report["evaluation_grades_used_to_select_policies"] if confirmation else report["dev_grades_used_to_select_policies"])
            if confirmation:
                self.assertTrue(report["query_document_self_matches_excluded"])
                self.assertEqual(report["dataset"], "BEIR SCIDOCS")
            for family in (("ridge11",) if confirmation else routers):
                self.assertEqual(report["methods"][family+"_r90"]["neural_calls_across_timed_repeats"], {"dense": 0, "splade": 0})
                self.assertEqual(report["methods"][family+"_r99"]["neural_calls_across_timed_repeats"], {"dense": 6, "splade": 6})
            zip_name = "scidocs_confirmation_outputs.zip" if confirmation else "nfcorpus_retention_benchmark_outputs.zip"
            with zipfile.ZipFile(output / zip_name) as archive:
                self.assertIn("quality_speed.csv", archive.namelist())
                self.assertIn("nfcorpus_frozen_retention_routers.json", archive.namelist())
                self.assertEqual(archive.read("nfcorpus_frozen_retention_routers.json"),
                    (folder / "artifacts" / "nfcorpus_frozen_retention_routers.json").read_bytes())
        finally:
            # Only this test-created directory, verified inside the tests folder.
            self.assertEqual(folder.resolve().parent, (PROJECT / "tests").resolve())
            shutil.rmtree(folder)


if __name__ == "__main__":
    unittest.main()
