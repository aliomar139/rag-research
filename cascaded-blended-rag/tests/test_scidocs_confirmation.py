import hashlib
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import scidocs_download
from src.query_self_exclusion import QueryContext, ExcludingBM25, ExcludingNeural

PROJECT = Path(__file__).resolve().parents[1]


class ConfirmationTests(unittest.TestCase):
    def test_self_exclusion_backfills_to_100_for_both_stages(self):
        class Search:
            dfs = {}
            def retrieve(self, query, top_k):
                self.requested = top_k
                return ([("self", 1.0)]+[(str(i), .5) for i in range(100)])[:top_k]
            retrieve_only = retrieve
        context = QueryContext()
        context.document_id = "self"
        for wrapper in (ExcludingBM25, ExcludingNeural):
            search = Search()
            wrapped = wrapper(search, context)
            result = wrapped.retrieve_only("query") if wrapper is ExcludingBM25 else wrapped.retrieve("query")
            self.assertEqual(search.requested, 101)
            self.assertEqual(len(result), 100)
            self.assertNotIn("self", [doc for doc, _ in result])

    def test_official_archive_check_and_named_member_extraction(self):
        folder = PROJECT / "tests" / (".tmp-scidocs-"+uuid.uuid4().hex)
        folder.mkdir()
        try:
            archive_path = folder / "scidocs.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
                    archive.writestr("scidocs/"+name, "verified "+name)
                archive.writestr("../unexpected.txt", "should not be extracted")
            digest = hashlib.md5(archive_path.read_bytes()).hexdigest()
            with patch.object(scidocs_download, "DATASET_MD5", digest):
                result = scidocs_download.ensure_scidocs(folder)
                self.assertEqual((result / "corpus.jsonl").read_text(), "verified corpus.jsonl")
                self.assertFalse((folder.parent / "unexpected.txt").exists())
                (result / "corpus.jsonl").write_text("modified")
                scidocs_download.ensure_scidocs(folder)
                self.assertEqual((result / "corpus.jsonl").read_text(), "verified corpus.jsonl")
            with patch.object(scidocs_download, "DATASET_MD5", "wrong"):
                with self.assertRaises(ValueError):
                    scidocs_download.ensure_scidocs(folder)
        finally:
            self.assertEqual(folder.resolve().parent, (PROJECT / "tests").resolve())
            shutil.rmtree(folder)


if __name__ == "__main__":
    unittest.main()
