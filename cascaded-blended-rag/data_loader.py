import os
import json
import csv
import ssl
import urllib.request
import zipfile
from typing import Dict, Tuple

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

def download_and_load_scifact(
    data_dir: str = "data/raw",
    split: str = "test",
) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Downloads BEIR SciFact and loads:
      - corpus: {doc_id: {"title": str, "abstract": str}}
      - queries: {query_id: query_text} (only queries present in the selected split qrels)
      - qrels: {query_id: {doc_id: relevance_score}}

    `split` may be "train" or "test". The default remains "test" for
    compatibility with the existing baseline scripts.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'")
    os.makedirs(data_dir, exist_ok=True)
    scifact_dir = os.path.join(data_dir, "scifact")
    zip_path = os.path.join(data_dir, "scifact.zip")

    if not os.path.exists(scifact_dir):
        print(f"Downloading SciFact dataset from {SCIFACT_URL}...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(SCIFACT_URL, context=ctx) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Extracting SciFact...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        os.remove(zip_path)
        print("SciFact download complete.")

    # 1. Load Corpus with separate 'title' and 'abstract' fields
    corpus = {}
    corpus_path = os.path.join(scifact_dir, "corpus.jsonl")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus[str(item["_id"])] = {
                "title": item.get("title", "").strip(),
                "abstract": item.get("text", "").strip()
            }

    # 2. Load Test Qrels (Ground Truth)
    qrels = {}
    qrels_path = os.path.join(scifact_dir, "qrels", f"{split}.tsv")
    with open(qrels_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # Skip header (query-id, corpus-id, score)
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, score = str(row[0]), str(row[1]), int(row[2])
            if score > 0:
                qrels.setdefault(qid, {})[did] = score

    # 3. Load Test Queries (300 evaluated queries)
    queries = {}
    queries_path = os.path.join(scifact_dir, "queries.jsonl")
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = str(item["_id"])
            if qid in qrels:
                queries[qid] = item["text"].strip()

    print(f"Loaded SciFact: {len(corpus)} docs | {len(queries)} {split} queries.")
    return corpus, queries, qrels

if __name__ == "__main__":
    download_and_load_scifact()