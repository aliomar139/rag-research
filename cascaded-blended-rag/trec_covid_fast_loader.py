import os
import json
import csv
import ssl
import urllib.request
import zipfile
from typing import Dict, Tuple

TREC_COVID_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"

def load_trec_covid_fast(data_dir: str = "data/raw", max_docs: int = 5000) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Subsets TREC-COVID for fast CPU evaluation:
      - 50 official queries
      - All highly relevant docs (score=2, ~1,500 docs) + negative samples up to max_docs
    """
    os.makedirs(data_dir, exist_ok=True)
    tc_dir = os.path.join(data_dir, "trec-covid")
    corpus_path = os.path.join(tc_dir, "corpus.jsonl")
    zip_path = os.path.join(data_dir, "trec-covid.zip")

    if not os.path.exists(corpus_path):
        print(f"Downloading TREC-COVID dataset from {TREC_COVID_URL}...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(TREC_COVID_URL, context=ctx) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Extracting TREC-COVID...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        os.remove(zip_path)
        print("TREC-COVID download complete.")
    
    # 1. Load Qrels: take top 25 highly relevant docs (score=2) per query for focused benchmark
    qrels = {}
    highly_rel_docs = set()
    qrels_path = os.path.join(tc_dir, "qrels", "test.tsv")
    with open(qrels_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, score = str(row[0]), str(row[1]), int(row[2])
            if score == 2:
                if len(qrels.get(qid, {})) < 25:
                    qrels.setdefault(qid, {})[did] = 1
                    highly_rel_docs.add(did)

    # 2. Load Queries
    queries = {}
    queries_path = os.path.join(tc_dir, "queries.jsonl")
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = str(item["_id"])
            if qid in qrels:
                queries[qid] = item["text"].strip()

    # 3. Stream corpus: exactly the selected gold relevant docs + negatives up to max_docs
    corpus = {}
    corpus_path = os.path.join(tc_dir, "corpus.jsonl")
    print(f"Sampling corpus: targeting {len(highly_rel_docs)} gold docs + negatives up to {max_docs}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            did = str(item["_id"])
            if did in highly_rel_docs or len(corpus) < max_docs:
                corpus[did] = {
                    "title": item.get("title", "").strip(),
                    "abstract": item.get("text", "").strip()
                }
            if len(corpus) >= max_docs and len(highly_rel_docs.difference(corpus.keys())) == 0:
                break



    print(f"Loaded TREC-COVID Fast Pool: {len(corpus)} docs | {len(queries)} queries.")
    return corpus, queries, qrels

if __name__ == "__main__":
    load_trec_covid_fast()

