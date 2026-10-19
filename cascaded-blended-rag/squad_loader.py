import os
import json
import ssl
import urllib.request
from typing import Dict, Tuple

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

def download_and_load_squad(data_dir: str = "data/raw") -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Downloads and parses SQuAD v1.1 dev set into:
      - corpus: {doc_id: {"title": article_title, "abstract": paragraph_text}}
      - queries: {query_id: question_text}
      - qrels: {query_id: {doc_id: 1}}
    """
    os.makedirs(data_dir, exist_ok=True)
    squad_path = os.path.join(data_dir, "squad_dev_v1.1.json")

    if not os.path.exists(squad_path):
        print(f"Downloading SQuAD v1.1 dev set from {SQUAD_DEV_URL}...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(SQUAD_DEV_URL, context=ctx) as response, open(squad_path, "wb") as f:
            f.write(response.read())
        print("SQuAD download complete.")

    with open(squad_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    corpus = {}
    queries = {}
    qrels = {}

    doc_counter = 0
    for article in data:
        article_title = article.get("title", "").replace("_", " ")
        for p in article["paragraphs"]:
            doc_id = f"doc_{doc_counter}"
            doc_counter += 1
            corpus[doc_id] = {
                "title": article_title,
                "abstract": p["context"].strip()
            }
            for qa in p["qas"]:
                qid = qa["id"]
                queries[qid] = qa["question"].strip()
                qrels[qid] = {doc_id: 1}

    print(f"Loaded SQuAD v1.1: {len(corpus)} paragraphs | {len(queries)} questions.")
    return corpus, queries, qrels

if __name__ == "__main__":
    download_and_load_squad()

