"""Calibrate a Ridge cascade gate on NFCorpus train and evaluate on dev.

The script never opens test.tsv. Train relevance labels are used to fit the
router and select an exit threshold with out-of-fold predictions. Dev labels
are read only after the static and conditional rankings have been fixed.

Run from cascaded-blended-rag/ with:
    python experiments/12_calibrate_nfcorpus_train_check_dev.py
"""

import csv
import hashlib
import json
import math
import os
import ssl
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from ranx import Qrels, Run, evaluate
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bm25_retriever import BM25Retriever, simple_tokenize
from src.dense_retriever import DenseRetriever
from src.splade_retriever import SPLADERetriever

DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
DATASET_MD5 = "a89dba18a62ef92f7d323ec890a0d38d"
TOP_K, RRF_K = 100, 60
TARGET_RECALL_FRACTION = 0.99
CV_FOLDS, CV_SEED = 5, 42
FEATURE_NAMES = [
    "query_token_count", "query_unique_token_count", "bm25_top1", "bm25_top2",
    "bm25_gap12", "bm25_top10_mean", "bm25_top10_std",
    "bm25_top1_per_query_token", "max_idf", "mean_idf", "idf_std",
]


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, *args, **kwargs):
    sync_cuda()
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    sync_cuda()
    return value, (time.perf_counter() - start) * 1000


def ensure_dataset(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = raw_dir / "nfcorpus"
    if (dataset_dir / "corpus.jsonl").is_file():
        return dataset_dir
    archive_path = raw_dir / "nfcorpus.zip"
    if not archive_path.is_file():
        print(f"Downloading official BEIR NFCorpus archive: {DATASET_URL}", flush=True)
        with urllib.request.urlopen(DATASET_URL, context=ssl.create_default_context(), timeout=120) as response:
            archive_path.write_bytes(response.read())
    actual_md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()
    if actual_md5 != DATASET_MD5:
        raise ValueError(f"NFCorpus MD5 mismatch: expected {DATASET_MD5}, got {actual_md5}")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        prefix = next((n.split("/", 1)[0] + "/" for n in names if n.endswith("/corpus.jsonl")), None)
        if prefix is None:
            raise FileNotFoundError("Archive has no corpus.jsonl")
        root = raw_dir.resolve()
        for name in names:
            if name.startswith(prefix):
                target = (raw_dir / name).resolve()
                if root not in target.parents and target != root:
                    raise ValueError(f"Unsafe archive path: {name}")
        archive.extractall(raw_dir)
    if not (dataset_dir / "corpus.jsonl").is_file():
        found = list(raw_dir.rglob("corpus.jsonl"))
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one corpus.jsonl, found {found}")
        dataset_dir = found[0].parent
    return dataset_dir


def load_corpus(dataset_dir):
    corpus = {}
    with (dataset_dir / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[str(row["_id"])] = {"title": row.get("title", "").strip(), "abstract": row.get("text", "").strip()}
    return corpus


def load_split(dataset_dir, split):
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    ids = set()
    with qrels_path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row:
                ids.add(str(row[0]))
    queries = {}
    with (dataset_dir / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row["_id"])
            if qid in ids:
                queries[qid] = row["text"].strip()
    if len(queries) != len(ids):
        raise ValueError(f"{split}: found {len(queries)} of {len(ids)} qrel query IDs")
    return queries, qrels_path


def load_qrels(path):
    result = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and int(row[2]) > 0:
                result[str(row[0])][str(row[1])] = int(row[2])
    return dict(result)


def doc_freqs(corpus):
    out = defaultdict(int)
    for doc in corpus.values():
        for token in set(simple_tokenize(doc["title"] + " " + doc["abstract"])):
            out[token] += 1
    return out


def features(query, ranking, dfs, n_docs):
    tokens = simple_tokenize(query)
    unique = list(dict.fromkeys(tokens))
    scores = [float(s) for _, s in ranking[:10]]
    top1, top2 = (scores + [0.0, 0.0])[:2]
    idfs = [math.log((n_docs - dfs.get(t, 0) + 0.5) / (dfs.get(t, 0) + 0.5) + 1) for t in unique]
    avg = mean(scores) if scores else 0.0
    idf_avg = mean(idfs) if idfs else 0.0
    return dict(query_token_count=len(tokens), query_unique_token_count=len(unique), bm25_top1=top1,
                bm25_top2=top2, bm25_gap12=top1-top2, bm25_top10_mean=avg,
                bm25_top10_std=float(np.std(scores)) if scores else 0.0,
                bm25_top1_per_query_token=top1/max(1, len(tokens)), max_idf=max(idfs, default=0.0),
                mean_idf=idf_avg, idf_std=float(np.std(idfs)) if idfs else 0.0)


def fuse(rankings):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, 1):
            scores[doc_id] += 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]


def recall(qrels, ranking):
    relevant = qrels
    if not relevant:
        return 0.0
    return len(set(relevant).intersection(doc for doc, _ in ranking[:TOP_K])) / len(relevant)


def score_run(qrels, rankings):
    run = {qid: {doc: score for doc, score in ranking} for qid, ranking in rankings.items()}
    values = evaluate(Qrels(qrels), Run(run), metrics=["recall@100", "ndcg@10", "map@10"])
    return {k: float(v) for k, v in values.items()}


def summary(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


def calibrate(examples):
    X = np.asarray([[e["features"][name] for name in FEATURE_NAMES] for e in examples])
    gains = np.asarray([e["gain"] for e in examples])
    bm25_recall = np.asarray([e["bm25_recall"] for e in examples])
    static_recall = np.asarray([e["static_recall"] for e in examples])
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    oof = np.maximum(cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=10.0)), X, gains, cv=cv, n_jobs=1), 0.0)
    target_loss = (1-TARGET_RECALL_FRACTION) * float(static_recall.mean())
    candidates = np.r_[np.nextafter(oof.min(), -np.inf), np.unique(oof)]
    feasible = []
    for threshold in candidates:
        exits = oof <= threshold
        loss = float(np.where(exits, gains, 0).mean())
        if loss <= target_loss:
            feasible.append((int(exits.sum()), -loss, float(threshold), exits, loss))
    if not feasible:
        raise RuntimeError("No threshold satisfies the train-only recall constraint, including zero exits")
    count, _, threshold, exits, loss = max(feasible)
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, gains)
    policy_recall = np.where(exits, bm25_recall, static_recall)
    report = {"source": "NFCorpus official train split only", "queries": len(examples),
              "router": "StandardScaler + Ridge(alpha=10)", "cv_folds": CV_FOLDS, "cv_seed": CV_SEED,
              "target_fraction_of_static_recall": TARGET_RECALL_FRACTION,
              "oof_static_recall@100": float(static_recall.mean()),
              "oof_target_recall@100": float(static_recall.mean()*TARGET_RECALL_FRACTION),
              "threshold": threshold, "oof_exit_rate": count/len(examples),
              "oof_policy_recall@100": float(policy_recall.mean()),
              "oof_policy_recall_ratio": float(policy_recall.mean()/static_recall.mean()),
              "oof_aggregate_recall_loss": loss}
    return model, threshold, report


def main():
    project = Path(__file__).resolve().parents[1]
    dataset_dir = ensure_dataset(project / "data" / "raw")
    corpus = load_corpus(dataset_dir)
    train_queries, train_qrels_path = load_split(dataset_dir, "train")
    dev_queries, dev_qrels_path = load_split(dataset_dir, "dev")
    print(f"Full corpus {len(corpus)} docs; train queries {len(train_queries)}; dev queries {len(dev_queries)}", flush=True)

    cache_dir = project / "data" / "processed_nfcorpus_router"
    bm25, dense, splade = BM25Retriever(corpus), DenseRetriever(corpus, cache_dir=str(cache_dir)), SPLADERetriever(corpus, cache_dir=str(cache_dir))
    dfs = doc_freqs(corpus)

    # Create train-only labels from the official train qrels, then fit the gate.
    train_qrels = load_qrels(train_qrels_path)
    examples = []
    for i, (qid, query) in enumerate(train_queries.items(), 1):
        b = bm25.retrieve_best_fields(query, TOP_K)
        d = dense.retrieve(query, TOP_K)
        s = splade.retrieve(query, TOP_K)
        static = fuse([b, d, s])
        rb, rs = recall(train_qrels.get(qid, {}), b), recall(train_qrels.get(qid, {}), static)
        examples.append({"features": features(query, b, dfs, len(corpus)), "bm25_recall": rb,
                         "static_recall": rs, "gain": max(0.0, rs-rb)})
        if i % 50 == 0 or i == len(train_queries):
            print(f"Train calibration retrievals: {i}/{len(train_queries)}", flush=True)
    model, threshold, calibration = calibrate(examples)

    # Fix dev routes and rankings before opening dev relevance grades.
    static_rankings, cascade_rankings, decisions = {}, {}, {}
    static_times, cascade_times = [], []
    repeats = max(1, int(os.environ.get("CASCADE_LATENCY_REPEATS", "2")))
    warm = list(dev_queries.values())[:3]
    for query in warm:
        bm25.retrieve_best_fields(query, TOP_K); dense.retrieve(query, TOP_K); splade.retrieve(query, TOP_K)
    for repeat in range(repeats):
        for i, (qid, query) in enumerate(dev_queries.items(), 1):
            start = time.perf_counter()
            b, _ = timed(bm25.retrieve_best_fields, query, TOP_K)
            d, _ = timed(dense.retrieve, query, TOP_K)
            s, _ = timed(splade.retrieve, query, TOP_K)
            static = fuse([b, d, s]); sync_cuda()
            static_times.append((time.perf_counter()-start)*1000)
            if repeat == 0:
                static_rankings[qid] = static
            start = time.perf_counter()
            b, _ = timed(bm25.retrieve_best_fields, query, TOP_K)
            x = features(query, b, dfs, len(corpus))
            pred = max(0.0, float(model.predict(np.asarray([[x[n] for n in FEATURE_NAMES]]))[0]))
            exit_now = pred <= threshold
            if exit_now:
                ranking = b
            else:
                d, _ = timed(dense.retrieve, query, TOP_K)
                s, _ = timed(splade.retrieve, query, TOP_K)
                ranking = fuse([b, d, s])
            sync_cuda(); cascade_times.append((time.perf_counter()-start)*1000)
            if repeat == 0:
                cascade_rankings[qid] = ranking
                decisions[qid] = {"route": "EARLY_EXIT" if exit_now else "ESCALATE", "predicted_gain": pred}
            if repeat == 0 and (i % 50 == 0 or i == len(dev_queries)):
                print(f"Dev routing pass: {i}/{len(dev_queries)}", flush=True)

    # Dev grades are consumed only after all routes and rankings are fixed.
    dev_qrels = load_qrels(dev_qrels_path)
    exits = sum(v["route"] == "EARLY_EXIT" for v in decisions.values())
    result = {"dataset": "BEIR NFCorpus", "calibration_split": "official train", "evaluation_split": "official dev",
              "corpus_documents": len(corpus), "train_queries": len(train_queries), "dev_queries": len(dev_queries),
              "corpus_source": "official full BEIR corpus; no qrel-based document selection",
              "test_split_opened": False, "calibration": calibration,
              "static_rrf": {"metrics": score_run(dev_qrels, static_rankings), "latency_ms": summary(static_times)},
              "cascade": {"metrics": score_run(dev_qrels, cascade_rankings), "early_exit_rate": exits/len(dev_queries),
                          "dense_calls_skipped_per_pass": exits, "splade_calls_skipped_per_pass": exits,
                          "latency_ms": summary(cascade_times), "per_query_first_pass": decisions},
              "hardware": {"cuda_available": torch.cuda.is_available(),
                           "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"},
              "protocol_note": "Router and threshold are selected using NFCorpus train only. Dev relevance grades are opened after dev routes and rankings are fixed. test.tsv is never read."}
    out = project / "results" / "nfcorpus_train_calibrated_dev.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "calibration": calibration,
                      "static_rrf": result["static_rrf"]["metrics"], "cascade": result["cascade"]["metrics"],
                      "exit_rate": result["cascade"]["early_exit_rate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
