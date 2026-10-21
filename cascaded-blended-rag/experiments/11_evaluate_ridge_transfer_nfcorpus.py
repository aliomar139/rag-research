"""Evaluate a frozen SciFact-trained Ridge router on BEIR NFCorpus test.

The router and exit threshold use SciFact train-only labels. NFCorpus test
qrels are used only after all retrieval and routing decisions are complete.
The official full NFCorpus corpus is loaded without qrel-based sampling.

Run from cascaded-blended-rag/:
    python experiments/11_evaluate_ridge_transfer_nfcorpus.py

The script downloads the official BEIR nfcorpus.zip if it is not already
present under data/raw/. Dense and SPLADE indexes use a dedicated cache path.
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
TOP_K = 100
RRF_K = 60
TRAIN_RECALL_FRACTION = 0.99
CV_FOLDS = 5
CV_SEED = 42
FEATURE_NAMES = [
    "query_token_count", "query_unique_token_count", "bm25_top1", "bm25_top2",
    "bm25_gap12", "bm25_top10_mean", "bm25_top10_std",
    "bm25_top1_per_query_token", "max_idf", "mean_idf", "idf_std",
]


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn, *args, **kwargs):
    sync_cuda()
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    sync_cuda()
    return value, (time.perf_counter() - start) * 1000.0


def find_training_file(project_dir):
    candidates = [
        project_dir / "results" / "router_training_scifact.jsonl",
        project_dir.parent / "router_training_scifact.jsonl",
        project_dir / "router_training_scifact.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"SciFact training examples not found. Checked:\n{expected}")


def make_router():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def calibrate_router(examples):
    X = np.asarray(
        [[row["features"][name] for name in FEATURE_NAMES] for row in examples],
        dtype=float,
    )
    gains = np.asarray(
        [row["recall_gain_from_escalation"] for row in examples], dtype=float
    )
    bm25_recall = np.asarray([row["bm25_recall@100"] for row in examples], dtype=float)
    static_recall = np.asarray([row["static_recall@100"] for row in examples], dtype=float)

    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    oof_gain = np.maximum(
        cross_val_predict(make_router(), X, gains, cv=folds, n_jobs=1), 0.0
    )
    static_mean = float(static_recall.mean())
    allowed_loss = (1.0 - TRAIN_RECALL_FRACTION) * static_mean
    candidates = np.r_[np.nextafter(oof_gain.min(), -np.inf), np.unique(oof_gain)]
    feasible = []
    for threshold in candidates:
        exits = oof_gain <= threshold
        loss = float(np.where(exits, gains, 0.0).mean())
        if loss <= allowed_loss:
            feasible.append((int(exits.sum()), -loss, float(threshold), exits, loss))
    exit_count, _, threshold, exits, loss = max(feasible)

    model = make_router().fit(X, gains)
    policy_recall = np.where(exits, bm25_recall, static_recall)
    calibration = {
        "source": "BEIR SciFact train examples only",
        "queries": len(examples),
        "model": "StandardScaler + Ridge(alpha=10), predicts nonnegative Recall@100 escalation gain",
        "folds": CV_FOLDS,
        "fold_seed": CV_SEED,
        "target_fraction_of_static_recall": TRAIN_RECALL_FRACTION,
        "static_recall@100": static_mean,
        "target_recall@100": TRAIN_RECALL_FRACTION * static_mean,
        "threshold": threshold,
        "oof_exit_rate": exit_count / len(examples),
        "oof_policy_recall@100": float(policy_recall.mean()),
        "oof_policy_recall_ratio": float(policy_recall.mean() / static_mean),
        "oof_aggregate_recall_loss": loss,
    }
    return model, threshold, calibration


def ensure_nfcorpus(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = raw_dir / "nfcorpus"
    if (dataset_dir / "corpus.jsonl").is_file():
        return dataset_dir

    archive_path = raw_dir / "nfcorpus.zip"
    if not archive_path.is_file():
        print(f"Downloading official BEIR NFCorpus archive: {DATASET_URL}", flush=True)
        context = ssl.create_default_context()
        with urllib.request.urlopen(DATASET_URL, context=context, timeout=120) as response:
            archive_path.write_bytes(response.read())

    archive_md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()
    if archive_md5 != DATASET_MD5:
        raise ValueError(
            f"NFCorpus archive MD5 mismatch: expected {DATASET_MD5}, got {archive_md5}"
        )

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        root_prefix = next(
            (name.split("/", 1)[0] + "/" for name in members if name.endswith("/corpus.jsonl")),
            None,
        )
        if root_prefix is None:
            raise FileNotFoundError("The nfcorpus.zip archive has no corpus.jsonl")
        for member in members:
            if not member.startswith(root_prefix):
                continue
            target = (raw_dir / member).resolve()
            if raw_dir.resolve() not in target.parents and target != raw_dir.resolve():
                raise ValueError(f"Unsafe path in NFCorpus archive: {member}")
        archive.extractall(raw_dir)

    if not (dataset_dir / "corpus.jsonl").is_file():
        found = list(raw_dir.rglob("corpus.jsonl"))
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one extracted corpus.jsonl; found {found}")
        dataset_dir = found[0].parent
    return dataset_dir


def load_nfcorpus_corpus_and_test_queries(dataset_dir):
    corpus = {}
    with (dataset_dir / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            corpus[str(row["_id"])] = {
                "title": row.get("title", "").strip(),
                "abstract": row.get("text", "").strip(),
            }

    qrels_path = dataset_dir / "qrels" / "test.tsv"
    test_query_ids = set()
    with qrels_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 1:
                test_query_ids.add(str(row[0]))

    queries = {}
    with (dataset_dir / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = str(row["_id"])
            if qid in test_query_ids:
                queries[qid] = row["text"].strip()
    if not corpus or not queries:
        raise ValueError("NFCorpus loader returned an empty corpus or test query set")
    return corpus, queries, qrels_path


def load_test_qrels(qrels_path):
    qrels = defaultdict(dict)
    with qrels_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and int(row[2]) > 0:
                qrels[str(row[0])][str(row[1])] = int(row[2])
    return dict(qrels)


def make_doc_freqs(corpus):
    frequencies = defaultdict(int)
    for doc in corpus.values():
        for token in set(simple_tokenize(f"{doc['title']} {doc['abstract']}")):
            frequencies[token] += 1
    return frequencies


def router_features(query, bm25_ranking, doc_freqs, total_docs):
    tokens = simple_tokenize(query)
    unique_tokens = list(dict.fromkeys(tokens))
    scores = [float(score) for _, score in bm25_ranking[:10]]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    idfs = []
    for token in unique_tokens:
        df = doc_freqs.get(token, 0)
        idfs.append(math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0))
    score_mean = mean(scores) if scores else 0.0
    idf_mean = mean(idfs) if idfs else 0.0
    return {
        "query_token_count": len(tokens),
        "query_unique_token_count": len(unique_tokens),
        "bm25_top1": top1,
        "bm25_top2": top2,
        "bm25_gap12": top1 - top2,
        "bm25_top10_mean": score_mean,
        "bm25_top10_std": (sum((x - score_mean) ** 2 for x in scores) / len(scores)) ** 0.5 if scores else 0.0,
        "bm25_top1_per_query_token": top1 / max(1, len(tokens)),
        "max_idf": max(idfs, default=0.0),
        "mean_idf": idf_mean,
        "idf_std": (sum((x - idf_mean) ** 2 for x in idfs) / len(idfs)) ** 0.5 if idfs else 0.0,
    }


def fuse_rankings(rankings):
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (RRF_K + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:TOP_K]


def score_run(qrels, rankings):
    run = {qid: {doc_id: score for doc_id, score in rank} for qid, rank in rankings.items()}
    values = evaluate(
        Qrels(qrels), Run(run), metrics=["recall@100", "ndcg@10", "map@10"]
    )
    return {key: float(value) for key, value in values.items()}


def percentiles(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()) if len(array) else 0.0,
        "p50": float(np.percentile(array, 50)) if len(array) else 0.0,
        "p95": float(np.percentile(array, 95)) if len(array) else 0.0,
        "p99": float(np.percentile(array, 99)) if len(array) else 0.0,
    }


def main():
    project_dir = Path(__file__).resolve().parents[1]
    training_path = find_training_file(project_dir)
    examples = [
        json.loads(line)
        for line in training_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model, threshold, calibration = calibrate_router(examples)

    raw_dir = project_dir / "data" / "raw"
    dataset_dir = ensure_nfcorpus(raw_dir)
    corpus, queries, qrels_path = load_nfcorpus_corpus_and_test_queries(dataset_dir)
    print(
        f"NFCorpus full corpus: {len(corpus)} docs | test queries: {len(queries)} | "
        f"GPU: {torch.cuda.is_available()}",
        flush=True,
    )

    cache_dir = project_dir / "data" / "processed_nfcorpus_router"
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, cache_dir=str(cache_dir))
    splade = SPLADERetriever(corpus, cache_dir=str(cache_dir))
    doc_freqs = make_doc_freqs(corpus)
    repeats = max(1, int(os.environ.get("CASCADE_LATENCY_REPEATS", "2")))

    warm_queries = list(queries.values())[:3]
    for query in warm_queries:
        bm25.retrieve_best_fields(query, top_k=TOP_K)
        dense.retrieve(query, top_k=TOP_K)
        splade.retrieve(query, top_k=TOP_K)

    static_rankings, static_latencies = {}, defaultdict(list)
    for repeat in range(repeats):
        for index, (qid, query) in enumerate(queries.items(), start=1):
            start = time.perf_counter()
            bm25_rank, bm25_ms = timed_call(bm25.retrieve_best_fields, query, top_k=TOP_K)
            dense_rank, dense_ms = timed_call(dense.retrieve, query, top_k=TOP_K)
            splade_rank, splade_ms = timed_call(splade.retrieve, query, top_k=TOP_K)
            fused = fuse_rankings([bm25_rank, dense_rank, splade_rank])
            sync_cuda()
            static_latencies[qid].append((time.perf_counter() - start) * 1000.0)
            if repeat == 0:
                static_rankings[qid] = fused
            if repeat == 0 and (index % 50 == 0 or index == len(queries)):
                print(f"Static NFCorpus pass: {index}/{len(queries)}", flush=True)

    cascade_rankings, per_query = {}, {}
    cascade_latencies = defaultdict(list)
    for repeat in range(repeats):
        for index, (qid, query) in enumerate(queries.items(), start=1):
            start = time.perf_counter()
            bm25_rank, bm25_ms = timed_call(bm25.retrieve_best_fields, query, top_k=TOP_K)
            route_start = time.perf_counter()
            features = router_features(query, bm25_rank, doc_freqs, len(corpus))
            vector = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=float)
            predicted_gain = max(0.0, float(model.predict(vector)[0]))
            exit_early = predicted_gain <= threshold
            route_ms = (time.perf_counter() - route_start) * 1000.0
            timings = {
                "bm25_ms": bm25_ms,
                "router_ms": route_ms,
                "dense_ms": 0.0,
                "splade_ms": 0.0,
                "fusion_ms": 0.0,
            }
            if exit_early:
                ranking = bm25_rank
            else:
                dense_rank, timings["dense_ms"] = timed_call(dense.retrieve, query, top_k=TOP_K)
                splade_rank, timings["splade_ms"] = timed_call(splade.retrieve, query, top_k=TOP_K)
                fusion_start = time.perf_counter()
                ranking = fuse_rankings([bm25_rank, dense_rank, splade_rank])
                timings["fusion_ms"] = (time.perf_counter() - fusion_start) * 1000.0
            sync_cuda()
            total_ms = (time.perf_counter() - start) * 1000.0
            cascade_latencies[qid].append(total_ms)
            if repeat == 0:
                cascade_rankings[qid] = ranking
                per_query[qid] = {
                    "route": "EARLY_EXIT" if exit_early else "ESCALATE",
                    "predicted_recall_gain": predicted_gain,
                    "timings_first_pass_ms": timings,
                }
            if repeat == 0 and (index % 50 == 0 or index == len(queries)):
                print(f"Cascade NFCorpus pass: {index}/{len(queries)}", flush=True)

    # Read relevance grades only after rankings and routes have been fixed.
    qrels = load_test_qrels(qrels_path)
    exits = sum(row["route"] == "EARLY_EXIT" for row in per_query.values())
    result = {
        "dataset": "BEIR NFCorpus",
        "evaluation_split": "official test",
        "corpus_documents": len(corpus),
        "test_queries": len(queries),
        "corpus_source": "official full BEIR NFCorpus corpus; no qrel-based sampling",
        "router_training_source": str(training_path),
        "router": "SciFact-trained StandardScaler + Ridge(alpha=10)",
        "threshold_calibration": calibration,
        "nfc_test_relevance_grades_read_before_routing": False,
        "test_query_ids_read_from_official_qrels_to_select_the_test_split": True,
        "latency_repeats_after_warmup": repeats,
        "static_rrf": {
            "metrics": score_run(qrels, static_rankings),
            "latency_ms": percentiles([v for values in static_latencies.values() for v in values]),
        },
        "cascade": {
            "metrics": score_run(qrels, cascade_rankings),
            "early_exit_rate": exits / len(queries),
            "queries_exiting_per_pass": exits,
            "dense_calls_skipped_per_pass": exits,
            "splade_calls_skipped_per_pass": exits,
            "dense_calls_skipped_across_timed_repeats": exits * repeats,
            "splade_calls_skipped_across_timed_repeats": exits * repeats,
            "latency_ms": percentiles([v for values in cascade_latencies.values() for v in values]),
            "per_query_first_pass": per_query,
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "protocol_note": (
            "Router and threshold were frozen from SciFact train-only data before NFCorpus test scoring. "
            "This measures cross-dataset transfer; it does not recalibrate on NFCorpus."
        ),
    }
    out_path = project_dir / "results" / "ridge_transfer_nfcorpus.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out_path),
        "static_rrf": result["static_rrf"]["metrics"],
        "cascade": result["cascade"]["metrics"],
        "exit_rate": result["cascade"]["early_exit_rate"],
        "static_latency_ms": result["static_rrf"]["latency_ms"],
        "cascade_latency_ms": result["cascade"]["latency_ms"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
