"""CPU-only, paired execution benchmark of old and indexed BM25 scoring.

Uses existing local corpora, no downloads, no qrels and no neural models.
Score/ranking agreement is recorded before any speed result is accepted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
import zipfile

import numpy as np
from rank_bm25 import BM25Okapi
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.bm25_retriever import simple_tokenize
from src.indexed_bm25 import IndexedBestFieldsBM25, stable_top_k


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("trec-covid", "nfcorpus"), default="trec-covid")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--query-limit", type=int, default=0, help="0 means all available queries; otherwise label-limited engineering benchmark")
    args = parser.parse_args()
    corpus_path = (PROJECT / "data/raw/trec-covid/corpus.jsonl" if args.dataset == "trec-covid"
                   else PROJECT.parent / "nfcorpus_corpus_for_features.jsonl")
    corpus = {}
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            corpus[str(row["_id"])] = {"title": row.get("title", "").strip(), "abstract": row.get("text", "").strip()}
    if args.dataset == "trec-covid":
        query_path = PROJECT / "data/raw/trec-covid/queries.jsonl"
        queries = {str(row["_id"]): row["text"].strip() for row in map(json.loads, query_path.open(encoding="utf-8"))}
    else:
        with zipfile.ZipFile(PROJECT.parent / "nfcorpus_train_diagnostics_outputs.zip") as archive:
            queries = {row["query_id"]: row["query"] for row in map(json.loads, archive.open("nfcorpus_train_router_examples.jsonl"))}
    ids = sorted(queries)
    if args.query_limit:
        ids = ids[:args.query_limit]
    print(f"{args.dataset}: {len(corpus)} documents, {len(ids)} queries, {args.repeats} repeats; CPU only", flush=True)
    doc_ids = list(corpus)
    start = time.perf_counter()
    title = BM25Okapi((simple_tokenize(corpus[q]["title"]) for q in doc_ids))
    abstract = BM25Okapi((simple_tokenize(corpus[q]["abstract"]) for q in doc_ids))
    reference_build_seconds = time.perf_counter()-start
    print(f"Reference field indexes ready: {reference_build_seconds:.1f}s; building cached postings", flush=True)
    start = time.perf_counter()
    indexed = IndexedBestFieldsBM25.from_models(doc_ids, title, abstract)
    postings_build_seconds = time.perf_counter()-start
    print(f"Postings ready: {postings_build_seconds:.1f}s", flush=True)

    def reference(query):
        tokens = simple_tokenize(query)
        scores = np.maximum(title.get_scores(tokens), abstract.get_scores(tokens))
        order = np.argsort(-scores, kind="stable")[:100]
        return scores, order

    def optimized(query):
        scores = indexed.score(query)
        return scores, stable_top_k(scores, 100)

    methods = {"reference": reference, "indexed": optimized}
    for qid in ids[:3]:
        for method in methods.values():
            method(queries[qid])
    timings = {name: {} for name in methods}
    max_score_difference = 0.0
    compared_queries = 0
    rng = random.Random(20261021)
    for repeat in range(args.repeats):
        order = ids.copy()
        rng.shuffle(order)
        for i, qid in enumerate(order, 1):
            names = list(methods)
            rng.shuffle(names)
            answers = {}
            for name in names:
                start = time.perf_counter()
                answers[name] = methods[name](queries[qid])
                elapsed = (time.perf_counter()-start)*1000
                timings[name].setdefault(qid, []).append(elapsed)
            old_scores, old_order = answers["reference"]
            new_scores, new_order = answers["indexed"]
            difference = float(np.max(np.abs(old_scores-new_scores)))
            max_score_difference = max(max_score_difference, difference)
            if not np.array_equal(old_order, new_order) or not np.array_equal(old_scores, new_scores):
                raise ValueError(f"Score/ranking disagreement for {qid}: max abs difference {difference}")
            compared_queries += 1
            if i % 25 == 0 or i == len(order):
                print(f"CPU benchmark repeat {repeat+1}/{args.repeats}: {i}/{len(order)}", flush=True)
    means = {name: float(np.mean([v for values in rows.values() for v in values])) for name, rows in timings.items()}
    result = {
        "dataset": args.dataset, "documents": len(corpus), "queries": len(ids), "repeats": args.repeats,
        "corpus_sha256": sha(corpus_path), "indexed_source_sha256": sha(PROJECT / "src/indexed_bm25.py"),
        "all_full_corpus_scores_and_top100_ids_identical": True,
        "query_repetitions_compared": compared_queries, "maximum_absolute_score_difference": max_score_difference,
        "reference_build_seconds": reference_build_seconds, "postings_build_seconds": postings_build_seconds,
        "mean_latency_ms": means, "measured_bm25_speedup": means["reference"]/means["indexed"],
        "timings_per_query_ms": timings, "python": platform.python_version(), "numpy": np.__version__,
        "processor": platform.processor(), "cpu_threads": 1,
        "scope": "Local CPU paired BM25 execution benchmark only; no claim about Kaggle end-to-end cascade speed or a novel scoring algorithm.",
    }
    output = PROJECT / "results" / ("indexed_bm25_cpu_"+args.dataset+".json")
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "timings_per_query_ms"}, indent=2), flush=True)
    print("SAVED:", output, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
