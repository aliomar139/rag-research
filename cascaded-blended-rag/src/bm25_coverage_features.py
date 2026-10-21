"""Corpus/query-only routing features; never uses relevance or neural ranks."""

from collections import Counter
import math
import time

import numpy as np

from src.bm25_retriever import BM25Retriever, simple_tokenize

STOPWORDS = frozenset("a an and are as at be by for from how in is it of on or that the this to was were what when where which who why with".split())


class BM25CoverageFeatures:
    def __init__(self, corpus):
        self.bm25 = BM25Retriever(corpus)
        self.doc_ids = self.bm25.doc_ids
        self.titles = [set(simple_tokenize(corpus[doc]["title"])) for doc in self.doc_ids]
        self.abstracts = [set(simple_tokenize(corpus[doc]["abstract"])) for doc in self.doc_ids]
        self.combined = [a | b for a, b in zip(self.titles, self.abstracts)]
        self.doc_lengths = [len(simple_tokenize(corpus[doc]["title"] + " " + corpus[doc]["abstract"]))
                            for doc in self.doc_ids]
        self.dfs = Counter(token for document in self.combined for token in document)
        n = len(self.doc_ids)
        self.idfs = {token: math.log((n-df+.5)/(df+.5)+1) for token, df in self.dfs.items()}

    def _retrieve_state(self, query, top_k=100):
        start = time.perf_counter()
        tokens = simple_tokenize(query)
        title_scores = np.asarray(self.bm25.bm25_title.get_scores(tokens), dtype=float)
        abstract_scores = np.asarray(self.bm25.bm25_abstract.get_scores(tokens), dtype=float)
        scores = np.maximum(title_scores, abstract_scores)
        # Stable descending order reproduces the original Python sorted ties.
        indices = np.argsort(-scores, kind="stable")[:top_k]
        ranking = [(self.doc_ids[i], float(scores[i])) for i in indices]
        bm25_ms = (time.perf_counter()-start)*1000
        return ranking, tokens, title_scores, abstract_scores, scores, indices, bm25_ms

    def retrieve_only(self, query, top_k=100):
        """Use the identical BM25 implementation without coverage features."""
        return self._retrieve_state(query, top_k)[0]

    def retrieve_with_features(self, query, top_k=100):
        ranking, tokens, title_scores, abstract_scores, scores, indices, bm25_ms = self._retrieve_state(query, top_k)
        start_features = time.perf_counter()
        result = {}
        unique = set(tokens)
        content = unique-STOPWORDS
        n_docs = len(self.doc_ids)
        default_idf = math.log((n_docs+.5)/.5+1)
        weights = {t: self.idfs.get(t, default_idf) for t in unique}
        total_weight = sum(weights.values())
        top = scores[indices]
        for k in (20, 50, 100):
            values = top[:k]
            result[f"score_mean_top{k}"] = float(values.mean()) if len(values) else 0.0
            result[f"score_std_top{k}"] = float(values.std()) if len(values) else 0.0
        top1 = float(top[0]) if len(top) else 0.0
        scale = max(abs(top1), float(np.max(np.abs(top))) if len(top) else 0.0, 1e-12)
        for k in (2, 5, 10, 100):
            other = float(top[min(k-1, len(top)-1)]) if len(top) else 0.0
            result[f"relative_score_gap_1_{k}"] = (top1-other)/scale
        positive = np.maximum(top, 0)
        mass = float(positive.sum())
        probabilities = positive/mass if mass else np.zeros_like(positive)
        nonzero = probabilities[probabilities > 0]
        result["score_entropy_top100"] = float(-(nonzero*np.log(nonzero)).sum()/math.log(max(2, len(top))))
        result["score_mass_top10_fraction"] = float(positive[:10].sum()/mass) if mass else 0.0
        result["positive_score_corpus_fraction"] = float((scores > 0).mean())
        result["query_oov_fraction"] = sum(t not in self.dfs for t in unique)/max(1, len(unique))
        result["query_stopword_fraction"] = len(unique & STOPWORDS)/max(1, len(unique))
        result["query_numeric_fraction"] = sum(t.isdigit() for t in unique)/max(1, len(unique))
        result["query_rare_fraction"] = sum(0 < self.dfs.get(t, 0) <= .01*n_docs for t in unique)/max(1, len(unique))
        for field, documents in (("combined", self.combined), ("title", self.titles), ("abstract", self.abstracts)):
            covered = [unique & documents[i] for i in indices[:10]]
            coverage = [len(matched)/max(1, len(unique)) for matched in covered]
            weighted = [sum(weights[t] for t in matched)/total_weight if total_weight else 0.0 for matched in covered]
            content_coverage = [len(content & documents[i])/max(1, len(content)) for i in indices[:10]]
            result[f"{field}_coverage_top1"] = coverage[0] if coverage else 0.0
            result[f"{field}_weighted_coverage_top1"] = weighted[0] if weighted else 0.0
            for k in (3, 10):
                result[f"{field}_coverage_mean_top{k}"] = float(np.mean(coverage[:k])) if coverage else 0.0
                result[f"{field}_coverage_max_top{k}"] = max(coverage[:k], default=0.0)
                result[f"{field}_weighted_coverage_mean_top{k}"] = float(np.mean(weighted[:k])) if weighted else 0.0
                result[f"{field}_content_coverage_mean_top{k}"] = float(np.mean(content_coverage[:k])) if content_coverage else 0.0
            result[f"{field}_full_coverage_fraction_top10"] = sum(value == 1 for value in coverage)/max(1, len(coverage)) if unique else 0.0
        title_order = np.argsort(-title_scores, kind="stable")
        abstract_order = np.argsort(-abstract_scores, kind="stable")
        for k in (10, 100):
            a, b = set(title_order[:k].tolist()), set(abstract_order[:k].tolist())
            result[f"title_abstract_rank_jaccard_top{k}"] = len(a & b)/max(1, len(a | b))
        result["title_abstract_same_top1"] = float(title_order[0] == abstract_order[0])
        result["title_dominates_top10_fraction"] = float((title_scores[indices[:10]] > abstract_scores[indices[:10]]).mean())
        result["title_score_at_top1_relative"] = float(title_scores[indices[0]]/scale)
        result["abstract_score_at_top1_relative"] = float(abstract_scores[indices[0]]/scale)
        result["top1_document_length"] = self.doc_lengths[indices[0]]
        result["mean_document_length_top10"] = float(np.mean([self.doc_lengths[i] for i in indices[:10]]))
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("Nonfinite BM25 coverage feature")
        feature_ms = (time.perf_counter()-start_features)*1000
        return ranking, result, {"bm25_ms": bm25_ms, "extra_feature_ms": feature_ms}
