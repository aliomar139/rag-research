"""Precomputed postings for the project's rank_bm25 Best Fields scorer.

This is an execution optimization, not a new BM25 formula. Tokenization,
rank_bm25 epsilon-adjusted IDFs, repeated query terms, float64 accumulation,
Best Fields max and corpus-order tie breaking are preserved.
"""
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi

from src.bm25_retriever import simple_tokenize


def stable_top_k(scores, top_k):
    """Return descending scores with original corpus order for every tie."""
    count = min(max(0, int(top_k)), len(scores))
    if not count:
        return np.empty(0, dtype=np.int64)
    if count == len(scores):
        return np.argsort(-scores, kind="stable")
    cutoff = np.partition(scores, len(scores)-count)[len(scores)-count]
    higher = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[:count-len(higher)]
    candidates = np.concatenate((higher, tied))
    return candidates[np.lexsort((candidates, -scores[candidates]))]


class FieldPostings:
    def __init__(self, reference):
        self.size = reference.corpus_size
        self.vocabulary = {term: i for i, term in enumerate(reference.idf)}
        lengths = np.zeros(len(self.vocabulary), dtype=np.int64)
        for frequencies in reference.doc_freqs:
            for term in frequencies:
                lengths[self.vocabulary[term]] += 1
        self.offsets = np.r_[0, np.cumsum(lengths)]
        total = int(self.offsets[-1])
        self.documents = np.empty(total, dtype=np.int32 if self.size < 2**31 else np.int64)
        frequencies = np.empty(total, dtype=np.int64)
        cursors = self.offsets[:-1].copy()
        for document, terms in enumerate(reference.doc_freqs):
            for term, frequency in terms.items():
                column = self.vocabulary[term]
                position = cursors[column]
                self.documents[position] = document
                frequencies[position] = frequency
                cursors[column] += 1
        self.weights = np.empty(total, dtype=np.float64)
        document_lengths = np.asarray(reference.doc_len)
        for term, column in self.vocabulary.items():
            start, end = self.offsets[column:column+2]
            tf = frequencies[start:end]
            doc_lengths = document_lengths[self.documents[start:end]]
            # Keep rank_bm25's ufunc order rather than algebraically changing
            # the denominator; original floating-point scores matter to gates.
            self.weights[start:end] = (reference.idf.get(term) or 0) * (
                tf * (reference.k1+1) /
                (tf + reference.k1*(1-reference.b+reference.b*doc_lengths/reference.avgdl)))

    def score(self, tokens):
        scores = np.zeros(self.size, dtype=np.float64)
        for term in tokens:  # Preserve duplicate tokens and accumulation order.
            column = self.vocabulary.get(term)
            if column is not None:
                start, end = self.offsets[column:column+2]
                scores[self.documents[start:end]] += self.weights[start:end]
        return scores


class IndexedBestFieldsBM25:
    @classmethod
    def from_models(cls, doc_ids, title, abstract):
        instance = cls.__new__(cls)
        instance.doc_ids = list(doc_ids)
        if title.corpus_size != len(doc_ids) or abstract.corpus_size != len(doc_ids):
            raise ValueError("Reference fields must match the corpus ordering")
        instance.title = FieldPostings(title)
        instance.abstract = FieldPostings(abstract)
        instance.dfs = Counter()
        for first, second in zip(title.doc_freqs, abstract.doc_freqs):
            instance.dfs.update(first.keys() | second.keys())
        return instance

    def __init__(self, corpus):
        doc_ids = list(corpus)
        title = BM25Okapi((simple_tokenize(corpus[q]["title"]) for q in doc_ids))
        abstract = BM25Okapi((simple_tokenize(corpus[q]["abstract"]) for q in doc_ids))
        built = self.from_models(doc_ids, title, abstract)
        self.__dict__.update(built.__dict__)

    def score(self, query):
        tokens = simple_tokenize(query)
        return np.maximum(self.title.score(tokens), self.abstract.score(tokens))

    def retrieve_only(self, query, top_k=100):
        scores = self.score(query)
        return [(self.doc_ids[i], float(scores[i])) for i in stable_top_k(scores, top_k)]

    retrieve_best_fields = retrieve_only
