import re
from typing import Dict, List, Tuple
from rank_bm25 import BM25Okapi

def simple_tokenize(text: str) -> List[str]:
    """Tokenize lowercased text into alphanumeric words."""
    return re.findall(r"\w+", text.lower())

class BM25Retriever:
    def __init__(self, corpus: Dict[str, dict]):
        """
        corpus: {doc_id: {"title": str, "abstract": str}}
        """
        self.doc_ids = list(corpus.keys())
        
        # Build Single-Field tokens (Title + Abstract)
        single_field_corpus = [
            simple_tokenize(f"{corpus[did]['title']} {corpus[did]['abstract']}")
            for did in self.doc_ids
        ]
        self.bm25_single = BM25Okapi(single_field_corpus)
        
        # Build Multi-Field (Best Fields) separate indexes
        title_corpus = [simple_tokenize(corpus[did]["title"]) for did in self.doc_ids]
        abstract_corpus = [simple_tokenize(corpus[did]["abstract"]) for did in self.doc_ids]
        
        self.bm25_title = BM25Okapi(title_corpus)
        self.bm25_abstract = BM25Okapi(abstract_corpus)

    def retrieve_single(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Standard BM25 over concatenated title + abstract."""
        tokens = simple_tokenize(query)
        scores = self.bm25_single.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]

    def retrieve_best_fields(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Blended RAG Best Fields strategy:
        Score(Q, d) = max(Score_title(Q, d), Score_abstract(Q, d))
        """
        tokens = simple_tokenize(query)
        scores_t = self.bm25_title.get_scores(tokens)
        scores_a = self.bm25_abstract.get_scores(tokens)
        
        # Element-wise max
        best_scores = [max(t, a) for t, a in zip(scores_t, scores_a)]
        top_indices = sorted(range(len(best_scores)), key=lambda i: best_scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(best_scores[i])) for i in top_indices]

