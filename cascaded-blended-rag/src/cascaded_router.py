import math
from typing import Dict, List, Tuple

class CascadedRouter:
    """
    Tier 1 Decision Gate for Cascaded Blended RAG:
    Evaluates BM25 candidate scores (s1, s2, gap) and query lexical specificity (MaxIDF)
    to determine whether to early-exit at Tier 1 or escalate to Tier 2 (Dense + SPLADE).
    """
    def __init__(self, doc_freqs: Dict[str, int], total_docs: int, tau_high: float = 24.0, tau_gap: float = 4.0, tau_idf: float = 3.5):
        self.doc_freqs = doc_freqs
        self.total_docs = total_docs
        self.tau_high = tau_high
        self.tau_gap = tau_gap
        self.tau_idf = tau_idf

    def compute_max_idf(self, query_tokens: List[str]) -> float:
        """Computes maximum inverse document frequency among query terms."""
        if not query_tokens:
            return 0.0
        max_idf = 0.0
        for token in query_tokens:
            df = self.doc_freqs.get(token, 0)
            # Standard Lucene/BM25 IDF
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            if idf > max_idf:
                max_idf = idf
        return max_idf

    def route_query(
        self,
        query_tokens: List[str],
        bm25_top_candidates: List[Tuple[str, float]]
    ) -> Tuple[str, dict]:
        """
        Decides routing based on:
          1. s1 >= tau_high (overwhelming lexical confidence)
          2. gap = (s1 - s2) >= tau_gap AND MaxIDF >= tau_idf (distinct winner on specific terms)
        
        Returns:
          ("EARLY_EXIT" or "ESCALATE", telemetry_dict)
        """
        if not bm25_top_candidates:
            return "ESCALATE", {"reason": "no_bm25_hits"}

        s1 = bm25_top_candidates[0][1]
        s2 = bm25_top_candidates[1][1] if len(bm25_top_candidates) > 1 else 0.0
        gap = s1 - s2
        max_idf = self.compute_max_idf(query_tokens)

        telemetry = {
            "s1": round(s1, 3),
            "s2": round(s2, 3),
            "gap": round(gap, 3),
            "max_idf": round(max_idf, 3)
        }

        # Rule 1: Absolute top score threshold
        if s1 >= self.tau_high:
            telemetry["decision"] = "early_exit_high_score"
            return "EARLY_EXIT", telemetry

        # Rule 2: Margin gap separation + term specificity
        if gap >= self.tau_gap and max_idf >= self.tau_idf:
            telemetry["decision"] = "early_exit_gap_margin"
            return "EARLY_EXIT", telemetry

        # Otherwise escalate to Tier 2
        telemetry["decision"] = "escalate_blended_rag"
        return "ESCALATE", telemetry

