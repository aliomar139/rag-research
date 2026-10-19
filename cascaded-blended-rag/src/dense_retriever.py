import os
import numpy as np
import faiss
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from src.cache_utils import corpus_fingerprint, cache_matches, write_cache_metadata

class DenseRetriever:
    def __init__(self, corpus: Dict[str, dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = "data/processed"):
        """
        Builds or loads cached dense embeddings for SciFact corpus using FAISS IndexFlatIP.
        """
        self.doc_ids = list(corpus.keys())
        self.model = SentenceTransformer(model_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        doc_emb_path = os.path.join(cache_dir, "dense_doc_embeddings.npy")
        metadata_path = doc_emb_path + ".json"
        fingerprint = corpus_fingerprint(corpus)
        
        if cache_matches(doc_emb_path, metadata_path, fingerprint, model_name):
            print(f"Loading cached dense embeddings from {doc_emb_path}...")
            embeddings = np.load(doc_emb_path)
            if embeddings.shape[0] != len(self.doc_ids):
                raise ValueError("Dense cache row count does not match the current corpus ordering.")
        else:
            print("Encoding corpus documents (title + abstract) with MiniLM...")
            texts = [f"{corpus[did]['title']} {corpus[did]['abstract']}".strip() for did in self.doc_ids]
            embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype="float32")
            np.save(doc_emb_path, embeddings)
            write_cache_metadata(metadata_path, fingerprint, model_name)
            print(f"Saved corpus embeddings to {doc_emb_path}.")
            
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} documents (dimension {dim}).")

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Single query encoding (batch_size=1) and FAISS inner product search."""
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")
        scores, indices = self.index.search(q_emb, top_k)
        
        return [(self.doc_ids[idx], float(scores[0][i])) for i, idx in enumerate(indices[0])]

