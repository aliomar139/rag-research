import os
import torch
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from src.cache_utils import corpus_fingerprint, cache_matches, write_cache_metadata

class SPLADERetriever:
    def __init__(
        self,
        corpus: Dict[str, dict],
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        cache_dir: str = "data/processed",
        batch_size: int = 16
    ):
        """
        SPLADE sparse neural retriever with disk caching (.npz).
        Encodes docs once, saves scipy.sparse.csr_matrix, and reloads in <0.2s.
        """
        self.doc_ids = list(corpus.keys())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SPLADE model ({model_name}) on device: {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        os.makedirs(cache_dir, exist_ok=True)
        index_path = os.path.join(cache_dir, "scifact_splade_index.npz")
        metadata_path = index_path + ".json"
        fingerprint = corpus_fingerprint(corpus)

        if cache_matches(index_path, metadata_path, fingerprint, model_name):
            print(f"Loading cached SPLADE sparse matrix from {index_path}...")
            self.doc_matrix = sp.load_npz(index_path)
            if self.doc_matrix.shape[0] != len(self.doc_ids):
                raise ValueError("SPLADE cache row count does not match the current corpus ordering.")
            print(f"Loaded SPLADE index: shape {self.doc_matrix.shape}.")
        else:
            print("Encoding corpus documents with SPLADE (this takes a few minutes on CPU)...")
            texts = [f"{corpus[did]['title']} {corpus[did]['abstract']}".strip() for did in self.doc_ids]
            self.doc_matrix = self._encode_documents(texts, batch_size=batch_size)
            sp.save_npz(index_path, self.doc_matrix)
            write_cache_metadata(metadata_path, fingerprint, model_name)
            print(f"Saved SPLADE sparse matrix to {index_path}.")

    def _encode_documents(self, texts: List[str], batch_size: int = 16) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding SPLADE"):
            batch_texts = texts[i : i + batch_size]
            tokens = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**tokens)
                logits = outputs.logits
                # SPLADE pooling: log(1 + ReLU(logits)) * attention_mask
                relu_log = torch.log1p(torch.relu(logits))
                attention_mask = tokens["attention_mask"].unsqueeze(-1)
                doc_reps, _ = torch.max(relu_log * attention_mask, dim=1)

            # Convert to CPU numpy sparse components
            doc_reps = doc_reps.cpu().numpy()
            for row_idx_in_batch, rep in enumerate(doc_reps):
                global_row_idx = i + row_idx_in_batch
                nonzero_indices = np.nonzero(rep)[0]
                nonzero_values = rep[nonzero_indices]
                
                rows.extend([global_row_idx] * len(nonzero_indices))
                cols.extend(nonzero_indices)
                data.extend(nonzero_values)

        vocab_size = self.tokenizer.vocab_size
        return sp.csr_matrix((data, (rows, cols)), shape=(len(texts), vocab_size), dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Single query encoding (batch_size=1) and dot-product search with doc sparse matrix."""
        tokens = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**tokens)
            logits = outputs.logits
            relu_log = torch.log1p(torch.relu(logits))
            attention_mask = tokens["attention_mask"].unsqueeze(-1)
            q_rep, _ = torch.max(relu_log * attention_mask, dim=1)
            q_rep = q_rep.squeeze(0).cpu().numpy()

        # Dot product with all docs: (N_docs, Vocab) x (Vocab, 1) -> (N_docs,)
        scores = self.doc_matrix.dot(q_rep)
        
        # Top-k indices
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self.doc_ids[idx], float(scores[idx])) for idx in top_indices]

