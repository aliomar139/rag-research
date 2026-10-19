import os
import pandas as pd
from pypdf import PdfReader

key_papers = [
    ("43", "Exact Adaptive Hybrid Retrieval Without Fixed Top-L Cutoffs"),
    ("2", "DAT: Dynamic Alpha Tuning for Hybrid Retrieval in Retrieval-Augmented Generation"),
    ("75", "Higress-RAG: A Holistic Optimization Framework for Enterprise Retrieval-Augmented Generation via Dual Hybrid Retrieval, Adaptive Routing, and CRAG"),
    ("7", "Unsupervised Query Routing for Retrieval Augmented Generation"),
    ("12", "Evidence-Unit Fairness and the Limits of Query-Adaptive Sparse-Dense Fusion in Financial Document Retrieval"),
    ("41", "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity"),
    ("69", "SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems"),
    ("51", "Adaptive Chunking: Optimizing Chunking-Method Selection for RAG"),
    ("9", "From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents"),
    ("16", "MoR: Better Handling Diverse Queries with a Mixture of Sparse, Dense, and Human Retrievers"),
    ("61", "SAC-RAG: Semantic Adaptive Context Compression for Retrieval-Augmented Generation")
]

index_df = pd.read_csv("downloaded_papers_index.csv")

for pid, title in key_papers:
    row = index_df[index_df['id'] == int(pid)]
    if row.empty:
        continue
    fn = row.iloc[0]['filename']
    path = os.path.join("downloaded_papers", fn)
    print("="*80)
    print(f"PAPER [{pid}]: {title}")
    if os.path.exists(path):
        try:
            reader = PdfReader(path)
            txt = ""
            for p in range(min(3, len(reader.pages))):
                txt += reader.pages[p].extract_text() or ""
            print(txt[:1200].replace('\n', ' '))
        except Exception as e:
            print("Error reading:", e)
    else:
        print("File not found on disk:", fn)

