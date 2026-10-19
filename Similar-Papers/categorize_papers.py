import pickle
import pandas as pd
import re

with open('parsed_papers_preview.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"Loaded {len(df)} parsed papers.")

# Let's categorize papers based on their core mechanism:
# 1. Hybrid Retrieval (Sparse + Dense / BM25 + Dense / Multi-field / Fusion / RRF)
# 2. Adaptive / Dynamic Weighting (Dynamic alpha / query-specific weights)
# 3. Adaptive Routing / Gating / Cascading (Deciding whether to retrieve, routing between retrievers, early exit)
# 4. Adaptive Chunking / Compression / Context Selection (Chunk size, summarization, context reduction)
# 5. Graph-RAG / Knowledge Graph + RAG (Graph + vector hybrid)
# 6. Domain-specific Adaptive RAG (Medical, Legal, Code, Education)

categories = {
    'Hybrid Retrieval & Multi-Field Fusion': [],
    'Dynamic Weighting & Score Tuning': [],
    'Adaptive Query Routing & Gating (Closest to Cascaded/Early-Exit)': [],
    'Adaptive Context & Dynamic Chunking': [],
    'Graph-RAG & Multi-Modal Hybrid': [],
    'Domain-Specific Adaptive RAG': []
}

for idx, r in df.iterrows():
    title = r['title']
    text = (r['title'] + " " + r['text_preview']).lower()
    
    # Check indicators
    is_hybrid_fusion = any(k in text for k in ['hybrid retrieval', 'sparse-dense', 'bm25', 'rrf', 'reciprocal rank', 'splade', 'multi-field', 'dense passage'])
    is_dynamic_weight = any(k in text for k in ['dynamic alpha', 'weight tuning', 'adaptive retriever weighting', 'rank fusion'])
    is_routing_gating = any(k in text for k in ['routing', 'router', 'gating', 'gate', 'cascade', 'early exit', 'decision', 'when to retrieve', 'adaptive retrieval'])
    is_chunking_ctx = any(k in text for k in ['chunking', 'compression', 'context selection', 'pruning', 'budget'])
    is_graph = any(k in text for k in ['graph', 'kg-', 'knowledge graph'])
    
    assigned = False
    if 'dynamic alpha' in text or 'weighting' in text and 'adaptive' in text:
        categories['Dynamic Weighting & Score Tuning'].append(r)
        assigned = True
    elif any(k in text for k in ['router', 'routing', 'gating', 'cascade', 'early exit', 'retrieval as a decision']):
        categories['Adaptive Query Routing & Gating (Closest to Cascaded/Early-Exit)'].append(r)
        assigned = True
    elif is_hybrid_fusion:
        categories['Hybrid Retrieval & Multi-Field Fusion'].append(r)
        assigned = True
    elif is_chunking_ctx:
        categories['Adaptive Context & Dynamic Chunking'].append(r)
        assigned = True
    elif is_graph:
        categories['Graph-RAG & Multi-Modal Hybrid'].append(r)
        assigned = True
    else:
        categories['Domain-Specific Adaptive RAG'].append(r)

print("\n--- CATEGORY BREAKDOWN ---")
for cat, items in categories.items():
    print(f"\n### {cat} (Count: {len(items)})")
    for it in items[:6]:
        print(f"  - [{it['id']}] {it['title']} ({it['year']})")

