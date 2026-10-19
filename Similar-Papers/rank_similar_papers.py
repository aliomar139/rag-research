import pickle
import pandas as pd
import re

with open('parsed_papers_preview.pkl', 'rb') as f:
    df = pickle.load(f)

# Find papers that explicitly mention:
# "blended", "hybrid retrieval", "sparse", "dense", "fusion", "rrf", "reciprocal rank", "routing", "adaptive-rag", "dat"
targets = []
for idx, r in df.iterrows():
    text = (r['title'] + " " + r['text_preview']).lower()
    score = 0
    reasons = []
    
    if any(k in text for k in ['blended rag', 'blended-rag']):
        score += 15
        reasons.append("mentions blended rag")
    if 'rrf' in text or 'reciprocal rank' in text:
        score += 8
        reasons.append("uses reciprocal rank fusion (RRF)")
    if 'splade' in text:
        score += 8
        reasons.append("uses SPLADE / learned sparse")
    if 'sparse-dense' in text or 'sparse and dense' in text or 'dense and sparse' in text:
        score += 6
        reasons.append("sparse + dense hybrid")
    if 'bm25' in text and ('dense' in text or 'semantic' in text):
        score += 5
        reasons.append("BM25 + dense hybrid")
    if 'multi-field' in text or 'structured field' in text:
        score += 7
        reasons.append("multi-field retrieval")
    if 'query routing' in text or 'adaptive routing' in text or 'early exit' in text:
        score += 7
        reasons.append("adaptive query routing")
    if 'adaptive-rag' in text:
        score += 6
        reasons.append("adaptive-rag comparison")
    if 'dynamic alpha' in text or 'dat' in text:
        score += 6
        reasons.append("dynamic alpha / hybrid tuning")
        
    if score >= 10:
        targets.append({
            'id': r['id'],
            'title': r['title'],
            'year': r['year'],
            'score': score,
            'reasons': ", ".join(reasons),
            'filename': r['filename']
        })

targets_df = pd.DataFrame(targets).sort_values(by='score', ascending=False)
print(f"Found {len(targets_df)} highly relevant comparator papers:")
print(targets_df.to_string(index=False))

