import pickle
import os
import pandas as pd
from pypdf import PdfReader

with open('parsed_papers_preview.pkl', 'rb') as f:
    df = pickle.load(f)

# Let's search across all papers for key comparative performance data:
# Datasets: NQ (Natural Questions), SQuAD, BEIR, SciFact, HotpotQA, MS MARCO
# Metrics: Recall@5, Recall@10, NDCG@10, MRR, Latency, EM, F1

data_records = []
for idx, r in df.iterrows():
    fn = r['filename']
    filepath = os.path.join("downloaded_papers", fn)
    if not os.path.exists(filepath):
        continue
    try:
        reader = PdfReader(filepath)
        full_txt = ""
        for p in range(min(7, len(reader.pages))):
            full_txt += reader.pages[p].extract_text() or ""
        
        has_nq = 'natural questions' in full_txt.lower() or ' nq ' in full_txt.lower()
        has_squad = 'squad' in full_txt.lower()
        has_beir = 'beir' in full_txt.lower()
        has_latency = 'latency' in full_txt.lower() or 'throughput' in full_txt.lower() or ' ms ' in full_txt.lower()
        has_rrf = 'reciprocal rank fusion' in full_txt.lower() or ' rrf ' in full_txt.lower()
        has_splade = 'splade' in full_txt.lower()
        
        data_records.append({
            'id': r['id'],
            'title': r['title'],
            'year': r['year'],
            'has_nq': has_nq,
            'has_squad': has_squad,
            'has_beir': has_beir,
            'has_latency': has_latency,
            'has_rrf': has_rrf,
            'has_splade': has_splade
        })
    except:
        pass

d_df = pd.DataFrame(data_records)
print(f"Total surveyed in depth: {len(d_df)}")
print("Papers testing on NQ:", d_df['has_nq'].sum())
print("Papers testing on SQuAD:", d_df['has_squad'].sum())
print("Papers testing on BEIR:", d_df['has_beir'].sum())
print("Papers reporting Latency/Cost:", d_df['has_latency'].sum())
print("Papers using RRF:", d_df['has_rrf'].sum())
print("Papers using SPLADE:", d_df['has_splade'].sum())

direct_competitors = d_df[(d_df['has_beir'] | d_df['has_nq']) & d_df['has_latency']]
print(f"Direct architectural and benchmark competitors ({len(direct_competitors)}):")
for _, r in direct_competitors.head(15).iterrows():
    print(f"  [{r['id']}] {r['title']} ({r['year']}) | RRF: {r['has_rrf']} | SPLADE: {r['has_splade']}")

