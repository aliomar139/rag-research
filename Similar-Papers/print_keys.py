import pickle
import sys

with open('parsed_papers_preview.pkl', 'rb') as f:
    df = pickle.load(f)

for pid in [43, 2, 75, 7, 12]:
    row = df[df['id'] == pid]
    if not row.empty:
        r = row.iloc[0]
        print('='*70)
        print(f"ID: {pid} | {r['title']} ({r['year']})")
        # clean non-ascii for safe printing
        safe_txt = r['text_preview'][:700].encode('ascii', errors='replace').decode('ascii')
        print(safe_txt)

