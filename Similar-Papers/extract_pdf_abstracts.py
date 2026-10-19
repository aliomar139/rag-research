import os
import glob
import re
import pandas as pd
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(BASE_DIR, "downloaded_papers")
INDEX_CSV = os.path.join(BASE_DIR, "downloaded_papers_index.csv")

df = pd.read_csv(INDEX_CSV)
print(f"Total entries in index: {len(df)}")
downloaded = df[df['status'] == 'Success']
print(f"Downloaded entries: {len(downloaded)}")

# Let's inspect titles and abstract / text
results = []

for idx, row in downloaded.iterrows():
    fn = row['filename']
    filepath = os.path.join(PAPERS_DIR, fn)
    if not os.path.exists(filepath):
        continue
    
    text_sample = ""
    try:
        reader = PdfReader(filepath)
        num_pages = len(reader.pages)
        # extract first 2 pages (title, abstract, intro)
        for p in range(min(2, num_pages)):
            text_sample += reader.pages[p].extract_text() or ""
    except Exception as e:
        text_sample = f"Error reading PDF: {e}"

    results.append({
        'id': row['id'],
        'title': row['title'],
        'year': row['year'],
        'venue': row['venue'],
        'filename': fn,
        'pages': num_pages if 'num_pages' in locals() else 0,
        'text_len': len(text_sample),
        'text_preview': text_sample[:1000].replace('\n', ' ')
    })

res_df = pd.DataFrame(results)
print(f"Successfully parsed text from {len(res_df)} PDFs.")
res_df.to_pickle(os.path.join(BASE_DIR, "parsed_papers_preview.pkl"))
print("Saved parsed_papers_preview.pkl")

