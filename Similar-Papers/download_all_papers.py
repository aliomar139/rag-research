import os
import re
import sys
import time
import shutil
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(BASE_DIR, "downloaded_papers")
INDEX_CSV = os.path.join(BASE_DIR, "downloaded_papers_index.csv")
REPORT_MD = os.path.join(BASE_DIR, "download_report.md")
DOCS_DIR = os.path.join(os.path.dirname(BASE_DIR), "docs")

os.makedirs(PAPERS_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 PaperResearchDownloader/1.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8"
}

session = requests.Session()
session.headers.update(HEADERS)

def clean_title(t):
    if not isinstance(t, str):
        return ""
    return t.strip().rstrip('.').strip()

def norm_title(t):
    t = clean_title(t).lower()
    return re.sub(r'[^a-z0-9]', '', t)

def sanitize_filename(name):
    # Remove invalid windows filename characters: < > : " / \ | ? *
    clean = re.sub(r'[<>:"/\\|?*]', '_', name)
    clean = re.sub(r'\s+', ' ', clean).strip()
    if len(clean) > 85:
        clean = clean[:85].rstrip(' _')
    return clean

def extract_arxiv_id(text):
    if not isinstance(text, str):
        return None
    m = re.search(r'(?:arxiv\.org/abs/|arxiv\.org/pdf/|arxiv\.|abs[-/])([0-9]{4}\.[0-9]{4,5}|[0-9]{4}-[0-9]{4,5})', text, re.I)
    if m:
        return m.group(1).replace('-', '.')
    m2 = re.search(r'\b([0-9]{4}\.[0-9]{4,5})\b', text)
    if m2:
        return m2.group(1)
    return None

def verify_and_save_pdf(resp_or_bytes, dest_path):
    if isinstance(resp_or_bytes, requests.Response):
        content = resp_or_bytes.content
    else:
        content = resp_or_bytes
    
    if len(content) < 5000:
        return False, "File too small (< 5KB)"
    
    # Check PDF magic bytes (allowing small BOM or leading whitespace)
    header = content[:1024]
    if b'%PDF-' not in header:
        return False, "Not a valid PDF header"
    
    temp_path = dest_path + ".tmp"
    try:
        with open(temp_path, "wb") as f:
            f.write(content)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)
        return True, f"Saved ({len(content) // 1024} KB)"
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False, str(e)

def download_url(url, dest_path, timeout=20):
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True, stream=True)
        if r.status_code == 200:
            ok, msg = verify_and_save_pdf(r, dest_path)
            if ok:
                return True, msg, r.url
            return False, msg, r.url
        return False, f"HTTP {r.status_code}", url
    except Exception as e:
        return False, str(e), url

def try_arxiv(arxiv_id, dest_path):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    ok, msg, final_url = download_url(url, dest_path)
    if ok:
        return True, "arXiv", final_url
    return False, msg, None

def search_arxiv_by_title(title):
    try:
        clean_q = re.sub(r'[^a-zA-Z0-9 ]', ' ', title).strip()
        url = f"https://export.arxiv.org/api/query?search_query=ti:%22{urllib.parse.quote(clean_q)}%22&max_results=3"
        r = session.get(url, timeout=12)
        if r.status_code == 200:
            root = ET.fromstring(r.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            norm_q = norm_title(title)
            for entry in root.findall('atom:entry', ns):
                entry_title = entry.find('atom:title', ns).text.strip()
                norm_entry = norm_title(entry_title)
                # Compare similarity
                if norm_entry in norm_q or norm_q in norm_entry or (len(norm_entry) > 10 and norm_entry[:30] == norm_q[:30]):
                    id_elem = entry.find('atom:id', ns).text.strip()
                    arx = extract_arxiv_id(id_elem)
                    if arx:
                        return arx
    except Exception as e:
        pass
    return None

def try_openalex(title, doi, dest_path):
    # Look up by DOI or Title
    data = None
    try:
        if doi:
            clean_doi = doi.replace("https://doi.org/", "").strip()
            url = f"https://api.openalex.org/works/https://doi.org/{clean_doi}"
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
        if not data and title:
            url = f"https://api.openalex.org/works?search={urllib.parse.quote(title)}&per_page=1"
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                results = r.json().get('results', [])
                if results:
                    top = results[0]
                    # verify title match
                    if norm_title(top.get('title', '')) == norm_title(title) or norm_title(title)[:30] in norm_title(top.get('title', '')):
                        data = top
    except Exception:
        pass

    if not data:
        return False, "Not found in OpenAlex", None

    # Check locations for direct PDF
    locations = data.get('locations', [])
    pdf_candidates = []
    
    # 1. best_oa_location
    best_oa = data.get('best_oa_location') or {}
    if best_oa.get('pdf_url'):
        pdf_candidates.append(best_oa['pdf_url'])
    if data.get('open_access', {}).get('oa_url'):
        pdf_candidates.append(data['open_access']['oa_url'])

    for loc in locations:
        if loc.get('pdf_url') and loc['pdf_url'] not in pdf_candidates:
            pdf_candidates.append(loc['pdf_url'])
        landing = loc.get('landing_page_url')
        if landing:
            # Check for AAAI OJS
            if 'ojs.aaai.org' in landing and '/view/' in landing:
                # OJS download link pattern
                # view/{id} or view/{id}/{galley}
                m = re.search(r'article/view/(\d+)(?:/(\d+))?', landing)
                if m:
                    art_id = m.group(1)
                    galley = m.group(2)
                    if galley:
                        pdf_candidates.append(f"https://ojs.aaai.org/index.php/AAAI/article/download/{art_id}/{galley}")
                    else:
                        pdf_candidates.append(f"https://ojs.aaai.org/index.php/AAAI/article/download/{art_id}")

    for purl in pdf_candidates:
        if not purl:
            continue
        # If it's an IEEE link, handle IEEE stamp
        if 'ieeexplore.ieee.org' in purl:
            m = re.search(r'arnumber=(\d+)|document/(\d+)', purl)
            if m:
                arnum = m.group(1) or m.group(2)
                stamp_url = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arnum}"
                ok, msg, f_url = download_ieee_stamp(stamp_url, dest_path)
                if ok:
                    return True, "OpenAlex -> IEEE OA", f_url

        ok, msg, f_url = download_url(purl, dest_path)
        if ok:
            return True, f"OpenAlex ({purl[:30]}...)", f_url
        
    return False, "No valid PDF in OpenAlex locations", None

def try_unpaywall(doi, dest_path):
    if not doi:
        return False, "No DOI", None
    clean_doi = doi.replace("https://doi.org/", "").strip()
    url = f"https://api.unpaywall.org/v2/{clean_doi}?email=student@academic.edu"
    try:
        r = session.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get('is_oa'):
                best_oa = data.get('best_oa_location') or {}
                pdf_url = best_oa.get('url_for_pdf') or best_oa.get('url')
                if pdf_url:
                    ok, msg, f_url = download_url(pdf_url, dest_path)
                    if ok:
                        return True, "Unpaywall", f_url
    except Exception:
        pass
    return False, "Not OA in Unpaywall", None

def download_ieee_stamp(stamp_url, dest_path):
    try:
        r = session.get(stamp_url, timeout=12)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            iframe = soup.find('iframe')
            if iframe and iframe.get('src'):
                pdf_src = iframe.get('src')
                ok, msg, f_url = download_url(pdf_src, dest_path)
                if ok:
                    return True, msg, f_url
    except Exception:
        pass
    return False, "IEEE stamp failed", None

def try_scrape_landing(landing_url, dest_path):
    if not landing_url:
        return False, "No landing URL", None
    try:
        r = session.get(landing_url, timeout=12)
        if r.status_code == 200:
            # Check for IEEE
            if 'ieeexplore.ieee.org/document/' in r.url:
                m = re.search(r'document/(\d+)', r.url)
                if m:
                    stamp_url = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={m.group(1)}"
                    ok, msg, f_url = download_ieee_stamp(stamp_url, dest_path)
                    if ok:
                        return True, "IEEE Scraper", f_url

            soup = BeautifulSoup(r.text, 'html.parser')
            # Look for meta citation_pdf_url
            meta_pdf = soup.find('meta', attrs={'name': lambda x: x and x.lower() == 'citation_pdf_url'})
            if meta_pdf and meta_pdf.get('content'):
                pdf_url = meta_pdf.get('content')
                ok, msg, f_url = download_url(pdf_url, dest_path)
                if ok:
                    return True, "Meta citation_pdf_url", f_url

            # Look for direct PDF download links
            for a in soup.find_all('a'):
                href = a.get('href')
                txt = a.get_text(strip=True).lower()
                if href and ('download' in txt or 'pdf' in txt) and ('pdf' in href.lower() or 'download' in href.lower()):
                    full_href = urllib.parse.urljoin(r.url, href)
                    ok, msg, f_url = download_url(full_href, dest_path)
                    if ok:
                        return True, "Page PDF Link", f_url
    except Exception:
        pass
    return False, "Scraping failed", None

def load_and_deduplicate():
    f1 = os.path.join(BASE_DIR, 'consensus_adaptive_or_hybird_rag.csv.csv')
    f2 = os.path.join(BASE_DIR, 'dblp_adaptive_or_hybird_rag.csv')
    
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    
    papers_by_norm = {}

    for idx, row in df1.iterrows():
        title = clean_title(row.get('Title'))
        if not title:
            continue
        nt = norm_title(title)
        doi = str(row.get('DOI')).strip() if pd.notna(row.get('DOI')) else None
        if doi and doi.lower() == 'nan':
            doi = None
        arx = extract_arxiv_id(doi) if doi else None
        if not arx and pd.notna(row.get('Consensus Link')):
            arx = extract_arxiv_id(str(row.get('Consensus Link')))
            
        if nt not in papers_by_norm:
            papers_by_norm[nt] = {
                'title': title,
                'authors': str(row.get('Authors')) if pd.notna(row.get('Authors')) else '',
                'year': int(row.get('Year')) if pd.notna(row.get('Year')) else '',
                'doi': doi,
                'arxiv_id': arx,
                'sources': ['Consensus'],
                'venue': str(row.get('Journal')) if pd.notna(row.get('Journal')) else '',
                'dblp_key': None,
                'dblp_link': None,
                'consensus_link': row.get('Consensus Link') if pd.notna(row.get('Consensus Link')) else None
            }
        else:
            p = papers_by_norm[nt]
            if 'Consensus' not in p['sources']:
                p['sources'].append('Consensus')
            if not p['doi'] and doi:
                p['doi'] = doi
            if not p['arxiv_id'] and arx:
                p['arxiv_id'] = arx

    for idx, row in df2.iterrows():
        title = clean_title(row.get('Title'))
        if not title:
            continue
        nt = norm_title(title)
        key = str(row.get('DBLP Key')).strip() if pd.notna(row.get('DBLP Key')) else None
        link = str(row.get('DBLP Link')).strip() if pd.notna(row.get('DBLP Link')) else None
        arx = extract_arxiv_id(key) or extract_arxiv_id(link)
        
        if nt not in papers_by_norm:
            papers_by_norm[nt] = {
                'title': title,
                'authors': str(row.get('Authors')) if pd.notna(row.get('Authors')) else '',
                'year': int(row.get('Year')) if pd.notna(row.get('Year')) else '',
                'doi': None,
                'arxiv_id': arx,
                'sources': ['DBLP'],
                'venue': str(row.get('Venue')) if pd.notna(row.get('Venue')) else '',
                'dblp_key': key,
                'dblp_link': link,
                'consensus_link': None
            }
        else:
            p = papers_by_norm[nt]
            if 'DBLP' not in p['sources']:
                p['sources'].append('DBLP')
            if not p['arxiv_id'] and arx:
                p['arxiv_id'] = arx
            if not p['dblp_key'] and key:
                p['dblp_key'] = key
            if not p['dblp_link'] and link:
                p['dblp_link'] = link
            if not p['venue'] and pd.notna(row.get('Venue')):
                p['venue'] = str(row.get('Venue'))

    return list(papers_by_norm.values())

def main():
    print("="*80)
    print("STARTING PAPER DOWNLOADER & DEDUPLICATOR")
    print("="*80)
    
    unique_papers = load_and_deduplicate()
    total = len(unique_papers)
    print(f"Loaded {total} unique deduplicated papers across both CSVs.\n")

    results = []
    success_count = 0

    # Also check existing Blended-Rag.pdf in docs
    existing_blended = os.path.join(DOCS_DIR, "Blended-Rag.pdf")

    for i, paper in enumerate(unique_papers, 1):
        title = paper['title']
        year = str(paper['year']) if paper['year'] else "Unknown"
        safe_t = sanitize_filename(title)
        filename = f"{i:03d}_{year}_{safe_t}.pdf"
        dest_path = os.path.join(PAPERS_DIR, filename)

        status = "Failed"
        download_source = "None"
        source_url = ""
        file_size_kb = 0

        print(f"[{i:03d}/{total:03d}] {title[:65]}...")

        # 0. Check if already exists in target directory
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 5000:
            with open(dest_path, "rb") as f:
                if b"%PDF-" in f.read(1024):
                    status = "Success (Cached)"
                    download_source = "Local Disk"
                    file_size_kb = os.path.getsize(dest_path) // 1024
                    print(f"  -> Already exists: {filename} ({file_size_kb} KB)")
                    success_count += 1
                    paper_record = dict(paper)
                    paper_record['id'] = i
                    paper_record['status'] = status
                    paper_record['download_source'] = download_source
                    paper_record['source_url'] = source_url
                    paper_record['filename'] = filename
                    paper_record['file_size_kb'] = file_size_kb
                    results.append(paper_record)
                    continue

        # 1. Check if Blended RAG matching existing doc
        if 'blended rag' in norm_title(title) and os.path.exists(existing_blended):
            shutil.copy2(existing_blended, dest_path)
            status = "Success"
            download_source = "Existing Project Doc (Blended-Rag.pdf)"
            file_size_kb = os.path.getsize(dest_path) // 1024
            print(f"  -> Copied from docs: {filename} ({file_size_kb} KB)")
            success_count += 1
            paper_record = dict(paper)
            paper_record['id'] = i
            paper_record['status'] = status
            paper_record['download_source'] = download_source
            paper_record['source_url'] = "Local /docs/Blended-Rag.pdf"
            paper_record['filename'] = filename
            paper_record['file_size_kb'] = file_size_kb
            results.append(paper_record)
            continue

        # 2. Try direct arXiv ID
        if paper.get('arxiv_id'):
            ok, src, url = try_arxiv(paper['arxiv_id'], dest_path)
            if ok:
                status = "Success"
                download_source = src
                source_url = url
                file_size_kb = os.path.getsize(dest_path) // 1024
                print(f"  -> Downloaded via arXiv ({paper['arxiv_id']}): {file_size_kb} KB")
                success_count += 1
                paper_record = dict(paper)
                paper_record['id'] = i
                paper_record['status'] = status
                paper_record['download_source'] = download_source
                paper_record['source_url'] = source_url
                paper_record['filename'] = filename
                paper_record['file_size_kb'] = file_size_kb
                results.append(paper_record)
                time.sleep(0.4)
                continue

        # 3. Try searching arXiv by title
        arx_id = search_arxiv_by_title(title)
        if arx_id:
            paper['arxiv_id'] = arx_id
            ok, src, url = try_arxiv(arx_id, dest_path)
            if ok:
                status = "Success"
                download_source = f"arXiv Search ({arx_id})"
                source_url = url
                file_size_kb = os.path.getsize(dest_path) // 1024
                print(f"  -> Downloaded via arXiv Search ({arx_id}): {file_size_kb} KB")
                success_count += 1
                paper_record = dict(paper)
                paper_record['id'] = i
                paper_record['status'] = status
                paper_record['download_source'] = download_source
                paper_record['source_url'] = source_url
                paper_record['filename'] = filename
                paper_record['file_size_kb'] = file_size_kb
                results.append(paper_record)
                time.sleep(0.4)
                continue

        # 4. Try OpenAlex
        ok, src, url = try_openalex(title, paper.get('doi'), dest_path)
        if ok:
            status = "Success"
            download_source = src
            source_url = url
            file_size_kb = os.path.getsize(dest_path) // 1024
            print(f"  -> Downloaded via {src}: {file_size_kb} KB")
            success_count += 1
            paper_record = dict(paper)
            paper_record['id'] = i
            paper_record['status'] = status
            paper_record['download_source'] = download_source
            paper_record['source_url'] = source_url
            paper_record['filename'] = filename
            paper_record['file_size_kb'] = file_size_kb
            results.append(paper_record)
            time.sleep(0.4)
            continue

        # 5. Try Unpaywall
        if paper.get('doi'):
            ok, src, url = try_unpaywall(paper['doi'], dest_path)
            if ok:
                status = "Success"
                download_source = src
                source_url = url
                file_size_kb = os.path.getsize(dest_path) // 1024
                print(f"  -> Downloaded via Unpaywall: {file_size_kb} KB")
                success_count += 1
                paper_record = dict(paper)
                paper_record['id'] = i
                paper_record['status'] = status
                paper_record['download_source'] = download_source
                paper_record['source_url'] = source_url
                paper_record['filename'] = filename
                paper_record['file_size_kb'] = file_size_kb
                results.append(paper_record)
                time.sleep(0.4)
                continue

        # 6. Try landing page scraping if DOI or Link exists
        landing_url = None
        if paper.get('doi'):
            clean_doi = paper['doi'].replace("https://doi.org/", "").strip()
            landing_url = f"https://doi.org/{clean_doi}"
        elif paper.get('dblp_link'):
            landing_url = paper['dblp_link']
            
        if landing_url:
            ok, src, url = try_scrape_landing(landing_url, dest_path)
            if ok:
                status = "Success"
                download_source = src
                source_url = url
                file_size_kb = os.path.getsize(dest_path) // 1024
                print(f"  -> Downloaded via {src}: {file_size_kb} KB")
                success_count += 1
                paper_record = dict(paper)
                paper_record['id'] = i
                paper_record['status'] = status
                paper_record['download_source'] = download_source
                paper_record['source_url'] = source_url
                paper_record['filename'] = filename
                paper_record['file_size_kb'] = file_size_kb
                results.append(paper_record)
                time.sleep(0.4)
                continue

        # If all failed, mark as Paywalled / Not Openly Accessible
        print(f"  -> [PAYWALLED / NO OPEN PDF AVAILABLE]")
        paper_record = dict(paper)
        paper_record['id'] = i
        paper_record['status'] = "Paywalled / Restricted"
        paper_record['download_source'] = "None"
        paper_record['source_url'] = landing_url or ""
        paper_record['filename'] = ""
        paper_record['file_size_kb'] = 0
        results.append(paper_record)
        time.sleep(0.3)

    # Save Results CSV
    res_df = pd.DataFrame(results)
    # Reorder columns
    cols = ['id', 'status', 'title', 'authors', 'year', 'venue', 'sources', 'arxiv_id', 'doi', 'download_source', 'filename', 'file_size_kb', 'source_url']
    res_df = res_df[[c for c in cols if c in res_df.columns]]
    res_df.to_csv(INDEX_CSV, index=False, encoding='utf-8-sig')

    # Generate Markdown Report
    with open(REPORT_MD, "w", encoding="utf-8") as rf:
        rf.write("# Download & Scraping Report: Similar Papers (Adaptive & Hybrid RAG)\n\n")
        rf.write(f"- **Total Unique Papers Deduplicated**: {total}\n")
        rf.write(f"- **Successfully Downloaded**: {success_count} ({success_count/total*100:.1f}%)\n")
        rf.write(f"- **Paywalled / Non-OA**: {total - success_count} ({(total - success_count)/total*100:.1f}%)\n")
        rf.write(f"- **Storage Directory**: `{PAPERS_DIR}`\n")
        rf.write(f"- **Catalog Index File**: `{INDEX_CSV}`\n\n")
        
        rf.write("## Download Summary Table\n\n")
        rf.write("| # | Status | Title | Year | Venue | Size (KB) | Download Source |\n")
        rf.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            t_short = r['title'].replace('|', '-')
            if len(t_short) > 50:
                t_short = t_short[:50] + "..."
            v_short = str(r['venue']).replace('|', '-')
            if len(v_short) > 20:
                v_short = v_short[:20] + "..."
            rf.write(f"| {r['id']} | {r['status']} | {t_short} | {r['year']} | {v_short} | {r['file_size_kb']} | {r['download_source']} |\n")

    print("\n" + "="*80)
    print("DOWNLOAD PROCESS COMPLETED!")
    print(f"Downloaded: {success_count} / {total} papers ({success_count/total*100:.1f}%)")
    print(f"PDFs Directory: {PAPERS_DIR}")
    print(f"Index CSV: {INDEX_CSV}")
    print(f"Report Markdown: {REPORT_MD}")
    print("="*80)

if __name__ == "__main__":
    main()

