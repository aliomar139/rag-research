"""Full-corpus dataset preparation for the Blended RAG dataset extension.

BEIR supplies NQ, TREC-COVID and HotpotQA. SQuAD 1.1 is converted to paragraph
retrieval over every train+dev paragraph, with dev questions as evaluation.
This is a declared adaptation, not the original paper's exact index setup.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

BEIR = {
    "trec-covid": {"md5": "ce62140cb23feb9becf6270d0d1fe6d1", "queries": 50, "documents": 171332},
    "nq": {"md5": "d4d3d2e48787a744b6f6e691ff534307", "queries": 3452, "documents": 2681468},
    "hotpotqa": {"md5": "f412724f78b0d91183a0e86805e16114", "queries": 7405, "documents": 5233329},
}
NAMES = (*BEIR, "squad")


def file_hash(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(url, target):
    target = Path(target)
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name+".part")
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=180) as source, partial.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024*1024)
    partial.replace(target)


def prepare_beir(name, raw):
    info = BEIR[name]
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    archive_path = raw / (name+".zip")
    download(url, archive_path)
    if file_hash(archive_path, "md5") != info["md5"]:
        raise ValueError(f"{name} checksum mismatch; remove the corrupt {archive_path} and retry")
    folder = raw / name
    with zipfile.ZipFile(archive_path) as archive:
        for relative in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Re-extract the three named files from the verified archive. No arbitrary paths.
            with archive.open(name+"/"+relative) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024*1024)
    return folder, {"dataset": "BEIR "+name, "split": "official test",
                    "source_url": url, "official_archive_md5": info["md5"],
                    "expected_queries": info["queries"], "expected_documents": info["documents"],
                    "corpus_scope": "Official full corpus; no relevance-dependent slicing"}


def prepare_squad(raw):
    urls = {split: f"https://rajpurkar.github.io/SQuAD-explorer/dataset/{split}-v1.1.json"
            for split in ("train", "dev")}
    folder = raw / "squad"
    folder.mkdir(parents=True, exist_ok=True)
    documents, queries, labels, source_hashes = {}, {}, {}, {}
    for split, url in urls.items():
        source = raw / f"squad-{split}-v1.1.json"
        download(url, source)
        source_hashes[split] = file_hash(source)
        data = json.loads(source.read_text(encoding="utf-8"))
        if str(data["version"]) != "1.1":
            raise ValueError("Expected official SQuAD 1.1")
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                identity = article["title"]+"\0"+paragraph["context"]
                did = "squad-paragraph-"+hashlib.sha256(identity.encode("utf-8")).hexdigest()
                documents[did] = {"_id": did, "title": article["title"], "text": paragraph["context"]}
                if split == "dev":
                    for question in paragraph["qas"]:
                        qid = "squad-question-"+question["id"]
                        if qid in queries:
                            raise ValueError("Duplicate SQuAD dev question ID")
                        queries[qid] = {"_id": qid, "text": question["question"]}
                        labels[qid] = did
    if len(queries) != 10570:
        raise ValueError(f"Expected 10570 SQuAD 1.1 dev questions, got {len(queries)}")
    for filename, values in (("corpus.jsonl", documents.values()), ("queries.jsonl", queries.values())):
        with (folder / filename).open("w", encoding="utf-8") as output:
            for row in values:
                output.write(json.dumps(row, ensure_ascii=False)+"\n")
    (folder / "qrels").mkdir(exist_ok=True)
    with (folder / "qrels/test.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows((qid, did, 1) for qid, did in labels.items())
    return folder, {"dataset": "SQuAD 1.1 paragraph retrieval", "split": "all official dev questions",
                    "source_urls": urls, "source_sha256": source_hashes,
                    "expected_queries": 10570, "expected_documents": len(documents),
                    "corpus_scope": "All unique title+paragraph pairs from official train and dev; no query-dependent filter",
                    "relevance_definition": "The annotated parent paragraph is relevant; alternative answer-bearing paragraphs are unjudged",
                    "annotation_scope": "Raw parent annotations converted to qrels before timing; never used for routing or calibration",
                    "original_paper_reproduction": False}


def prepare_dataset(name, raw_dir):
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if name in BEIR:
        return prepare_beir(name, raw)
    if name == "squad":
        return prepare_squad(raw)
    raise ValueError("Unsupported dataset: "+name)
