# Paste into one Kaggle cell. No retrieval models or GPU are needed.
from pathlib import Path
import hashlib
import io
import os
import shutil
import urllib.request
import zipfile
from IPython.display import FileLink, display

expected = "10cc83ef1826b1425e6a87090b5140b39b27755d5a27e48215a88611c899991f"
destination = Path("/kaggle/working/nfcorpus_corpus_for_features.jsonl")
candidates = [p for p in Path("/kaggle/working").rglob("corpus.jsonl")
              if p.parent.name == "nfcorpus"]
source = next((p for p in candidates
               if hashlib.sha256(p.read_bytes()).hexdigest() == expected), None)
if source:
    shutil.copyfile(source, destination)
else:
    # If the earlier session ended, fetch only the official data archive.
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
    with urllib.request.urlopen(url, timeout=120) as response:
        archive_bytes = response.read()
    assert hashlib.md5(archive_bytes).hexdigest() == "a89dba18a62ef92f7d323ec890a0d38d"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        destination.write_bytes(archive.read("nfcorpus/corpus.jsonl"))

assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected
print(f"Corpus verified. Download: {destination}")
os.chdir("/kaggle/working")
display(FileLink(destination.name))
