# Run the original-paper dataset extension

## First action: upload this file

```text
C:\Users\user\Desktop\rag-research-alone\notebooks\kaggle_paper_datasets_v1\paper_datasets_frozen_source_20261020_a7c4_v1.zip
```

Use this new private Kaggle dataset title:

```text
paper-datasets-frozen-r90-20261020-a7c4-v1
```

This title differs from every earlier title in this project. If your account
already contains it, open that existing dataset instead of creating a second
one. The launcher finds files by name, so the mount folder/slug does not matter.

## Five steps

1. Create the private source dataset using the ZIP above. Upload the ZIP only;
   its archive paths use forward slashes. Kaggle may unpack it automatically.
2. Create a Python notebook. Enable Internet and a T4 GPU; use one GPU for the
   benchmark. Add the new source dataset through Add Input.
3. Paste all of `RUN_IN_KAGGLE.py` into one cell. Leave `DATASET = "trec-covid"`
   and `ALLOW_LARGE_CORPUS = False` for the first run.
4. Run the cell. Corpus encoding is setup work, not measured query latency.
   Budget roughly 1–4 hours for this first TREC-COVID run; this is an estimate,
   not a measured runtime guarantee. Wait for the download link.
5. Download `artifacts/downloads/trec_covid_paper_benchmark_outputs.zip` and put it in
   `C:\Users\user\Desktop\rag-research-alone\`. Send its filename in chat.

If you interrupt the cell, rerun it with the same dataset while its working
files remain available. Completed query repetitions are checkpointed; earlier
queries are not treated as inference caches. Index caches can also be reused.
Closing a Kaggle session can lose working files unless saved. A resumed run
records each timing session; multiple sessions must be disclosed with results.

## Other datasets: same bundle, separate runs

| DATASET value | Evaluation scope | Execution guidance |
|:--|:--|:--|
| `trec-covid` | All 50 BEIR test queries, all 171,332 documents | First Kaggle run |
| `squad` | All 10,570 SQuAD 1.1 dev questions; all unique train+dev title/paragraph pairs | Next separate run; change only DATASET to `squad` |
| `nq` | All 3,452 BEIR test queries, all 2,681,468 documents | Full corpus needs substantial RAM, disk and encoding time; not a normal short Kaggle job |
| `hotpotqa` | All 7,405 BEIR test queries, all 5,233,329 documents | Same limitation, larger still |

NQ and HotpotQA have runnable loader/benchmark code, but full execution has not
been demonstrated within Kaggle's resource limits. Do not set the large-corpus
option merely to bypass this limitation on an ordinary notebook. Their current
BM25/SPLADE indexes are built in memory. A provisioned machine must have enough
memory and disk for the full implementation; this package does not establish
an exact minimum. No automatic corpus or query reduction is performed.

After extracting the source bundle on a provisioned machine with dependencies
installed and a CUDA PyTorch build, the commands are:

```bash
python experiments/20_benchmark_paper_datasets.py --dataset nq --allow-large-corpus
python experiments/20_benchmark_paper_datasets.py --dataset hotpotqa --allow-large-corpus
```

Each dataset writes to a different output/cache directory. SQuAD returns
`squad_paper_benchmark_outputs.zip`; NQ and HotpotQA use the same naming pattern.

## What this evaluates

- Frozen NFCorpus-trained original 11-feature gate at 90/92/95/99% thresholds;
  no new training or threshold adjustment on these evaluation queries.
- Eight methods: static RRF, BM25, Dense, SPLADE and the four cascade policies.
  The nominal 90% policy is primary for this extension, selected before its
  outcomes are inspected. The other policies remain reported secondary points.
- Recall@100 retention, nDCG@10, MAP@10, hit@10, hit@20 and Recall@10;
  actual mean/p95/p99 retrieval latency, branch counts and paired uncertainty.
- Full corpora, three warmed randomized timing repetitions; no per-query
  ranking sharing across methods, no relevance-filtered document pools.

The ZIP contains the source and frozen gate; the notebook downloads the datasets
and model assets. It does not contain precomputed evaluation rankings.

## How to judge success

At least 90% observed recall retention with reduced latency supports a measured
trade-off against our full-fusion baseline on that dataset. Report interval
bounds too. A lower retention bound below 90% means a 90% floor is uncertain.
The magnitude of time saved matters; 1% and 40% reductions are different results.

Standalone baselines must stay in the main tables on every dataset. If a single
retriever is faster with equal or better quality, the cascade is not the best
measured operating point there. A win against fusion alone does not establish
a major algorithmic contribution or superiority over the original paper.
Report every tested dataset, including NFCorpus and SCIDOCS, with its proper
development/evaluation status. Do not select only favorable datasets or the
best threshold after inspecting results and call that a preselected success.

## Reproduction boundaries

This evaluates our MiniLM/SPLADE/RRF adaptation on dataset families used by
Blended RAG. The original paper uses Elasticsearch/ELSER and different query
configurations. It is not an exact reproduction, and absolute scores across
the two implementations are not a controlled speed comparison.

SQuAD is converted to paragraph retrieval using official train+dev paragraphs
as the index and all dev questions as queries. Only the annotated parent
paragraph is judged relevant. Raw annotations are converted before timing,
but not used to choose routes, thresholds or corpus members. This conversion
and corpus scope may differ from the original paper's document construction.
No generated answers or answer EM/F1 are measured.

Sources: [Blended RAG](https://arxiv.org/html/2404.07220v1),
[BEIR dataset table](https://github.com/beir-cellar/beir#beers-available-datasets),
[official SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).
