# All measured indexed speedups

The tables show every frozen cascade setting, including the fastest settings
that miss the 90% recall floor. The two speedup columns use different references:
original fusion measures indexing plus routing; indexed fusion measures the
additional conditional-routing gain. Standalone methods are separate baselines.

![Full measured speedup progression](figures/speedup_progression.png)

## Headline numbers and their limits

- TREC cascade maximum: **2.26x** vs indexed fusion (**36.96x** vs original
  fusion), retaining **84.53%** recall. This point misses the 90% floor.
- TREC cascade maximum with point retention above 90%: **1.64x** vs indexed
  fusion (**26.83x** vs original fusion), retaining **91.95%**. This is secondary
  r95; its recall interval is 86.88% to 95.76%, crossing below 90%.
- TREC conservative primary: **1.24x** vs indexed fusion (**20.21x** vs original
  fusion), retaining **96.68%**, with recall interval 93.58% to 98.75%.
- SQuAD cascade maximum: **1.59x** vs indexed fusion (**7.91x** vs original
  fusion on the same questions), retaining **99.22%** on the fixed sample.

These are measured means with prepared corpus indexes. Percent time saved is
`1 - 1/speedup`: 19.04% less time equals 1.24x speedup, not 19x. Cross-run
original/indexed ratios include run variability. Original TREC and indexed
TREC have three passes each; original SQuAD has one complete pass and indexed
SQuAD has three. SQuAD comparisons use exactly the same 256 questions.

## TREC-COVID, all 50 queries

| Configuration | Recall retained | Mean ms | Speedup vs original fusion | Speedup vs indexed fusion |
|:--|--:|--:|--:|--:|
| Original fusion, same queries | 100.00% | 1493.23 | 1.00x | N/A |
| Indexed fusion | 100.00% | 91.28 | 16.36x | 1.00x |
| r99 (primary) | 96.68% | 73.90 | 20.21x | 1.24x |
| r95 (secondary) | 91.95% | 55.66 | 26.83x | 1.64x |
| r92 (secondary) | 84.53% | 41.99 | 35.56x | 2.17x |
| r90 (secondary) | 84.53% | 40.41 | 36.96x | 2.26x |

**Standalone baselines, reported separately from cascade results:**

| Method | Recall retained | Mean ms | Speedup vs original fusion | Speedup vs indexed fusion |
|:--|--:|--:|--:|--:|
| BM25 | 71.38% | 8.11 | 184.02x | 11.25x |
| Dense | 69.81% | 30.41 | 49.10x | 3.00x |
| SPLADE | 104.69% | 52.77 | 28.29x | 1.73x |

## SQuAD, 256 of 10,570 questions

| Configuration | Recall retained | Mean ms | Speedup vs original fusion | Speedup vs indexed fusion |
|:--|--:|--:|--:|--:|
| Original fusion, same queries | 100.00% | 147.01 | 1.00x | N/A |
| Indexed fusion | 100.00% | 29.55 | 4.98x | 1.00x |
| r99 (primary) | 100.00% | 28.31 | 5.19x | 1.04x |
| r95 (secondary) | 100.00% | 24.82 | 5.92x | 1.19x |
| r92 (secondary) | 100.00% | 21.31 | 6.90x | 1.39x |
| r90 (secondary) | 99.22% | 18.59 | 7.91x | 1.59x |

**Standalone baselines, reported separately from cascade results:**

| Method | Recall retained | Mean ms | Speedup vs original fusion | Speedup vs indexed fusion |
|:--|--:|--:|--:|--:|
| BM25 | 95.70% | 1.03 | 142.25x | 28.59x |
| Dense | 95.70% | 10.82 | 13.58x | 2.73x |
| SPLADE | 98.83% | 16.96 | 8.67x | 1.74x |

## BM25 scorer speedups with identical rankings

The TREC BM25 scorer itself changes from 1410.14 to 8.11 ms across the T4
pipeline runs: **173.78x**, preserving its old rankings and scores on all 50
queries. This reference is the old BM25 scorer, rather than fusion. Its
Recall@100 remains 71.38% of fusion, so scorer equivalence is not a 90%
fusion-recall result. The matched SQuAD BM25 scorer ratio is **113.74x**.

Separate local paired CPU benchmarks show **134.85x** scorer speedup on
TREC and **34.59x** on NFCorpus train, with identical full score vectors.
Those local scorer measurements have different hardware and timing scope.

## What can be written in the report

> With indexing and conditional execution combined, the TREC pipeline reached
> up to 26.83x observed speedup over our original full-fusion implementation
> while retaining 91.95% of its Recall@100 at a secondary operating point.
> Against equally indexed fusion, that point gives 1.64x speedup. Its recall
> interval crosses below 90%; the conservative primary retains 96.68% at
> 1.24x speedup against the optimized reference.

The absolute maximum TREC cascade point (2.26x vs indexed fusion) is also
reported, with its failed 90% floor. SPLADE and BM25 standalone speedups do
not become cascade claims. No original-paper ELSER or new-algorithm priority
claim follows from comparisons with our original Python implementation.

Regenerate locally from the workspace root; no Kaggle run or upload:

```powershell
& '.\cascaded-blended-rag\venv\Scripts\python.exe' '.\cascaded-blended-rag\experiments\32_prepare_report_assets.py'
```
