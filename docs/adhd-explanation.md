# How we got here: the project explanation

Updated 21 October 2026.

**We started by trying to skip unnecessary neural searches. We ended with a faster retrieval implementation, a measured speed-quality trade-off, and clear evidence of where the cascade helps.**

Here is the story in order.

## 1. What were we originally trying to improve?

Our retrieval system combines three search methods:

| Method | What it searches for |
|---|---|
| **BM25** | Matching words and terms |
| **Dense** | Similar meaning |
| **SPLADE** | Matching terms expanded by a learned model |

The normal system runs **all three for every question**, then merges their rankings. We call that **static fusion**.

Our idea was:

> Run BM25 first. If its result looks sufficient, return it immediately. Otherwise, run Dense and SPLADE and produce the full fused ranking.

That is the **cascade**. The small decision model is the **gate**.

We trained the gate, rather than new versions of Dense or SPLADE.

### What quality were we trying to preserve?

We measured **Recall@100**: how much of the known relevant material appears within the first 100 retrieved documents.

**Recall retention compares our shortcut's average recall with fusion's average recall.**

For example:

- Fusion achieves 50% recall.
- The shortcut achieves 45%.
- The shortcut retains **90% of fusion recall**, because `45 / 50 = 90%`.

It does **not** mean 90% document-list overlap or 90% answer accuracy.

## 2. Why did the first versions disappoint us?

Several different problems appeared.

### A. Some early results were exploratory

The earlier NQ and cached SciFact experiments helped explore the idea, but some used estimated timings, precomputed neural results, or restricted candidate corpora.

Those results could not establish real end-to-end acceleration. We needed actual conditional execution: **the skipped neural calls must genuinely never happen.**

### B. Keeping almost all recall made skipping difficult

Initially, we aimed to retain roughly **98-99%** of fusion quality.

In the corrected SciFact experiment, the tested rules could not satisfy the quality constraint, so the safe fallback ran full fusion for every query.

**Result: no useful speedup.**

### C. A more complicated gate added too much overhead

We tried a learned Random Forest gate. It skipped many neural calls, but deciding whether to skip was itself expensive.

The overall system became slower despite those skipped calls.

We therefore investigated cheaper models and selected **Ridge**, a small regression model.

### D. A gate trained on one dataset did not reliably transfer

A SciFact-trained Ridge gate became faster on NFCorpus, but retained only about **85.2%** of fusion recall.

That missed the near-99% objective.

We then trained and calibrated on **NFCorpus train**, checked **dev**, and left the already-inspected NFCorpus test split out of further tuning.

### E. The strict NFCorpus gate preserved quality but barely accelerated

The train-calibrated gate retained **99.27%** of fusion recall on dev, but exited early on only about **4% of questions**.

That left very little computation to save.

The diagnostic ZIPs and corpus feature file we downloaded helped investigate these decisions. Richer features and alternative models did not produce a convincing solution to the strict target.

**Our lesson at this stage: preserving almost everything was leaving too little room for useful shortcuts.**

## 3. You changed the objective to a practical trade-off

You asked whether retaining approximately **90-92%** could be acceptable if it produced meaningful acceleration.

We therefore prepared four fixed settings:

**90%, 92%, 95% and 99%.**

These are **training calibration targets**. They are not promises that every later dataset will achieve those exact percentages.

We froze the gate and thresholds, then measured them with real retrieval.

### What the original implementation showed

| Dataset | Setting used as primary | Recall retained | Mean time saved |
|---|---|---:|---:|
| NFCorpus dev | 92% | 94.74% | 23.00% |
| SCIDOCS | 92% | 90.01% | 5.53% |
| TREC-COVID | 90% | 84.53% | 3.56% |

Two findings mattered:

1. **Calibration did not transfer reliably.** All four nominal targets missed on SCIDOCS and TREC.
2. **Skipping neural calls did not necessarily save much total time.**

SCIDOCS was an additional transfer check we introduced. You then asked to evaluate datasets used in the original paper, which led to the TREC-COVID and SQuAD runs.

Those are evaluations of **our MiniLM/SPLADE adaptation**. The original paper uses a different retrieval implementation, including ELSER. [Blended RAG paper](https://arxiv.org/html/2404.07220v2).

## 4. We found the main execution bottleneck

This was the turning point.

On the original TREC implementation:

- **BM25 alone took about 1,410 ms.**
- **Full fusion took about 1,493 ms.**

The cascade always paid that BM25 cost **before making its decision**.

Even when it skipped both neural retrievers, it had already spent most of the query time.

### What we changed

We implemented an indexed BM25 scorer that precomputes term contributions during index preparation.

It avoids repeating expensive work during each query.

Crucially, we gave this optimization to **both fusion and the cascade**.

| TREC component | Original mean | Indexed mean |
|---|---:|---:|
| BM25 | 1,410.14 ms | 8.11 ms |
| Full fusion | 1,493.23 ms | 91.28 ms |

The retrieved rankings and scores remained identical.

This produced two distinct improvements:

- **Indexing:** execute the existing retrieval computation much faster.
- **Routing:** additionally avoid neural retrieval on selected questions.

### What happened to the long SQuAD run?

It was processing **10,570 questions across eight methods**, with three planned timing passes.

We preserved its completed first pass and the additional partial repetitions. We scored the completed pass without rerunning retrieval.

We then used the cached corpus indexes for a shorter indexed benchmark:

**256 questions, the full document corpus, three completed timing passes.**

So the earlier work was preserved; we avoided finishing hours of repeated timing merely to investigate the indexing change.

## 5. What have we actually reached?

### TREC-COVID: the complete progression

These results cover **all 50 queries and all 171,332 documents**.

| Configuration | Recall retained | Speedup vs our original fusion | Extra speedup vs indexed fusion |
|---|---:|---:|---:|
| Indexed fusion | 100% | 16.36x | 1.00x |
| Cascade 99% setting | 96.68% | 20.21x | 1.24x |
| Cascade 95% setting | **91.95%** | **26.83x** | **1.64x** |
| Cascade 92% setting | 84.53% | 35.56x | 2.17x |
| Cascade 90% setting | 84.53% | 36.96x | 2.26x |

The two speedup columns answer different questions:

- **Against our original implementation:** how much did indexing and routing together improve runtime?
- **Against indexed fusion:** how much additional benefit did the cascade itself provide?

For the 95% setting:

```text
Original fusion:   1493.23 ms
Indexed fusion:      91.28 ms
Indexed cascade:     55.66 ms

1493.23 / 55.66 = 26.83x combined improvement
  91.28 / 55.66 =  1.64x additional cascade improvement
```

### Which point should we trust most?

The conservative **99% setting** was the preselected primary for the indexed follow-up.

It achieves:

**96.68% recall retained at 1.24x speedup over indexed fusion.**

Its recall uncertainty interval is **93.58-98.75%**, above your 90% floor in this measured experiment. It still misses its nominal 99% training target.

The faster **95% setting** achieves:

**91.95% retained at 1.64x speedup.**

Its uncertainty interval is **86.88-95.76%**, so we cannot present it as a dependable minimum of 90%.

The maximum **2.26x** cascade point retains only **84.53%**, so it fails your floor.

The original/indexed runtime comparisons span separate runs and include run variability. The reported gains concern mean query retrieval time with prepared indexes; they exclude index construction and answer generation. A mean speedup does not establish a p95 improvement.

### SQuAD: a stronger quality-preserving shortcut

On the fixed 256-question sample:

**1.59x speedup over indexed fusion while retaining 99.22% recall.**

Including indexing, that is **7.91x over our original fusion on those same questions**.

Those indexed numbers apply to the sample, not all 10,570 questions.

### The remaining limitation: standalone retrievers can still be better

This prevents us from declaring an overall victory:

- On TREC, **SPLADE alone is faster with higher recall and nDCG** than the conservative cascade. The cascade has higher MAP.
- On SQuAD, **BM25 alone already retains 95.70% recall at 1.03 ms**, making it the fastest listed option for your 90% floor.
- On NFCorpus and SCIDOCS, Dense provides stronger recall/time choices than the cascades.

We therefore demonstrated a useful improvement **relative to fusion**, while overall superiority to the relevant standalone choices remains unestablished.

## What the project now contributes

We have:

1. **A working conditional retrieval system** with genuinely skipped neural calls.
2. **A compatible execution optimization** that preserves measured rankings and scores.
3. **A complete measured speed-quality curve**, including successful and failed operating points.
4. **An empirical finding:** first-stage execution cost and strong standalone baselines determine whether a cascade is worthwhile.

A major new algorithm or universally better retrieval system has not been established. Full NQ/HotpotQA and generated-answer quality also remain outside the completed evidence.

**The clearest project summary is:**

> We made our retrieval implementation substantially faster, then measured how much additional acceleration selective neural retrieval provides at different recall budgets. The approach improves on full fusion, while its practical value depends on the dataset and competing standalone retrievers.

## Where to continue

Use the revised Abstract in the [consolidated report](FINAL_MANUSCRIPT.md) as the starting point for the final write-up.

The [full speedup tables](SPEEDUP_SUMMARY.md), [current results](FINAL_RESULTS.md) and [growing speedup figure](figures/speedup_progression.png) contain the supporting measurements.
