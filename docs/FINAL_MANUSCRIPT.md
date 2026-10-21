# Conditional Hybrid Retrieval under Recall Budgets

## An Indexed Cascade Study

Research project report. Completed experimental scope, 21 October 2026.

## Abstract

Hybrid retrieval improves coverage by executing complementary retrievers, but its query cost can limit the benefit of conditional routing. We study a BM25-first Ridge gate that either returns lexical results or executes Dense MiniLM and SPLADE before reciprocal rank fusion. Four policies, calibrated on NFCorpus train at nominal recall-retention targets of 90%, 92%, 95% and 99%, remain frozen during transfer. Initial measurements identify an expensive lexical stage and quality targets that fail to transfer. We implement compatible indexed BM25 scoring and apply it equally to the reference and cascade. On all 50 TREC-COVID queries, routing adds 1.24x speedup over indexed fusion at 96.68% Recall@100 retention for the preselected primary. A secondary setting reaches 1.64x at 91.95%, although its confidence interval crosses below 90%. On a fixed 256-question SQuAD sample, the fastest cascade reaches 1.59x at 99.22% retention. Saved rankings agree exactly across backends, and independent raw-label scoring reproduces the quality metrics. Standalone retrievers still offer stronger recall/time points in these collections. The contribution is a measured execution optimization and an analysis of conditional retrieval under explicit quality budgets.

**Keywords:** hybrid retrieval, conditional computation, BM25 indexing, recall retention, retrieval evaluation.

## 1. Original paper and research context

Sawarkar, Mangal and Solanki's *Blended RAG* studies keyword, dense-vector and learned-sparse retrieval with hybrid field queries [1]. Its sparse implementation uses ELSER with Elasticsearch. The paper evaluates retrieval on several collections and generated answers on NQ and SQuAD. Its reported TREC-COVID nDCG@10 of 0.87 and SQuAD answer EM/F1 of 57.63/68.40 are published reference results, not measurements reproduced here. CoQA appears in its analysis of documents without useful metadata.

This project adopts the motivation that different retrieval signals can supply useful context, then asks whether every query needs all branches. We implement MiniLM, SPLADE and reciprocal rank fusion in Python. The completed study therefore constitutes a retrieval adaptation and engineering extension. It measures our pipeline's cost and quality, without attributing its speedups to the original ELSER system.

Query-dependent selection already has precedents. Arabzadeh et al. train a classifier to choose sparse, dense or hybrid retrieval for individual queries [2]. Eager sparse scoring also predates this project: BM25S moves scoring work into index construction [3]. Our gate and postings representation do not establish algorithmic priority over those ideas. The empirical question concerns their interaction under the frozen policies, datasets and execution settings evaluated here.

## 2. Motivation and contribution

A cascade saves time only when the work it avoids exceeds the work paid before its decision. A lexical first stage can be accurate enough for some queries, but an expensive implementation can consume nearly the entire fusion runtime. Counting skipped neural calls then overstates the benefit. Our initial TREC-COVID measurements illustrate this constraint: BM25 takes 1410.14 ms against 1493.23 ms for static fusion.

Quality budgets make the decision measurable. We investigate policies calibrated to retain 90%, 92%, 95% or 99% of fusion's mean Recall@100, while reporting actual retention on later collections. Calibration is an empirical training procedure. A nominal target names the fitted policy and does not promise that the same ratio will hold on a new dataset.

The project supplies four related results: a frozen budgeted cascade, a compatible BM25 execution change, an evaluation that separates indexing gains from conditional-routing gains, and evidence that standalone retrievers can make fusion-relative savings an incomplete objective. We retain failed targets and faster low-quality points so readers can inspect the full trade-off. Claims of competitive superiority, a new routing principle or preserved generated-answer quality require further evidence.

## 3. Adaptation, reproduction and metric differences

### 3.1 What we reproduced and changed

Our reproduction target is the use of complementary retrieval signals and Best Fields lexical search. We substitute accessible local model and engine choices, construct a declared paragraph corpus for SQuAD, and add conditional execution. These changes produce a different experimental system. Table 1 specifies the comparison before presenting results.

**Table 1. Relationship to the original paper. Published quality values are contextual references rather than matched baselines.**

| Dimension | Original Blended RAG [1] | This project |
|:--|:--|:--|
| Retrieval engine | Elasticsearch with ELSER and field-query combinations | Python BM25, MiniLM, SPLADE and RRF |
| Main question | Retrieval and RAG quality with blended field queries | Query latency under frozen recall-retention budgets |
| Retrieval reporting | Top-k accuracy and nDCG@10 | Recall@100, retention, nDCG@10, MAP@10 and latency |
| Generated answers | NQ/SQuAD EM and F1 | Outside completed evaluation |
| Timing reference | Paper's own system | Our original fusion and equally indexed fusion |
| Completed overlap | Paper includes TREC-COVID and SQuAD | Full TREC queries and declared SQuAD evaluations |

### 3.2 Why numerical metrics differ

The metric definition determines what a number means. Top-k hit accuracy asks whether at least one relevant document occurs among the first k results. Recall@100 measures the fraction of all judged-relevant documents recovered in the first 100. A TREC query can have many relevant documents, so a method can score a hit while recovering a small fraction of that total. nDCG@10 rewards useful ordering near the top; its value is neither a recall-retention percentage nor answer accuracy.

We define retention as the ratio of mean recalls: `rho = mean Recall_method@100 / mean Recall_fusion@100`. It can exceed 100% when a standalone method retrieves more relevant material than fusion. It does not count matching document IDs, and it does not guarantee that every query retains the same quality. For example, retaining 96.68% of a reference recall of 0.12250 yields recall near 0.11844, not 96.68% of all relevant TREC documents.

Dataset construction and models also differ. Our SQuAD corpus includes unique title/paragraph pairs from official train and dev data [5]; each dev question has its annotated parent paragraph as the relevance label. Alternative answer-bearing paragraphs remain unjudged. Different models, preprocessing, document units and label rules can change quality even under a shared metric name. We have not isolated each factor in a matched ablation, so they are comparison limits rather than proven explanations for a particular numerical gap.

Timing adds another distinction. Every reported speedup divides latencies measured inside our implementation. The original-fusion reference includes the benefit of the indexed backend across separate runs, whereas the indexed-fusion reference measures extra routing savings with the same backend. Neither ratio measures acceleration relative to the published Blended RAG engine. We do not compare our retrieval-retention percentages with the paper's EM/F1 scores.

## 4. Method


### 4.1 Retrieval actions

The static reference executes Best Fields BM25, Dense MiniLM
(`sentence-transformers/all-MiniLM-L6-v2`), using sentence embeddings [7], and SPLADE
(`naver/splade-cocondenser-ensembledistil`) [6]. BM25 scores title and abstract
fields separately and takes their maximum. Each branch contributes 100
results to reciprocal rank fusion, which sums `1 / (60 + rank)` and returns
100 documents. The confirmation and paper-dataset runners request 101
candidates, remove an identical query/document ID and keep 100 before
features or fusion. Actual ID overlap is zero on the reported confirmations.

The cascade always executes BM25 first. A gate then chooses between returning
that ranking and running both neural branches before the same fusion.
Fallback retrieval reuses that query's BM25 ranking. Timed methods do not
share neural query rankings or precomputed query results.

```mermaid
flowchart LR
    Q[Query] --> B[Best Fields BM25]
    B --> G[Ridge gate on 11 features]
    G -->|Predicted gain at or below threshold| E[Return BM25 top 100]
    G -->|Predicted gain above threshold| N[Run Dense and SPLADE]
    N --> F[RRF with the existing BM25 ranking]
    F --> R[Return fused top 100]
```

### 4.2 Gate and calibration

The original 11 features describe query length, term specificity and the
BM25 score distribution. They include token and unique-token counts;
top-one and top-two scores, their gap, the top-ten mean and standard
deviation, and top-one score per query token; plus maximum, mean and standard
deviation of query-term IDF. Exact definitions and feature order remain in
experiment 12 and the exported router artifact.

A standardized Ridge regressor with alpha 10 predicts
`g(q) = max(Recall_static(q) - Recall_BM25(q), 0)`.
Inference clips negative predictions to zero. The gate exits when its
prediction is at or below the selected threshold. For target `t`, calibration
limits the sum of observed clipped losses on selected exits to
`(1 - t) * sum_q Recall_static(q)`. Favorable BM25 differences do not offset
losses during this calibration procedure.

Thresholds use five-fold out-of-fold predictions on all 2,590 NFCorpus train
queries, with seed 42; the exported model then fits all training queries.
Earlier nested comparisons use five outer folds and four inner folds.
A 69-feature coverage ablation remains development evidence; the original
11-feature model is frozen for confirmation. This empirical calibration
procedure provides no distribution-free guarantee on later collections.

### 4.3 Compatible BM25 execution

The original Python scorer repeatedly visits document term-frequency
structures. The revised scorer precomputes per-term field contributions
into postings and uses partitioning with a corpus-order tie rule to select
the top results. It preserves tokenization, epsilon-adjusted IDFs, field
normalization, duplicate query-token contributions and float64 accumulation
order. Both reference fusion and all cascades receive the change.

Local CPU comparisons verify identical full score vectors and top-100
rankings on 50 TREC queries and 2,590 NFCorpus train queries. The separate
Kaggle follow-ups verify every method's saved top-100 IDs, scores, routes and
recalls against the original runs. This lets us attribute changed runtime
to execution while retaining the same measured retrieval behavior.

## 5. Experimental protocol


### 5.1 Collections and evaluation scope

NFCorpus, SCIDOCS and TREC-COVID use BEIR distributions, retaining their full
corpora. BEIR supplies a heterogeneous retrieval benchmark; the split and
inspection status of each project run are specified below.
See BEIR [4].

**Table 2. Completed evaluation contexts. Full corpora are retained in every row.**

| Study | Queries evaluated | Documents | Complete timing passes | Primary in that study |
|:--|--:|--:|--:|:--|
| NFCorpus dev, inspected development | 324 | 3,633 | 3 | r92 |
| SCIDOCS official test, frozen confirmation | 1,000 | 25,657 | 3 | r92 |
| Original TREC-COVID | 50 | 171,332 | 3 | r90 |
| Original SQuAD paragraph retrieval | 10,570 | 20,958 | 1 | r90 |
| Indexed TREC-COVID follow-up | All 50 | 171,332 | 3 | r99 against 90% floor |
| Indexed SQuAD follow-up | 256 of 10,570 | 20,958 | 3 | r99 against 90% floor |

The SQuAD corpus contains all unique title/paragraph pairs from official
SQuAD 1.1 train and dev data. Dev questions query the corpus; the annotated
parent paragraph is relevant and alternative answer-bearing paragraphs are
unjudged. This is retrieval evaluation rather than answer EM/F1. The original
long timing job was shortened before quality scoring after one complete pass
and 4,649 additional query repetitions. The complete pass supplies its means;
additional records remain separate and do not establish three-pass stability.

The indexed SQuAD sample is the first 256 query IDs after sorting by
SHA256(`squad` + NUL + query ID), as declared before its outcomes. Its quality
and timing apply to that sample. Full-query original SQuAD quality must not
be substituted for indexed sample metrics. Full NQ/HotpotQA evaluation remains pending, and CoQA has no completed run; earlier qrel-informed NQ simulations
and historical SciFact runs do not strengthen this report's main claims.

### 5.2 Fixed decisions and inspection status

Weights and thresholds use NFCorpus train only. NFCorpus dev had already
been inspected; the r92 model/threshold and confirmation protocol were frozen
before SCIDOCS outcomes. NFCorpus test had been inspected earlier and is
excluded from this calibration and confirmation sequence.

The original paper-dataset extension declared r90 primary before its outcomes.
The indexed engineering follow-up declared the existing conservative r99
policy primary against a 90% floor, using already-inspected NFCorpus/SCIDOCS/
TREC evidence before indexed outcomes and current full-SQuAD outcome
inspection. Earlier studies retain their original primaries. All four
settings remain reported. The indexed TREC result is an engineering
follow-up on known quality, rather than a new independent generalization test.

### 5.3 Timings, metrics and comparison references

Kaggle measurements use a Tesla T4, one CPU thread, randomized query/method
ordering, warmups and GPU synchronization around each timed query. Timings
include query retrieval and routing, while excluding corpus encoding,
index preparation, model loading, dataset download and answer generation.
Three passes are complete except for the explicitly amended original SQuAD
run. Datasets and backend runs have separate sessions.

Recall retention is `mean Recall_method@100 / mean Recall_fusion@100`.
It measures relative retrieval quality and can exceed 100%; it does not
represent overlap of retrieved lists or a per-query guarantee. nDCG@10 and
MAP@10 report ranking quality. Query-paired bootstrap intervals use 1,000
resamples and repeat-averaged query times, without covering all hardware or
session variation. Count-matched random routing supplies an offline recall
reference with 200 assignments, not a timed system.

We report two speedup references. Original-fusion latency divided by indexed
method latency describes indexing and routing combined, across runs.
Indexed-fusion latency divided by indexed-cascade latency describes the
additional conditional-routing gain with the same scorer. SQuAD cross-run
comparisons use the exact same 256 questions. Percent time saved is
`1 - 1/speedup`; a 1.24x speedup corresponds to 19.04% less mean time here.

## 6. Results


### 6.1 Full speedup progression

![Full speedup progression with recall at every measured point](figures/speedup_progression.png)

**Figure 1.** Every cascade setting, ordered from conservative to more
permissive. Top panels compare with the original implementation; bottom
panels compare with equally indexed fusion. Percentages beside points are
Recall@100 retained. Colors distinguish a failed 90% point, a point above
90% whose interval crosses below it, and an interval lower bound above 90%.
These are measured configurations, not evidence of scaling to larger corpora.

On TREC, indexing alone changes fusion from 1493.23 to 91.28 ms, an observed
16.36x cross-run speedup. Indexing plus r99/r95/r92/r90 reaches
20.21x/26.83x/35.56x/36.96x relative to original fusion, retaining
96.68%/91.95%/84.53%/84.53% recall. The largest two points fail the requested
90% floor. The 26.83x secondary point clears 90% at its observed mean but
has a recall interval crossing below that floor. These larger ratios include
the backend change and must not all be attributed to routing.

On the same SQuAD sample, indexed fusion reaches 4.98x over original fusion.
The four cascade settings then reach 5.19x, 5.92x, 6.90x and 7.91x, retaining
100%, 100%, 100% and 99.22% recall. The comparisons use one original timing
pass versus three indexed passes. Complete tables, including the faster
standalone baselines, are in [SPEEDUP_SUMMARY.md](SPEEDUP_SUMMARY.md).

### 6.2 Indexed TREC-COVID

**Table 3. All 50 TREC queries and the full corpus; the reference is equally indexed fusion.**

| Method | Recall retained | nDCG@10 | MAP@10 | Mean ms | Speedup vs indexed fusion |
|:--|--:|--:|--:|--:|--:|
| Indexed fusion | 100.00% | 0.76323 | 0.01861 | 91.28 | 1.00x |
| BM25 | 71.38% | 0.54969 | 0.01145 | 8.11 | 11.25x |
| Dense | 69.81% | 0.47254 | 0.01054 | 30.41 | 3.00x |
| SPLADE | 104.69% | 0.72899 | 0.01662 | 52.77 | 1.73x |
| r99 (primary) | 96.68% | 0.72281 | 0.01769 | 73.90 | 1.24x |
| r95 (secondary) | 91.95% | 0.70561 | 0.01742 | 55.66 | 1.64x |
| r92 (secondary) | 84.53% | 0.65315 | 0.01499 | 41.99 | 2.17x |
| r90 (secondary) | 84.53% | 0.65115 | 0.01499 | 40.41 | 2.26x |

The preselected r99 primary yields 1.24x speedup, retaining 96.68% recall.
Its mean time saving is 19.04% with a query-bootstrap interval of 9.47% to
30.11%; its retention interval is 93.58% to 98.75%. Both lower bounds pass
the follow-up's positive-saving and 90%-retention criteria in this setting.
The primary exits on 11 queries, avoiding 33 invocations of each neural
branch across three passes. It still misses its nominal 99% calibration
label, as do all other nominal TREC targets.

The fastest secondary cascade point with observed retention above 90% is
r95: 1.64x speedup at 91.95% retained. Its recall interval is 86.88% to
95.76%, and its time-saving interval is 26.54% to 51.94%. The absolute maximum
cascade speedup is 2.26x at r90, retaining 84.53%; r92 also fails 90%. These
points remain visible without replacing the preselected primary.

SPLADE alone reaches 1.73x speedup, retaining 104.69% of fusion recall. It
has higher recall and nDCG with lower mean time than the primary. The primary
has higher MAP, 0.01769 versus 0.01662, a descriptive relative difference of
6.43% without a significance claim. Under a recall/time objective SPLADE
remains the stronger constant choice. This MAP difference cannot justify
changing the original objective retrospectively to establish superiority.

Primary absolute Recall@100 is 0.11844 versus fusion's 0.12250; many documents
are judged relevant per TREC topic. The 96.68% figure is a retention ratio.
Primary p95 is 99.41 ms versus fusion's 97.66 ms, so the measured mean gain
does not establish a p95 improvement.

### 6.3 Indexed SQuAD sample

**Table 4. The fixed 256-question sample, full corpus and three timing passes.**

| Method | Recall retained | nDCG@10 | MAP@10 | Mean ms | Speedup vs indexed fusion |
|:--|--:|--:|--:|--:|--:|
| Indexed fusion | 100.00% | 0.80931 | 0.77059 | 29.55 | 1.00x |
| BM25 | 95.70% | 0.77869 | 0.75038 | 1.03 | 28.59x |
| Dense | 95.70% | 0.68778 | 0.63040 | 10.82 | 2.73x |
| SPLADE | 98.83% | 0.85440 | 0.82831 | 16.96 | 1.74x |
| r99 (primary) | 100.00% | 0.81020 | 0.77082 | 28.31 | 1.04x |
| r95 (secondary) | 100.00% | 0.80277 | 0.76464 | 24.82 | 1.19x |
| r92 (secondary) | 100.00% | 0.80287 | 0.76757 | 21.31 | 1.39x |
| r90 (secondary) | 99.22% | 0.80092 | 0.76757 | 18.59 | 1.59x |

Additional cascade speedup grows from 1.04x at r99 to 1.19x, 1.39x and 1.59x
as the nominal setting becomes more permissive. Observed recall retention is
100% at the first three points and 99.22% at r90. The primary's mean saving
is 4.19% (interval 1.90% to 7.04%). Secondary r90 saves 37.08% (31.58% to
42.69%) and retains 99.22% (98.05% to 100%). The primary remains r99.

BM25 alone retains 95.70% at 1.03 ms, reaching 28.59x speedup over indexed
fusion; it is the fastest listed method meeting the user's 90% floor here.
SPLADE has the highest nDCG and MAP at 16.96 ms, while retaining 98.83%
recall. These baselines limit a claim of practical cascade superiority.
Zero sampled losses and empirical intervals of [1, 1] at r92/r95/r99 do not
guarantee zero loss on unmeasured questions. r90 p95 is 31.48 ms versus
fusion's 31.37 ms, so a tail-latency improvement is not shown.

### 6.4 Earlier calibration and original-backend findings

**Table 5. Original study primaries, kept separate from indexed study selection.**

| Original backend study | Primary setting | Recall retained | Fusion mean ms | Cascade mean ms | Mean time saved |
|:--|:--|--:|--:|--:|--:|
| NFCorpus dev | r92 | 94.74% | 23.61 | 18.18 | 23.00% |
| SCIDOCS test | r92 | 90.01% | 159.51 | 150.69 | 5.53% |
| TREC-COVID | r90 | 84.53% | 1493.23 | 1440.00 | 3.56% |
| SQuAD full queries, one pass | r90 | 99.38% | 148.26 | 139.19 | 6.12% |

All nominal targets pass on NFCorpus dev and full-query SQuAD. All fail on
SCIDOCS and TREC. SCIDOCS r92's retention interval, 88.27% to 91.61%, crosses
90%; the original TREC r90 point fails even at its observed 84.53% retention.
No recalibration follows these results. The original full-query SQuAD result
has one complete timing pass, despite its 10,570-query quality coverage.

Dense has higher recall and lower time than every cascade on NFCorpus dev
and SCIDOCS. It retains 99.77% of fusion recall at 6.62 ms on NFCorpus; on
SCIDOCS it retains 109.65% at 10.14 ms and leads all three quality metrics.
These outcomes remain part of the project. Per-policy original-backend
values, the development feature ablation and random-assignment references
remain in the detailed findings and combined machine-readable table.

### 6.5 Compatibility and component execution

All 400 TREC and 2,048 sampled SQuAD first-pass rankings match the original
runs exactly: document IDs and order, scores, routes and per-query recalls.
Maximum saved-ranking score difference is zero. Independent raw-label
scoring reproduces all three aggregate quality metrics for every method on
both indexed collections and on original full-query SQuAD/TREC.

Across TREC pipeline runs, BM25 alone changes from 1410.14 to 8.11 ms, an
observed 173.78x scorer ratio. Dense and SPLADE standalone means remain
close. This scorer result preserves BM25 quality; BM25 still retains only
71.38% of fusion recall. The larger scorer ratio is not an end-to-end cascade
speedup. Separate paired local CPU measurements show 134.85x scorer speedup
on TREC and 34.59x on NFCorpus train, with identical full score vectors.
Those measurements have different hardware and scope.

![Quality and latency of all indexed methods](figures/indexed_quality_latency.png)

**Figure 2.** Mean query time and recall retention for all indexed methods.
The complete reference corpus is common within each panel. Error bars are
95% query-bootstrap intervals. The SQuAD panel is a declared sample.

Relative to original fusion, indexed standalone BM25, Dense and SPLADE reach 184.02x, 49.10x and 28.29x on TREC, retaining 71.38%, 69.81% and 104.69% of fusion recall respectively. On the same SQuAD sample, their original-fusion ratios are 142.25x, 13.58x and 8.67x, retaining 95.70%, 95.70% and 98.83%. These are standalone comparisons; the BM25 scorer ratio, combined cascade ratios and additional routing gains remain distinct quantities.

## 7. Implementation and reproducibility

### 7.1 Modules and execution paths

The implementation keeps retrieval, policy decisions and scoring separate. `src/indexed_bm25.py` implements the compatible postings backend. `src/bm25_retriever.py`, `src/dense_retriever.py` and `src/splade_retriever.py` expose the branch implementations. The frozen weights and thresholds live in `results/nfcorpus_frozen_retention_routers.json`; the indexed protocol records model revisions and source fingerprints. Experiments 20 and 25 execute original and indexed benchmark pipelines respectively.

The gate consumes only query and initial BM25 features at inference time. It either returns those lexical results or runs Dense and SPLADE, reusing the already-computed BM25 ranking during fusion. A static reference executes all branches for every query. Evaluated methods do not reuse another method's neural query rankings, which would hide the actual cost of conditional execution. Corpus representations and indexes are prepared before the query-timing window.

### 7.2 Evidence and rebuilding the report

The evidence supports both artifact consistency and independent quality scoring. Experiment 22 imports result archives and checks frozen assets, timings, branch decisions and rankings. Experiment 30 compares original/indexed outputs on identical query IDs. Experiments 29 and 31 reconstruct SQuAD labels or read exact BEIR relevance labels to rescore saved rankings. These checks strengthen the reported retrieval results, while clock accuracy and broader hardware variability remain experimental limitations.

Experiment 32 rebuilds the complete speedup figures and CSV tables from compact reports without retrieval. The submission builders create this paper, the offline HTML/PDF showcase and the editable presentation from the same recorded evidence. Current navigation and reproduction commands appear in `docs/reproducibility/README.md`. Raw datasets, neural caches and bulk per-query records remain excluded from the source-sharing ZIP; the artifact catalog records the original local export hashes needed to identify matching evidence.

## 8. Discussion and limitations

First-stage cost controls how much a gate can save. On original TREC, BM25
alone consumes 1410.14 ms against fusion's 1493.23 ms. After compatible
indexing, that first stage takes 8.11 ms and the same gate can avoid a larger
fraction of the remaining query cost. The study changes reference and
proposed method together, rather than giving the execution improvement only
to the cascade. Original/indexed comparisons are separate runs, so their
ratios include run variability.

Training calibration does not ensure transfer. Identical rankings after
indexing preserve the original quality misses, while the indexed primary
choice uses already-inspected evidence. The work supports an engineering
follow-up, not a new independent guarantee. Standalone quality/time comparisons
also matter: fusion itself need not be the most efficient useful action.

The measurements cover one GPU type, small numbers of timing sessions and
prepared indexes. They exclude production concurrency, energy, memory/update
cost and generated answers. We have not measured modern adaptive retrieval
baselines, full NQ/HotpotQA, CoQA or the original paper's ELSER implementation.
Method novelty, publication competitiveness and overall superiority remain
unestablished. A future method study should define its quality/cost objective
on training and validation data before evaluating new independent queries.

### 8.1 Deferred evaluation

Full-corpus Natural Questions, HotpotQA and CoQA remain outside this release. Earlier NQ results use a qrel-informed candidate pool and simulated routing, so they cannot establish full-corpus savings of the present cascade. Generated-answer EM/F1, modern adaptive-retrieval baselines, concurrency, energy, and index memory/update costs also remain unmeasured. The indexed SQuAD study covers its fixed 256-question sample, not all 10,570 questions.

Further testing should preserve this release and declare the new protocol before inspecting outcomes. Additional datasets can assess transfer, but would not by themselves establish a new algorithm or superiority over stronger retrieval strategies. A later study should retain the strongest standalone methods and use a new validation boundary for any revised gate.

## 9. Conclusion


Compatible indexing makes conditional neural skipping deliver larger mean
savings against an equally optimized fusion reference. TREC cascade speedup
ranges from 1.24x to 2.26x while recall retention declines from 96.68% to
84.53%. Its fastest secondary point above 90% observed retention reaches
1.64x at 91.95%, with uncertainty crossing the floor; the conservative primary
retains 96.68% at 1.24x. Against our original fusion implementation, indexing
and routing together reach 26.83x at that 91.95% secondary point and 36.96x
at the 84.53% point. On sampled SQuAD, the cascade reaches 1.59x over indexed
fusion at 99.22% retention. Strong constant baselines remain preferable under
the recall/time objective on these datasets. The contribution is a measured
execution optimization and empirical analysis of its quality/time limits.

## References

[1] Kunal Sawarkar, Abhilasha Mangal and Shivam Raj Solanki. 2024. *Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers*. arXiv:2404.07220v2. [Paper](https://arxiv.org/html/2404.07220v2).

[2] Negar Arabzadeh, Xinyi Yan and Charles L. A. Clarke. 2021. *Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection*. arXiv:2109.10739. [Paper](https://arxiv.org/abs/2109.10739).

[3] Xing Han Lu. 2024. *BM25S: Orders of magnitude faster lexical search via eager sparse scoring*. arXiv:2407.03618. [Paper](https://arxiv.org/abs/2407.03618).

[4] Nandan Thakur, Nils Reimers, Andreas Ruckle, Abhishek Srivastava and Iryna Gurevych. 2021. *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS Datasets and Benchmarks. [Paper](https://arxiv.org/abs/2104.08663).

[5] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev and Percy Liang. 2016. *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. EMNLP. [Paper](https://arxiv.org/abs/1606.05250).

[6] Thibault Formal, Benjamin Piwowarski and Stephane Clinchant. 2021. *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*. SIGIR. [Paper](https://arxiv.org/abs/2107.05720).

[7] Nils Reimers and Iryna Gurevych. 2019. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP. [Paper](https://arxiv.org/abs/1908.10084).

## Appendix A. Evidence entry points

All implementation paths here are relative to `cascaded-blended-rag/`.

| Evidence | Location |
|:--|:--|
| All-query indexed TREC-COVID | `results/indexed_trec-covid_q0/` |
| Fixed-sample indexed SQuAD | `results/indexed_squad_q256/` |
| Original full-query SQuAD | `results/paper_squad_one_pass/` |
| Combined metrics | `results/final_quality_speed.csv` |
| Frozen policies and protocol | `results/nfcorpus_frozen_retention_routers.json`, `results/indexed_cascade_protocol.json` |

Frozen weights SHA256: `3efd7382cd91cfc43bd7ae5d4eaf27571ae92657dec97e3eaf85fbb36131a5d1`.
Indexed runner SHA256: `ddb7b5656ed20e0ab20cb2199f544137e3679c161b99cb935c5f14919e44a104`.

The [reproduction guide](reproducibility/README.md), [complete speedup tables](SPEEDUP_SUMMARY.md) and [release review](review/PAPER_RELEASE_REVIEW.md) document commands, claim support and the answered five-part self-review. Original exports are identified in the [artifact catalog](reproducibility/artifact_catalog.json).
