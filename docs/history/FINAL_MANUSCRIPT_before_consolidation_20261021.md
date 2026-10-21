> **Historical document.** This snapshot predates the consolidated submission. Use [the current manuscript](../FINAL_MANUSCRIPT.md) for conclusions.

> **Scope update, 21 October 2026:** the main text reports the original
> NFCorpus/SCIDOCS sequence. Addenda record TREC-COVID, full-query SQuAD with
> shortened timing, the verified 256-query indexed SQuAD follow-up, and the
> full-query indexed TREC-COVID follow-up.
> Full NQ/HotpotQA are pending. Read `docs/FINAL_RESULTS.md` for the current state.

# Recall-Retention Budgets for Conditional Hybrid Retrieval: Development Gains and Transfer Limits

Project manuscript, 20 October 2026. All numerical claims below refer to the
saved local experiments. Literature positioning and venue suitability require
separate review; this draft makes no claim of method priority.

## Abstract

Conditional hybrid retrieval seeks to reduce query latency by executing
additional retrievers only when their expected benefit justifies the cost.
We evaluate a BM25-first cascade that predicts the recall benefit of escalating
to a Dense–SPLADE–BM25 fusion. A compact Ridge gate is trained on NFCorpus train
and calibrated at four recall-retention targets: 90%, 92%, 95% and 99%.
On NFCorpus dev, the primary 92% policy retains 94.74% of full-fusion Recall@100
and reduces mean retrieval latency by 23.00%. In a frozen SCIDOCS confirmation,
the same policy retains 90.01% and reduces mean latency by 5.53%; all four
policies miss their nominal targets. BM25 scoring dominates the SCIDOCS
runtime, limiting the benefit of skipping neural retrieval. Dense alone
exceeds full fusion on Recall@100, nDCG@10 and MAP@10 on SCIDOCS while running
15.73 times faster. These measurements establish the tested cascade's
trade-offs and transfer limits, and show why standalone retrievers and
first-stage cost must be included in conditional-retrieval evaluations.

## 1. Research question and contribution

Combining retrievers can increase query cost. A cascade attempts to avoid
unnecessary work by deciding whether a cheaper ranking is sufficient. Its
practical value depends on both the quality of that decision and how much
execution time remains after the decision can be made.

We ask whether a small BM25-based gate can retain a specified fraction of
full-fusion recall while reducing measured retrieval latency, and whether
the chosen thresholds transfer to another collection without calibration.
The contribution is a reproducible implementation and empirical study:
four explicit quality budgets, real conditional execution, repeated timings,
standalone baselines and a frozen cross-dataset confirmation. The study
does not establish a new routing principle or superior retrieval method.

## 2. Method

The static reference retrieves 100 documents from each of Best Fields BM25,
Dense MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) and SPLADE
(`naver/splade-cocondenser-ensembledistil`). Reciprocal rank fusion sums
`1 / (60 + rank)` across the three rankings and keeps 100 documents.

The conditional method first runs BM25 and computes the original 11 query
and ranking features defined in `experiments/12_calibrate_nfcorpus_train_check_dev.py`.
Feature order, standardization constants and fitted coefficients are stored
in `results/nfcorpus_frozen_retention_routers.json`. A standardized Ridge
regressor with alpha 10 predicts the nonnegative benefit of escalation.
If its clipped prediction is at or below the selected threshold, the method
returns BM25. Otherwise it executes both neural branches and returns the
same fusion as the static reference. Neural rankings are not precomputed
per query or shared between timed methods.

For query q, the training target is
`g(q) = max(Recall_static(q) - Recall_BM25(q), 0)`.
For desired retention t, threshold calibration limits the sum of these clipped
losses over selected exits to `(1-t) * sum_q Recall_static(q)`.
This is conservative with respect to favorable BM25 differences on the
calibration examples. It provides no distribution-free guarantee on new
queries or collections.

Final thresholds use five-fold out-of-fold predictions on NFCorpus train
(seed 42); the exported model is then fitted on all 2,590 training queries.
Nested train-only comparisons use five outer folds and four inner folds.
A 69-feature coverage ablation is evaluated on development data, but the
original 11-feature family is frozen for confirmation. Exact fold seeds,
feature definitions and threshold tie handling are in experiments 12–16
and `src/router_diagnostics.py`.

## 3. Evaluation protocol

NFCorpus dev contains 324 queries and the full 3,633-document corpus. It had
already been inspected and is treated as development evidence. SCIDOCS
confirmation contains 1,000 official test queries and all 25,657 documents.
No relevance-dependent corpus slicing is used. NFCorpus test had been
inspected in an earlier project experiment and is excluded from this final
calibration and confirmation procedure.

The original gate, all four thresholds and the primary 92% candidate were
frozen before SCIDOCS scoring. The saved confirmation protocol records their
hashes, dataset dimensions, baselines and outcome rules. The runner reads
SCIDOCS relevance grades for scoring after completing all timed rankings;
no refitting or threshold selection uses those grades. Each SCIDOCS branch
requests 101 candidates, excludes an identical query/document ID and keeps
100 before features or fusion. Actual query/corpus ID overlap is zero.

Each collection is measured on a Tesla T4 using three warmed repetitions,
randomized query and method ordering, GPU synchronization around each timed
query, and one CPU thread. Corpus representations and indexes can be reused;
query results cannot. Timings cover query-time retrieval and routing, excluding
model loading, corpus encoding/index construction, dataset download and
answer generation. Dataset sessions are separate, so cross-dataset absolute
latencies do not isolate a causal effect of corpus size.

Metrics are mean Recall@100, nDCG@10 and MAP@10. Recall retention is
`mean Recall_method / mean Recall_static`; it is neither retrieved-list
overlap nor a per-query guarantee. Values above 100% are possible. Speedup
is static mean latency divided by method mean latency. Query-paired bootstrap
intervals use 1,000 resamples and repeat-averaged query times; they do not
capture all hardware or session variability. Count-matched random routing
is an offline recall reference with 200 assignments, not a timed system.

## 4. Results

**Table 1. Original frozen gate. Targets are training calibration settings, not observed guarantees.**

| Target | NFCorpus retention | NFCorpus mean (ms) | SCIDOCS retention | SCIDOCS mean (ms) | SCIDOCS exits |
|:--|--:|--:|--:|--:|--:|
| 90% | 92.33% | 17.07 | 88.67% | 149.43 | 39.90% |
| 92% (primary) | 94.74% | 18.18 | 90.01% | 150.69 | 35.00% |
| 95% | 96.23% | 19.67 | 92.10% | 152.87 | 27.10% |
| 99% | 99.27% | 23.14 | 97.44% | 157.22 | 11.00% |
| Static RRF | 100.00% | 23.61 | 100.00% | 159.51 | 0.00% |

Relaxing the budget produces a measurable development trade-off. The primary
policy reduces NFCorpus mean time by 23.00%, with retention interval
92.66–96.92%. On SCIDOCS, all four policies miss their nominal targets.
The primary policy's retention interval is 88.27–91.61%, and its mean time
reduction is 5.53% (interval 5.00–6.08%). Its p95 falls only from 231.74 ms
to 229.98 ms. The observed 90.01% point does not establish a dependable 90%
floor. No policy is relabeled or recalibrated after observing confirmation.

**Table 2. Standalone baselines and the primary cascade on SCIDOCS. Higher quality and lower time are preferable.**

| Method | Recall@100 | nDCG@10 | MAP@10 | Mean (ms) |
|:--|--:|--:|--:|--:|
| Static RRF | 0.46523 | 0.18591 | 0.10976 | 159.51 |
| BM25 | 0.32378 | 0.13534 | 0.07626 | 132.51 |
| Dense | 0.51012 | 0.21640 | 0.12937 | 10.14 |
| SPLADE | 0.37342 | 0.15856 | 0.09245 | 16.39 |
| Cascade, nominal 92% | 0.41873 | 0.17082 | 0.09932 | 150.69 |

Dense exceeds static fusion and every cascade on all three SCIDOCS quality
metrics while running 15.73 times faster than fusion. On NFCorpus it also
beats every cascade in recall and mean latency: Recall@100 is 0.30266
(99.77% retention) at 6.62 ms. NFCorpus static fusion has recall 0.30337 at
23.61 ms. SPLADE gives the highest NFCorpus nDCG@10 (0.33164) and MAP@10
(0.10852), versus fusion's 0.31872 and 0.10129, at 10.26 ms. The cascade's
savings against fusion do not establish a preferred operating point against
these standalone alternatives.

At the primary policy, learned routing retains more recall than the mean
count-matched random assignment: 94.74% versus 89.54% on NFCorpus, and 90.01%
versus 89.42% on SCIDOCS. The smaller transfer gap is descriptive; no formal
test of that gap is reported. The development coverage-feature ablation
offers little practical improvement and misses its 99% target (98.75%
observed retention). It is not selected for confirmation.

## 5. Discussion and limitations

Skipping branches does not imply proportional wall-clock savings. The primary
SCIDOCS gate skips 35% of Dense and SPLADE executions: 1,050 calls per branch
over three passes. It still runs BM25 for every query. BM25 alone takes
132.51 ms against fusion's 159.51 ms, leaving relatively little time to save.
This finding applies to the tested full-corpus BM25 scoring implementation.
An optimized inverted index could change both the baseline and the cascade;
its performance is unmeasured here.

The calibration budget controls empirical training loss, not target-domain
risk. Lower retention on SCIDOCS shows that a train-calibrated threshold does
not automatically transfer. This study identifies the failure but does not
isolate whether feature shift, relevance structure or another dataset change
causes it. Further changes tested on SCIDOCS would use inspected evidence.

The evaluation covers two collections in this final sequence and one
hardware type. It has no production-load measurement, energy measurement,
modern adaptive-retrieval baseline or generated-answer evaluation. Earlier
SciFact experiments and qrel-informed NQ simulations are historical evidence;
they do not strengthen the current timing or generalization claims. The
project supports an empirical contribution within the measured setting.
Novelty and publication competitiveness remain unestablished.

## 6. Conclusion

The tested BM25-first cascade reduces measured time against full fusion at
relaxed recall budgets, but its calibrated targets fail in frozen transfer
and its operating points are inferior to standalone Dense in recall and
latency on both measured collections. On SCIDOCS, Dense also leads all listed
ranking metrics. A conditional-retrieval study therefore needs explicit
first-stage cost accounting and standalone retriever comparisons alongside
branch-skipping rates. Any future system revision should first address
first-stage cost and fallback choice, then receive a new independent
evaluation. Those revisions are outside this completed experiment sequence.

## Reproducibility and claim review

Machine-readable results are in `results/final_quality_speed.csv`; original
reports and per-query records are in `results/retention_tradeoff_dev/` and
`results/scidocs_confirmation/`, relative to `cascaded-blended-rag/`.
`experiments/19_finalize_confirmation.py` validates the downloaded confirmation
and regenerates the combined table without running retrieval. Exact asset
hashes and validation scope are saved in `local_verification.json`.
The section outline, claim-evidence map and answered five-part self-review
are in `docs/review/FINAL_WRITING_AUDIT.md`.


## Addendum: original-paper dataset extension, TREC-COVID

After the NFCorpus/SCIDOCS sequence, we evaluated the unchanged gate on all
50 official BEIR TREC-COVID test queries and 171,332 documents. The extension
preselected the nominal 90% policy as primary and retained all four policies.
This change of primary was declared before the extension outcomes and does
not alter the earlier SCIDOCS primary or its failed target. Three randomized
warmed repetitions ran on a Tesla T4 in one timing session.

**Table 3. TREC-COVID frozen policies. Mean query time excludes corpus indexing.**

| Policy | Recall retained | Mean time (ms) | Mean time reduction |
|:--|--:|--:|--:|
| Static fusion | 100.00% | 1493.23 | Reference |
| Nominal 90% primary | 84.53% | 1440.00 | 3.56% |
| Nominal 92% | 84.53% | 1443.98 | 3.30% |
| Nominal 95% | 91.95% | 1456.90 | 2.43% |
| Nominal 99% | 96.68% | 1475.73 | 1.17% |
| SPLADE | 104.69% | 52.98 | 96.45% |

All four policies miss their nominal calibration targets. The primary policy
also misses the relaxed 90% floor. Although the secondary 95% policy retains
91.95% at its point estimate, its retention interval is 86.88?95.76% and its
mean latency saving is only 2.43%. The nominal 99% policy's retention interval
is 93.58?98.75%, but its mean time saving is only 1.17%.

Standalone SPLADE has higher observed recall than fusion and all cascades at
28.19x fusion speed. Fusion has higher nDCG (0.76323 versus 0.72899) and MAP
(0.01861 versus 0.01662). The 95% and 99% cascades also exceed SPLADE in MAP,
so recall/latency dominance must not be generalized to every quality metric.
SPLADE's recall-retention interval, 97.75?112.87%, crosses 100%.

BM25 alone takes 1410.14 ms, compared with fusion's 1493.23 ms. This supports
the implementation-specific first-stage bottleneck interpretation, despite
62% primary-policy early exits. The experiment does not establish a major
acceleration at a 90% recall floor. It is an evaluation of our adaptation on
a paper-relevant dataset, not a timing comparison with the original ELSER
system. The subsequent SQuAD evaluations are recorded in the addenda below;
NQ and HotpotQA remain pending subject to full-corpus resources. Evidence is in
`docs/findings/TREC_COVID_FINDINGS.md`.


## Addendum: SQuAD paragraph retrieval, shortened timing protocol

We evaluated all 10,570 SQuAD 1.1 dev questions against 20,958 unique
train+dev title/paragraph documents. Only the annotated parent paragraph is
relevant. The original three-pass timing job was shortened for runtime after
one complete pass and 4,649 additional query repetitions, before quality
scoring. The complete pass supplies headline results; later records are
preserved separately. The original protocol and execution amendment are
retained. These timings do not establish three-pass stability.

**Table 4. Full-query SQuAD results with the original slow BM25 backend.**

| Method | Recall retained | nDCG@10 | MAP@10 | Mean time (ms) |
|:--|--:|--:|--:|--:|
| Static fusion | 100.00% | 0.82897 | 0.79284 | 148.26 |
| BM25 | 97.11% | 0.78831 | 0.75540 | 119.45 |
| Dense | 96.12% | 0.68247 | 0.63339 | 10.72 |
| SPLADE | 99.35% | 0.85367 | 0.82425 | 17.24 |
| Cascade nominal 90% | 99.38% | 0.81896 | 0.78335 | 139.19 |
| Cascade nominal 92% | 99.50% | 0.82223 | 0.78655 | 140.87 |
| Cascade nominal 95% | 99.70% | 0.82539 | 0.78953 | 143.38 |
| Cascade nominal 99% | 99.95% | 0.82821 | 0.79214 | 146.92 |

Every frozen nominal target is met here. Primary r90 saves 6.12% mean time
while retaining 99.38% of fusion recall. SPLADE is much faster with slightly
lower recall and higher nDCG/MAP than this cascade; it also improves nDCG/MAP
over static fusion. BM25 alone already satisfies a 90% recall floor. These
results do not establish a practically superior cascade.

Local raw SQuAD dev data reconstruct the exact evaluation qrels fingerprint.
Independent parent-label scoring reproduces all three reported quality
metrics for every method. This verifies retrieval evaluation, not answer
EM/F1. The full-query result is distinct from the 256-query indexed
execution follow-up. Detailed evidence: `docs/findings/SQUAD_FINDINGS.md`.

## Addendum: indexed SQuAD execution with frozen policies

We replaced repeated whole-corpus BM25 token scoring with precomputed term
postings while retaining its formula, features and ranking order. Both static
fusion and every cascade use this optimization. We measured 256 questions
selected by a declared SHA256 rule from all 10,570 SQuAD dev questions, against
the complete 20,958-document corpus. Three warmed randomized repetitions ran
in one T4 session with one CPU thread. Corpus preparation took 9.43 seconds
and is excluded from query timings. This is an engineering follow-up on
inspected dataset families; it supplies no new independent transfer claim.

**Table 5. Indexed execution on the fixed SQuAD sample.** The nominal 99% gate
was preselected as primary against a 90% required floor; all other gates are
secondary. Retention is relative to fusion Recall@100 of 1.0 on this sample.

| Method | Recall retained | nDCG@10 | MAP@10 | Mean ms | Mean time saved |
|:--|--:|--:|--:|--:|--:|
| Static fusion | 100.00% | 0.80931 | 0.77059 | 29.55 | Reference |
| BM25 | 95.70% | 0.77869 | 0.75038 | 1.03 | 96.50% |
| Dense | 95.70% | 0.68778 | 0.63040 | 10.82 | 63.37% |
| SPLADE | 98.83% | 0.85440 | 0.82831 | 16.96 | 42.61% |
| Nominal 90%, secondary | 99.22% | 0.80092 | 0.76757 | 18.59 | 37.08% |
| Nominal 92%, secondary | 100.00% | 0.80287 | 0.76757 | 21.31 | 27.89% |
| Nominal 95%, secondary | 100.00% | 0.80277 | 0.76464 | 24.82 | 15.99% |
| Nominal 99%, primary | 100.00% | 0.81020 | 0.77082 | 28.31 | 4.19% |

The primary retains every sampled parent hit and saves 4.19% mean time, with
a query-bootstrap 95% interval of 1.90% to 7.04%. Secondary r90 retains
99.22% (98.05% to 100.00%) and saves 37.08% (31.58% to 42.69%); r92 saves
27.89% with zero observed sample recall loss. These results demonstrate
conditional mean time savings after first-stage cost is reduced. They do
not establish a p95 improvement: r90 p95 is 31.48 ms against fusion's 31.37 ms.
Zero sampled losses do not imply a population retention guarantee.

All 2,048 first-pass rankings and scores, routes and per-query recalls exactly
match the original run on the same questions. Independent reconstruction of
SQuAD parent labels reproduces all three quality metrics. The old and indexed
fusion means on these same questions are 147.01 and 29.55 ms, respectively;
these cross-run observations compare one original pass with three indexed
passes and include session variation. The within-indexed cascade comparison
above avoids giving the optimization exclusively to the proposed method.

BM25 is still the fastest listed method satisfying a 90% recall floor and
retains over 90% of fusion nDCG and MAP. SPLADE leads ranking metrics and is
faster than every cascade, with slightly lower recall. Thus the study supports
score-preserving execution improvement and a bounded trade-off against fusion,
while superiority over the relevant constant baselines remains unestablished.
The sample's perfect fusion recall is not an indexing quality improvement;
neither its quality nor its time-saving percentage is a full-query indexed
result. This adaptation does not establish an acceleration of the original
paper's Elasticsearch/ELSER system. Evidence and claim review are in
`docs/findings/SQUAD_INDEXED_FINDINGS.md` and `docs/review/FINAL_WRITING_AUDIT.md`.

## Addendum: indexed TREC-COVID on the complete evaluation set

We applied the same frozen indexed execution to all 50 official BEIR
TREC-COVID queries and the full 171,332-document corpus. All eight methods
ran for three warmed randomized timing passes in one T4 session with one
CPU thread. Preparation reused existing neural corpus indexes and took
62.91 seconds, excluded from query timings. The preselected r99 policy is
assessed against a 90% recall-retention floor. Its selection used inspected
data before the indexed outcomes; this engineering follow-up supplies no
new independent quality-generalization claim.

**Table 6. Indexed TREC-COVID, all queries and full corpus.** Static fusion,
standalone BM25 and every cascade use the same optimized BM25 scorer.

| Method | Recall retained | nDCG@10 | MAP@10 | Mean ms | Mean time saved |
|:--|--:|--:|--:|--:|--:|
| Static fusion | 100.00% | 0.76323 | 0.01861 | 91.28 | Reference |
| BM25 | 71.38% | 0.54969 | 0.01145 | 8.11 | 91.11% |
| Dense | 69.81% | 0.47254 | 0.01054 | 30.41 | 66.68% |
| SPLADE | 104.69% | 0.72899 | 0.01662 | 52.77 | 42.19% |
| Nominal 90%, secondary | 84.53% | 0.65115 | 0.01499 | 40.41 | 55.74% |
| Nominal 92%, secondary | 84.53% | 0.65315 | 0.01499 | 41.99 | 54.00% |
| Nominal 95%, secondary | 91.95% | 0.70561 | 0.01742 | 55.66 | 39.02% |
| Nominal 99%, primary | 96.68% | 0.72281 | 0.01769 | 73.90 | 19.04% |

The primary saves 19.04% mean retrieval time with a query-bootstrap 95%
interval of 9.47% to 30.11%, retaining 96.68% of fusion recall with an interval
of 93.58% to 98.75%. This supports a positive mean saving above the requested
90% recall floor in the measured setting. The gate exits on 11 of 50 queries,
avoiding 33 invocations of each neural branch over three passes. Its nominal
99% calibration target still fails, as do all other nominal TREC targets.
The secondary r95 point saves 39.02% at 91.95% retention, but its interval
crosses below 90%. Primary p95 is 99.41 ms against fusion's 97.66 ms; this
study does not establish a p95 reduction or a universal retention guarantee.

Every method's rankings, scores, routes and per-query recall exactly match
the original TREC run on all 50 queries. Independent scoring from the exact
raw relevance labels reproduces Recall@100, nDCG@10 and MAP@10 for all eight
methods. Original and indexed fusion means are 1493.23 and 91.28 ms across
separate three-pass runs. BM25 falls from 1410.14 to 8.11 ms while neural
standalone times remain close. These observations support the execution-cost
diagnosis. The additional conditional gain must be reported against indexed
fusion, rather than attributing the entire backend improvement to routing.

SPLADE alone is faster than the primary and has higher recall and nDCG.
The primary has higher MAP@10, 0.01769 versus 0.01662; the descriptive 6.43%
relative MAP difference has no significance claim. Under the original
recall/time objective, SPLADE remains the stronger constant choice. The MAP
trade-off cannot justify changing the objective retrospectively to establish
superiority. The experiment supports a score-preserving execution improvement
and a conditional quality/time trade-off against fusion. Novelty, superiority
to modern adaptive baselines and acceleration of the original ELSER system
remain unestablished. Evidence: `docs/findings/TREC_COVID_INDEXED_FINDINGS.md`.

## Conclusion across the completed extension

Reducing first-stage implementation cost makes neural skipping yield larger
mean savings against equally optimized fusion. The frozen conservative gate
retains 96.68% recall with 19.04% mean savings on full-query TREC-COVID, and
100% observed recall with 4.19% mean savings on the declared SQuAD sample.
More permissive secondary settings offer larger savings with dataset-dependent
quality loss. Strong constant baselines remain decisive: SPLADE offers a
better recall/time point on TREC, and BM25 already exceeds a 90% floor on
SQuAD at much lower latency. These findings form a reproducible engineering
evaluation with calibration and baseline limits. No additional experiment is
required to report this scope. Full NQ/HotpotQA and any stronger method claim
remain outside the completed evidence.
