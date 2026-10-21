# NFCorpus four-target benchmark: measured findings

Source: workspace-root `artifacts/downloads/nfcorpus_retention_benchmark_outputs.zip`. Extracted
artifacts are in `cascaded-blended-rag/results/retention_tradeoff_dev/`.
This is the already inspected official dev split, not an untouched test.

The run covers 324 queries, the full 3,633-document corpus, a Tesla T4, and
three randomized warmed repetitions per method. All 3,888 first-pass records,
11,664 query timings, branch counts, frozen model bytes, and source fingerprint
were validated locally. No ranking changed on later repetitions.

## Original compact gate

| Training target | Observed recall retained | Mean latency | Mean latency reduction | Early exits |
|:--|--:|--:|--:|--:|
| 90% | 92.33% | 17.07 ms | 27.70% | 39.20% |
| 92% | 94.74% | 18.18 ms | 23.00% | 32.10% |
| 95% | 96.23% | 19.67 ms | 16.70% | 23.77% |
| 99% | 99.27% | 23.14 ms | 2.00% | 4.01% |

Static RRF: Recall@100 0.30337; mean latency 23.61 ms; p95 33.97 ms.
The r92 gate's recall-retention bootstrap interval is 92.66-96.92%, and its
mean-latency-reduction interval is 19.33-26.71%. These paired query-bootstrap
intervals do not capture every source of session/hardware variability. The
r90 recall interval is 89.47-95.04%, so it is not a high-confidence 90% floor.
Tail improvements are small: r92 p95 is 32.79 ms, about 3.5% below static.

## Baselines change the practical conclusion

| Method | Recall@100 | Retained recall | nDCG@10 | MAP@10 | Mean latency |
|:--|--:|--:|--:|--:|--:|
| Static RRF | 0.30337 | 100% | 0.31872 | 0.10129 | 23.61 ms |
| BM25 | 0.20472 | 67.48% | 0.25895 | 0.07594 | 6.44 ms |
| Dense | 0.30266 | 99.77% | 0.30215 | 0.09448 | 6.62 ms |
| SPLADE | 0.29654 | 97.75% | 0.33164 | 0.10852 | 10.26 ms |

Dense is faster and has higher Recall@100 than every cascade point on this
collection. That does not mean Dense wins every metric: SPLADE has the highest
nDCG@10 and MAP@10. SPLADE also has better recall, nDCG, MAP, and latency than
both families' 90/92/95% cascade points. The cascade is not the best practical
quality/latency choice demonstrated by this run.

## What the gate does establish here

- Conditional skipping is real: at r90 it skips 381 Dense and 381 SPLADE calls
  across three timed passes. Mean latency falls against static RRF.
- Learned routing preserves more recall than count-matched random skipping:
  r92 retains 94.74%, versus 89.54% averaged over 200 random assignments with
  the same 104 exits per pass. Random latency was not measured.
- Relaxed targets expose a quality/cost curve that the 99% constraint hid.
- The 69-feature ablation offers little practical improvement for its overhead.
  Its r99 point retains 98.75%, missing its nominal target, with a timing
  interval that includes no mean benefit.

These support a conditional-routing evaluation against full fusion. They do
not support superiority over strong single retrievers, universal retention
guarantees, novel-method priority, or generated-answer-quality claims.

## Historical finalization plan (now completed)

Freeze the original gate at all four existing thresholds. Use r92 as the
development-selected primary candidate and report the complete curve. Run one
predeclared full-corpus SCIDOCS confirmation with the strong baselines, without
training or threshold changes. The ready upload/run instructions are in
`notebooks/kaggle_scidocs_confirmation_v1/START_HERE.md`. Then consolidate both
datasets into the final paper/report, including any unfavorable outcomes.

## Confirmation completed

The downloaded `artifacts/downloads/scidocs_confirmation_outputs.zip` has been validated. All four
original-gate targets fail on SCIDOCS; Dense is faster with higher quality.
See [FINAL_RESULTS.md](../FINAL_RESULTS.md) for the complete outcome and final steps.
