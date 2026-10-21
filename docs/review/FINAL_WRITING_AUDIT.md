# Final manuscript review

## Section outline and paragraph roles

1. Abstract: question, implemented gate, development result, frozen failure,
   strong baseline and bounded takeaway.
2. Research question: motivate conditional cost; define the empirical contribution.
3. Method: reference retrieval; real branch decision; calibrated loss;
   train-only model and threshold fitting.
4. Protocol: data scope; frozen confirmation; timing scope; metric meanings.
5. Results: complete target curve; transfer uncertainty; standalone comparison;
   random-routing reference and coverage ablation.
6. Discussion: measured first-stage bottleneck; unresolved transfer mechanism;
   evaluation limits. Conclusion: findings and a bounded future direction.

Reverse-outline check: each paragraph has one of these roles. Historical NQ
simulation numbers are excluded from the principal result tables. Claims of
superior retrieval and universal quality preservation were removed.

## Claim-evidence map

| Claim | Evidence | Boundaries |
|:--|:--|:--|
| Primary gate saves 23.00% mean time on NFCorpus dev | `results/retention_tradeoff_dev/nfcorpus_retention_benchmark.json`, `ridge11_r92` | Inspected development split, this implementation and run |
| All four targets fail on SCIDOCS | `results/scidocs_confirmation/scidocs_confirmation_benchmark.json`, four `observed_target_met` values | Frozen confirmation; no post-result recalibration |
| Primary transfer point retains 90.01% and saves 5.53% mean time | Same report, primary metrics and bootstrap intervals | Lower retention interval crosses 90%; no quality guarantee |
| Dense has higher SCIDOCS quality at 15.73x fusion speed | Same report, standalone Dense and static rows | Listed models/metrics; no universal ranking claim |
| First-stage cost limits savings | BM25 132.51 ms versus fusion 159.51 ms; branch counts | Descriptive timing, no optimized BM25 counterfactual |
| Frozen assets match the prepared files | `results/scidocs_confirmation/local_verification.json`; script 19 | Artifact consistency, not independent raw-qrel or clock verification |

Paths beginning `results/` are relative to `cascaded-blended-rag/`.

## Answered five-dimensional self-review

1. **Contribution — bounded.** The study supplies a reproducible implementation
   and measured findings about budgets, transfer and runtime bottlenecks.
   A novel algorithm or competitive new retriever is not established. The
   manuscript states this limit in the abstract, introduction and discussion.
2. **Writing clarity — revised.** Retention, loss, threshold, execution path,
   time scope and split status are defined. Saved feature order and source
   locations permit implementation inspection. The paper is a project
   manuscript; literature review and venue formatting are separate work.
3. **Experimental strength — mixed.** Real execution and three timing passes
   support savings against fusion. Strong standalone baselines beat the
   cascade, and all transfer targets fail. Those outcomes are reported rather
   than presented as competitive improvements.
4. **Evaluation completeness — limited.** Four targets, two feature families
   on development, random-exit recall references and four baselines are
   available. Adaptive baselines, more hardware, energy and answer quality are
   absent. No claim requiring them remains in the manuscript.
5. **Method soundness — failure exposed.** Train-only fitting and frozen
   confirmation reduce selection bias in the final sequence. The gate does
   not give a target-domain guarantee and its net practical value is weak
   against the measured alternatives. Further tuning on inspected SCIDOCS
   would require a new independent evaluation for a new generalization claim.

## Completion decision

The experiments can be finalized as an honest empirical project. The evidence
does not justify a stronger success or publication claim. No additional
experiment is necessary to report the completed study and its limitations.


## TREC-COVID addendum review

- Contribution: no major improvement claimed; the primary floor fails and
  secondary gains are small. The earlier empirical scope remains appropriate.
- Clarity: the new primary is distinguished from SCIDOCS and all nominal
  targets remain visible. Point retention and uncertainty are separate.
- Experimental strength: one full-corpus T4 experiment with three passes;
  only 50 topics, so the quality intervals are wide.
- Completeness: all frozen targets and standalone baselines are retained;
  subsequent SQuAD results are reviewed below. Full NQ/HotpotQA remain pending.
- Soundness: no threshold change; SPLADE recall/time advantage is not extended
  to MAP/nDCG or presented as an exact original-paper comparison.

Claim evidence for all addendum numbers:
`results/paper_trec-covid/paper_dataset_benchmark.json`; consistency checks:
`results/paper_trec-covid/local_verification.json`, under `cascaded-blended-rag/`.


## SQuAD addendum review

1. Contribution: quality targets pass here; small old-backend time savings do
   not establish superiority over standalone retrievers.
2. Clarity: 10,570-query quality and one-pass timing are explicit; additional
   partial records are preserved rather than pooled unevenly.
3. Experimental strength: exact qrels reconstruction and independent metric
   reproduction support quality; timing has no completed repeat-run check.
4. Completeness: all four policies, four baselines, metrics and the runtime
   amendment remain visible. Indexed follow-up outcomes are reviewed below.
5. Soundness: no threshold refit; indexed primary was fixed before inspecting
   SQuAD outcomes. BM25 already passing the floor remains a required baseline.

Evidence: `results/paper_squad_one_pass/paper_dataset_benchmark.json`,
`local_verification.json` and `independent_metric_verification.json`, relative
to `cascaded-blended-rag/`.

## Indexed SQuAD addendum: outline and claim-evidence alignment

Paragraph sequence: execution-cost motivation; frozen optimization and sample
design; all-method results; primary and secondary uncertainty; exact ranking
agreement; constant-baseline limitation. The evidence supports each paragraph
topic without requiring a full-query indexed or new-generalization claim.

| Claim | Evidence under `results/indexed_squad_q256/` | Status |
|:--|:--|:--|
| Primary r99 saves 4.19% mean time with all sample parent hits | `indexed_cascade_benchmark.json`, r99 metrics and intervals | Supported on 256 questions, three passes, one T4 session |
| Secondary r90 saves 37.08%, retaining 99.22% | Same report, r90 metrics and intervals | Supported secondary result; not a replacement primary |
| All saved results match the original backend exactly | `paired_backend_comparison.json`, all eight comparison rows | Supported on the same 256 questions only |
| Quality metrics reproduce from raw parent labels | `independent_metric_verification.json` | Supported for all sample methods; no generated-answer claim |
| Cascade is best for a 90% floor | BM25 95.70% retained at 1.03 ms | Unsupported; explicitly rejected |

### Answered five-dimensional self-review

1. Contribution: execution improvement and conditional mean savings are
   supported. New algorithmic priority and competitive novelty remain open.
2. Clarity: the sample, full corpus, three repetitions, frozen r99 primary,
   secondary r90/r92 results and mean versus p95 are stated explicitly.
3. Strength: exact ranking agreement and independently reproduced metrics
   accompany real conditional timing. One session and 256 questions limit
   generalization; zero-loss bootstrap intervals are not guarantees.
4. Completeness: all eight methods and every target are retained. BM25 and
   SPLADE remain decisive baselines. Modern adaptive comparisons, memory,
   update cost and generated-answer evaluation are absent, so no claims
   requiring them are made.
5. Soundness: the optimization is also applied to fusion and standalone BM25.
   All policies remain frozen, the declared query hash is verified and earlier
   transfer failures remain visible. Practical superiority at the requested
   floor is not established. The subsequent indexed TREC result is reviewed below.

## Indexed TREC addendum: outline and claim-evidence alignment

Paragraph sequence: full-query frozen engineering design; all-method table;
primary success against the 90% floor with nominal-target failure; exact
ranking and independent metric verification; SPLADE comparison; conclusion
limited to measured execution and quality/time trade-offs.

| Claim | Evidence under `results/indexed_trec-covid_q0/` | Status |
|:--|:--|:--|
| Primary saves 19.04% mean time and retains 96.68% recall | `indexed_cascade_benchmark.json`, r99 metrics and intervals | Supported for all 50 queries, three passes, one T4 session |
| Both primary interval lower bounds pass the follow-up criteria | Same report: retention 93.58%, time saved 9.47% | Supported query-bootstrap evidence; not a universal guarantee |
| All nominal TREC targets still fail | Four `observed_target_met` values | Supported; passing 90% does not mean passing nominal 99% |
| All old/new saved results exactly match | `paired_backend_comparison.json`, all eight methods | Supported across the full 50-query evaluation |
| Raw-label scoring reproduces all quality metrics | `independent_metric_verification.json` | Supported; timing instrumentation not rerun |

### Answered five-dimensional self-review

1. Contribution: mean latency improvement against equally optimized fusion
   is measured; eager scoring and routing are not claimed as new principles.
   SOTA, overall retriever superiority and exact-paper acceleration remain
   unsupported and are explicitly excluded from the claims.
2. Clarity: all 50 queries, full corpus, three passes, primary r99, the 90%
   evaluation floor, nominal-target failure and mean versus p95 are explicit.
   Absolute recall and relative retention are distinguished in the findings.
3. Strength: no new retuning, exact old/new rankings and independent raw-qrel
   scoring accompany actual conditional execution. One hardware/session and
   50 topics limit generalization; inspected-data selection is acknowledged.
4. Completeness: all four gates and strong constant baselines are included.
   SPLADE's recall/nDCG/time advantage and the cascade's MAP advantage are both
   reported without changing the original recall objective. Adaptive SOTA,
   generated answers and full NQ/HotpotQA are not evaluated.
5. Soundness: reference and cascades share the optimization. Original primary
   failures remain visible. Larger cross-run backend ratios are separated
   from the fair 19.04% within-indexed conditional saving. These completed
   results can be reported as a bounded engineering study; no further GPU
   benchmark is needed merely to write up the existing evidence.

## Consolidated report and full speedup presentation, 21 October 2026

The user requested a growing presentation through the highest speedup,
including every gate rather than only the primary 19.04% saving. The report
now has one abstract and conclusion covering the complete experiment story.
The previous addendum-based version is archived in
`docs/history/FINAL_MANUSCRIPT_before_consolidation_20261021.md`.

### Outline and paragraph roles

1. Abstract/introduction: state the retrieval-cost question, full measured
   speedup range and limits of the empirical contribution.
2. Related work/method: identify existing strategy selection and eager scoring,
   describe the frozen gate, then explain the compatible execution change.
3. Protocol: establish split/inspection status, primary decisions, the two
   speedup references, and complete versus sampled evaluation scopes.
4. Results: show the growing configuration progression, all indexed methods,
   previous failed targets, exact compatibility and independent quality checks.
5. Discussion/conclusion: explain first-stage cost and strong constant
   baselines without turning engineering gains into method-priority claims.

### Claim-evidence map

| Claim | Evidence | Status |
|:--|:--|:--|
| All TREC cascade points reach 1.24x/1.64x/2.17x/2.26x over indexed fusion | Indexed report means, all four methods | Supported; last two retain only 84.53% |
| Combined old-fusion ratios grow through 16.36x/20.21x/26.83x/35.56x/36.96x | Original full-query TREC fusion mean divided by indexed means; experiment 32 export | Supported across separate runs; indexing and routing combined |
| The secondary point reaches 26.83x at 91.95% retained | Same-query source means; r95 recall interval | Supported point; lower interval crosses below 90% |
| SQuAD cascade maximum is 1.59x over indexed fusion, 7.91x over original fusion | Exactly matched 256 query IDs; original one pass, indexed three | Supported for sample only |
| Strategy selection and eager scoring predate this project | Primary papers arXiv:2109.10739 and arXiv:2407.03618, checked 21 October 2026 | Supported; no new-principle claim |

### Answered five-part review

1. Contribution: the report now shows all measured execution improvements,
   rather than reducing its contribution to one percentage. New algorithmic
   priority and best-method status remain unproven.
2. Clarity: every ratio names its reference. The figures use actual measured
   configurations; the presentation does not imply growth with corpus size.
   Observed recall accompanies every point, including the failed floor.
3. Strength: cross-run means are labeled as such, and the fair conditional
   comparison shares the indexed scorer. Existing artifact and independent
   metric checks remain the evidence; no new benchmark was run for plotting.
4. Completeness: all gate settings, full standalone tables and separate scorer
   ratios appear. Original failed targets and incomplete larger-dataset scope
   remain explicit. The report cites verified primary sources for its limited
   related-work discussion; unverified historical literature claims are unused.
5. Soundness: no policy changed. The 36.96x maximum is not presented as a 90%
   quality success, and the 26.83x point retains its uncertainty caveat. The
   preselected conservative point remains primary despite larger secondary gains.
