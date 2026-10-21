# Paper release review

## Story and section outline

1. Original paper: establish the research source and the adaptation boundary.
2. Motivation: explain why first-stage cost can prevent useful early exits.
3. Reproduction and metrics: distinguish hit accuracy, absolute recall,
   retention, nDCG and answer EM/F1 before presenting any comparisons.
4. Method and protocol: define the gate, calibration, compatible indexing,
   fixed decisions, timing window and sample boundaries.
5. Results: report all four settings, both speedup references and the strong
   standalone alternatives. Implementation, limitations and conclusion follow.

## Paragraph roles and reverse outline

| Section | Paragraph roles in order | Link to the central claim |
|:--|:--|:--|
| Abstract | Problem, mechanism, protocol, measured evidence, baseline limitation | Bounds the execution contribution |
| Original paper | Published setting, adaptation boundary, prior methods | Prevents an exact-reproduction or priority claim |
| Motivation | Runtime bottleneck, quality budget, contributions | Defines what counts as a useful improvement |
| Reproduction | Changed system, metric definitions, labels, comparison limits | Prevents mismatched numerical comparisons |
| Method | Actions, prediction target, thresholds, compatible scorer | Makes conditional computation inspectable |
| Protocol | Collections, inspection status, timing, estimands | Identifies what each measurement can support |
| Results | Complete progression, primary/secondary outcomes, baselines, compatibility | Supports the bounded latency/recall claim |
| Implementation | Modules, execution boundaries, validation, rebuilding | Connects prose to source and saved evidence |
| Discussion | Cost interpretation, transfer failures, omitted evaluations | Restricts extrapolation |
| Conclusion | Supported gains and stronger constant alternatives | Answers the question without changing its objective |

## Claim-evidence map

Paths beginning `results/` are relative to `cascaded-blended-rag/`.

| Claim | Evidence | Status |
|:--|:--|:--|
| TREC primary retains 96.68% at 1.24x vs indexed fusion | `results/indexed_trec-covid_q0/indexed_cascade_benchmark.json`, r99 | Supported for 50 queries/three passes |
| Secondary TREC r95 reaches 1.64x at 91.95% | Same report, r95 and its retention interval | Supported point estimate; interval crosses 90% |
| Maximum cascade reaches 2.26x indexed / 36.96x original at 84.53% | Same report and source CSV under `docs/figures/` | Supported; fails the 90% floor |
| Fastest sampled SQuAD cascade reaches 1.59x at 99.22% | `results/indexed_squad_q256/indexed_cascade_benchmark.json`, r90 | Supported for the fixed 256-question sample |
| Original and indexed saved rankings agree | `results/indexed_*/` compatibility records, experiment 30 | Supported on shared query IDs |
| Saved quality metrics reproduce from raw labels | Experiments 29/31 and recorded verification reports | Supported; no independent clock measurement |
| Models/metrics differ from the paper | Original paper [1], source modules and frozen protocols | Supported difference; causal size of each gap unmeasured |
| Cascade is the best recall/time method | Standalone comparisons contradict this in completed collections | Removed |
| Remaining datasets or answer quality have been tested | No completed current NQ/HotpotQA/CoQA or EM/F1 run | Explicitly outside scope |

## Answered five-dimension self-review

1. **Contribution.** What new evidence does the project supply? It measures
   the interaction of a frozen gate and a compatible lexical backend, while
   retaining transfer failures and stronger standalone choices. Algorithmic
   novelty and SOTA competitiveness need a separate study.
2. **Writing clarity.** Can readers identify the references and estimands?
   The manuscript defines retention and both speedup denominators before
   results, labels original/full versus indexed/sample SQuAD, and supplies
   code/evidence paths. Paragraphs carry one stated message.
3. **Experimental strength.** Do results support useful execution gains?
   Three indexed timing passes, exact ranking agreement and independent
   scoring support the bounded claim. One GPU type, few sessions and known
   TREC outcomes restrict broader inference. Standalone methods remain strong.
4. **Evaluation completeness.** Is the scope complete enough for every claim?
   Yes for the declared engineering study. Full NQ, HotpotQA, CoQA, generated
   answers, modern adaptive baselines and production measurements remain
   deferred. The release does not silently count prepared runners as results.
5. **Method design soundness.** Is the comparison matched? All methods share
   compatible indexing. The primary and all secondary settings retain their
   original labels. Selection uses previously inspected TREC evidence, so
   its follow-up supports execution analysis rather than fresh generalization.

## Remaining administrative details

The project report uses no invented author or institution. Add the required
author/course details and institution template before formal submission.
The current evidence is sufficient for this bounded report, while stronger
scientific claims still need additional experiments.
