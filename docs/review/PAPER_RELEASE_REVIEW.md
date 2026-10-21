# Paper release review

## Story and section outline

1. Original paper: establish the research source and adaptation boundary.
2. Motivation: explain why the cost of the first search limits savings.
3. Reproduction and metrics: distinguish hit accuracy, absolute recall,
   retention, nDCG and answer EM/F1 before presenting any comparisons.
4. Method: explain the retrieval choices, gate, calibration and indexed scorer.
5. Protocol: state datasets, inspection status, timing window and metrics.
6. Results: compare all four settings, both speedup references and standalone
   methods. Discussion and conclusion define the limits of the finding.

The revised source uses shorter sentences and defines technical terms at first
use. The paper PDF is nine pages, including references, with result tables and
figures retained.

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
| TREC primary retains 96.68% at 1.24x vs indexed fusion; combined observed ratio is 20.21x vs earlier Python fusion | `results/indexed_trec-covid_q0/indexed_cascade_benchmark.json`, r99, and original timing report | Supported within our implementation across separate runs; not a comparison with the original Blended RAG system |
| Secondary TREC r95 reaches 1.64x at 91.95% | Same report, r95 and its retention interval | Supported point estimate; interval crosses 90% |
| Maximum cascade reaches 2.26x indexed / 36.96x original at 84.53% | Same report and source CSV under `docs/figures/` | Supported; fails the 90% floor |
| Fastest sampled SQuAD cascade reaches 1.59x vs indexed fusion and 7.91x combined vs earlier Python fusion at 99.22% | `results/indexed_squad_q256/indexed_cascade_benchmark.json`, r90, and matched 256-query original timing slice | Supported for the fixed 256-question sample across separate runs |
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
   The manuscript explains retention and both speedup references before the
   results, distinguishes full and sampled SQuAD, and keeps the method and
   timing details needed to interpret the findings. Paragraphs have one main
   point and technical terms are explained in context.
3. **Experimental strength.** Do results support useful execution gains?
   Three indexed timing passes, exact ranking agreement and independent
   scoring support the bounded claim. One GPU type, few sessions and known
   TREC outcomes restrict broader inference. Standalone methods remain strong.
4. **Evaluation completeness.** Is the scope complete enough for every claim?
   Yes for the declared engineering study. Full NQ, HotpotQA, CoQA, generated
   answers, modern adaptive baselines and production measurements remain
   deferred. The release does not silently count prepared runners as results.
5. **Method design soundness.** Is the comparison matched? The indexed
   cascade and fusion use the same compatible backend. The primary and
   secondary settings retain their original labels. Equal-indexed comparisons
   isolate routing; earlier-fusion ratios combine backend and routing changes
   across sessions.

## Remaining administrative details

The paper, showcase, and deck now list the authors, supervisor, and university. Add course metadata and the institutional template if required before hand-in.
The evidence supports a bounded course report if claims stay within our implementation. It does not show a direct improvement over the published Elasticsearch/ELSER system or preserved generated-answer quality.
