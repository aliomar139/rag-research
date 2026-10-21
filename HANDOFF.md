# Session handoff: Conditional Hybrid Retrieval

Updated: **21 October 2026**. Read this before continuing work in this repository.

## 1. Current state and immediate next action

**The current experimental scope is finalized. The paper, showcase and presentation have been revised for a master's-level explanation and a clearer research story. No new Kaggle run was used or is needed to share these results.**

The latest request asked to simplify the paper and align the HTML, presentation
and PDFs. That revision is complete. The HTML now moves from the selected paper
to the reproduction and its results, explains why the numbers differ, then
covers the indexed implementation, its results and references. The deck has
19 main slides and four appendices. Wait for the next specific request before
starting experiments or changing the release further.

Active workspace: `C:\Users\user\Desktop\rag-research-alone`, Windows PowerShell.
The older IDE path under `M2-Porjects\rag-research` is not this active workspace.

At handoff creation, before adding this file:

- Branch: `main`; Git reported a clean working tree.
- HEAD: `dfa104a` — Remove outdated queries and cascade source bundle files to streamline project resources.
- Commit count: **3**. Older conversation snapshots mentioning two commits or hundreds of pending changes are stale.
- This handoff task creates `HANDOFF.md`; it does not commit, push, rebuild the sharing ZIP or run tests.

## 2. How to work with this user

The user explicitly enabled the **i-have-adhd** skill. Its source is
`C:\Users\user\.agents\skills\i-have-adhd\SKILL.md`. Read it if available.

1. Lead with the answer or a concrete next action. Explain research in plain language, with short sections and small tables.
2. Give numbered implementation steps and exact copyable code when work is required. End unfinished work with one specific next action.
3. For Kaggle work, provide the exact local file to upload, a fresh dataset title/slug, attachment steps and a robust notebook cell. Earlier sessions repeatedly failed on ZIP names and mount paths.
4. When the user sends an output filename, they mean it has been downloaded into the project. Find and inspect it before asking them to download it again.
5. Preserve completed work. Do not launch lengthy inference, encoding, timing jobs or unit tests unless the active request authorizes them. Do not request repeated permission for already authorized routine work.

The user wants a credible contribution and a fast finish. Be direct about
what succeeded and what the evidence cannot support. Do not inflate results
to reassure them.

## 3. Files to read first

| Need | Current source |
|:--|:--|
| Plain-language project history | [docs/adhd-explanation.md](docs/adhd-explanation.md) |
| Authoritative manuscript and results | [docs/FINAL_MANUSCRIPT.md](docs/FINAL_MANUSCRIPT.md), [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md) |
| All measured speedups and quality trade-offs | [docs/SPEEDUP_SUMMARY.md](docs/SPEEDUP_SUMMARY.md), [CSV](docs/figures/all_measured_speedups.csv) |
| Deliverables and rebuild instructions | [docs/submission/README.md](docs/submission/README.md) |
| Reproduction, claim review and completion status | [reproduction guide](docs/reproducibility/README.md), [paper review](docs/review/PAPER_RELEASE_REVIEW.md), [finish checklist](docs/review/PROJECT_FINISH_CHECKLIST.md) |

Current indexed evidence is under:

```text
cascaded-blended-rag/results/indexed_trec-covid_q0/
cascaded-blended-rag/results/indexed_squad_q256/
```

Each contains `indexed_cascade_benchmark.json`, `quality_speed.csv`,
`paired_backend_comparison.json`, `independent_metric_verification.json`
and supporting records. Large per-query files remain local and ignored.
Consult saved reports rather than treating this handoff as a replacement
for the experimental evidence.

## 4. What the approach does

```text
Question -> indexed Best Fields BM25 -> learned Ridge gate
                                      | enough predicted quality: return BM25
                                      | otherwise: run Dense + SPLADE, then RRF
```

- **Static fusion** always runs BM25, Dense and SPLADE.
- **Indexed fusion** does the same retrieval with a faster compatible BM25 implementation.
- **Indexed cascade** uses that same efficient BM25 and adds the decision to skip the two neural branches.

Dense uses `sentence-transformers/all-MiniLM-L6-v2`; SPLADE uses
`naver/splade-cocondenser-ensembledistil`. Best Fields scores title and abstract
separately and takes the maximum. Fusion is RRF with constant 60 and top 100.

The gate is standardized Ridge regression, alpha 10, with 11 query/BM25
features. It predicts `max(Recall_fusion - Recall_BM25, 0)` and clips negative
predictions to zero. It exits when the prediction is at or below a threshold.
Thresholds were calibrated using five-fold out-of-fold predictions on all
**2,590 NFCorpus train queries**, seed 42; the final model fits all train.
NFCorpus test had already been inspected and was excluded from this sequence.

**r90, r92, r95 and r99 are frozen training-target settings of this same
cascade. Their names are not guaranteed measured retention on new datasets.**
The indexed follow-up's preselected primary is r99; the faster settings
remain secondary. Earlier studies retain their own originally declared primaries.

## 5. The two speedup charts: preserve this distinction

The user needed this explained twice. Both panels show the **same methods**;
they change the reference latency used in the division.

| Chart | Calculation | Meaning |
|:--|:--|:--|
| Left: vs our original fusion | Original fusion time / method time | Total improvement from indexing plus routing, across separate runs |
| Right: vs equally indexed fusion | Indexed fusion time / method time | Additional routing benefit when both methods already have efficient BM25 |

Example: **the same fixed 256 SQuAD questions**:

| Version | Mean query time | Speedup vs original fusion | Speedup vs indexed fusion |
|:--|--:|--:|--:|
| Original fusion | 147.01 ms | 1.00x | Not the reference here |
| Indexed fusion | 29.55 ms | 4.98x | 1.00x |
| Indexed cascade r90 | 18.59 ms | 7.91x | 1.59x |

Thus **7.91x is the combined improvement**, while **1.59x is the extra benefit
of the gate over already indexed fusion** (37.08% less mean time).
The r90 point retains 99.22% of fusion's mean Recall@100 on this sample.

"Original fusion" means **our initial implementation**, not the published
paper's engine. The larger cross-run ratios include session variability.
The growing bars are configurations with more aggressive skipping, not an
experiment showing speedup as corpus size grows.

## 6. What recall retention means

```text
Retention = mean Recall_method@100 / mean Recall_fusion@100
Speedup = reference mean latency / method mean latency
Time saved = 1 - method mean latency / reference mean latency
```

Retention is **not document-list overlap**, the percentage of questions
answered correctly, or a per-query guarantee. It can exceed 100% when a
standalone retriever has higher mean recall than fusion.

For example, TREC fusion recall is about **0.12250**. The primary cascade's
recall is about **0.11844**, which is **96.68% retention**. It does not retrieve
96.68% of all relevant TREC documents.

Top-k hit accuracy, Recall@100, nDCG@10 and answer EM/F1 measure different
things. Generated-answer EM/F1 was not evaluated in the current study.

## 7. Main measured results

### Indexed TREC-COVID

**All 50 queries, full 171,332-document corpus, three warmed T4 timing passes.**

| Configuration | Recall retained | Mean ms | Speedup vs original fusion | Speedup vs indexed fusion |
|:--|--:|--:|--:|--:|
| Indexed fusion | 100.00% | 91.28 | 16.36x | 1.00x |
| r99, primary | 96.68% | 73.90 | 20.21x | 1.24x |
| r95, secondary | 91.95% | 55.66 | 26.83x | 1.64x |
| r92, secondary | 84.53% | 41.99 | 35.56x | 2.17x |
| r90, secondary | 84.53% | 40.41 | 36.96x | 2.26x |

Primary r99 saves **19.04% mean time** versus indexed fusion. Its query-bootstrap
95% intervals are **93.58–98.75% retention** and **9.47–30.11% time saved**.
It clears the user's 90% floor but misses its nominal 99% target.

Secondary r95 saves 39.02%, but its retention interval, **86.88–95.76%**,
crosses below 90%. r92/r90 fail the 90% floor. The largest 36.96x value must
always be accompanied by its 84.53% retention and failed floor.

**SPLADE alone is an important stronger alternative:** 104.69% retention,
52.77 ms, higher recall/nDCG and lower mean time than the primary cascade.
The primary has higher MAP@10, so this is a trade-off rather than overall superiority.
Primary p95 latency is 99.41 ms versus fusion's 97.66 ms: the benefit is mean time.

TREC quality had already been inspected before the indexed follow-up. This
is an execution follow-up on known quality, not a fresh independent generalization test.

### Indexed SQuAD

**256 fixed SHA-selected questions, full 20,958-document corpus, three warmed T4 passes.**

| Configuration | Recall retained | Mean ms | Speedup vs original fusion | Speedup vs indexed fusion |
|:--|--:|--:|--:|--:|
| Indexed fusion | 100.00% | 29.55 | 4.98x | 1.00x |
| r99, primary | 100.00% | 28.31 | 5.19x | 1.04x |
| r95, secondary | 100.00% | 24.82 | 5.92x | 1.19x |
| r92, secondary | 100.00% | 21.31 | 6.90x | 1.39x |
| r90, secondary | 99.22% | 18.59 | 7.91x | 1.59x |

r90's 95% intervals are **98.05–100% retention** and **31.58–42.69% time saved**.
Its p95 is 31.48 ms versus fusion's 31.37 ms, so do not claim a tail-latency gain.

**BM25 alone already retains 95.70% at about 1.03 ms**, much faster than the
cascade at the user's 90% floor. SPLADE has the strongest nDCG/MAP.

Original full-query SQuAD covers **all 10,570 dev questions, one complete
timing pass**, with partial extra records retained separately. Its r90 retains
99.38% and saves 6.12%. Do not combine those full-query means or quality
values with indexed sample metrics. Cross-run charts use the same 256 IDs.

### Earlier supporting studies

| Study | Reported original primary | Recall retained | Mean time saved |
|:--|:--|--:|--:|
| NFCorpus dev, 324 queries | r92 | 94.74% | 23.00% |
| SCIDOCS frozen confirmation, 1,000 queries | r92 | 90.01% | 5.53% |
| Original TREC-COVID | r90 | 84.53% | 3.56% |
| Original full SQuAD | r90 | 99.38% | 6.12% |

SCIDOCS r92's interval crosses below 90%; Dense alone is stronger on
NFCorpus/SCIDOCS recall and time. All nominal targets miss on SCIDOCS and TREC.

## 8. Claim boundaries and unfinished evaluation

The original paper is Sawarkar, Mangal and Solanki, **Blended RAG** (2024):
[arXiv 2404.07220](https://arxiv.org/abs/2404.07220).
Its system uses Elasticsearch/ELSER. Ours uses Python/MiniLM/SPLADE/RRF.
This is an **adaptation and measured engineering contribution**, not an exact
reproduction, a demonstrated new routing principle, or a speedup of the paper's engine.

Completed overlap with the paper's datasets: **TREC-COVID and SQuAD**, with
the query scopes above. Additional full current **NQ, HotpotQA and CoQA**
remain uncompleted. Earlier qrel-informed NQ simulations do not count as a
full-corpus live benchmark. HotpotQA has preparation code but no completed run.
Generated-answer EM/F1 and modern adaptive-baseline comparisons remain deferred.

Indexing alone accounts for much of the large speedup. Standalone retrievers
often offer a better recall/time choice. Do not claim overall method superiority,
state of the art, guaranteed 90% retention, or preserved answer quality.

Timings cover query retrieval/routing, excluding corpus encoding, index
preparation, downloads, model loading and answer generation. No production
throughput or scaling study was completed.

## 9. Finished deliverables

All are under [docs/submission](docs/submission/README.md):

| File | What it contains |
|:--|:--|
| `conditional_retrieval_paper.pdf` | 10-page academic paper; editable source is `docs/FINAL_MANUSCRIPT.md` |
| `research_showcase.html` / `research_showcase.pdf` | Offline research summary and matching 10-page PDF |
| `research_presentation.pptx` / `research_presentation.pdf` | 23 slides: 19 main plus 4 appendices, for a 15-20 minute presentation |
| `presentation_notes.md` | Speaker notes, also inside PPTX |
| `RELEASE_MANIFEST.json` | File SHA256 values and frozen-asset fingerprints |

The showcase covers the selected paper, the reproduction, its results and
comparison limits, then the indexed implementation, results and references.
The audience/tone remain an editorial research account for the
supervisor/examiners.
The PPTX has editable text, native tables and 2 native charts with embedded
data. Install the four static Newsreader/IBM Plex Sans fonts from
`docs/submission/assets/fonts/` for its intended appearance. Font licenses
are included. HTML embeds fonts. Paper/showcase PDFs have selectable text;
the presentation PDF consists of high-resolution slide renders.

Materials were visually reviewed and locally rendered. The final PPTX was
imported and rendered by the authoring runtime; native PowerPoint was not opened.
Authors (Ali Omar, Abdallah Mestrah, Abbas Yazbek), supervisor (Dr. Ahmad Abboud), and Lebanese University affiliation are listed in the deliverables. Add course metadata and the required institutional template before formal hand-in. A repository license is not selected.

Existing sharing ZIP:
`artifacts/submission/rag_research_submission_20261021.zip` (about 7.73 MiB).
It contains source, compact evidence and deliverables, excluding large local
datasets/records/environments. **It predates this handoff; regenerate it if
the new root file must be included.**

## 10. Evidence integrity and repository organization

Earlier artifact checks found identical original/indexed first-pass IDs,
scores, routes and recalls for **400 TREC method/query records** and
**2,048 sampled SQuAD records**. All **1,200 TREC** and **6,144 SQuAD** timing
records were checked for consistency. Independent raw-label rescoring
reproduced quality metrics. These checks do not independently certify clocks
or production performance. No inference or timing was rerun to make the materials.

Preserve frozen bytes. `.gitattributes` disables newline conversion for
Python/JSON/JSONL/TSV/notebooks because frozen files mix CRLF and LF.
Never reformat protected files or update a protocol hash merely to accept
a changed source file.

| Important protected file | SHA256 |
|:--|:--|
| `cascaded-blended-rag/experiments/25_benchmark_indexed_cascade.py` | `ddb7b5656ed20e0ab20cb2199f544137e3679c161b99cb935c5f14919e44a104` |
| `cascaded-blended-rag/src/indexed_bm25.py` | `33ce9636fcf36cc2dc0e57b661c04c0e557af9092a965179e8e0778caf0327e9` |
| `cascaded-blended-rag/results/nfcorpus_frozen_retention_routers.json` | `3efd7382cd91cfc43bd7ae5d4eaf27571ae92657dec97e3eaf85fbb36131a5d1` |
| `cascaded-blended-rag/results/indexed_cascade_protocol.json` | `b1f9f79192a9600492e2446130d29eac4925733db6ed2b236825344efb385e0f` |

The release manifest and audit cover additional protected assets. Saved
reports include model revisions and runtime details. Current dependency
ranges are not an exact Kaggle environment lockfile.

Repository cleanup is already complete. Keep the current top-level structure:
code/results in `cascaded-blended-rag`, current writing in `docs`, historical
drafts in `docs/history`, versioned Kaggle instructions in `notebooks`, and
publication/sharing utilities in `tools`.

Local raw data, caches, environments, downloaded ZIPs and large rankings are
ignored, not disposable. Original downloads are under `artifacts/downloads`;
source packages under `artifacts/source_bundles`; old exports under
`artifacts/legacy_exports`. `PROJECT_ANCHOR.md` is an ignored local historical
note, not the shared entry point. The current explanation filename is
`docs/adhd-explanation.md`, not the older misspelling `adhdd-explanation.md`.

## 11. Safe continuation and rebuild instructions

For a conceptual question, answer from the current documents and evidence;
there is no need to rebuild materials or run retrieval.

For requested edits to the paper/showcase/deck:

1. Edit the relevant source. Preserve frozen experiments, weights, protocols and measured report JSON.
2. Rebuild only affected artifacts using [the submission guide](docs/submission/README.md); inspect changed pages/slides.
3. Regenerate `docs/submission/RELEASE_MANIFEST.json` after final artifact bytes are settled.
4. If preparing a new shared bundle, run the repository audit and ZIP builder below. These do not execute unit tests or retrieval.
5. Report changed files and any limits. Commit or push only when requested.

```powershell
python tools/check_repository.py
python tools/build_submission.py
```

Relevant builders:

```text
tools/build_publication_figures.py       Saved-report chart assets
tools/build_research_materials.py        Paper PDF and offline HTML
tools/showcase.template.html             Showcase content/layout source
tools/render_showcase.mjs                Local HTML rendering and PDF
tools/build_research_presentation.mjs    Editable PPTX
tools/export_presentation_pdf.py         PDF from rendered final slides
tools/write_release_manifest.py          Final release fingerprints
```

The JavaScript builders depend on the bundled Codex Artifact Tool/Playwright
runtime. Exact PowerShell commands and environment overrides are in the
submission guide. Windows runtime binaries currently live under:

```text
C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe
C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

In this Windows sandbox, deck authoring renders the candidate and all slides,
then can fail when Node's synchronous child-process call tries to launch the
bundled Python validators (`EPERM`). For this 23-slide release, the package and
layout validators were run directly on the candidate and passed with zero
findings before the checked bytes were copied to the deliverable. The
`--render-existing` path also succeeded. If rebuilding, inspect validator
results and rendered slides. Private render files live in `.local/release_build`.

For later dataset work, declare a fresh protocol before examining outcomes,
preserve the current release, retain BM25/Dense/SPLADE references and record
new results separately. Use `notebooks/README.md` to select the appropriate
workflow. ZIP entries must use `/`, not Windows `\`; discover actual Kaggle
mounts instead of relying on a fixed `/kaggle/input/<slug>` layout.

**No additional experiment is currently authorized by this handoff. The next
session should read this file, inspect the user's new request and continue
from the finished release.**
