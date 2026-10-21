# Final research materials

**Start by opening [the offline HTML showcase](research_showcase.html).**
No server, Kaggle session or model download is needed to read these files.

| File | Use |
|:--|:--|
| [Research paper](conditional_retrieval_paper.pdf) | 11-page academic report with methods, protocol, results, limitations and references |
| [HTML showcase](research_showcase.html) / [matching PDF](research_showcase.pdf) | A visual account of the original paper, motive, adaptation, metric differences, results and implementation |
| [Editable presentation](research_presentation.pptx) / [presentation PDF](research_presentation.pdf) | 18 main slides for a 15–20 minute talk, plus four evidence/reproduction appendices |
| [Speaker notes](presentation_notes.md) | Detailed explanations and source references, also embedded in the PPTX |

The editable paper source is [FINAL_MANUSCRIPT.md](../FINAL_MANUSCRIPT.md).
The [release review](../review/PAPER_RELEASE_REVIEW.md) records the section
outline, paragraph roles, claim-evidence map and answered self-review.
The [release manifest](RELEASE_MANIFEST.json) fingerprints the final materials
and their frozen protocol assets.

## Opening and presenting

1. Open the HTML in a browser to read the visual story. It embeds fonts and
   figures; adjacent PDF/deck links work when the folder stays together.
2. Use the paper PDF for detailed reading and the showcase PDF for a shorter
   illustrated explanation.
3. Install the four static font faces in [assets/fonts](assets/fonts/README.md)
   before opening the PPTX for its intended appearance. The deck's text,
   eleven tables and three data charts remain editable. The presentation
   PDF preserves the inspected appearance without requiring font installation.

The presentation PDF uses high-resolution slide renders. The paper and
showcase PDFs retain selectable text. The PPTX carries source references in
speaker notes and literal chart-data workbooks. Chart values round to six
decimals for workbook compatibility; display values use two decimals, and
original reports retain full precision.

Author and institution details have not been invented. Add personal/course
metadata and the required institutional template before formal hand-in.

## Completed scope

The release reports retrieval and query timing on the completed datasets.
Full current NQ, HotpotQA and CoQA, generated-answer EM/F1 and modern adaptive
baselines remain deferred. The indexed SQuAD result uses 256 fixed questions;
its full corpus is retained. Original SQuAD evidence covers all 10,570
questions with one complete timing pass. TREC covers all 50 queries.

The larger original-fusion ratios combine indexing and routing across runs.
Additional cascade gains use equally indexed fusion. All four settings,
failed quality floors, uncertainty and stronger standalone alternatives
remain visible. This is an engineering study with a bounded contribution.

## Rebuild paper, HTML and figures

From the repository root with Python 3.12 or newer:

```powershell
python -m pip install -r requirements/publication.txt
python cascaded-blended-rag/experiments/32_prepare_report_assets.py
python tools/build_publication_figures.py
python tools/build_research_materials.py
```

These commands read saved results. They do not rerun retrieval or timing.
The template is `tools/showcase.template.html`; visual tokens are in the
root `tokens.css`. The included licensed font files allow offline generation.

## Rebuild the deck and print showcase

JavaScript authoring uses the bundled Codex primary runtime's Artifact Tool
and Playwright. It needs that runtime rather than an npm substitute. The
defaults locate the installed Windows runtime; set `RUNTIME_NODE_MODULES`,
`RUNTIME_PYTHON` and `PRESENTATIONS_SKILL_DIR` for another installation.

For this workspace, the complete commands are:

```powershell
$releaseNode = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
$releasePython = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $releaseNode tools/build_research_presentation.mjs
& $releaseNode tools/build_research_presentation.mjs --render-existing
& $releasePython tools/export_presentation_pdf.py
& $releaseNode tools/render_showcase.mjs
python tools/write_release_manifest.py
```

The deck builder validates a new private revision, then copies its checked
bytes to the stable deliverable. Previews and validation records stay in
ignored `.local/release_build/`. The headless browser renders the local HTML
to the matching PDF. No site is deployed.

## Sharing and later evaluation

Run `python tools/build_submission.py` to rebuild the source/report sharing
ZIP. It includes these materials and compact evidence, while excluding
local data, environments and large result archives. Preserve this release
before conducting additional datasets. Declare a fresh protocol before
examining their outcomes, retain the standalone references and record new
results separately rather than replacing this release's claims.
