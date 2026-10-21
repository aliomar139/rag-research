# Conditional Hybrid Retrieval

**A master's research project on deciding when a search system should spend the extra cost of neural retrieval.**

This repository contains our reproduction of the blended retrieval approach from the selected paper, experiments with a BM25-first cascade, and the final paper and presentation. The cascade uses an explicit decision rule to decide whether to run a second retrieval stage. We also implemented indexed BM25 fusion, which reduced the cost of the lexical baseline.

**Authors:** Ali Omar, Abdallah Mestrah, and Abbas Yazbek. **Supervisor:** Dr. Ahmad Abboud. **Institution:** Lebanese University.

## Start here

- [Final results and experiment index](final_results/README.md): paper, showcase, slide deck, PDFs, and links to all 32 experiment scripts.
- [Research paper](final_results/conditional_retrieval_paper.pdf)
- [HTML research showcase](final_results/research_showcase.html)
- [Editable presentation](final_results/research_presentation.pptx)
- [Numbered experiment scripts](cascaded-blended-rag/experiments/)
- [Code and implementation notes](cascaded-blended-rag/README.md)
- [Reproduction guide](docs/reproducibility/README.md)

## What we measured

The results below come from our Python implementation and compare retrieval configurations on the same project datasets. They show an improvement over our reproduced baseline in these experiments; they are not direct timing measurements of the original Elasticsearch and ELSER system.

| Evaluation | Comparison | Result |
|:--|:--|--:|
| TREC-COVID, all 50 queries | Selected cascade vs equally indexed fusion | **1.24x faster**, retaining 96.68% of fusion Recall@100 |
| SQuAD, declared 256-question sample | Fastest tested cascade vs indexed fusion | **1.59x faster**, retaining 99.22% of fusion Recall@100 |

The indexed fusion implementation also changes the baseline cost, so speedups against our original, non-indexed fusion include both the indexing work and the cascade decision. The table uses equally indexed fusion to show the added cascade gain separately. The comparisons measure retrieval time and recall; they don't establish answer quality or speedups on the original paper's ELSER stack. Full NQ and HotpotQA evaluations remain pending. See the [final results guide](final_results/README.md) and [paper](final_results/conditional_retrieval_paper.pdf) for protocols, other settings, and limitations.

## Repository map

| Path | Contents |
|:--|:--|
| [`cascaded-blended-rag/src/`](cascaded-blended-rag/src/) | Retrieval, indexed BM25, and routing implementation |
| [`cascaded-blended-rag/experiments/`](cascaded-blended-rag/experiments/) | Numbered experiment scripts; descriptions and links to all runs are in the final results guide |
| [`cascaded-blended-rag/results/`](cascaded-blended-rag/results/) | Saved reports, protocols, and validation records |
| [`final_results/`](final_results/) | Paper, showcase, presentation, and release guide |
| [`docs/`](docs/README.md) | Manuscript source, reproducibility notes, findings, and project history |
| [`notebooks/`](notebooks/README.md) | Kaggle workflows and legacy notebooks |
| [`requirements/`](requirements/) | Dependency lists for analysis and benchmarking |

## Reproduce and inspect

To review the saved evidence and regenerate report figures, follow the [reproduction guide](docs/reproducibility/README.md). It describes the required inputs and how to restore ignored data. The final PDFs and reports are already included, so reading the results does not require running the benchmarks.

Run the repository audit and build a shareable archive with:

```powershell
python tools/check_repository.py
python tools/build_submission.py
```

The audit checks repository links, shareable files, and frozen source fingerprints. The archive is written to `artifacts/submission/`.

## Study scope

This is an engineering reproduction and extension using our Python retrieval setup. Its measurements support conclusions about the evaluated datasets and configurations. They do not by themselves show that the proposed cascade improves the original paper's system or preserves downstream answer quality.
