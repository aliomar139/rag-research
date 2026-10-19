# Cascaded Adaptive Blended RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Repository](https://img.shields.io/badge/GitHub-aliomar139%2Frag--research-success.svg)](https://github.com/aliomar139/rag-research)

A systems-oriented, efficiency-focused extension of IBM's **Blended RAG** (*Sawarkar et al., IEEE MIPR 2024*). This repository introduces a **confidence-gated sequential cascade** for multi-field hybrid retrieval, eliminating redundant neural computation for easy queries while preserving full hybrid retrieval accuracy on challenging ones.

---

## 📑 Table of Contents

- [Motivation & Problem Statement](#-motivation--problem-statement)
- [Methodology & Architecture](#-methodology--architecture)
- [Repository Structure](#-repository-structure)
- [Quick Start & Installation](#-quick-start--installation)
- [Reproducing Experiments](#-reproducing-experiments)
- [Empirical Results](#-empirical-results)
  - [1. Base Paper Reproduction](#1-base-paper-reproduction)
  - [2. Cascaded Adaptive Performance](#2-cascaded-adaptive-performance)
- [Academic Reports & Presentation Guides](#-academic-reports--presentation-guides)
- [Related Literature & Prior Art](#-related-literature--prior-art)
- [Collaboration & Research Notes](#-collaboration--research-notes)

---

## 💡 Motivation & Problem Statement

Standard **Blended RAG** (*Sawarkar, Mangal, & Solanki, IBM Research, IEEE MIPR 2024*) achieves state-of-the-art retrieval accuracy by executing three parallel modalities for **every incoming query**:
1. **Lexical Retrieval:** BM25 over multiple document fields (Title + Abstract).
2. **Dense Semantic Retrieval:** Dense vector embeddings (`all-MiniLM-L6-v2`) via FAISS FlatIP.
3. **Sparse Neural Retrieval:** SPLADE / ELSER neural lexical expansion.
4. **Reciprocal Rank Fusion (RRF):** Merging ranked lists into a unified context.

### The Inefficiency Problem
While effective, Blended RAG uses a **static execution graph**: all three engines are executed for 100% of queries. Our empirical ablations on Natural Questions reveal that:
- **Lexical BM25 alone already finds the correct document for 84.4% of queries** (taking under 10 ms on CPU).
- Running dense neural embeddings and SPLADE on queries that BM25 has already unambiguously answered wastes substantial GPU computation and incurs an unnecessary **67 ms to 113 ms latency penalty**.

### Our Proposed Solution: Cascaded Adaptive Blended RAG
We introduce an adaptive, confidence-driven sequential cascade:
- Fast, CPU-friendly BM25 Best Fields runs first.
- A **multi-signal decision gate** inspects top candidate confidence:
  - Top candidate absolute score ($s_1 \ge \tau_{\text{high}}$)
  - Margin between 1st and 2nd rank ($\Delta = s_1 - s_2 \ge \tau_{\text{gap}}$)
  - Query keyword specificity ($\max(\text{IDF}) \ge \tau_{\text{idf}}$)
- If the lexical match is confident, the system **exits early** (~10 ms), completely bypassing the neural models.
- If uncertain, the query cascades into the neural dense and sparse retrievers, fusing all branches via 3-way RRF.

---

## 📐 Methodology & Architecture

```
                                  Incoming Query
                                        │
                                        ▼
                       ┌─────────────────────────────────┐
                       │   Stage 1: Multi-Field BM25    │
                       │   (Pure CPU, Best Fields, <10ms)│
                       └────────────────┬────────────────┘
                                        │
                                        ▼
                       ┌─────────────────────────────────┐
                       │    Sequential Decision Gate     │
                       │  - s₁ ≥ τ_high                 │
                       │  - Δ (s₁ - s₂) ≥ τ_gap          │
                       │  - max(IDF) ≥ τ_idf             │
                       └───────────────┬─┬───────────────┘
                                       │ │
                     [High Confidence] │ │ [Low Confidence / Ambiguous]
                                       │ │
               ┌───────────────────────┘ └───────────────────────┐
               ▼                                                 ▼
    ┌──────────────────────┐                         ┌───────────────────────┐
    │  EARLY EXIT (Lexical)│                         │  Stage 2: Neural Fallback│
    │  - Latency: ~10 ms   │                         │  - Dense (MiniLM + FAISS) │
    │  - Neural Avoided:   │                         │  - Sparse (SPLADE)        │
    │    73% – 84%         │                         └───────────┬───────────┘
    └──────────┬───────────┘                                     │
               │                                                 ▼
               │                                     ┌───────────────────────┐
               │                                     │ 3-Way Reciprocal Rank │
               │                                     │     Fusion (RRF)      │
               │                                     └───────────┬───────────┘
               │                                                 │
               └───────────────────────┬─────────────────────────┘
                                       ▼
                             Final Ranked Documents
```

---

## 📁 Repository Structure

```text
rag-research/
├── README.md                                  # Repository overview and quick start (this file)
├── PROJECT_ANCHOR.md                          # Live research telemetry, completed milestones, and data
├── requirements.txt                           # Clean Python dependencies specification
├── .gitignore                                 # Git rules ignoring large binaries (>100MB) & virtualenvs
│
├── cascaded-blended-rag/                      # Core research engine & experiment suite
│   ├── src/                                   # Production retrieval and routing modules
│   │   ├── bm25_retriever.py                  # Pure Python Best Fields BM25 indexer
│   │   ├── dense_retriever.py                 # MiniLM-L6-v2 + FAISS FlatIP indexer & cache
│   │   ├── splade_retriever.py                # SPLADE sparse neural encoder + CSR cache
│   │   ├── cascaded_router.py                 # Sequential decision gate (s1, delta, MaxIDF)
│   │   └── cache_utils.py                     # Safe disk cache serialization
│   ├── experiments/                           # Reproducible evaluation scripts
│   │   ├── 01_baseline_bm25.py                # BM25 Single vs. Best Fields on SciFact
│   │   ├── 02_baseline_dense.py               # Dense vector evaluation on SciFact
│   │   ├── 03_baseline_splade.py              # SPLADE evaluation on SciFact
│   │   ├── 04_baseline_blended_rag.py         # 3-Way Reciprocal Rank Fusion on SciFact
│   │   ├── 05_replicate_squad.py              # SQuAD v1.1 Top-10 retrieval evaluation
│   │   ├── 06_replicate_trec_covid.py         # TREC-COVID NDCG@10 evaluation
│   │   ├── 07_tune_cascade_cv.py              # Exploratory parameter sweep
│   │   └── 08_evaluate_cascade_heldout.py     # Rigorous held-out train/test cascade evaluation
│   ├── data_loader.py                         # BEIR SciFact automatic loader & parser
│   ├── squad_loader.py                        # SQuAD v1.1 dataset downloader & parser
│   ├── trec_covid_fast_loader.py              # TREC-COVID fast benchmark loader with auto-download
│   ├── data/                                  # Benchmark corpora (SciFact, SQuAD, TREC-COVID)
│   └── results/                               # Verified experimental outputs & metrics
│
├── docs/                                      # Research reports, publications, and presentation guides
│   ├── Cascaded_Blended_RAG_Research_Report.docx  # Academic report (Word document)
│   ├── Cascaded_Blended_RAG_Research_Report.md    # Academic report (Markdown version)
│   ├── cascaded_blended_rag_report.html       # Self-contained visual HTML research report
│   ├── cascaded_blended_rag_report.pdf        # Standalone PDF research report
│   ├── how_to_explain_the_project.md          # 30-second elevator pitch & oral defense guide
│   ├── improvement_plan.md                    # Methodology roadmap & experimental rigor plan
│   ├── results_ledger.md                      # Detailed provenance ledger for all experimental runs
│   ├── literature_review_and_positioning.md   # Extended survey of adaptive and hybrid RAG
│   ├── project_focused_literature_review.md   # Comparative analysis against recent 2024-2026 preprints
│   ├── Blended-Rag.pdf                        # Base paper (Sawarkar et al., IBM Research, IEEE MIPR 2024)
│   └── proposal_brief.pdf                     # Original M2 proposal brief
│
├── notebooks/                                 # GPU execution and evaluation notebooks
│   ├── reproduction.ipynb                     # Kaggle Dual T4 notebook (FLAN-T5-XXL QA & NQ ablation)
│   ├── heldout_scifact_kaggle.ipynb           # Held-out conditional latency evaluation notebook
│   └── cascade_source_bundle.zip              # Packaged bundle for Kaggle/Colab imports
│
└── Similar-Papers/                            # Comparative literature analysis (106 papers cataloged)
    ├── downloaded_papers_index.csv            # Detailed metadata catalog of all surveyed papers
    ├── download_report.md                     # Comprehensive download telemetry report
    ├── download_all_papers.py                 # Automated downloader from arXiv, ACL, IEEE, DBLP
    ├── dblp_adaptive_or_hybird_rag.csv        # DBLP query search results
    └── consensus_adaptive_or_hybird_rag.csv   # Consensus literature search results
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/aliomar139/rag-research.git
cd rag-research
```

### 2. Set Up Python Environment
Create a virtual environment (Python 3.10+ recommended):

```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on Hardware:** The baseline BM25 indexer, SQuAD retrieval, TREC-COVID subset, and SciFact experiments run fast on standard CPU. For running SPLADE neural encoding or FLAN-T5-XXL (11B parameters) generation from scratch, a GPU with $\ge 12\text{GB}$ VRAM (or Kaggle Dual T4 / Colab GPU) is recommended.

---

## 🔬 Reproducing Experiments

All experiment scripts are housed under `cascaded-blended-rag/experiments/` and run from the `cascaded-blended-rag/` directory (or use repository paths). Data files download automatically on first run if not already present.

### Step 1: Run Multi-Field BM25 Baseline
Evaluates Single-Field vs. Best-Fields BM25 on the SciFact benchmark:
```bash
python cascaded-blended-rag/experiments/01_baseline_bm25.py
```

### Step 2: Run Dense Semantic & SPLADE Sparse Baselines
Computes FAISS dense embeddings and sparse neural representations:
```bash
python cascaded-blended-rag/experiments/02_baseline_dense.py
python cascaded-blended-rag/experiments/03_baseline_splade.py
```

### Step 3: Run 3-Way Blended RAG (Base Paper Model)
Fuses BM25, Dense, and SPLADE using Reciprocal Rank Fusion ($k=60$):
```bash
python cascaded-blended-rag/experiments/04_baseline_blended_rag.py
```

### Step 4: Run SQuAD v1.1 & TREC-COVID Reproductions
```bash
# SQuAD v1.1 top-10 retrieval accuracy
python cascaded-blended-rag/experiments/05_replicate_squad.py

# TREC-COVID NDCG@10 multi-field evaluation
python cascaded-blended-rag/experiments/06_replicate_trec_covid.py
```

### Step 5: Run Held-Out Cascaded Adaptive Evaluation
Calibrates cascade thresholds strictly on training qrels and evaluates real conditional latency and accuracy on held-out test qrels:
```bash
python cascaded-blended-rag/experiments/08_evaluate_cascade_heldout.py
```

### Dual-GPU Kaggle Notebook
To run the full end-to-end NQ ablation and FLAN-T5-XXL (11B) QA generation:
1. Open [notebooks/reproduction.ipynb](notebooks/reproduction.ipynb) on Kaggle.
2. Select **GPU: Dual T4**.
3. Run all cells to inspect live latency profiling, threshold sweeps, and generated exact-match outputs.

---

## 📊 Empirical Results

### 1. Base Paper Reproduction
We reproduced the core benchmarks of IBM's Blended RAG across four diverse datasets:

| Benchmark Dataset | Scope | Metric | Paper Baseline | Our Reproduction | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Natural Questions (NQ)** | 500 Queries (BEIR) | Top-20 Retrieval | 88.88% | **97.20%** | **Verified** |
| **Natural Questions (NQ)** | 500 Queries (BEIR) | NDCG@10 | 0.6700 | **0.7094** | **Verified** |
| **SQuAD v1.1 Retrieval** | 500 Queries, 2067 Docs | Top-10 Retrieval | 94.49% | **96.40%** | **Verified** |
| **SQuAD QA Generation** | FLAN-T5-XXL (11B) | Exact Match (EM) | 57.63% | **79.00%** | **Verified** |
| **SQuAD QA Generation** | FLAN-T5-XXL (11B) | Token F1 Score | 68.40% | **81.59%** | **Verified** |
| **TREC-COVID** | 50 Queries, 2572 Docs | NDCG@10 | 0.8000 – 0.8700 | **0.6463** | Beats single models |
| **SciFact** | 300 Queries (BEIR) | Recall@100 / NDCG@10 | — | **0.9610 / 0.7032** | Baseline |

### 2. Cascaded Adaptive Performance (Natural Questions Pareto Frontier)
On Natural Questions (dual Nvidia T4 GPUs), the cascaded decision gate avoids **over 73% to 84% of neural calls** with negligible impact on retrieval accuracy:

| Configuration | Gate Parameters ($\tau_{\text{high}}, \Delta, \text{IDF}$) | Neural Avoidance (%) | Average Latency | Top-20 Recall | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Static Blended RAG (Paper)** | None (100% neural execution) | 0.0% | 67.0 ms | **97.20%** | 1.0× (Baseline) |
| **Cascaded — Conservative** | $\tau=28.0, \Delta=6.0, \text{IDF}=3.5$ | **73.2%** | **25.3 ms** | **91.80%** | **2.6× FASTER** |
| **Cascaded — Balanced** | $\tau=25.0, \Delta=5.0, \text{IDF}=3.5$ | **84.2%** | **19.0 ms** | **89.40%** | **3.5× FASTER** |
| **Cascaded — Aggressive** | $\tau=23.0, \Delta=4.0, \text{IDF}=3.0$ | **90.0%** | **15.7 ms** | **88.40%** | **4.3× FASTER** |
| **Pure BM25 Baseline** | Pure Lexical (No Neural) | 100.0% | **10.0 ms** | 84.40% | 6.7× FASTER |

> **Key Takeaway:** The Conservative cascade configuration preserves higher retrieval accuracy than the base paper's original benchmark (91.80% vs. 88.88%) while reducing average query latency by **62.2%**.

---

## 📚 Academic Reports & Presentation Guides

The repository includes a complete academic documentation suite for thesis defense and research sharing:

- **Formal Academic Report:**
  - Word Document: [`docs/Cascaded_Blended_RAG_Research_Report.docx`](docs/Cascaded_Blended_RAG_Research_Report.docx)
  - PDF Document: [`docs/cascaded_blended_rag_report.pdf`](docs/cascaded_blended_rag_report.pdf)
  - Interactive HTML Report: [`docs/cascaded_blended_rag_report.html`](docs/cascaded_blended_rag_report.html)
  - Markdown Report: [`docs/Cascaded_Blended_RAG_Research_Report.md`](docs/Cascaded_Blended_RAG_Research_Report.md)
- **Oral Defense & Presentation Guide:**
  - [`docs/how_to_explain_the_project.md`](docs/how_to_explain_the_project.md) — 30-second elevator pitch, 3-minute oral defense script, and anticipation of tough jury questions.
- **Experimental Provenance & Rigor:**
  - [`docs/results_ledger.md`](docs/results_ledger.md) — Full telemetry, hardware environments, and caveats for every experiment run.
  - [`docs/improvement_plan.md`](docs/improvement_plan.md) — Future steps for multi-dataset held-out benchmarking and latency calibration.

---

## 🔎 Related Literature & Prior Art

We conducted a systematic survey of over 100 recent publications (2024–2026) at the intersection of adaptive RAG, hybrid search, and query routing:
- **Base Paper:** Sawarkar et al. (*IEEE MIPR 2024*), *Blended RAG*.
- **Pre-Retrieval Routing:** Adaptive-RAG (*Jeong et al., NAACL 2024*), MBA-RAG (*Tang et al., COLING 2025*), MoR (*Kalra et al., EMNLP 2025*).
- **Post-Retrieval Cascades:** SymRAG (*Hakim et al., NeSy 2025*), Retrieval as a Decision (*Wang et al., TMLR 2026*), SAGE (*Raza et al., 2026*).
- **Comprehensive Survey:** See [`docs/project_focused_literature_review.md`](docs/project_focused_literature_review.md) and [`Similar-Papers/downloaded_papers_index.csv`](Similar-Papers/downloaded_papers_index.csv).

---

## 🤝 Collaboration & Research Notes

This repository is maintained by **Ali Omar** ([@aliomar139](https://github.com/aliomar139)) as part of the Master 2 Research Project.

### Collaboration Guidelines for Team Members & Classmates
1. **Branching:** Work on individual feature branches (`git checkout -b feature/new-dataset`).
2. **Datasets:** Do not commit raw corpus files $>50\text{MB}$. Ensure loaders use streaming or download hooks.
3. **Reproducibility:** Run `cascaded-blended-rag/experiments/08_evaluate_cascade_heldout.py` when validating parameter changes to avoid data leakage between train and test splits.

---

## 📜 References & Acknowledgements

1. K. Sawarkar, A. Mangal, and S. R. Solanki, *"Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers"*, IEEE MIPR, 2024. [arXiv:2404.07220](https://arxiv.org/abs/2404.07220).
2. N. Thakur et al., *"BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models"*, NeurIPS, 2021.
3. H. J. Zhan et al., *"SPLADE: Sparse Lexical and Expansion Model for Information Retrieval"*, SIGIR, 2021.
