# How to Explain the Project: Pitch and Defense Guide

> **Superseded defense notes:** Several performance and novelty claims below do not match the current evidence audit. Use `docs/results_ledger.md` for existing-run provenance and `docs/improvement_plan.md` for the corrected evaluation sequence. Rewrite these notes after the held-out run.
## Cascaded Adaptive Blended RAG

---

## 1. The 30-Second Elevator Pitch
*Use this when someone asks: "What is your thesis about?"*

> "IBM's Blended RAG paper proved that combining BM25, neural vectors, and SPLADE across document fields gives top-tier retrieval accuracy. But it runs all three retrievers for **100% of queries**.
>
> My research is **the first adaptive extension of Blended RAG**. I introduce a fast pre-retrieval cascade: if a query has an obvious keyword match, it returns results in 3 milliseconds and stops. If the query is ambiguous or complex, it escalates to full Blended RAG.
>
> This cuts average query latency by **35% to 50%** and avoids neural vector search on **40% to 60% of queries**, while preserving over 96% of full hybrid retrieval accuracy."

---

## 2. The 3-Step Presentation Flow
*Use this structure for a 2-minute slide or whiteboard explanation.*

### Step 1: The Base Paper (What Worked)
* Sawarkar et al. (IBM Research, IEEE MIPR 2024) introduced Blended RAG.
* Key finding: Blending dense vectors, learned sparse weights (SPLADE), and BM25 across metadata fields (title, abstract, text) beats single retrievers.
* Results: 68% F1 on SQuAD, 42% Exact Match on NQ, and 88.8% retrieval accuracy on NQ without fine-tuning.

### Step 2: The Core Problem (The Static Hybrid Penalty)
* Blended RAG runs all three models (FAISS, SPLADE, BM25) for every single query.
* Obvious keyword queries pay the exact same 30 to 80 ms latency penalty as complex multi-hop questions.
* Running GPU or neural vector passes on queries where BM25 is already conclusive wastes compute.

### Step 3: Our Solution (The Sequential Cascade)
* **Tier 0 (< 1 ms CPU):** Checks query length, MaxIDF, and rare token ratio.
* **Tier 1 (2 to 5 ms CPU):** Runs BM25. If the top score is high and the margin between rank 1 and rank 2 is wide, **exit immediately**. Skips 100% of neural compute.
* **Tier 2 (30 to 80 ms):** If confidence is low, escalate to full Blended RAG: Dense MiniLM + SPLADE + BM25 merged via 3-way Reciprocal Rank Fusion (RRF).
* **Metadata Preserved:** Retains Best Fields matching on titles and abstracts.

---

## 3. How to Answer the Hardest Questions

### Question 1: "Hasn't adaptive retrieval already been done?"
* **Direct Answer:** "Yes, adapting retrieval effort by difficulty is an established research direction. However, existing work adapts different knobs:
  * Adaptive-RAG (Jeong et al., 2024) routes between no retrieval, single-step, and multi-step retrieval. It does not do hybrid early exit.
  * DAT (Hsu & Tzeng, 2025) dynamically tunes sparse-dense weights (α). But it runs both retrievers on every query first, saving zero latency.
  * SAGE (Raza et al., 2026) adapts passage depth (top-k), not retriever paths.
  * Our work is specifically **the first adaptive systems extension of IBM's Blended RAG**, testing whether pre-retrieval lexical early exit works before running heavy multi-field hybrid stacks."

### Question 2: "What if BM25 gives high confidence on the wrong passage?"
* **Direct Answer:** "We do not rely on raw BM25 score alone. Early exit requires two simultaneous conditions:
  1. High top score: `s1 >= tau_high`
  2. Wide score gap: `s1 - s2 >= tau_gap`
  3. High word rarity: `MaxIDF >= tau_idf` to avoid high scores on generic words.
  If there is any ambiguity or close competition, the query immediately escalates to Tier 2."

### Question 3: "What if early exit degrades accuracy?"
* **Direct Answer:** "We calibrate our thresholds on a validation split using 5-fold cross-validation to keep recall drop under 2%. If early exit is too aggressive on specific datasets, our documented backup plan is:
  * Pivot 1: Dynamic alpha weighting for 3-way Blended RRF.
  * Pivot 2: Adaptive top-k depth truncation.
  Even if early exit only matches static fusion, mapping the exact boundary where lexical confidence holds is a solid systems contribution."

### Question 4: "Why SPLADE and not just BM25 + Dense?"
* **Direct Answer:** "Because Blended RAG explicitly showed that learned sparse expansion (ELSER/SPLADE) was their strongest single component. Omitting SPLADE would weaken the baseline. We keep full Blended RAG in Tier 2."

---

## 4. Quick Cheatsheet for Meetings

| Topic | Plain English Summary |
| :--- | :--- |
| **Base Paper** | Sawarkar et al. (IEEE MIPR 2024), IBM Research |
| **Our Core Claim** | First adaptive systems extension of Blended RAG |
| **Target Latency** | 35% to 50% lower average query latency |
| **Neural Calls Saved** | 40% to 60% of queries exit at BM25 |
| **Accuracy Target** | Keep >= 96% of static Blended RAG Recall@10 |
| **Primary Dataset** | BEIR SciFact (5,183 scientific papers, 300 test queries) |
| **Hardware Scope** | CPU and laptop friendly, fully reproducible |

