> **Historical document.** This file predates the consolidated submission and may contain superseded claims or commands. Use [the current manuscript](../../FINAL_MANUSCRIPT.md) for conclusions.

> **Priority revised on 21 October 2026 after both indexed measurements.**
> Primary r99 saves 19.04% mean time and retains 96.68% recall on full-query
> TREC; it saves 4.19% with zero observed recall loss on sampled SQuAD.
> Strong constant baselines remain: SPLADE on TREC and BM25 on SQuAD.
> Next is the bounded engineering write-up; no further GPU run is required
> to report these results. Read [FINAL_RESULTS.md](../../FINAL_RESULTS.md).
> The direct selector below remains an unimplemented research direction.

# Concrete redesign after the TREC-COVID result

## Current action

Both SQuAD exports and indexed TREC are received and verified. Use the
completed evidence in the write-up. The execution optimization works and
primary mean savings are measured against a fair optimized reference; superior
routing against the relevant standalone choices remains unestablished.
The remaining sections preserve the earlier redesign rationale, rather than
instructing a new training or GPU run. Further method work is optional and
would require its own protocol and evidence.

## Two problems require different fixes

The current first stage calls rank_bm25 scoring over all documents for each
query token. On TREC-COVID, BM25 alone takes 1410.14 ms versus fusion's
1493.23 ms. A gate after this stage cannot avoid that paid work. A postings
index is the engineering remedy. Any revised timing comparison must optimize
the static baseline too. Lucene/Pyserini is an existing indexed implementation,
but its analysis/scoring defaults can change rankings; it cannot silently
replace the frozen scorer and be treated as an identical result.

Separately, the first retriever is not consistently the best quality/time
choice. Dense retains 99.77% of fusion recall on NFCorpus dev and 109.65% on
SCIDOCS. SPLADE retains 104.69% on TREC-COVID. These inspected results motivate
adaptive branch selection; they cannot train and independently validate that
selector on the same queries.

## Candidate research direction: select the branch before retrieval

Use cheap query features to choose Dense or SPLADE directly. A more expensive
fusion action should be added only if training/validation evidence shows
useful gains. Do not always run Dense before deciding whether to run SPLADE:
that pays Dense's cost even on queries routed to SPLADE.

Train the decision with both measured cost and retrieval loss, using only
training/validation labels. Compare it with Dense, SPLADE, an inexpensive
constant selector and optimized fusion. Evaluate a preselected policy on
independent queries, with Recall@100 and ranking metrics. Recall retention
should also be shown relative to the strongest relevant single baseline,
so a weak fusion reference cannot manufacture success.

This direction has prior art. Its novelty is not established merely by
changing the first retriever or using a classifier. The precise objective,
features, calibration and empirical result need comparison with existing
strategy-selection methods.

## Saved-data illustration, not a claimed result

Using the inspected TREC-COVID labels to choose routes retrospectively, a
simple clipped-loss budget assigns 30 queries to Dense and 20 to SPLADE.
It retains 93.27% of SPLADE Recall@100. Summing those queries' standalone
times estimates 39.19 ms versus SPLADE's measured 52.98 ms, about 26% lower.
Running Dense first and adding SPLADE on the same 20 queries instead estimates
51.41 ms. Both are arithmetic estimates without selector overhead or actual
routed execution. No trained model has achieved this result. nDCG/MAP for this
diagnostic have not been scored.

This illustration shows an opportunity to investigate a direct selector,
not proof that a learnable selector generalizes or a new contribution exists.
It must not enter a paper's measured-results table as a system result.

Regenerate the saved-data analysis locally without a GPU:

```powershell
& '.\cascaded-blended-rag\venv\Scripts\python.exe' '.\cascaded-blended-rag\experiments\23_analyze_direct_routing_headroom.py'
```

## Work order

1. Receive the running SQuAD result and preserve all current evidence.
2. Implement an indexed BM25 version and establish score/ranking compatibility
   before timing the revised baseline. This is an engineering fix.
3. Audit available training examples for Dense/SPLADE labels and cheap query
   features. Define the selector, quality/cost objective and split protocol
   before spending GPU time on missing training labels.
4. Train and validate a direct selector. Reject it if it cannot improve the
   trade-off against strong constant baselines on validation.
5. Only if validation supports it, package a frozen measured confirmation on
   independent queries. Report failures and previous unfavorable datasets.

Sources: [Pyserini indexed retrieval](https://github.com/castorini/pyserini/blob/master/docs/usage-search.md),
[prior dense/sparse strategy selection](https://arxiv.org/abs/2109.10739).
