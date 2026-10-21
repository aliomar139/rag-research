# Retrieval modules

| Module | Role |
|:--|:--|
| `indexed_bm25.py` | Compatible postings-based Best Fields scoring |
| `bm25_retriever.py`, `dense_retriever.py`, `splade_retriever.py` | Original retrieval primitives preserved for recorded runs |
| `retention_tradeoff.py`, `router_diagnostics.py` | Portable Ridge decision and train calibration utilities |
| `tradeoff_analysis.py` | Timing summaries, paired intervals and random-exit reference |
| `paper_datasets.py`, `scidocs_download.py`, `query_self_exclusion.py`, `cache_utils.py` | Dataset, self-match and cache support |

`bm25_coverage_features.py` supports the development feature ablation.
`cascaded_router.py` is the original heuristic, not the final Ridge policy.
These files remain at fixed paths for imports and frozen provenance.
