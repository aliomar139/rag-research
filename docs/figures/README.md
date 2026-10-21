# Report figures and numerical exports

- `speedup_progression.png` / `.svg`: every frozen gate, both speedup references,
  observed recall and flags for the 90% floor.
- `indexed_quality_latency.png` / `.svg`: all indexed methods, including the
  standalone baselines, with query-bootstrap recall intervals.
- `all_measured_speedups.csv`: all indexed methods and both speedup ratios.
- `indexed_quality_latency.csv`: quality and latency values behind the plots.
- `*_sources.json`: fingerprints of the reports used for each export.

Regenerate from the repository root:

```powershell
python cascaded-blended-rag/experiments/32_prepare_report_assets.py
```

Install `requirements/analysis.txt` first. This reads completed reports;
it does not perform inference, train a gate or measure new latency.
