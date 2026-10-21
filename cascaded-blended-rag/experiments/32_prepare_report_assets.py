"""Export the completed indexed results as a figure and compact CSV.

Reads saved results only; performs no retrieval, training or timing.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, FuncFormatter

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT.parent / "docs/figures"


def export_speedups(reports):
    original_dirs = {"trec-covid": "paper_trec-covid", "squad": "paper_squad_one_pass"}
    names = ["static_rrf", "ridge11_r99", "ridge11_r95", "ridge11_r92", "ridge11_r90"]
    labels = ["Indexed\nfusion", "r99*", "r95", "r92", "r90"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    rows, summaries, source_hashes = [], [], {}
    for column, (dataset, report) in enumerate(reports):
        old_path = PROJECT / "results" / original_dirs[dataset] / "paper_dataset_benchmark.json"
        old = json.loads(old_path.read_text(encoding="utf-8"))
        for key in ("corpus_sha256", "evaluation_qrels_sha256", "frozen_router_sha256"):
            if old[key] != report[key]:
                raise ValueError("Cannot compare different assets: "+key)
        selected = report["selected_query_ids"]
        old_times = old["methods"]["static_rrf"]["timings_per_query_ms"]
        old_mean = sum(sum(old_times[q])/len(old_times[q]) for q in selected)/len(selected)
        source_hashes[dataset] = hashlib.sha256(old_path.read_bytes()).hexdigest()
        methods = report["methods"]
        combined = [1.0]+[old_mean/methods[name]["latency_ms"]["mean"] for name in names]
        fair = [methods[name]["mean_speedup_factor"] for name in names]
        retentions = [1.0]+[methods[name]["recall_retention"] for name in names]
        statuses = ["supported"]
        for name in names:
            m = methods[name]
            statuses.append("below" if m["recall_retention"] < .90 else
                            ("uncertain" if m["recall_retention_95_interval"][0] < .90 else "supported"))
        palette = {"supported": "#2459a6", "uncertain": "#a86108", "below": "#b53636"}
        for row_index, (speeds, ticks, retains, states) in enumerate((
            (combined, ["Original\nfusion"]+labels, retentions, statuses),
            (fair, labels, retentions[1:], statuses[1:]),
        )):
            ax = axes[row_index, column]
            ax.plot(range(len(speeds)), speeds, color="#8b8b8b", linewidth=1.3)
            for i, (speed, retention, status) in enumerate(zip(speeds, retains, states)):
                primary = ticks[i] == "r99*"
                color = palette[status]
                ax.scatter(i, speed, color=color, marker="*" if primary else "o", s=120 if primary else 40, zorder=3)
                ax.annotate(f"{speed:.2f}x\n{retention:.2%}", (i, speed),
                            xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.5, color=color)
            ax.set_xticks(range(len(ticks)), ticks)
            ax.set(xlim=(-.5, len(ticks)-.5), ylim=(0, max(speeds)*1.37))
            ax.axhline(1, color="#999999", linestyle="--", linewidth=.7)
            ax.grid(axis="y", alpha=.15)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}x"))
            title = "TREC-COVID: all 50 queries" if dataset == "trec-covid" else "SQuAD: fixed 256-question sample"
            ax.set_title(title, fontsize=11)
            ax.set_ylabel("Speedup vs original fusion" if row_index == 0 else "Speedup vs indexed fusion")
        heading = "TREC-COVID, all 50 queries" if dataset == "trec-covid" else "SQuAD, 256 of 10,570 questions"
        table = [f"## {heading}", "", "| Configuration | Recall retained | Mean ms | Speedup vs original fusion | Speedup vs indexed fusion |",
                 "|:--|--:|--:|--:|--:|", f"| Original fusion, same queries | 100.00% | {old_mean:.2f} | 1.00x | N/A |"]
        for name in names:
            m = methods[name]
            title = "Indexed fusion" if name == "static_rrf" else name.split("_")[-1]+(" (primary)" if name.endswith("r99") else " (secondary)")
            table.append(f"| {title} | {m['recall_retention']:.2%} | {m['latency_ms']['mean']:.2f} | {old_mean/m['latency_ms']['mean']:.2f}x | {m['mean_speedup_factor']:.2f}x |")
        table += ["", "**Standalone baselines, reported separately from cascade results:**", "",
                  "| Method | Recall retained | Mean ms | Speedup vs original fusion | Speedup vs indexed fusion |",
                  "|:--|--:|--:|--:|--:|"]
        for name in ("bm25", "dense", "splade"):
            m = methods[name]
            table.append(f"| {name.upper() if name != 'dense' else 'Dense'} | {m['recall_retention']:.2%} | {m['latency_ms']['mean']:.2f} | {old_mean/m['latency_ms']['mean']:.2f}x | {m['mean_speedup_factor']:.2f}x |")
        for name, m in methods.items():
            rows.append({"dataset": dataset, "queries": len(selected), "method": name,
                         "retention": m["recall_retention"], "mean_ms": m["latency_ms"]["mean"],
                         "original_fusion_same_queries_ms": old_mean,
                         "speedup_vs_original_fusion": old_mean/m["latency_ms"]["mean"],
                         "speedup_vs_indexed_fusion": m["mean_speedup_factor"],
                         "point_recall_floor_met": m["recall_retention"] >= .90,
                         "retention_interval_lower_at_least_90": m["recall_retention_95_interval"][0] >= .90})
        summaries.append("\n".join(table))
    fig.suptitle("Full measured speedup progression as the gate becomes more permissive", fontsize=14, y=.985)
    legend = [Line2D([0], [0], color=color, marker="o", linestyle="", label=label)
              for color, label in (("#2459a6", "Recall interval lower bound >=90%"),
                                   ("#a86108", "Point >=90%; interval crosses below"),
                                   ("#b53636", "Point recall below 90%"))]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .085), ncol=3, fontsize=8.5, frameon=False)
    fig.subplots_adjust(left=.075, right=.98, bottom=.20, top=.91, hspace=.38, wspace=.23)
    fig.text(.075, .057, "Top: indexing + routing vs the original implementation, across runs. Bottom: additional routing gain vs equally indexed fusion.", fontsize=9)
    fig.text(.075, .027, "Percentages: Recall@100 retained. r99*: preselected primary. rXX: training calibration setting. Points are measured configurations.", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"speedup_progression.{extension}", dpi=200)
    plt.close(fig)
    with (OUTPUT / "all_measured_speedups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    intro = """# All measured indexed speedups

The tables show every frozen cascade setting, including the fastest settings
that miss the 90% recall floor. The two speedup columns use different references:
original fusion measures indexing plus routing; indexed fusion measures the
additional conditional-routing gain. Standalone methods are separate baselines.

![Full measured speedup progression](figures/speedup_progression.png)

## Headline numbers and their limits

- TREC cascade maximum: **2.26x** vs indexed fusion (**36.96x** vs original
  fusion), retaining **84.53%** recall. This point misses the 90% floor.
- TREC cascade maximum with point retention above 90%: **1.64x** vs indexed
  fusion (**26.83x** vs original fusion), retaining **91.95%**. This is secondary
  r95; its recall interval is 86.88% to 95.76%, crossing below 90%.
- TREC conservative primary: **1.24x** vs indexed fusion (**20.21x** vs original
  fusion), retaining **96.68%**, with recall interval 93.58% to 98.75%.
- SQuAD cascade maximum: **1.59x** vs indexed fusion (**7.91x** vs original
  fusion on the same questions), retaining **99.22%** on the fixed sample.

These are measured means with prepared corpus indexes. Percent time saved is
`1 - 1/speedup`: 19.04% less time equals 1.24x speedup, not 19x. Cross-run
original/indexed ratios include run variability. Original TREC and indexed
TREC have three passes each; original SQuAD has one complete pass and indexed
SQuAD has three. SQuAD comparisons use exactly the same 256 questions.

"""
    tail = """

## BM25 scorer speedups with identical rankings

The TREC BM25 scorer itself changes from 1410.14 to 8.11 ms across the T4
pipeline runs: **173.78x**, preserving its old rankings and scores on all 50
queries. This reference is the old BM25 scorer, rather than fusion. Its
Recall@100 remains 71.38% of fusion, so scorer equivalence is not a 90%
fusion-recall result. The matched SQuAD BM25 scorer ratio is **113.74x**.

Separate local paired CPU benchmarks show **134.85x** scorer speedup on
TREC and **34.59x** on NFCorpus train, with identical full score vectors.
Those local scorer measurements have different hardware and timing scope.

## What can be written in the report

> With indexing and conditional execution combined, the TREC pipeline reached
> up to 26.83x observed speedup over our original full-fusion implementation
> while retaining 91.95% of its Recall@100 at a secondary operating point.
> Against equally indexed fusion, that point gives 1.64x speedup. Its recall
> interval crosses below 90%; the conservative primary retains 96.68% at
> 1.24x speedup against the optimized reference.

The absolute maximum TREC cascade point (2.26x vs indexed fusion) is also
reported, with its failed 90% floor. SPLADE and BM25 standalone speedups do
not become cascade claims. No original-paper ELSER or new-algorithm priority
claim follows from comparisons with our original Python implementation.

Regenerate locally from the workspace root; no Kaggle run or upload:

```powershell
& '.\\cascaded-blended-rag\\venv\\Scripts\\python.exe' '.\\cascaded-blended-rag\\experiments\\32_prepare_report_assets.py'
```
"""
    (PROJECT.parent / "docs/SPEEDUP_SUMMARY.md").write_text(intro+"\n\n".join(summaries)+tail, encoding="utf-8")
    (OUTPUT / "speedup_progression_sources.json").write_text(json.dumps({"original_report_sha256": source_hashes,
        "scope": "Ratios of saved same-query means; no new retrieval or timing."}, indent=2)+"\n", encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inputs = [
        ("trec-covid", PROJECT / "results/indexed_trec-covid_q0/indexed_cascade_benchmark.json"),
        ("squad", PROJECT / "results/indexed_squad_q256/indexed_cascade_benchmark.json"),
    ]
    reports = [(name, json.loads(path.read_text(encoding="utf-8"))) for name, path in inputs]
    with (OUTPUT / "indexed_quality_latency.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["dataset", "queries", "all_queries", "corpus_documents", "method", "primary",
                   "recall@100", "recall_retention", "ndcg@10", "map@10", "mean_ms", "mean_time_saved"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for dataset, report in reports:
            for name, method in report["methods"].items():
                writer.writerow({"dataset": dataset, "queries": report["queries"],
                                 "all_queries": report["all_available_queries"],
                                 "corpus_documents": report["corpus_documents"], "method": name,
                                 "primary": name == report["primary_outcome"]["policy"],
                                 "recall@100": method["metrics"]["recall@100"],
                                 "recall_retention": method["recall_retention"],
                                 "ndcg@10": method["metrics"]["ndcg@10"], "map@10": method["metrics"]["map@10"],
                                 "mean_ms": method["latency_ms"]["mean"], "mean_time_saved": method["mean_latency_reduction"]})
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    colors = {"static_rrf": "#454545", "bm25": "#7b4895", "dense": "#ac5817", "splade": "#15734b"}
    labels = {"static_rrf": "Fusion", "bm25": "BM25", "dense": "Dense", "splade": "SPLADE"}
    offsets = {
        "trec-covid": {"static_rrf": (6, -15), "bm25": (6, 7), "dense": (6, -13), "splade": (7, 6),
                       "ridge11_r90": (-34, -16), "ridge11_r92": (6, 7), "ridge11_r95": (6, -16), "ridge11_r99": (6, 8)},
        "squad": {"static_rrf": (5, -17), "bm25": (5, -16), "dense": (5, -16), "splade": (-30, -16),
                  "ridge11_r90": (-26, 10), "ridge11_r92": (-13, 12), "ridge11_r95": (-13, -22), "ridge11_r99": (3, 12)},
    }
    for ax, (dataset, report) in zip(axes, reports):
        methods = report["methods"]
        gates = [methods[f"ridge11_r{t}"] for t in (90, 92, 95, 99)]
        ax.plot([m["latency_ms"]["mean"] for m in gates], [m["recall_retention"] for m in gates],
                color="#2459a6", alpha=.65, linewidth=1, zorder=2)
        for name, method in methods.items():
            x, y = method["latency_ms"]["mean"], method["recall_retention"]
            lo, hi = method["recall_retention_95_interval"]
            gate = name.startswith("ridge11")
            primary = name == report["primary_outcome"]["policy"]
            color = "#2459a6" if gate else colors[name]
            ax.errorbar(x, y, yerr=[[max(0, y-lo)], [max(0, hi-y)]], fmt="none",
                        color=color, alpha=.4, capsize=2.5, linewidth=.8, zorder=1)
            ax.scatter(x, y, color=color, marker="*" if primary else ("o" if gate else "s"),
                       s=135 if primary else 35, zorder=3)
            label = (name.split("_")[-1]+("*" if primary else "")) if gate else labels[name]
            ax.annotate(label, (x, y), xytext=offsets[dataset][name], textcoords="offset points",
                        fontsize=9, color=color)
        ax.axhline(.90, color="#777777", linestyle="--", linewidth=.8)
        ax.grid(axis="y", alpha=.16)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("Mean retrieval latency (ms; lower is faster)")
        if dataset == "trec-covid":
            ax.set(xlim=(0, 110), ylim=(.59, 1.16), title="TREC-COVID: all 50 queries\n171,332 documents")
        else:
            ax.set(xlim=(0, 38), ylim=(.895, 1.035), title="SQuAD: 256 of 10,570 queries\n20,958 documents")
        ax.set_ylabel("Recall@100 retained relative to fusion")
    fig.suptitle("Indexed retrieval: quality and measured execution cost", fontsize=14, y=.98)
    fig.subplots_adjust(left=.075, right=.98, bottom=.23, top=.81, wspace=.27)
    fig.text(.075, .09, "r99* is the preselected primary. Dashed line: 90% floor. All methods use the same corpus per panel.", fontsize=9)
    fig.text(.075, .045, "Bars: 95% query-bootstrap intervals; three warmed T4 passes. SQuAD results cover the declared sample only.", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"indexed_quality_latency.{extension}", dpi=200)
    plt.close(fig)
    metadata = {"source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs},
                "scope": "Exports saved measurements only. No new inference, scoring, timing or policy selection."}
    (OUTPUT / "indexed_quality_latency_sources.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    export_speedups(reports)
    print("Saved PNG, SVG, CSV and source fingerprints:", OUTPUT)


if __name__ == "__main__":
    main()
