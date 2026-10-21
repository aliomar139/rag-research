"""Hindsight Dense/SPLADE routing illustration using saved TREC-COVID results.

This is diagnostic analysis of inspected labels. No router is trained or
evaluated. Summed standalone times are estimates, not measured router latency.
"""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    folder = PROJECT / "results/paper_trec-covid"
    source = folder / "paper_dataset_benchmark.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    recall = {}
    for line in (folder / "retrieval_first_pass.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        recall.setdefault(row["method"], {})[row["query_id"]] = row["recall@100"]
    ids = list(recall["splade"])
    loss_budget = .10*sum(recall["splade"].values())
    ordered = sorted(ids, key=lambda q: max(0, recall["splade"][q]-recall["dense"][q]))
    selected, loss = set(), 0.0
    for qid in ordered:
        clipped = max(0, recall["splade"][qid]-recall["dense"][qid])
        if loss+clipped > loss_budget:
            break
        loss += clipped
        selected.add(qid)

    def mean_time(name, qid):
        values = report["methods"][name]["timings_per_query_ms"][qid]
        return sum(values)/len(values)

    direct = sum(mean_time("dense" if q in selected else "splade", q) for q in ids)/len(ids)
    sequential = sum(mean_time("dense", q)+(0 if q in selected else mean_time("splade", q)) for q in ids)/len(ids)
    base = report["methods"]["splade"]["latency_ms"]["mean"]
    result = {
        "status": "HINDSIGHT DIAGNOSTIC ONLY; not a deployable or evaluated router",
        "dataset": "inspected TREC-COVID test", "queries": len(ids),
        "source_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection": "Sort queries by known clipped recall loss of substituting Dense for SPLADE; maximize substitution count within a 10% clipped-loss budget. Not a latency-optimal selection.",
        "dense_choices": len(selected), "splade_choices": len(ids)-len(selected),
        "hindsight_recall_retention_vs_splade": sum(recall["dense"][q] if q in selected else recall["splade"][q] for q in ids)/sum(recall["splade"].values()),
        "estimated_direct_choice_mean_ms": direct,
        "estimated_dense_then_conditional_splade_mean_ms": sequential,
        "measured_standalone_splade_mean_ms": base,
        "estimated_direct_mean_reduction": 1-direct/base,
        "limitations": [
            "Relevance labels choose every route retrospectively. This is not held-out evidence for a learned policy.",
            "Latency estimates sum separately measured standalone query times; classifier overhead, execution-order effects and a real routed run are absent.",
            "Only Recall@100 is constrained; nDCG/MAP and generated answers are not evaluated for this diagnostic.",
            "Any classifier needs separate training/validation and independent evaluation; this inspected test cannot validate its fitted decisions.",
        ],
    }
    output = PROJECT / "results/direct_routing_headroom_trec_covid.json"
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("SAVED:", output)


if __name__ == "__main__":
    main()
