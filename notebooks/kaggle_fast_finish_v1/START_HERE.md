# Finish the existing full-query pass and run the short indexed follow-up

> **Completed and verified, 21 October 2026:** both exports have been received.
> The indexed TREC export is also received and verified. These instructions
> reproduce completed runs. Current next step: use
> [the verified findings](../../docs/findings/TREC_COVID_INDEXED_FINDINGS.md) in the
> write-up. No new Kaggle run or upload is required for the reported results.

The original run has completed all 10,570 questions once and is partway through
the second timing pass. Completing all three slow passes is no longer needed
for the next engineering decision. This utility preserves the first pass and
supplemental later records, then scores without running any retriever.

## Upload

```text
C:\Users\user\Desktop\rag-research-alone\notebooks\kaggle_fast_finish_v1\fast_finish_indexed_20261021_d4a9_v1.zip
```

New private Kaggle dataset title:

```text
fast-finish-indexed-20261021-d4a9-v1
```

Use this combined bundle instead of the earlier indexed-only bundle for this
shortened-run workflow. Keep the original paper-dataset source input attached.

## Steps

1. Click Stop on the currently running cell. Keep the notebook session and
   working files; do not reset/delete the working directory.
2. Upload the ZIP above as a private Kaggle dataset and attach it with Add Input.
3. Paste `PASTE_CELL.py` into a new cell with `ACTION = "finalize_original"`.
   Run it. It stops only a lingering subprocess executing the exact original
   benchmark script, reads the checkpoint and requires all 10,570 first-pass
   queries. It performs scoring/export only, with no neural inference or
   document encoding. Allow roughly 2–10 minutes; not measured end to end here.
4. Download `artifacts/downloads/squad_one_pass_outputs.zip` to the local project root. If the
   original full run already completed, it returns that full result instead.
5. After the download, change only the first line to
   `ACTION = "indexed_followup"` and rerun the same cell. This uses existing
   corpus indexes, all eight methods, three passes on 256 fixed-hash questions
   and the full corpus. Download `artifacts/downloads/squad_indexed_cascade_outputs.zip` afterward.

The indexed follow-up is estimated at 5–20 minutes with existing caches.
Missing/mismatched caches cause a clear stop rather than expensive re-encoding.

## Evidence handling

- First-pass quality covers all 10,570 questions. Headline old-backend latency
  uses one complete timing pass, not three. The original protocol file is
  preserved and a runtime amendment records this change before scoring grades.
- Later complete query records are exported to `additional_repetitions.jsonl`
  as supplemental data. They are not unevenly pooled into headline means.
  An incomplete final checkpoint line is ignored and recorded explicitly.
- Query-bootstrap intervals from one pass do not establish repeated-run
  timing stability. The original three-pass run must not be claimed completed.
- Original checkpoint files and neural corpus caches remain in place.
  Scored one-pass output is in a separate `paper_squad_one_pass` directory.
- The indexed follow-up retains the conservative r99 policy as primary against
  a 90% floor, reports all four policies and uses equally optimized baselines.
  Its quality/timing metrics cover its 256-query sample, not all questions.

The first-pass finalizer reuses the exact checksum-verified original runner's
scoring/export section. The finalizer and package have syntax/archive checks;
the user's live checkpoint has not been accessed or executed locally.
