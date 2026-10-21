"""Restore two ignored root inputs required by historical train-only scripts."""
import shutil

from _repository import ROOT, sha256


def main():
    for name in ("nfcorpus_train_diagnostics_outputs.zip", "nfcorpus_corpus_for_features.jsonl"):
        source, target = ROOT / "artifacts/downloads" / name, ROOT / name
        if not source.is_file():
            raise FileNotFoundError(f"Receive the original artifact first: {source}")
        if target.exists():
            if sha256(source) != sha256(target):
                raise ValueError(f"Different existing input retained: {target}")
        else:
            shutil.copyfile(source, target)
        print("Historical compatibility input:", target)
    print("Both files are ignored. No training, inference or dataset download was performed.")


if __name__ == "__main__":
    main()
