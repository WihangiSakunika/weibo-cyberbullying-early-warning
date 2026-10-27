"""PLANNED, NOT YET RUN: Chinese transformer baseline.

This module is intentionally a documented placeholder. It will fine-tune a
Hugging Face Chinese encoder (e.g. `hfl/chinese-bert-wwm-ext`) on the same
incident-level split used by the TF-IDF baselines. It requires a GPU and
`pip install torch transformers` (uncomment those lines in requirements.txt).

Status: NOT executed in this repository. No metrics are reported for it.
Running (once implemented / on a GPU machine):

    python src/train_transformer.py --model hfl/chinese-bert-wwm-ext --epochs 3
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-bert-wwm-ext")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    raise NotImplementedError(
        "Transformer baseline is planned but not yet implemented in this repo. "
        "The TF-IDF baselines (src/train.py) are the currently validated pipeline."
    )


if __name__ == "__main__":
    main()
