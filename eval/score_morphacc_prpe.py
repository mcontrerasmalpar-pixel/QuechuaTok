"""Recompute PRPE MorphAcc on the 15-word gold set. No trained SentencePiece models required."""
from __future__ import annotations

import json
from pathlib import Path

from prpe import load_suffixes, morpheme_boundary_accuracy, segment_prpe

HERE = Path(__file__).resolve().parent


def main() -> None:
    gold = json.loads((HERE / "morphacc_gold.json").read_text(encoding="utf-8"))
    suffixes = load_suffixes()
    scores = []
    print(f"{'word':<12} {'gold':<28} {'prpe':<28} {'acc'}")
    print("-" * 80)
    for item in gold["items"]:
        pred = segment_prpe(item["word"], suffixes)
        acc = morpheme_boundary_accuracy(pred, item["morphemes"])
        scores.append(acc)
        print(
            f"{item['word']:<12} "
            f"{' | '.join(item['morphemes']):<28} "
            f"{' | '.join(pred):<28} "
            f"{acc:.2%}"
        )
    overall = round(sum(scores) / len(scores) * 100, 2)
    print("-" * 80)
    print(f"PRPE MorphAcc%: {overall}")
    print("Paper reports 83.33% on this 15-word set.")


if __name__ == "__main__":
    main()
