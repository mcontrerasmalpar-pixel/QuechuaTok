"""Recompute PRPE MorphAcc on a MorphAcc gold file.

Default: the frozen 15-word paper set (must still print 83.33).
Pass --gold eval/morphacc_gold_500.json for the SQUOIA-treebank set.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from prpe import load_suffixes, morpheme_boundary_accuracy, segment_prpe

HERE = Path(__file__).resolve().parent
PAPER_GOLD = HERE / "morphacc_gold.json"


def _load(path: Path) -> dict:
    """Load a gold JSON; if it lists `parts`, concatenate those JSONL/JSON shards."""
    gold = json.loads(path.read_text(encoding="utf-8"))
    parts = gold.get("parts")
    if not parts:
        return gold
    items = []
    base = path.parent
    for rel in parts:
        p = base / rel
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".jsonl":
            items.extend(json.loads(line) for line in text.splitlines() if line.strip())
        else:
            chunk = json.loads(text)
            items.extend(chunk if isinstance(chunk, list) else chunk["items"])
    gold["items"] = items
    return gold


def score(gold: dict, suffixes: list[str], *, verbose: bool) -> list[float]:
    scores: list[float] = []
    by_stratum: dict[str, list[float]] = defaultdict(list)
    if verbose:
        print(f"{'word':<28} {'gold':<36} {'prpe':<36} {'acc'}")
        print("-" * 110)
    for item in gold["items"]:
        pred = segment_prpe(item["word"], suffixes)
        acc = morpheme_boundary_accuracy(pred, item["morphemes"])
        scores.append(acc)
        stratum = item.get("stratum", "")
        if stratum:
            by_stratum[stratum].append(acc)
        if verbose:
            print(
                f"{item['word']:<28} "
                f"{' | '.join(item['morphemes']):<36} "
                f"{' | '.join(pred):<36} "
                f"{acc:.2%}"
            )
    overall = round(sum(scores) / len(scores) * 100, 2) if scores else 0.0
    if verbose:
        print("-" * 110)
    print(f"PRPE MorphAcc%: {overall}  (n={len(scores)})")
    if by_stratum:
        print("Per stratum:")
        for name, vals in sorted(by_stratum.items(), key=lambda kv: kv[0]):
            pct = round(sum(vals) / len(vals) * 100, 2)
            print(f"  {name:<24} n={len(vals):3}  {pct:.2f}%")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=PAPER_GOLD,
        help="Gold JSON (default: eval/morphacc_gold.json, the frozen 15-word paper set)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the overall (and per-stratum) summary",
    )
    args = parser.parse_args()
    gold_path = args.gold if args.gold.is_absolute() else (Path.cwd() / args.gold)
    if not gold_path.exists():
        gold_path = HERE / args.gold.name
    gold = _load(gold_path)
    suffixes = load_suffixes()
    verbose = (not args.quiet) and len(gold.get("items", [])) <= 30
    score(gold, suffixes, verbose=verbose)
    if gold_path.resolve() == PAPER_GOLD.resolve() or gold.get("n_words") == 15:
        print("Paper reports 83.33% on this 15-word set.")


if __name__ == "__main__":
    main()
