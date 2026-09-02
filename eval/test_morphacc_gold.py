"""Sanity checks for the frozen 15-word gold and MorphAcc-500."""
from __future__ import annotations

import json
from pathlib import Path

from prpe import load_suffixes, morpheme_boundary_accuracy, segment_prpe

HERE = Path(__file__).resolve().parent


def _load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def test_paper_15_frozen() -> None:
    gold = _load("morphacc_gold.json")
    assert gold["n_words"] == 15
    assert len(gold["items"]) == 15
    suffixes = load_suffixes()
    scores = [
        morpheme_boundary_accuracy(segment_prpe(it["word"], suffixes), it["morphemes"])
        for it in gold["items"]
    ]
    overall = round(sum(scores) / len(scores) * 100, 2)
    assert overall == 83.33, overall


def test_gold_500_shape() -> None:
    gold15 = _load("morphacc_gold.json")
    gold500 = _load("morphacc_gold_500.json")
    assert gold500["n_words"] == 500
    items = gold500["items"]
    assert len(items) == 500
    words = [it["word"] for it in items]
    assert len(set(words)) == 500
    for it in items:
        assert "".join(it["morphemes"]) == it["word"], it["word"]
    for a, b in zip(gold15["items"], items[:15], strict=True):
        assert a["word"] == b["word"]
        assert a["morphemes"] == b["morphemes"]
        assert a["squoia"] == b["squoia"]


if __name__ == "__main__":
    test_paper_15_frozen()
    test_gold_500_shape()
    print("ok")
