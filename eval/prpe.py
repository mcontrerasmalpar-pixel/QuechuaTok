"""PRPE morphological segmenter for Southern Quechua (QuechuaTok)."""
from __future__ import annotations

import json
from pathlib import Path

_SUFFIXES_PATH = Path(__file__).with_name("prpe_suffixes.json")


def load_suffixes(path: Path | None = None) -> list[str]:
    data = json.loads((path or _SUFFIXES_PATH).read_text(encoding="utf-8"))
    return sorted(data["suffixes"], key=len, reverse=True)


def segment_prpe(word: str, suffixes: list[str] | None = None) -> list[str]:
    """Greedy longest-suffix strip. Matches QuechuaTok_v5_final.ipynb."""
    suffixes = suffixes if suffixes is not None else load_suffixes()
    if len(word) <= 3:
        return [word]
    segments: list[str] = []
    remaining = word
    while len(remaining) > 3:
        found = False
        for suf in suffixes:
            if remaining.endswith(suf) and len(remaining) - len(suf) >= 3:
                segments.insert(0, suf)
                remaining = remaining[: -len(suf)]
                found = True
                break
        if not found:
            break
    segments.insert(0, remaining)
    return segments


def morpheme_boundary_accuracy(pred_segs: list[str], silver_morphemes: list[str]) -> float:
    pred_set = set(pred_segs)
    silver_set = set(silver_morphemes)
    if not silver_set:
        return 0.0
    return round(len(pred_set & silver_set) / len(silver_set), 4)
