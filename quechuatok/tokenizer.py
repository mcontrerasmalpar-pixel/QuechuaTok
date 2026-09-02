"""Minimal tokenizer-style wrapper around PRPE. No Hugging Face dependency."""
from __future__ import annotations

from quechuatok.prpe import load_suffixes, segment_prpe


class PrpeTokenizer:
    """Greedy morphological tokenizer for Southern Quechua.

    Same algorithm as ``eval/prpe.py``. Not a Hugging Face
    ``PreTrainedTokenizer``; ``tokenize`` is the public API.
    """

    def __init__(self) -> None:
        self.suffixes = load_suffixes()

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for word in text.split():
            tokens.extend(segment_prpe(word, self.suffixes))
        return tokens

    def __call__(self, text: str) -> dict[str, list[str]]:
        return {"tokens": self.tokenize(text)}
