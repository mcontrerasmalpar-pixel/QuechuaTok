"""QuechuaTok: morphological PRPE tokenizer for Southern Quechua."""

from quechuatok.prpe import (
    load_suffixes,
    morpheme_boundary_accuracy,
    segment_prpe,
)
from quechuatok.tokenizer import PrpeTokenizer

__all__ = [
    "PrpeTokenizer",
    "load_suffixes",
    "morpheme_boundary_accuracy",
    "segment_prpe",
]
__version__ = "0.1.0"
