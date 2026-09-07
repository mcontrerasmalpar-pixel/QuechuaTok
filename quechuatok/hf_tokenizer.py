"""Hugging Face-compatible PRPE tokenizer for QuechuaBERT (optional deps)."""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from quechuatok.prpe import load_suffixes, segment_prpe

SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


def _require_transformers():
    try:
        from transformers import PreTrainedTokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "quechuatok HF tokenizer needs transformers. "
            'Install with: pip install -e ".[mlm]"'
        ) from e
    return PreTrainedTokenizer


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip().lower())


def prpe_pieces(text: str, suffixes: list[str] | None = None) -> list[str]:
    suffixes = suffixes if suffixes is not None else load_suffixes()
    pieces: list[str] = []
    for word in normalize_text(text).split():
        pieces.extend(segment_prpe(word, suffixes))
    return pieces


def build_vocab(
    texts: Iterable[str],
    *,
    min_freq: int = 1,
    max_size: int | None = 16000,
) -> dict[str, int]:
    """Build a piece vocab from PRPE segments over a text corpus."""
    suffixes = load_suffixes()
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(prpe_pieces(text, suffixes))
    vocab: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for piece, freq in counts.most_common():
        if freq < min_freq:
            continue
        if piece in vocab:
            continue
        if max_size is not None and len(vocab) >= max_size:
            break
        vocab[piece] = len(vocab)
    return vocab


def PrpeHfTokenizer_from_vocab(vocab: dict[str, int], **kwargs):
    """Factory so callers need not import transformers until use."""
    PreTrainedTokenizer = _require_transformers()

    class PrpeHfTokenizer(PreTrainedTokenizer):
        """Word-level PreTrainedTokenizer over PRPE morpheme pieces."""

        model_input_names = ["input_ids", "attention_mask"]

        def __init__(self, vocab_map: dict[str, int], **kw):
            self._vocab = dict(vocab_map)
            self._id_to_token = {i: t for t, i in self._vocab.items()}
            self.prpe_suffixes = load_suffixes()
            kw.setdefault("pad_token", "[PAD]")
            kw.setdefault("unk_token", "[UNK]")
            kw.setdefault("cls_token", "[CLS]")
            kw.setdefault("sep_token", "[SEP]")
            kw.setdefault("mask_token", "[MASK]")
            super().__init__(**kw)

        @property
        def vocab_size(self) -> int:
            return len(self._vocab)

        def get_vocab(self) -> dict[str, int]:
            return dict(self._vocab)

        def _tokenize(self, text: str) -> list[str]:
            return prpe_pieces(text, self.prpe_suffixes)

        def _convert_token_to_id(self, token: str) -> int:
            return self._vocab.get(token, self._vocab[self.unk_token])

        def _convert_id_to_token(self, index: int) -> str:
            return self._id_to_token.get(index, self.unk_token)

        def convert_tokens_to_string(self, tokens: list[str]) -> str:
            return " ".join(tokens)

        def build_inputs_with_special_tokens(
            self, token_ids_0: list[int], token_ids_1: list[int] | None = None
        ) -> list[int]:
            cls = [self.cls_token_id]
            sep = [self.sep_token_id]
            if token_ids_1 is None:
                return cls + token_ids_0 + sep
            return cls + token_ids_0 + sep + token_ids_1 + sep

        def get_special_tokens_mask(
            self,
            token_ids_0: list[int],
            token_ids_1: list[int] | None = None,
            already_has_special_tokens: bool = False,
        ) -> list[int]:
            if already_has_special_tokens:
                return super().get_special_tokens_mask(
                    token_ids_0=token_ids_0,
                    token_ids_1=token_ids_1,
                    already_has_special_tokens=True,
                )
            if token_ids_1 is None:
                return [1] + ([0] * len(token_ids_0)) + [1]
            return (
                [1]
                + ([0] * len(token_ids_0))
                + [1]
                + ([0] * len(token_ids_1))
                + [1]
            )

        def save_vocabulary(
            self, save_directory: str, filename_prefix: str | None = None
        ) -> tuple[str]:
            path = Path(save_directory)
            path.mkdir(parents=True, exist_ok=True)
            prefix = f"{filename_prefix}-" if filename_prefix else ""
            vocab_file = path / f"{prefix}vocab.json"
            vocab_file.write_text(
                json.dumps(self._vocab, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return (str(vocab_file),)

    return PrpeHfTokenizer(vocab_map=vocab, **kwargs)


def PrpeHfTokenizer_from_texts(
    texts: Sequence[str],
    *,
    min_freq: int = 1,
    max_size: int | None = 16000,
    **kwargs,
):
    vocab = build_vocab(texts, min_freq=min_freq, max_size=max_size)
    return PrpeHfTokenizer_from_vocab(vocab, **kwargs)
