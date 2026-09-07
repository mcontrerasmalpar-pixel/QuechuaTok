"""Offline smoke: PRPE HF vocab + BertForMaskedLM forward (skip without mlm extras)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import BertConfig, BertForMaskedLM

from quechuatok.hf_tokenizer import PrpeHfTokenizer_from_texts, build_vocab, prpe_pieces


def test_prpe_pieces_basic():
    pieces = prpe_pieces("purisqanchikmanta")
    assert pieces[0] == "puri"
    assert "manta" in pieces


def test_build_vocab_and_mask_forward():
    texts = [
        "wasipi rimani",
        "purisqanchikmanta hamuni",
        "wasiykikunapiqa kachkan",
        "munakuwarqanki nuqa",
    ]
    vocab = build_vocab(texts, max_size=500)
    assert "[MASK]" in vocab
    tok = PrpeHfTokenizer_from_texts(texts, max_size=500)
    enc = tok("wasipi rimani", return_tensors="pt")
    assert enc["input_ids"].ndim == 2

    config = BertConfig(
        vocab_size=tok.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
        pad_token_id=tok.pad_token_id,
        type_vocab_size=1,
    )
    model = BertForMaskedLM(config)
    model.eval()
    with torch.no_grad():
        out = model(**enc)
    assert out.logits.shape[-1] == tok.vocab_size
