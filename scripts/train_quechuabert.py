#!/usr/bin/env python3
"""Smoke / full MLM train for QuechuaBERT with PRPE tokenizer (Refs #8)."""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

QUECHUA_SUFFIX_RE = re.compile(
    r"(kuna|pi|manta|wan|paq|ta|qa|pas|ni|nki|nchik|nku|mi|si|cha|rqa|sqa|nqa|spa|na|yki)$"
)


def normalize_line(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip().lower())
    text = re.sub(r"https?://\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def keep_quechua_line(line: str) -> bool:
    toks = line.split()
    if len(toks) < 2:
        return False
    hits = sum(1 for t in toks if QUECHUA_SUFFIX_RE.search(t))
    return (hits / len(toks)) >= 0.1


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.corpus_file:
        raw = Path(args.corpus_file).read_text(encoding="utf-8").splitlines()
        lines = [normalize_line(x) for x in raw]
        return [x for x in lines if x and keep_quechua_line(x)][: args.max_samples]

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    out: list[str] = []
    for row in ds:
        text = row.get("text") or row.get("content") or ""
        line = normalize_line(str(text))
        if line and keep_quechua_line(line):
            out.append(line)
        if len(out) >= args.max_samples:
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-file", type=str, default=None, help="Local UTF-8 lines (smoke)")
    p.add_argument("--dataset", default="Llamacha/monolingual-quechua-iic")
    p.add_argument("--split", default="train")
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--intermediate-size", type=int, default=512)
    p.add_argument("--output-dir", type=str, default="models/quechuabert-smoke")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", type=str, default=None)
    args = p.parse_args()

    import torch
    from transformers import (
        BertConfig,
        BertForMaskedLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    from quechuatok.hf_tokenizer import PrpeHfTokenizer_from_texts

    texts = load_texts(args)
    if not texts:
        raise SystemExit("No training lines after filter; try --corpus-file or raise --max-samples")

    tokenizer = PrpeHfTokenizer_from_texts(texts, max_size=8000)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_length,
        pad_token_id=tokenizer.pad_token_id,
        type_vocab_size=1,
    )
    model = BertForMaskedLM(config)

    def encode(batch_texts: list[str]):
        return tokenizer(
            batch_texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_attention_mask=True,
        )

    class LineDataset(torch.utils.data.Dataset):
        def __init__(self, lines: list[str]):
            self.lines = lines

        def __len__(self) -> int:
            return len(self.lines)

        def __getitem__(self, idx: int):
            enc = encode([self.lines[idx]])
            return {k: v[0] for k, v in enc.items()}

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=max(1, args.max_steps // 5),
        save_steps=max(args.max_steps, 1),
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=LineDataset(texts),
        data_collator=collator,
    )
    trainer.train()
    tokenizer.save_pretrained(out)
    model.save_pretrained(out)
    print(f"Saved QuechuaBERT smoke checkpoint to {out}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise SystemExit("--push-to-hub requires --hub-model-id")
        model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print(f"Pushed to https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
