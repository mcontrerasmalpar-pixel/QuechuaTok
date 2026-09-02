#!/usr/bin/env python3
"""Rebuild eval/morphacc_gold_500.json from the published SQUOIA treebank zip.

The committed JSON is the release artifact (seed 42, paper 15 locked as prefix).
This script regenerates a stratified 500 from the same zip; bit-identity with
the committed file is not guaranteed if CoNLL row order changes, but the
constraints match issue #9.

Usage:
  python eval/build_morphacc_gold_500.py
  python eval/build_morphacc_gold_500.py --zip /path/to/SQUOIA_quz_treebanks_31-07-2015.zip
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ZIP_URL = (
    "https://github.com/a-rios/squoia/releases/download/"
    "31-07-2015/SQUOIA_quz_treebanks_31-07-2015.zip"
)
SEED = 42
PRPE = [
    "ni", "nki", "n", "nchik", "nkichik", "nku", "rqa", "sqa", "nqa", "rqan",
    "sqan", "kuna", "pi", "manta", "wan", "paq", "ta", "qa", "pas", "mi",
    "si", "cha", "stin", "spa", "na",
]
EVID = {"mi", "si", "cha", "m", "s", "chá"}
QUECHUA_ORTH = re.compile(r"(qh|ph|th|chh|ll|ñ|q'|k'|p'|ch|sh|w|y|q|kh)")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def load_paper_15() -> list[dict]:
    gold = json.loads((HERE / "morphacc_gold.json").read_text(encoding="utf-8"))
    return gold["items"]


def parse_conll(text: str) -> list[list[dict]]:
    sents: list[list[dict]] = []
    sent: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            if sent:
                sents.append(sent)
                sent = []
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        sent.append(
            {
                "form": parts[1],
                "xpos": parts[4],
                "feats": parts[5] if len(parts) > 5 else "_",
            }
        )
    if sent:
        sents.append(sent)
    return sents


def reconstruct(sent: list[dict]) -> list[dict]:
    words: list[dict] = []
    current = None
    for tok in sent:
        form = tok["form"]
        if form in {"VROOT", "ROOT"} or tok["xpos"] in {"$.", "PUNCT"} or form in {
            ".", ",", ";", ":", "!", "?", "\"", "'",
        }:
            if current:
                words.append(current)
                current = None
            continue
        if form.startswith("-") and len(form) > 1:
            if current is None:
                continue
            current["morphemes"].append(form[1:])
            current["feats"].append(tok["feats"])
            current["xpos"].append(tok["xpos"])
        else:
            if current:
                words.append(current)
            current = {
                "morphemes": [form],
                "feats": [tok["feats"]],
                "xpos": [tok["xpos"]],
            }
    if current:
        words.append(current)
    for w in words:
        w["word"] = "".join(w["morphemes"])
        w["n"] = len(w["morphemes"])
    return words


def is_es(feats: list[str]) -> bool:
    blob = " ".join(feats)
    return "NRootES" in blob or "RootES" in blob


def quechua_stem(stem: str, feats: list[str]) -> bool:
    blob = " ".join(feats)
    if "NRootES" in blob or "NRootNUM" in blob or "Root=NP" in blob:
        return False
    if re.search(r"[áéíóúü]", stem) and "chá" not in stem:
        return False
    if "Root=NRoot" in blob or "Root=VRoot" in blob or QUECHUA_ORTH.search(stem):
        return True
    return False


def collect_from_zip(zf: zipfile.ZipFile) -> list[dict]:
    by_word: dict[str, list[dict]] = defaultdict(list)
    names = [n for n in zf.namelist() if n.startswith("conll/") and n.endswith(".conll")]
    # Prefer genre files over the union dump.
    names = [n for n in names if not n.endswith("squoia_qu.conll")] or names
    for name in names:
        text = zf.read(name).decode("utf-8", errors="replace")
        for sent in parse_conll(text):
            for w in reconstruct(sent):
                word = nfc(w["word"].lower())
                morphs = [nfc(m.lower()) for m in w["morphemes"]]
                if "".join(morphs) != word:
                    continue
                if not re.fullmatch(r"[a-záéíóúüñ']+", word) or len(word) < 3:
                    continue
                by_word[word].append(
                    {
                        "word": word,
                        "morphemes": morphs,
                        "n": len(morphs),
                        "feats": w["feats"],
                        "xpos": w["xpos"],
                        "file": Path(name).name,
                    }
                )
    pool = []
    for word, occs in by_word.items():
        splits = {"|".join(o["morphemes"]) for o in occs}
        if len(splits) != 1:
            continue
        w0 = occs[0]
        stem = w0["morphemes"][0]
        if len(stem) < 3 or stem in PRPE:
            continue
        pool.append(
            {
                **w0,
                "freq": len(occs),
                "stem": stem,
                "has_es": is_es(w0["feats"]),
                "quechua_stem": quechua_stem(stem, w0["feats"]),
                "has_evid": any(m in EVID for m in w0["morphemes"][1:]),
                "has_vpers": any("VPers=" in t for t in w0["feats"]),
                "has_cas": any("Cas=" in t for t in w0["feats"]),
                "prpe_oov": any(m not in set(PRPE) for m in w0["morphemes"][1:]),
                "simplex_stem": (w0["xpos"] or [""])[0] == "Root",
            }
        )
    return pool


def take(pool, pred, k, name, used, rng):
    opts = [p for p in pool if p["word"] not in used and pred(p)]
    by_stem: dict[str, list] = defaultdict(list)
    for p in opts:
        by_stem[p["stem"]].append(p)
    stems = list(by_stem)
    rng.shuffle(stems)
    chosen = []
    i = 0
    while len(chosen) < k and stems:
        stem = stems[i % len(stems)]
        bucket = by_stem[stem]
        if bucket:
            bucket.sort(
                key=lambda p: (
                    0 if "gregorio" in p["file"] else 1,
                    0 if 2 <= p["freq"] <= 20 else 1,
                    -p["n"],
                )
            )
            p = bucket.pop(0)
            p = dict(p)
            p["stratum"] = name
            chosen.append(p)
            used.add(p["word"])
        if not bucket:
            stems = [s for s in stems if s != stem]
            i = 0
        else:
            i += 1
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "morphacc_gold_500.json")
    args = parser.parse_args()
    if args.zip:
        data = args.zip.read_bytes()
    else:
        print("Downloading", ZIP_URL)
        data = urlopen(ZIP_URL, timeout=120).read()
    paper = load_paper_15()
    locked = {it["word"] for it in paper}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        pool = [p for p in collect_from_zip(zf) if p["word"] not in locked]
    rng = random.Random(SEED)
    used: set[str] = set()
    buckets = [
        ("simplex_quechua", 25, lambda p: p["n"] == 1 and p["quechua_stem"] and p["simplex_stem"] and not p["has_es"] and len(p["word"]) >= 4),
        ("loan_plus_suffix", 45, lambda p: p["has_es"] and p["n"] >= 2),
        ("evidential_stack", 60, lambda p: p["has_evid"] and p["n"] >= 3 and p["quechua_stem"]),
        ("four_plus_morphs", 80, lambda p: p["n"] >= 4 and p["quechua_stem"] and not p["has_es"]),
        ("three_morphs", 85, lambda p: p["n"] == 3 and p["quechua_stem"] and p["simplex_stem"] and not p["has_es"]),
        ("noun_case_2", 50, lambda p: p["n"] == 2 and p["has_cas"] and p["quechua_stem"] and not p["has_es"] and not p["has_vpers"]),
        ("verb_person_2", 50, lambda p: p["n"] == 2 and p["has_vpers"] and p["quechua_stem"] and not p["has_es"]),
        ("prpe_oov_suffix", 50, lambda p: p["prpe_oov"] and p["n"] >= 3 and p["quechua_stem"] and not p["has_es"]),
        ("gregorio_oral_3plus", 40, lambda p: "gregorio" in p["file"] and p["n"] >= 3 and p["quechua_stem"]),
    ]
    selected: list[dict] = []
    for name, k, pred in buckets:
        got = take(pool, pred, k, name, used, rng)
        print(f"{name:24} {len(got):3}/{k}")
        selected.extend(got)
    need = 500 - 15 - len(selected)
    if need > 0:
        selected.extend(take(pool, lambda p: p["n"] >= 2 and p["quechua_stem"], need, "fill_n2plus", used, rng))
    items = []
    for g in paper:
        items.append(
            {
                "word": g["word"],
                "squoia": g["squoia"],
                "morphemes": g["morphemes"],
                "check": "paper_15_locked_SQUOIA_FST_manual",
                "stratum": "paper_15",
                "source": "eval/morphacc_gold.json",
            }
        )
    for s in selected:
        items.append(
            {
                "word": s["word"],
                "squoia": "|".join(s["feats"]),
                "morphemes": s["morphemes"],
                "check": "SQUOIA_treebank_gold_CoNLL",
                "stratum": s["stratum"],
                "source_file": s["file"],
                "freq_in_treebank": s["freq"],
            }
        )
    assert len(items) == 500, len(items)
    out = {
        "source": (
            "Original 15 locked from morphacc_gold.json (SQUOIA FST + manual). "
            "Remaining 485 sampled from SQUOIA Quechua treebank CoNLL "
            "(a-rios/squoia release 31-07-2015). Morphemes reconstructed from "
            "hyphen-suffix CoNLL tokens, NFC lowercased. Gold, not FST silver."
        ),
        "n_words": 500,
        "metric": (
            "morpheme_boundary_accuracy = |pred_morphemes \u2229 silver_morphemes| "
            "/ |silver_morphemes| (string overlap, not character offsets)"
        ),
        "sampling": {
            "seed": SEED,
            "treebank": ZIP_URL,
            "variety": "quz (Cuzco), SQUOIA QIIC",
            "locked_paper_15": True,
        },
        "items": items,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out, "n=", len(items))


if __name__ == "__main__":
    main()
