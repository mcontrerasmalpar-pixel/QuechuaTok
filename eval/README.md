# Eval artifacts

Checked-in pieces of the paper-matching v1. Trained SentencePiece / WordPiece binaries are **not** in git: the notebook writes them under `models/` (gitignored) because they are regenerated from the Hugging Face corpora.

| File | What |
|---|---|
| `morphacc_gold.json` | 15-word MorphAcc silver set from `QuechuaTok_v5_final.ipynb`, verified against SQUOIA (Rios, 2016) |
| `prpe_suffixes.json` | PRPE suffix lexicon from the notebook |
| `prpe.py` | Greedy segmenter used in the paper |
| `score_morphacc_prpe.py` | Recompute PRPE MorphAcc without training |

```bash
pip install -r requirements.txt
python eval/score_morphacc_prpe.py
```

To regenerate BPE / Unigram / WordPiece models, run `QuechuaTok_v5_final.ipynb` top to bottom (needs network, `foma-bin` for the SQUOIA section).
