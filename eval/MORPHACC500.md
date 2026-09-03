# MorphAcc-500

`morphacc_gold.json` is frozen. `python eval/score_morphacc_prpe.py` must still print **83.33** on those 15 words (paper table).

`morphacc_gold_500.json` is a manifest. The 15 locked prefix plus 485 new forms live in `eval/morphacc_gold_500/part_*.jsonl` (500 total), sampled from the [SQUOIA Quechua treebank CoNLL](https://github.com/a-rios/squoia/releases/tag/31-07-2015) (Cuzco `quz`). The CoNLL is already morpheme-tokenized (`ka` + `-ni` → `kani`). This is **treebank gold**, not FST silver. Derivational fusions in the stem (`suyukuna|manta`) are kept as annotated.

```bash
python eval/score_morphacc_prpe.py
python eval/score_morphacc_prpe.py --gold eval/morphacc_gold_500.json
python eval/test_morphacc_gold.py
```

PRPE is expected to drop on 500 (preview ~39% vs 83.33% on 15). That is the point of issue #9.

Rebuild (optional, downloads the zip):

```bash
python eval/build_morphacc_gold_500.py
```
