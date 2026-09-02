# QuechuaTok

**Morphological Tokenization Benchmark for Southern Quechua**

**Author:** Maria Contreras ([@macmaky](https://www.kaggle.com/macmaky))
**Paper:** https://arxiv.org/abs/2606.23943
**Code:** [GitHub](https://github.com/mcontrerasmalpar-pixel/QuechuaTok) · [Kaggle](https://www.kaggle.com/code/macmaky/quechuatok)

QuechuaTok is a comparative benchmark of subword tokenizers for Southern
Quechua, a low-resource, highly agglutinative language. It trains and
evaluates BPE, Unigram LM, and WordPiece tokenizers alongside a custom
rule-based morphological segmenter (**PRPE**), and shows that the metric
most commonly used to judge tokenizers — fertility rate — is not enough
on its own: a tokenizer can achieve very low fertility purely by
memorizing frequent strings, while still producing linguistically
incorrect morpheme boundaries.

## Install

The PRPE segmenter is installable without training anything:

```bash
pip install "git+https://github.com/mcontrerasmalpar-pixel/QuechuaTok.git"
```

From a clone:

```bash
pip install -e .
```

```python
from quechuatok import segment_prpe, PrpeTokenizer

segment_prpe("purisqanchikmanta")
# ['puri', 'sqa', 'nchik', 'manta']

tok = PrpeTokenizer()
tok.tokenize("wasipi rimani")
# ['wasi', 'pi', 'rima', 'ni']
```

`python eval/score_morphacc_prpe.py` still prints `PRPE MorphAcc%: 83.33` on the
15-word gold set. The notebook is unchanged.

## Why this matters

Quechua words are built by stacking many suffixes onto a root
(person, tense/aspect/mood, evidentiality, case, etc.), e.g.:

```
purisqanchikmanta  →  puri | sqa | nchik | manta   (walk + PST + 1Pl.Incl + from)
```

Standard subword tokenizers (BPE, Unigram, WordPiece) are trained purely on
statistical co-occurrence and have no notion of morphology, so they tend to
split agglutinative words at the wrong boundaries — even when their
fertility rate (tokens per word) looks good. QuechuaTok quantifies this gap
with a **morphological boundary accuracy** metric, evaluated against a gold
standard derived from the SQUOIA finite-state morphological analyzer
(Rios, 2016).

## Dataset

- **Primary corpus:** [`Llamacha/monolingual-quechua-iic`](https://huggingface.co/datasets/Llamacha/monolingual-quechua-iic) — Southern Quechua monolingual text (Wikipedia + OSCAR).
- **Expansion corpus:** [`somosnlp-hackathon-2022/spanish-to-quechua`](https://huggingface.co/datasets/somosnlp-hackathon-2022/spanish-to-quechua) — used to grow the training corpus for the 16k-vocabulary experiments.
- **Preprocessing:** NFC normalization, lowercasing, URL/mention/hashtag stripping, punctuation removal, and a heuristic filter that keeps only lines where ≥10% of tokens end in a known Quechua suffix (`kuna`, `pi`, `manta`, `wan`, `paq`, `ta`, `qa`, `pas`, `ni`, `nki`, `n`, `nchik`, `nku`, `mi`, `si`, `cha`, `rqa`, `sqa`, `nqa`, `y`, `q`, `na`, `spa`), followed by deduplication.

## Methods compared

| Method | Type | Description |
|---|---|---|
| **BPE** | Statistical | SentencePiece byte-pair encoding, trained at 4k / 8k / 16k vocab sizes |
| **Unigram LM** | Statistical | SentencePiece unigram language model, same vocab sizes |
| **WordPiece** | Statistical | HuggingFace `tokenizers` WordPiece, 4k vocab |
| **PRPE** | Rule-based | Custom greedy morphological segmenter that strips a curated, length-sorted list of Quechua suffixes (verbal person markers, TAM, nominal case, evidentials, derivational morphemes) from the end of each word |

## Evaluation metrics

1. **Fertility rate** — average number of tokens produced per word (lower = more compact).
2. **OOV%** — percentage of tokens mapped to the unknown token.
3. **Morphological boundary accuracy (MorphAcc%)** — overlap between a tokenizer's predicted segment boundaries and a 15-word gold-standard set of Quechua verb/noun forms, cross-checked against SQUOIA analyzer output.
4. **Bigram perplexity** — held-out perplexity of a simple bigram language model built on top of each tokenizer's output, as a proxy for how predictable the resulting token sequences are.

## Results

### Fertility, OOV, and morphological accuracy

| Model        | Fertility ↓ | OOV% ↓ | MorphAcc% ↑ |
|--------------|-------------|--------|-------------|
| BPE 4k       | 2.2331      | 0.0    | 6.67        |
| BPE 8k       | 1.9177      | 0.0    | 6.67        |
| BPE 16k      | 1.6360      | 0.0    | 6.67        |
| Unigram 4k   | 2.3178      | 0.0    | 66.67       |
| Unigram 8k   | 1.9533      | 0.0    | 26.67       |
| Unigram 16k  | 1.6378      | 0.0    | 33.33       |
| **PRPE**     | 1.7971      | 0.0 *(by design)* | **83.33** |

### Full benchmark (fertility + morphology + perplexity)

| Model        | Fertility ↓ | MorphAcc% ↑ | Perplexity ↓ |
|--------------|-------------|-------------|--------------|
| BPE 4k       | 2.2331      | 6.67        | 1556.02      |
| BPE 16k      | 1.6360      | 6.67        | 2553.05      |
| Unigram 4k   | 2.3178      | 66.67       | 1344.34      |
| Unigram 16k  | 1.6378      | 33.33       | 2091.81      |
| **PRPE**     | 1.7971      | **83.33**   | 1879.36      |

**Key findings:**

1. **BPE achieves the lowest fertility but the worst morphological accuracy** (6.67%) — it minimizes token count by memorizing frequent strings, not by finding real morpheme boundaries.
2. **PRPE achieves the highest morphological accuracy** (83.33%) by construction, with competitive fertility.
3. **Unigram 4k has the lowest perplexity** (1344.34), suggesting its segmentation is the most predictable for a downstream language model, despite not having the best fertility or MorphAcc.
4. **No single tokenizer wins on every metric** — fertility rate alone is an insufficient criterion for evaluating tokenizers on agglutinative, low-resource languages, and should be reported alongside morphological accuracy.

### Qualitative segmentation examples (8k vocab)

From paper Table 2. The notebook's 16k qualitative cell did not finish (16k training failed in that run), so this table uses the 8k splits rather than empty 16k cells.

| Word | Gold segmentation | BPE 8k | Unigram 8k |
|---|---|---|---|
| `rimankichikmi` | rima \| nkichik \| mi | riman \| kichikmi | rimanki \| chikmi |
| `wasiykikunapiqa` | wasi \| yki \| kuna \| pi \| qa | wasi \| yki \| kunapiqa | wasi \| yki \| kunapiqa |
| `purisqanchikmanta` | puri \| sqa \| nchik \| manta | puris \| qanchikmanta | puri \| sqanchikmanta |

Gold for `munakuwarqanki` is `muna | ku | wa | rqa | nki` (notebook examples; BPE/Unigram 8k not reported in the paper table).

## Repository contents

- [`quechuatok/`](quechuatok/) — installable PRPE package (`pip install -e .`)
- [`pyproject.toml`](pyproject.toml) — package metadata
- [`QuechuaTok_v5_final.ipynb`](QuechuaTok_v5_final.ipynb) — end-to-end pipeline:
  1. Corpus download (HuggingFace `datasets`)
  2. Preprocessing and filtering
  3. Tokenizer training (BPE, Unigram, WordPiece at multiple vocab sizes)
  4. Fertility / OOV evaluation and visualization
  5. PRPE morphological segmenter and qualitative comparison
  6. Corpus expansion with a second dataset and retraining at 16k vocab
  7. Final benchmark table across all metrics
  8. SQUOIA finite-state analyzer setup (Rios, 2016) and silver-standard morphological evaluation
  9. Bigram perplexity evaluation
- [`eval/`](eval/) — 15-word MorphAcc gold set, PRPE suffix lexicon, and a script to recompute PRPE MorphAcc without training
- [`LICENSE`](LICENSE) — MIT
- [`CITATION.cff`](CITATION.cff)
- [`requirements.txt`](requirements.txt)

Trained SentencePiece / WordPiece binaries are **not** checked in. The notebook writes them under `models/` (gitignored). Regenerate by running the notebook.

## Reproducing the results

```bash
pip install -r requirements.txt
python eval/score_morphacc_prpe.py
```

Then open and run `QuechuaTok_v5_final.ipynb` top to bottom (originally on Kaggle; needs internet). It will:
- Download the corpora from HuggingFace,
- Create `data/processed/` and `models/` directories,
- Train all tokenizers locally,
- Clone and compile the [SQUOIA](https://github.com/ariosquoia/squoia) analyzer (requires `foma-bin`) for the silver-standard morphological evaluation,
- Save comparison charts as PNGs.

## Citation

If you use this work, please cite:

```bibtex
@misc{contreras2026quechuatok,
  title={QuechuaTok: Morphological Boundary Accuracy as a Necessary Metric for Tokenizer Evaluation in Agglutinative Low-Resource Languages},
  author={Contreras, Maria},
  year={2026},
  eprint={2606.23943},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2606.23943}
}
```

## References

- Rios, A. (2016). *A Basic Language Technology Toolkit for Quechua.* — [SQUOIA repository](https://github.com/ariosquoia/squoia)

**Links:** [arXiv](https://arxiv.org/abs/2606.23943) | [GitHub](https://github.com/mcontrerasmalpar-pixel/QuechuaTok) | [Kaggle](https://www.kaggle.com/code/macmaky/quechuatok)
