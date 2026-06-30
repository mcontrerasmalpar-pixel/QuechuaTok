# QuechuaTok

Morphological Tokenization Benchmark for Southern Quechua

**Author:** Maria Contreras ([@macmaky](https://www.kaggle.com/macmaky))
**Paper:** https://arxiv.org/abs/2606.23943

Comparative benchmark of BPE, Unigram LM, WordPiece, and a custom morphological
segmenter (PRPE) for Southern Quechua, a low-resource agglutinative language.
Uses the `Llamacha/monolingual-quechua-iic` corpus and evaluates fertility
rate, OOV%, and morphological boundary accuracy against the SQUOIA
finite-state morphological analyzer (Rios, 2016).

## Contents

- [`QuechuaTok_v5_final.ipynb`](QuechuaTok_v5_final.ipynb) — full pipeline:
  corpus download and preprocessing, tokenizer training (BPE, Unigram,
  WordPiece, PRPE), and comparative evaluation.

## Key findings

| Model        | Fertility ↓ | MorphAcc% ↑ | Perplexity ↓ |
|--------------|-------------|-------------|--------------|
| BPE 4k       | 2.2331      | 6.67        | 1556.02      |
| BPE 16k      | 1.6360      | 6.67        | 2553.05      |
| Unigram 4k   | 2.3178      | 66.67       | 1344.34      |
| Unigram 16k  | 1.6378      | 33.33       | 2091.81      |
| PRPE         | 1.7971      | 83.33       | 1879.36      |

No single tokenizer wins on all metrics — fertility rate alone is
insufficient to evaluate tokenizers for agglutinative languages.

## Citation

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

**Links:** [arXiv](https://arxiv.org/abs/2606.23943) | [Kaggle](https://kaggle.com/macmaky)
