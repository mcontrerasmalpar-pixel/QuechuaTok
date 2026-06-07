# QuechuaTok: Morphological Tokenization for Southern Quechua

**Author:** Maky (Maria Contreras) | [@macmaky](https://www.kaggle.com/macmaky)

A comparative benchmark of subword tokenizers — BPE, Unigram LM, and WordPiece — for Southern Quechua, a low-resource agglutinative language. This project introduces **PRPE** (Pattern-based Root-Prefix-Suffix Encoding), a morphology-aware tokenizer that significantly outperforms general-purpose baselines on morphological boundary accuracy.

---

## Results

| Model        | Fertility ↓ | OOV% ↓ | MorphAcc% ↑ |
|:-------------|:-----------:|:-------:|:-----------:|
| BPE 4k       | 2.2331      | 0.0     | 6.67        |
| BPE 8k       | 1.9177      | 0.0     | 6.67        |
| BPE 16k      | 1.6360      | 0.0     | 6.67        |
| Unigram 4k   | 2.3178      | 0.0     | 66.67       |
| Unigram 8k   | 1.9533      | 0.0     | 26.67       |
| Unigram 16k  | 1.6378      | 0.0     | 33.33       |
| **PRPE**     | **1.7971**  | **0.0** | **83.33**   |

> ↓ lower is better &nbsp;&nbsp; ↑ higher is better

**Key finding:** PRPE achieves 83.33% morphological boundary accuracy — a **10× improvement** over BPE and **25% higher** than the best Unigram baseline — while maintaining competitive fertility.

---

## Project Structure

```
QuechuaTok/
├── quechuatok.ipynb          # Main notebook (full pipeline)
├── quechuatok_paper.tex      # LaTeX paper source
├── quechuatok.bib            # Bibliography
├── quechuatok_figure1.png    # Results figure
└── README.md
```

---

## Installation

```bash
pip install datasets sentencepiece tokenizers unicodedata2 pandas matplotlib
```

Set your Hugging Face token before running the notebook:

```python
import os
os.environ["HF_TOKEN"] = "your_hf_token_here"
```

---

## Pipeline

1. **Corpus** — Downloads `Llamacha/monolingual-quechua-iic` (~Wikipedia + OSCAR, Southern Quechua)
2. **Preprocessing** — Unicode NFC normalization, lowercasing, URL/mention removal
3. **Tokenizer training** — BPE and Unigram LM at 4k / 8k / 16k vocab sizes via SentencePiece; WordPiece via HuggingFace Tokenizers
4. **Evaluation** — Fertility rate, OOV%, morphological boundary accuracy against a hand-annotated gold set
5. **PRPE** — Pattern-based morphological tokenizer using Quechua suffix inventories and the SQUOIA morphological analyzer (Rios et al.)

---

## Links

- **Kaggle Notebook:** [kaggle.com/code/macmaky/quechuatok](https://www.kaggle.com/code/macmaky/quechuatok)
- **Paper (LaTeX):** [`quechuatok_paper.tex`](./quechuatok_paper.tex)
- **Dataset:** [Llamacha/monolingual-quechua-iic](https://huggingface.co/datasets/Llamacha/monolingual-quechua-iic) on Hugging Face
- **SQUOIA Morphological Analyzer:** [github.com/ariosramirez/squoia](https://github.com/ariosramirez/squoia)

---

## Citation

```bibtex
@misc{contreras2025quechuatok,
  title   = {QuechuaTok: Morphological Tokenization for Southern Quechua},
  author  = {Contreras, Maria},
  year    = {2025},
  note    = {GitHub repository: https://github.com/mcontrerasmalpar-pixel/QuechuaTok}
}
```

---

*Part of ongoing research on NLP tools for low-resource indigenous languages of the Andes.*
