# Neural Language Modeling

Comparison of **LSTM** and **Transformer Decoder** architectures for neural language modeling on the WikiText-2 benchmark, with BPE tokenization, ablation experiments, and text generation.

*Deep Learning course project — Ho Chi Minh City University of Economics and Finance*

---

## Overview

This project implements and compares two neural language model architectures from scratch using PyTorch:

| | Model 1 (Baseline) | Model 2 (Advanced) |
|---|---|---|
| **Architecture** | LSTM | Transformer Decoder (GPT-style) |
| **Dataset** | WikiText-2 | WikiText-2 |
| **Tokenization** | BPE (vocab = 10,000) | BPE (vocab = 10,000) |
| **Metric** | Perplexity (PPL ↓) | Perplexity (PPL ↓) |

---

## Dataset — WikiText-2

WikiText-2 is a standard benchmark for language modeling, sourced from high-quality Wikipedia articles (Good + Featured).

| Split | Segments |
|---|---|
| Train | ~23,000 passages |
| Validation | ~2,400 passages |
| Test | ~2,700 passages |

Key challenges addressed:
- **Zipf distribution** — high imbalance ratio; rare words handled via BPE
- **Long-range dependencies** — LSTM struggles; Transformer handles via attention
- **Noise** — Wikipedia headings (`= ... =`) and special tokens filtered in preprocessing

---

## Architecture Details

### Model 1 — LSTM Baseline

| Hyperparameter | Value |
|---|---|
| Embedding dim | 256 |
| Hidden size | 1024 |
| Layers | 2 |
| Dropout | 0.5 |
| Optimizer | Adam (lr = 5e-4) |
| Sequence length | 128 |

### Model 2 — Transformer Decoder Mini

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| Attention heads | 4 |
| Layers | 4 |
| FFN dim | 1024 |
| Dropout | 0.15 |
| Optimizer | Adam (lr = 5e-4) |
| Sequence length | 128 |

Both models use **sliding window** sequences (stride = 64) and **early stopping** with validation PPL monitoring.

---

## Tech Stack

- **PyTorch** — model implementation
- **HuggingFace `datasets`** — WikiText-2 loading
- **HuggingFace `tokenizers`** — BPE tokenizer training
- **Matplotlib** — training curves, EDA plots

---

## Experiments

Beyond the main LSTM vs Transformer comparison, an ablation study is included examining the effect of:
- Optimizer choice
- Regularization strength (dropout)
- Training duration (early stopping behavior)

---

## Project Structure

```
neural-language-modeling/
├── Language_Modeling_Mini.ipynb   # Full pipeline notebook
└── README.md
```

**Notebook sections:**
1. Environment setup
2. Library imports & config
3. WikiText-2 loading & EDA
4. BPE tokenization & dataset
5. LSTM Baseline — training & evaluation
6. Transformer Decoder Mini — training & evaluation
7. Model comparison
8. Ablation: optimizer & regularization
9. Text generation & error analysis
10. Conclusion

---

## Usage

Designed to run in **Google Colab** with GPU (T4 recommended):

1. `Runtime > Change runtime type > T4 GPU`
2. Mount Google Drive (for checkpoint saving)
3. Run cells sequentially — training checkpoints are saved automatically

---

## Skills Demonstrated

- LSTM and Transformer Decoder implementation from scratch in PyTorch
- BPE tokenization pipeline
- Perplexity-based evaluation and training curve analysis
- Ablation study design
- Text generation with temperature sampling
