# Transformer-based Disease Prediction from Gene Expression Data

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **end-to-end, interpretable** deep-learning project that classifies **tumor vs normal** from bulk **gene expression data** using a **Transformer encoder** — evaluated rigorously under leave-one-dataset-out (LODO) — and explains every prediction with **SHAP**, **attention**, and **pathway enrichment**.

> *"Not building AI for AI's sake, but building AI to answer a real biomedical question — and explain why."*

---

## ✨ Highlights

- **Interpretability-first**: SHAP feature attribution + `[CLS]` attention heatmaps + GO/KEGG pathway enrichment answer *"why does the model say tumor?"*
- **Honest cross-platform evaluation**: leave-one-dataset-out (LODO) exposes batch leakage — the naive 100 % accuracy was platform, not biology.
- **End-to-end**: raw expression matrix → normalization → feature selection → Transformer → explanation.
- **Baseline comparison**: Random Forest / SVM baselines quantify the gain of the Transformer.
- **Fully reproducible**: public GEO data, fixed random seeds, `config.yaml`-driven experiments.

---

## 🧬 Pipeline

```
Gene Expression Data       (3 GEO platforms: tumor + normal)
        │
        ▼
Normalization              (log2(x+1) + per-gene z-score)
        │
        ▼
Feature Selection          (per-fold HVG top-5000)
        │
        ▼
Transformer Encoder        (Gene Embedding → multi-head self-attention → [CLS])
        │
        ▼
Tumor/Normal Classification ([CLS] → MLP head → softmax)
        │
        ▼
Interpretation             (SHAP + [CLS] attention → consensus genes → GO/KEGG enrichment)
```

---

## 🧪 Model Architecture

| Component | Description |
|---|---|
| Input | Gene expression vector `x ∈ R^G` (`G` = number of selected genes) |
| Gene Embedding | Learnable `d`-dim embedding per gene, scaled by the expression value |
| Positional Encoding | Learnable position embedding (gene order by variance / chromosome position) |
| `[CLS]` token | Prepended learnable token whose output summarizes the whole sample |
| Transformer Encoder | `N` layers, multi-head self-attention, feed-forward, LayerNorm, Dropout |
| MLP Head | `[CLS]` → Linear → GELU → Dropout → Linear → softmax over tumor/normal |

---

## 📁 Project Structure

```
gene-transformer-disease-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── config.yaml                     # all hyperparameters + paths + seed
├── .gitignore
├── CLAUDE.md
├── data/
│   ├── raw/                        # raw GEO/TCGA downloads
│   └── processed/                  # normalized sample × gene matrices + labels
├── src/
│   ├── data_loader.py              # download & parse GEO expression matrices
│   ├── preprocess.py               # normalization + feature selection
│   ├── dataset.py                  # PyTorch Dataset / DataLoader
│   ├── model.py                    # GeneTransformer (embedding + encoder + head)
│   ├── train.py                    # training loop (early stopping + LR schedule)
│   ├── evaluate.py                 # metrics, confusion matrix, ROC
│   ├── baseline.py                 # RandomForest / SVM baselines
│   ├── cross_dataset.py            # LODO split / batch diagnosis
│   ├── tumor_normal.py             # binary tumor/normal dataset + training
│   ├── explain_tn.py               # SHAP + attention (binary, inference-only)
│   ├── enrichment.py               # GO/KEGG/Reactome enrichment via Enrichr
│   └── utils.py                    # seed, metrics, plotting helpers
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_interpretability.ipynb
├── results/
│   ├── figures/                    # loss/AUC, confusion matrix, SHAP, attention heatmaps
│   └── checkpoints/                # best_model.pt
├── report/                         # technical report (English + 中文, Markdown)
└── tests/                          # unit tests (pytest)
```

---

## 🚀 Quick Start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install pandas numpy scikit-learn matplotlib seaborn
pip install GEOparse scanpy pydeseq2 shap captum pyyaml
pip install torch                  # add --index-url for CUDA if you have a GPU
```

### 2. Download data

```bash
# Three tumor/normal platforms: GSE31210 (lung), GSE39582 (colorectal), GSE45827 (breast)
python -c "import GEOparse; GEOparse.get_GEO(geo='GSE45827', destdir='data/raw')"
```

See `src/data_loader.py` for the full accession list and parsing.

### 3. Preprocess (normalize + feature selection)

```bash
python -m src.preprocess            # reads config.yaml, writes data/processed/*.npy
```

### 4. Train baselines (RF / SVM)

```bash
python -m src.baseline              # prints macro-F1 / AUC per model
```

### 5. Train the Transformer

```bash
python -m src.train                 # best checkpoint saved to results/checkpoints/
```

### 6. Evaluate

```bash
python -m src.evaluate              # confusion matrix, macro-F1, ROC-AUC
```

### 7. Explain (SHAP + attention) & enrich

```bash
python -m src.explain_tn            # SHAP summary + [CLS] attention heatmaps (binary)
python -m src.enrichment            # GO/KEGG/Reactome enrichment of consensus genes
```

---

## 📊 Results

The delivered model is a **binary tumor/normal classifier**, trained and evaluated
**leave-one-dataset-out (LODO)** across three independent platforms (lung GSE31210,
colorectal GSE39582, breast GSE45827). **PR-AUC / ROC-AUC are the primary,
threshold-free metrics** (class imbalance ≈ 18:1).

| Model (LODO) | PR-AUC (mean ± std) | ROC-AUC (mean ± std) |
|---|---|---|
| Random Forest | 0.9961 ± 0.0025 | 0.9486 ± 0.0250 |
| SVM (RBF) | 0.9986 ± 0.0012 | 0.9776 ± 0.0154 |
| **Gene Transformer** | **0.9963 ± 0.0029** | **0.9637 ± 0.0192** |

The Transformer reaches PR-AUC parity with SVM and exceeds the RF on ROC-AUC, while
staying end-to-end differentiable — the property that makes the interpretation below
possible. Hard labels use **quantile-aligned thresholds** per fold (see the
[technical report](report/technical_report.md) · [中文报告](report/technical_report_zh.md));
`class_weight` does *not* fix the cross-platform score shift.

---

## 🔍 Interpretability

Two complementary, inference-only views (no retraining — run on CPU from local
checkpoints):

- **SHAP (Deep SHAP)** — Gradient × Input attribution to `logit[tumor] − logit[normal]`,
  memory-bounded (batch=1) because self-attention over 5001 tokens is O(n²).
- **Attention** — the `[CLS]` token's self-attention to each gene (per layer/head),
  computed exactly from the layer's Q/K weights.

The cross-fold **consensus genes** (rank-based, absent-fold-penalized) are a
**tumor-stroma / extracellular-matrix signature**: *PFKFB3, CDO1, LCN2, COL11A1,
DCN, DPT, CA4, COL4A5, COL10A1, FBLN2, …* — plus a metabolic component
(PFKFB3 glycolysis, LPL, PDK4).

**Pathway enrichment** (Enrichr: GO / KEGG / Reactome) confirms it — the one
FDR-significant signal is **collagen / extracellular-matrix organization**
(GO:CC "Collagen-Containing Extracellular Matrix" adj. p = 3.9 × 10⁻⁶; Reactome
"Extracellular Matrix Organization" adj. p = 0.043). KEGG and GO Biological Process
are trending but do not survive multiple-testing correction.

![Pathway enrichment bubble plot](results/figures/enrichment_bubble.png)

> Run it yourself: `python -m src.enrichment` (top-100 consensus genes → Enrichr),
> `python -m src.explain_tn` (SHAP + attention across the three folds). Full
> write-up, consensus list, and the three clearest figures: see the
> [technical report](report/technical_report.md) · [中文报告](report/technical_report_zh.md).

---

## 📚 References

- GEO — Gene Expression Omnibus: https://www.ncbi.nlm.nih.gov/geo/
- TCGA — The Cancer Genome Atlas: https://www.cancer.gov/ccg/research/genome-sequencing/tcga
- GEOparse — https://github.com/guma44/GEOparse
- PyDESeq2 — https://github.com/owkin/PyDESeq2
- SHAP — https://github.com/shap/shap
- Captum — https://captum.ai/

## 📄 License

MIT — see [LICENSE](LICENSE).
