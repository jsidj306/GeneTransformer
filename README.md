# Transformer-based Disease Prediction from Gene Expression Data

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **end-to-end, interpretable** deep-learning project that predicts cancer type (breast BRCA, lung LUAD, colorectal COAD) from bulk **gene expression data** using a **Transformer encoder**, and explains every prediction with **SHAP** and **attention visualization**.

> *"Not building AI for AI's sake, but building AI to answer a real biomedical question — and explain why."*

---

## ✨ Highlights

- **Interpretability-first**: SHAP feature attribution + attention heatmaps answer *"why does the model think this patient is sick?"*
- **End-to-end**: from raw expression matrix → normalization → feature selection → model → explanation.
- **Baseline comparison**: Random Forest / SVM baselines quantify the gain of the Transformer.
- **Biology-aware features**: differential expression genes (DEG) + highly variable genes (HVG) inject domain prior.
- **Fully reproducible**: public GEO/TCGA data, fixed random seeds, `config.yaml`-driven experiments.

---

## 🧬 Pipeline

```
Gene Expression Data   (GEO / TCGA: BRCA, LUAD, COAD)
        │
        ▼
Normalization           (log2(x+1) + per-gene z-score)
        │
        ▼
Feature Selection       (HVG top-N  /  DEG via PyDESeq2)
        │
        ▼
Transformer Encoder     (Gene Embedding → multi-head self-attention → [CLS])
        │
        ▼
Disease Classification  ([CLS] → MLP head → softmax)
        │
        ▼
Model Interpretation    (SHAP + attention heatmaps → key genes & pathways)
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
| MLP Head | `[CLS]` → Linear → ReLU → Dropout → Linear → softmax over cancer types |

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
│   ├── data_loader.py              # download & parse GEO/TCGA expression matrices
│   ├── preprocess.py               # normalization + feature selection
│   ├── dataset.py                  # PyTorch Dataset / DataLoader
│   ├── model.py                    # GeneTransformer (embedding + encoder + head)
│   ├── train.py                    # training loop (early stopping + LR schedule)
│   ├── evaluate.py                 # metrics, confusion matrix, ROC
│   ├── baseline.py                 # RandomForest / SVM baselines
│   ├── explain.py                  # SHAP + attention visualization
│   └── utils.py                    # seed, metrics, plotting helpers
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_interpretability.ipynb
├── results/
│   ├── figures/                    # loss/AUC, confusion matrix, SHAP, attention heatmaps
│   └── checkpoints/                # best_model.pt
├── report/                         # technical report (Markdown/PDF)
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
# GEO example (per dataset: BRCA / LUAD / COAD)
python -c "import GEOparse; GEOparse.get_GEO(geo='GSE_XXXXX', destdir='data/raw')"
```

Or use TCGA via `gdc-client`. See `data/README.md` for the exact accession numbers.

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

### 7. Explain (SHAP + attention)

```bash
python -m src.explain               # SHAP summary/force plots + attention heatmaps
```

---

## 📊 Results (example)

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| Random Forest | 0.87 | 0.85 | 0.93 |
| SVM (RBF) | 0.88 | 0.86 | 0.94 |
| **Gene Transformer** | **0.92** | **0.91** | **0.96** |

*Replace with your own results from `results/baseline_metrics.csv` and the Transformer evaluation.*

---

## 🔍 Interpretability

- **SHAP** (`shap.DeepExplainer`) → `summary_plot` shows which genes matter globally and how high/low expression shifts the prediction; `force_plot` explains a single patient.
- **Attention** → per-layer attention weights are visualized as gene–gene heatmaps; the `[CLS]` row ranks the most-attended genes.
- **Cross-validation of explanations**: genes flagged by *both* SHAP and attention are reported as the most trustworthy, then validated via GO/KEGG pathway enrichment and comparison with known cancer driver genes (e.g. *TP53*, *EGFR*, *BRCA1*).

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
