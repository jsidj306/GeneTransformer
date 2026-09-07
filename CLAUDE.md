# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

End-to-end, interpretable deep-learning project that predicts **cancer type** (BRCA / LUAD / COAD) from bulk **gene expression data** using a **Transformer encoder**, then explains each prediction with **SHAP** and **attention visualization**.

Pipeline: raw expression matrix → normalization (log2 + z-score) → feature selection (DEG / highly-variable genes) → Transformer encoder → classification → interpretation.

## Environment

- Python 3.11 in `.venv`. Activate (Windows): `.venv\Scripts\activate`
- **CPU-only PyTorch** — this machine has no NVIDIA GPU (`nvidia-smi` absent), so training runs on CPU. Keep models small (few thousand genes, d_model ≤ 256) so CPU training stays tractable.
- Dependencies pinned in `requirements.txt`. Note: `pandas` is locked to 2.3.3 (scanpy / pydeseq2 are not yet pandas-3.x compatible) — do not upgrade it.

## Directory Structure

```
data/
  raw/            # raw GEO/TCGA downloads (gitignored)
  processed/      # normalized sample×gene matrices + labels (gitignored)
src/
  __init__.py
  data_loader.py  # download & parse GEO/TCGA expression matrices
  preprocess.py   # normalization + feature selection
  dataset.py      # PyTorch Dataset / DataLoader
  model.py        # GeneTransformer (embedding + encoder + [CLS] + MLP head)
  train.py        # training loop (early stopping + LR schedule)
  evaluate.py     # metrics, confusion matrix, ROC
  baseline.py     # RandomForest / SVM baselines
  explain.py      # SHAP + attention visualization
  utils.py        # seed, metrics, plotting helpers
notebooks/        # 01_data_exploration / 02_baseline / 03_interpretability
results/
  figures/        # loss/AUC, confusion matrix, SHAP, attention heatmaps
  checkpoints/    # best_model.pt (gitignored)
report/           # technical report (Markdown/PDF)
tests/            # pytest unit tests
```

## Common Commands

```bash
# activate environment
.venv\Scripts\activate

# install deps (already done, locked in requirements.txt)
pip install -r requirements.txt

# preprocessing (reads config.yaml, writes data/processed/*.npy)
python -m src.preprocess

# baselines (RF / SVM)
python -m src.baseline

# train Transformer
python -m src.train

# evaluate
python -m src.evaluate

# explain (SHAP + attention)
python -m src.explain

# tests
pytest
```

## Conventions

- **All hyperparameters and paths live in `config.yaml`** — scripts load it, never hardcode.
- **Fixed random seed** — `utils.py` sets `seed` (from config) for python/numpy/torch.
- Preprocessed outputs land in `data/processed/`; model checkpoints in `results/checkpoints/`; both are gitignored (reproducible, not committed).
- Feature selection reduces genes from ~20–50k down to ~2000–5000 before feeding the Transformer.
- Evaluate with **macro-F1** (class imbalance); baselines (RF/SVM) are the reference the Transformer must beat.
