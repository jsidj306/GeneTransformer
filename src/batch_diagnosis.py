"""
batch_diagnosis.py -- visual + quantitative evidence of batch confound

Produces:
  1. PCA of log2 expression colored by BATCH (GSE) and by CANCER, before/after
     full-data batch-only ComBat. If samples separate by GSE and not by cancer,
     the "100% accuracy" was batch effect.
  2. Batch-identity prediction: a RandomForest trained to predict GSE id from
     expression. ~100% accuracy is direct evidence the data is batch-separable.

Note: full-data ComBat is used here for VISUALIZATION only (batch-only, no
covariate -- see cross_dataset.combat_correct for why the cancer covariate is
collinear with batch). The leak-free LODO evaluation never runs ComBat on the
test split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .cross_dataset import load_processed
from .utils import load_config


def _top_var_genes(X, n=5000):
    idx = np.argsort(np.var(X, axis=0))[::-1][:n]
    return X[:, idx]


def apply_combat_full(X, batch):
    """Full-data batch-only ComBat (visualization only)."""
    from pycombat import Combat

    combat = Combat()
    return combat.fit_transform(X, np.asarray(batch))


def plot_pca(X, color, color_name, title, save_path):
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in np.unique(color):
        ax.scatter(Z[color == cls, 0], Z[color == cls, 1], s=10, alpha=0.6, label=str(cls))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=8, title=color_name)
    sns.despine()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return pca.explained_variance_ratio_


def run_batch_diagnosis(config):
    X, y_str, batch, genes = load_processed(config)
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    Xv = _top_var_genes(X, n=5000)  # visualization on top HVGs (global)

    # 1. Before ComBat
    plot_pca(Xv, batch, "GSE", "PCA by batch (before ComBat)", fig_dir / "pca_by_batch_before.png")
    plot_pca(Xv, y_str, "cancer", "PCA by cancer (before ComBat)", fig_dir / "pca_by_cancer_before.png")

    # 2. After full-data batch-only ComBat
    Xc = apply_combat_full(Xv, batch)
    plot_pca(Xc, batch, "GSE", "PCA by batch (after ComBat)", fig_dir / "pca_by_batch_after.png")
    plot_pca(Xc, y_str, "cancer", "PCA by cancer (after ComBat)", fig_dir / "pca_by_cancer_after.png")

    # 3. Batch-identity prediction (leakage evidence), before AND after ComBat
    le = LabelEncoder()
    b_int = le.fit_transform(batch)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    batch_scores = cross_val_score(rf, X, b_int, cv=skf, scoring="accuracy")
    batch_acc = float(batch_scores.mean())
    batch_std = float(batch_scores.std())
    print(f"Batch-identity prediction BEFORE ComBat (RF, 5-fold CV): "
          f"{batch_acc:.4f} +/- {batch_std:.4f}")

    batch_scores_c = cross_val_score(rf, Xc, b_int, cv=skf, scoring="accuracy")
    batch_acc_c = float(batch_scores_c.mean())
    print(f"Batch-identity prediction AFTER  ComBat (RF, 5-fold CV): "
          f"{batch_acc_c:.4f} +/- {float(batch_scores_c.std()):.4f}")

    # cancer-identity prediction for contrast (same protocol -> also ~100%,
    # because cancer == batch here)
    le2 = LabelEncoder()
    c_int = le2.fit_transform(y_str)
    cancer_scores = cross_val_score(rf, X, c_int, cv=skf, scoring="accuracy")
    cancer_acc = float(cancer_scores.mean())
    print(f"Cancer-identity prediction (RF, 5-fold CV): {cancer_acc:.4f} "
          f"+/- {float(cancer_scores.std()):.4f}")

    summary = {
        "batch_identity_accuracy_before_combat": {"mean": batch_acc, "std": batch_std},
        "batch_identity_accuracy_after_combat": {"mean": batch_acc_c,
                                                 "std": float(batch_scores_c.std())},
        "cancer_identity_accuracy": {"mean": cancer_acc, "std": float(cancer_scores.std())},
        "note": "batch and cancer are 1:1 confounded, so both are trivially "
                "predictable before ComBat; batch-only ComBat removes the batch "
                "(and thus class) signal. Only leave-one-dataset-out reveals the confound.",
    }
    with open("results/batch_diagnosis.yaml", "w", encoding="utf-8") as fh:
        yaml.dump(summary, fh, default_flow_style=False, allow_unicode=True)

    print(f"\n[OK] figures -> {fig_dir}/pca_by_*_before/after.png")
    print(f"[OK] summary -> results/batch_diagnosis.yaml")
    return summary


if __name__ == "__main__":
    import sys
    import traceback

    try:
        config = load_config()
        run_batch_diagnosis(config)
    except Exception as e:  # noqa: BLE001
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
