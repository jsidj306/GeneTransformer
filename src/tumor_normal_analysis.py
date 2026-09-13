"""
tumor_normal_analysis.py -- deeper analysis of the binary LODO baseline

Answers three questions about the tumor-vs-normal result with figures + text:

1. Does the model actually RANK tumor vs normal in the held-out platform?
   -> ROC + PR curves per fold. ROC shows ranking (AUC); PR shows how hard a
      hard decision is under 922:50 imbalance.

2. WHY does the train-fit threshold degenerate (normal recall = 0)?
   -> Score-shift diagnosis: compare train vs test score distributions per fold.
      The cross-platform scale shift moves the test scores away from the train
      threshold, so everything lands on one side.

3. Is the signal biological, and which genes drive it?
   -> RF feature importance aggregated across folds (top genes), as a bridge to
      the later SHAP/attention analysis.

Outputs -> results/:
    figures/tumor_normal_roc_pr.png
    figures/tumor_normal_score_shift.png
    figures/tumor_normal_top_genes.png
    tumor_normal_top_genes.csv
    tumor_normal_analysis.txt
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc as roc_auc, precision_recall_curve, average_precision_score

from .utils import load_config, set_seed
from .cross_dataset import prepare_fold
from .baseline import make_models
from .tumor_normal import load_binary, predict_score, optimal_threshold, class_weight_dict

FIG_DIR = Path("results/figures")
RES_DIR = Path("results")


def collect_fold_data(config):
    """Run LODO folds, return per-fold scores, labels, gene importances, thresholds."""
    X, y, batch, genes = load_binary(config)
    seed = config.get("seed", 42)
    set_seed(seed)
    cw = class_weight_dict(config)

    folds = []
    gene_importance = {g: [] for g in genes}  # gene -> importance per fold

    for held_out in np.unique(batch):
        te = batch == held_out
        tr = ~te
        X_tr, X_te, idx = prepare_fold(X[tr], X[te], batch[tr], batch[te], config)
        y_tr, y_te = y[tr], y[te]
        fold_genes = genes[idx]

        rec = {"held_out": held_out, "y_tr": y_tr, "y_te": y_te,
               "models": {}, "fold_genes": fold_genes}
        for name, model in make_models(seed, class_weight=cw).items():
            model.fit(X_tr, y_tr)
            s_tr = predict_score(model, X_tr)
            s_te = predict_score(model, X_te)
            thr, _ = optimal_threshold(y_tr, s_tr)
            rec["models"][name] = {"s_tr": s_tr, "s_te": s_te, "threshold": thr}

            if name == "RandomForest":
                imp = model.feature_importances_
                for g, v in zip(fold_genes, imp):
                    gene_importance[g].append(v)
        folds.append(rec)

    # aggregate importances (0 for genes not selected in a fold)
    imp_table = {}
    n_folds = len(folds)
    for g, vals in gene_importance.items():
        imp_table[g] = float(np.mean(vals)) if vals else 0.0
    imp_series = pd.Series(imp_table).sort_values(ascending=False)

    return folds, imp_series


def plot_roc_pr(folds):
    """Figure 1: ROC (top row) + PR (bottom row), one curve per fold, RF/SVM columns."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    model_names = ["RandomForest", "SVM"]
    for j, mname in enumerate(model_names):
        ax_roc, ax_pr = axes[0, j], axes[1, j]
        for rec in folds:
            y_te = rec["y_te"]
            s_te = rec["models"][mname]["s_te"]
            fpr, tpr, _ = roc_curve(y_te, s_te)
            a = roc_auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"held_out={rec['held_out']} (AUC={a:.3f})")
            prec, rec_, _ = precision_recall_curve(y_te, s_te)
            ap = average_precision_score(y_te, s_te)
            ax_pr.plot(rec_, prec, label=f"held_out={rec['held_out']} (AP={ap:.3f})")

        ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax_roc.set_title(f"{mname} — ROC (ranking)")
        ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
        ax_roc.legend(fontsize=8); ax_roc.grid(alpha=0.3)

        # baseline = prevalence of tumor (positive class)
        prev = np.mean(np.concatenate([rec["y_te"] for rec in folds]))
        ax_pr.axhline(prev, color="k", ls="--", lw=1, alpha=0.5,
                      label=f"chance (prevalence={prev:.2f})")
        ax_pr.set_title(f"{mname} — PR curve (imbalanced)")
        ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
        ax_pr.legend(fontsize=8); ax_pr.grid(alpha=0.3)

    fig.suptitle("Tumor vs normal — LODO per-fold ROC & PR", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "tumor_normal_roc_pr.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")
    return out


def plot_score_shift(folds):
    """Figure 2: score distributions (RF) — train vs test, tumor vs normal, per fold."""
    n = len(folds)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, rec in zip(axes, folds):
        y_tr, y_te = rec["y_tr"], rec["y_te"]
        m = rec["models"]["RandomForest"]
        bins = np.linspace(0, 1, 41)
        ax.hist(m["s_tr"][y_tr == 1], bins=bins, alpha=0.45, color="tab:red",
                label="train tumor", density=True)
        ax.hist(m["s_tr"][y_tr == 0], bins=bins, alpha=0.45, color="tab:blue",
                label="train normal", density=True)
        ax.hist(m["s_te"][y_te == 1], bins=bins, alpha=0.4, color="tab:red",
                label="test tumor", histtype="step", lw=2, density=True)
        ax.hist(m["s_te"][y_te == 0], bins=bins, alpha=0.4, color="tab:blue",
                label="test normal", histtype="step", lw=2, density=True)
        ax.axvline(m["threshold"], color="k", ls="--", lw=1.2,
                   label=f"train threshold={m['threshold']:.2f}")
        ax.set_title(f"held_out = {rec['held_out']}")
        ax.set_xlabel("RF tumor score")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density")
    fig.suptitle("Score-shift diagnosis: train-fit threshold does not transfer "
                 "cross-platform", fontsize=13, y=1.03)
    fig.tight_layout()
    out = FIG_DIR / "tumor_normal_score_shift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")
    return out


def plot_top_genes(imp_series, top_n=20):
    """Figure 3: top discriminating genes (RF importance aggregated over folds)."""
    top = imp_series.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(np.arange(len(top))[::-1], top.values, color="#2E7D32")
    ax.set_yticks(np.arange(len(top))[::-1])
    ax.set_yticklabels(top.index)
    ax.set_xlabel("mean RF importance across LODO folds")
    ax.set_title(f"Top {top_n} genes driving tumor vs normal (RF)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "tumor_normal_top_genes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")
    return out


def write_summary(folds, imp_series, top_n=20):
    """Text summary: train/test AUC + score-shift numbers + top genes."""
    lines = ["Tumor vs normal — baseline analysis", "=" * 60, ""]
    lines.append("[1] Ranking (train AUC vs held-out AUC)")
    for rec in folds:
        y_tr, y_te = rec["y_tr"], rec["y_te"]
        for mname, m in rec["models"].items():
            fpr_tr, tpr_tr, _ = roc_curve(y_tr, m["s_tr"])
            fpr_te, tpr_te, _ = roc_curve(y_te, m["s_te"])
            lines.append(
                f"  held_out={rec['held_out']:<9} {mname:<13} "
                f"train_AUC={roc_auc(fpr_tr, tpr_tr):.3f} "
                f"test_AUC={roc_auc(fpr_te, tpr_te):.3f}"
            )
    lines.append("")
    lines.append("[2] Score shift (RF) — median score per class, train vs test")
    lines.append("  (large train->test shift on tumor scores explains threshold failure)")
    for rec in folds:
        y_tr, y_te = rec["y_tr"], rec["y_te"]
        m = rec["models"]["RandomForest"]
        lines.append(
            f"  held_out={rec['held_out']:<9} "
            f"train tumor={np.median(m['s_tr'][y_tr == 1]):.3f} "
            f"train normal={np.median(m['s_tr'][y_tr == 0]):.3f} | "
            f"test tumor={np.median(m['s_te'][y_te == 1]):.3f} "
            f"test normal={np.median(m['s_te'][y_te == 0]):.3f} | "
            f"threshold={m['threshold']:.2f}"
        )
    lines.append("")
    lines.append(f"[3] Top {top_n} discriminating genes (RF importance, mean over folds)")
    for g, v in imp_series.head(top_n).items():
        lines.append(f"  {g:<14} {v:.5f}")
    lines.append("")
    lines.append(
        "Interpretation: test AUC >> 0.5 (and often ~train AUC) shows the model "
        "genuinely ranks tumor vs normal cross-platform; the score shift explains "
        "why a fixed threshold degenerates. Top genes should be checked against "
        "known cancer biology (proliferation / immune / tissue markers)."
    )

    out = RES_DIR / "tumor_normal_analysis.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[OK] {out}")
    return out


def save_top_genes_csv(imp_series, top_n=100):
    df = imp_series.head(top_n).rename("mean_rf_importance").reset_index()
    df.columns = ["gene", "mean_rf_importance"]
    out = RES_DIR / "tumor_normal_top_genes.csv"
    df.to_csv(out, index=False)
    print(f"[OK] {out}")
    return out


if __name__ == "__main__":
    import sys
    import traceback

    try:
        config = load_config()
        folds, imp_series = collect_fold_data(config)
        plot_roc_pr(folds)
        plot_score_shift(folds)
        plot_top_genes(imp_series)
        write_summary(folds, imp_series)
        save_top_genes_csv(imp_series)
        print("\n[SUCCESS] tumor/normal analysis complete")
    except Exception as e:  # noqa: BLE001
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
