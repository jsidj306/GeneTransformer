"""
explain_tn.py -- SHAP + attention interpretability for the binary tumor/normal Transformer.

Explains the three LODO binary Transformer checkpoints
(results/checkpoints/tn_lodo_GSE*), which classify tumor (1) vs normal (0)
across platforms. Two complementary, inference-only views:

1. SHAP (Deep SHAP / Gradient x Input linearization): per-gene contribution to
   the "tumor-ness" score (logit[tumor] - logit[normal]), aggregated to a
   global importance ranking. This is exactly what shap.DeepExplainer reduces
   to for this model, but computed in memory-bounded chunks because the model's
   self-attention is O(n_genes^2) (5001 tokens) and this machine has ~5 GB of
   free RAM -- a stock DeepExplainer would OOM on a full background.

2. Attention: the [CLS] token's self-attention to each gene across layers and
   heads -- which genes the classifier "reads" to make its decision.

Both run as pure inference from the already-downloaded checkpoints -- no
training, hence no Colab needed.

Usage:
    python -m src.explain_tn                 # all 3 folds
    python -m src.explain_tn GSE39582        # a single fold
"""
from __future__ import annotations

import gc
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .model import create_model
from .utils import load_config, set_seed, get_device
from .tumor_normal import load_binary, TUMOR_NORMAL_CLASSES
from .cross_dataset import prepare_fold

import shap

FIG_DIR = Path("results/figures")
TOP_K = 20          # genes shown in bar/summary plots
N_BG = 20           # background samples for the Deep-SHAP mean gradient
N_TEST = 30         # max test samples explained per fold
MAX_GENE_COLS = 30  # genes in the attention heatmap

# Validated default palette (dataviz skill): blue sequential ramp + ink/surface.
_SEQ_BLUE = ["#cde2fb", "#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
_CMAP = LinearSegmentedColormap.from_list("blue", _SEQ_BLUE)
_INK = "#0b0b0b"
_MUTED = "#898781"
_SURFACE = "#fcfcfb"


# --------------------------------------------------------------------------- #
# Fold loading
# --------------------------------------------------------------------------- #
def load_fold_model(config, held_out):
    """Rebuild a fold's binary model and load its best checkpoint."""
    ckpt = Path(f"results/checkpoints/tn_lodo_{held_out}/best_model.pt")
    state = torch.load(ckpt, map_location="cpu")
    cfg = state.get("config", config)  # saved config holds the exact n_genes
    cfg = {**cfg, "model": {**cfg.get("model", {}), "n_classes": 2}}
    model = create_model(cfg)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, cfg


def fold_data(config, held_out):
    """Reproduce the fold's train/test split + selected gene names."""
    X, y, batch, genes = load_binary(config)
    te = batch == held_out
    tr = ~te
    X_tr, X_te, idx = prepare_fold(X[tr], X[te], batch[tr], batch[te], config)
    return X_tr, X_te, y[tr], y[te], np.asarray(genes)[idx]


def stratified_subset(X, y, n_max, rng):
    """Up to n_max samples, half normal / half tumor (as balanced as possible)."""
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    half = max(1, n_max // 2)
    k0 = min(len(idx0), half)
    k1 = min(len(idx1), n_max - k0)
    sel = np.concatenate([
        rng.choice(idx0, k0, replace=False),
        rng.choice(idx1, k1, replace=False),
    ])
    return sel


# --------------------------------------------------------------------------- #
# Deep SHAP (Gradient x Input, memory-bounded)
# --------------------------------------------------------------------------- #
def _tumor_score(model, xb):
    """Scalar 'tumor-ness': logit[tumor] - logit[normal]."""
    logits = model(xb)
    return logits[:, 1] - logits[:, 0]


def deep_shap_importance(model, X_bg, X_test):
    """Deep-SHAP linearization of the tumor-vs-normal decision.

    mean_grad[i] = mean over background of d(tumor_score)/d x_i, then
    SHAP[s, i] = (x[s, i] - mean(bg)[i]) * mean_grad[i]. This is the gradient x
    input attribution with a background reference -- what shap.DeepExplainer
    computes for this architecture -- but chunked to stay in RAM.

    Returns (shap_matrix (n_test, n_genes), mean_grad (n_genes,), bg_mean).
    """
    model.eval()
    grads = []
    for i in range(0, len(X_bg), 1):  # chunk = 1: keep O(n_genes^2) attention small
        xb = torch.FloatTensor(X_bg[i:i + 1])
        xb.requires_grad_(True)
        score = _tumor_score(model, xb).sum()
        g = torch.autograd.grad(score, xb)[0]
        grads.append(g.detach().numpy()[0])
        del xb, g, score
        gc.collect()
    mean_grad = np.mean(grads, axis=0)  # (n_genes,)
    bg_mean = X_bg.mean(axis=0)          # (n_genes,)
    shap = (X_test - bg_mean) * mean_grad  # (n_test, n_genes)
    return shap.astype(np.float32), mean_grad, bg_mean


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
def attention_importance(model, X, top_k=TOP_K):
    """Mean [CLS]->gene attention over layers, heads and samples.

    Returns (mean_attn (n_genes,), per_layer_head (n_layers*n_heads, n_genes)).
    """
    model.eval()
    rows = []  # (n_samples, n_layers, n_heads, n_genes)
    with torch.no_grad():
        for i in range(0, len(X), 1):  # chunk = 1 for the same memory reason
            xb = torch.FloatTensor(X[i:i + 1])
            _, cls_attn = model.forward_attention(xb)  # list over layers of (B, H, L)
            # drop the [CLS] self-token, keep ->genes
            cls2gene = torch.stack([a[:, :, 1:] for a in cls_attn], dim=1)  # (1, L, H, G)
            rows.append(cls2gene.squeeze(0).numpy())  # (L, H, G)
            del xb, cls_attn, cls2gene
            gc.collect()
    A = np.stack(rows, axis=0)                        # (N, L, H, G)
    mean_attn = A.mean(axis=(0, 1, 2))                # (G,)
    per_lh = A.mean(axis=0).reshape(-1, A.shape[-1])  # (L*H, G)
    return mean_attn, per_lh


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _top_gene_bar(values, genes, title, xlabel, save_path, top_k=TOP_K):
    order = np.argsort(values)[::-1][:top_k]
    names = genes[order]
    plt.figure(figsize=(8, max(4, 0.32 * top_k)), facecolor=_SURFACE)
    plt.barh(range(top_k), values[order][::-1], color="#2a78d6")
    plt.yticks(range(top_k), names[::-1], color=_INK)
    plt.xlabel(xlabel, color=_MUTED)
    plt.title(title, color=_INK)
    plt.gca().tick_params(colors=_MUTED)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=_SURFACE)
    plt.close()


def _attention_heatmap(per_lh, genes, top_genes_idx, held, save_path):
    g = genes[top_genes_idx]
    vals = per_lh[:, top_genes_idx]  # (L*H, n_top)
    n_lh = vals.shape[0]
    fig, ax = plt.subplots(figsize=(0.42 * vals.shape[1], max(2.5, 0.28 * n_lh)),
                           facecolor=_SURFACE)
    im = ax.imshow(vals, aspect="auto", cmap=_CMAP)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g, rotation=90, fontsize=6, color=_INK)
    ax.set_yticks(range(n_lh))
    ax.set_yticklabels([f"L{l // 4 + 1}-H{l % 4 + 1}" for l in range(n_lh)],
                       fontsize=6, color=_INK)
    ax.set_title(f"[CLS] attention to genes -- held_out={held}", color=_INK)
    ax.tick_params(colors=_MUTED)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_color("#c3c2b7")
    cb = fig.colorbar(im, ax=ax, fraction=0.02)
    cb.set_label("mean attention", fontsize=8, color=_MUTED)
    cb.ax.tick_params(labelsize=7, colors=_MUTED)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=_SURFACE)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _fold_cache_path(held):
    return FIG_DIR / f"tn_{held}_cache.npz"


def run_explain(config, held_out=None):
    set_seed(config["seed"])
    get_device()

    X_all, y_all, batch_all, _ = load_binary(config)
    batches = [str(held_out)] if held_out else [str(b) for b in np.unique(batch_all)]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(config["seed"])

    # gene -> list of per-fold scores / ranks (1 = most important)
    acc_shap = {}
    acc_attn = {}
    shap_ranks = {}
    attn_ranks = {}
    report = ["Tumor/normal Transformer interpretability (SHAP + attention)",
              "=" * 60, ""]

    for held in batches:
        cache = _fold_cache_path(held)
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            genes = z["genes"]
            shap_mat, X_te_sub = z["shap_mat"], z["X_te_sub"]
            per_lh = z["per_lh"]
            n_normals, n_tumors = int(z["n_normals"]), int(z["n_tumors"])
            print(f"\n{'='*60}\nfold held_out={held}: [cache] loaded")
        else:
            model, cfg = load_fold_model(config, held)
            X_tr, X_te, y_tr, y_te, genes = fold_data(config, held)
            genes = np.asarray(genes)
            print(f"\n{'='*60}\nfold held_out={held}: train {X_tr.shape}, "
                  f"test {X_te.shape}, genes {len(genes)}")

            bg_idx = stratified_subset(X_tr, y_tr, N_BG, rng)
            te_idx = stratified_subset(X_te, y_te, N_TEST, rng)
            X_bg, X_te_sub = X_tr[bg_idx], X_te[te_idx]
            n_normals = int((y_te[te_idx] == 0).sum())
            n_tumors = int((y_te[te_idx] == 1).sum())
            print(f"  background {X_bg.shape}, explain {X_te_sub.shape} "
                  f"(test normals {n_normals}, tumors {n_tumors})")

            print("  [1/2] Deep SHAP (gradient x input)...")
            shap_mat, _, _ = deep_shap_importance(model, X_bg, X_te_sub)
            print("  [2/2] [CLS] attention...")
            _, per_lh = attention_importance(model, X_te_sub)
            del model
            gc.collect()

            np.savez(cache, genes=genes, shap_mat=shap_mat, X_te_sub=X_te_sub,
                     per_lh=per_lh, n_normals=n_normals, n_tumors=n_tumors)
            print(f"  [cache] saved -> {cache.name}")

        shap_imp = np.abs(shap_mat).mean(axis=0)   # (n_genes,)
        attn_imp = per_lh.mean(axis=0)             # (n_genes,) mean over layers/heads
        order = np.argsort(shap_imp)[::-1]
        attn_order = np.argsort(attn_imp)[::-1]

        # ---- figures ----
        plt.figure(figsize=(9, 7))
        shap.summary_plot(shap_mat, X_te_sub, feature_names=genes.tolist(),
                          max_display=TOP_K, show=False)
        plt.title(f"SHAP summary -- tumor vs normal ({held})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"tn_shap_summary_{held}.png", dpi=300, bbox_inches="tight")
        plt.close()
        _top_gene_bar(shap_imp, genes, f"Top genes by |SHAP| (tumor vs normal) -- {held}",
                      "mean |SHAP|", FIG_DIR / f"tn_shap_top_{held}.png")

        top_idx = attn_order[:MAX_GENE_COLS]
        _attention_heatmap(per_lh, genes, top_idx, held,
                           FIG_DIR / f"tn_attention_{held}.png")
        _top_gene_bar(attn_imp, genes, f"Top genes by mean [CLS] attention -- {held}",
                      "mean attention", FIG_DIR / f"tn_attention_top_{held}.png")

        # ---- record per-fold top genes ----
        report.append(f"[{held}]  test normals={n_normals} tumors={n_tumors}")
        report.append(f"  Top SHAP  : {', '.join(genes[order[:15]])}")
        report.append(f"  Top attend: {', '.join(genes[attn_order[:15]])}")
        report.append("")

        # ---- accumulate scores + ranks ----
        shap_rank = np.empty(len(genes), dtype=float)
        attn_rank = np.empty(len(genes), dtype=float)
        shap_rank[order] = np.arange(1, len(genes) + 1)
        attn_rank[attn_order] = np.arange(1, len(genes) + 1)
        for gi, g in enumerate(genes):
            acc_shap[g] = acc_shap.get(g, []) + [float(shap_imp[gi])]
            acc_attn[g] = acc_attn.get(g, []) + [float(attn_imp[gi])]
            shap_ranks[g] = shap_ranks.get(g, []) + [float(shap_rank[gi])]
            attn_ranks[g] = attn_ranks.get(g, []) + [float(attn_rank[gi])]

    # ---- rank-based aggregation (robust to scale/outliers) ----
    # A gene absent from a fold's HVG selection gets the worst possible rank
    # (n_genes + 1), so the mean penalizes genes that are not consistently
    # selected across platforms instead of rewarding single-fold outliers.
    n_total_folds = len(batches)
    n_g = config.get("model", {}).get("n_genes", 5000)
    penalty = n_g + 1.0

    def _mean_rank(rank_dict, g):
        ranks = rank_dict.get(g, [])
        padded = list(ranks) + [penalty] * (n_total_folds - len(ranks))
        return float(np.mean(padded))

    genes_all = np.array(sorted(acc_shap.keys()))
    mean_shap = np.array([np.mean(acc_shap[g]) for g in genes_all])
    mean_attn = np.array([np.mean(acc_attn[g]) for g in genes_all])
    mean_shap_rank = np.array([_mean_rank(shap_ranks, g) for g in genes_all])
    mean_attn_rank = np.array([_mean_rank(attn_ranks, g) for g in genes_all])
    n_folds = np.array([len(acc_shap[g]) for g in genes_all])

    # combined = mean rank across folds (absent -> worst rank), summed over the
    # two methods; lower is better. Both are on the same [1, n_genes+1] scale.
    combined = mean_shap_rank + mean_attn_rank
    comb_order = np.argsort(combined)

    out_csv = Path("results/tn_interpretability_top_genes.csv")
    lines = ["gene,mean_shap_rank,mean_attn_rank,combined_rank_score,n_folds,"
             "mean_shap_importance,mean_attention"]
    for gi in comb_order:
        g = genes_all[gi]
        lines.append(f"{g},{mean_shap_rank[gi]:.2f},{mean_attn_rank[gi]:.2f},"
                     f"{combined[gi]:.2f},{n_folds[gi]},"
                     f"{mean_shap[gi]:.6g},{mean_attn[gi]:.6g}")
    out_csv.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] consensus genes -> {out_csv}")

    # ---- overlap with the RF proliferation signature ----
    rf_sig = _load_rf_signature()
    overlap = [
        "Overlap with RF proliferation signature (tumor_normal_top_genes.csv)",
        "  gene  : Transformer SHAP-rank / attn-rank (n_folds)",
    ]
    for g in rf_sig:
        if g in shap_ranks:
            overlap.append(f"  {g:<8}: SHAP-rank {np.mean(shap_ranks[g]):.0f} / "
                           f"attn-rank {np.mean(attn_ranks[g]):.0f} "
                           f"(n_folds={len(shap_ranks[g])})")
        else:
            overlap.append(f"  {g:<8}: not in any fold's HVG top-5000")
    overlap.append("")

    report += [
        "Aggregate top genes (rank-based mean over folds, SHAP + attention)",
        "-" * 60,
        ", ".join(genes_all[comb_order[:20]]),
        "",
        *overlap,
        "Method notes:",
        "  - SHAP = Deep SHAP (Gradient x Input, background = stratified train mean),",
        "    attributed to logit[tumor]-logit[normal] (the tumor-vs-normal decision).",
        "  - Attention = mean [CLS]->gene self-attention across layers/heads/samples.",
        "  - Aggregation = per-gene mean RANK across folds (1 = most important),",
        "    averaged over the two methods; absent-from-fold genes get the worst",
        "    rank, so the consensus rewards genes consistent across platforms.",
        "  - Both are inference-only (no retraining), run on CPU from local checkpoints.",
        "  - HVG selects genes per fold on TRAIN variance, so a gene may be absent",
        "    from some folds (n_folds < 3) -- such genes are penalized in the rank.",
    ]
    out_txt = Path("results/tn_interpretability_report.txt")
    out_txt.write_text("\n".join(report), encoding="utf-8")
    print(f"[OK] report -> {out_txt}")
    return combined, genes_all


def _load_rf_signature(n=15):
    """Top-n genes from the RF feature-importance reference, if present."""
    p = Path("results/tumor_normal_top_genes.csv")
    if not p.exists():
        return []
    genes = []
    for line in p.read_text(encoding="utf-8").splitlines()[1:n + 1]:
        genes.append(line.split(",")[0].strip())
    return genes


if __name__ == "__main__":
    import sys
    import traceback

    try:
        config = load_config()
        held = sys.argv[1] if len(sys.argv) > 1 else None
        run_explain(config, held)
        print("\n[SUCCESS] interpretability complete")
    except Exception as e:  # noqa: BLE001
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
