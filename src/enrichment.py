"""
enrichment.py -- GO/KEGG pathway enrichment of the Transformer consensus genes.

Loads the cross-fold consensus genes (rank-based, from
results/tn_interpretability_top_genes.csv), submits them to Enrichr, and renders
an enrichment bubble plot for the README + technical report.

Uses the Enrichr REST API directly (multipart addList + enrich GET) -- same
endpoints and payload as gseapy, but with no gseapy dependency.

    python -m src.enrichment            # top-100 consensus genes, all libraries
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .utils import load_config

BASE_URL = "https://maayanlab.cloud/Enrichr"
LIBS = [
    "KEGG_2021_Human",
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
    "Reactome_2022",
]

LIB_TAG = {
    "KEGG_2021_Human": "KEGG",
    "GO_Biological_Process_2023": "GO:BP",
    "GO_Molecular_Function_2023": "GO:MF",
    "GO_Cellular_Component_2023": "GO:CC",
    "Reactome_2022": "Reactome",
}

# Validated default palette (dataviz skill): blue sequential ramp + ink/surface.
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
CMAP = LinearSegmentedColormap.from_list("blue", SEQ_BLUE)

FDR = 0.05


def _add_list(genes, description):
    r = requests.post(
        f"{BASE_URL}/addList",
        files={"list": (None, "\n".join(genes)), "description": (None, description)},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["userListId"]


def _enrich(user_list_id, lib):
    r = requests.get(
        f"{BASE_URL}/enrich",
        params={"userListId": user_list_id, "backgroundType": lib},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()[lib]


def parse_rows(lib, data):
    """Enrichr row -> dict. Row = [rank, term, p, odds_ratio, combined, genes,
    adj_p, old_p, old_adj_p]."""
    rows = []
    for row in data:
        rows.append({
            "library": lib,
            "term": row[1],
            "p_value": float(row[2]),
            "odds_ratio": float(row[3]),
            "combined_score": float(row[4]),
            "overlap": len(row[5]),
            "genes": row[5],
            "adj_p_value": float(row[6]),
        })
    return rows


def short_term(term):
    """Strip the GO / Reactome accession suffix for cleaner labels."""
    for sep in (" (GO:", " R-HSA-", " (Homo sapiens)"):
        if sep in term:
            term = term.split(sep)[0]
    return term


def load_consensus_genes(n=100):
    p = Path("results/tn_interpretability_top_genes.csv")
    df = pd.read_csv(p)
    # combined_rank_score is ascending (best first); take the top n.
    return df.sort_values("combined_rank_score")["gene"].head(n).astype(str).tolist()


def run_enrichment(config, n_genes=100, top_terms=15):
    genes = load_consensus_genes(n=n_genes)
    print(f"Consensus genes submitted: {len(genes)}")
    print(f"  first 10: {', '.join(genes[:10])}")

    uid = _add_list(genes, "tumor_normal_transformer_consensus")
    print(f"userListId = {uid}")

    all_rows = []
    for lib in LIBS:
        try:
            rows = parse_rows(lib, _enrich(uid, lib))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {lib}: {e}")
            continue
        rows.sort(key=lambda r: r["adj_p_value"])
        n_sig = sum(1 for r in rows if r["adj_p_value"] < FDR)
        print(f"  {lib}: {n_sig} significant terms (adj p < {FDR}) "
              f"/ {len(rows)} total")
        all_rows.extend(rows)
        time.sleep(1.5)

    # ---- save full results CSV ----
    rows_out = []
    for r in all_rows:
        rows_out.append({
            "library": r["library"],
            "term": short_term(r["term"]),
            "overlap": r["overlap"],
            "p_value": r["p_value"],
            "adj_p_value": r["adj_p_value"],
            "significant": r["adj_p_value"] < FDR,
            "odds_ratio": r["odds_ratio"],
            "combined_score": r["combined_score"],
            "genes": ";".join(r["genes"]),
        })
    out_csv = Path("results/enrichment_results.csv")
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"[OK] results -> {out_csv}")

    # ---- summary of the significant terms ----
    sig = [r for r in all_rows if r["adj_p_value"] < FDR]
    print("\nSignificant terms (adj p < 0.05):")
    for r in sig:
        print(f"  {LIB_TAG[r['library']]:8s} {short_term(r['term'])} "
              f"(adj p={r['adj_p_value']:.2g}, overlap={r['overlap']})")

    # ---- bubble figure (combined, top terms across all libraries) ----
    plot_bubbles(all_rows, top_terms=top_terms)
    return all_rows


def plot_bubbles(all_rows, top_terms=15, out=None):
    out = out or Path("results/figures/enrichment_bubble.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    d = sorted(all_rows, key=lambda r: r["adj_p_value"])[:top_terms][::-1]
    labels = [f"{LIB_TAG[r['library']]}  {short_term(r['term'])}" for r in d]
    x = -np.log10(np.clip([r["adj_p_value"] for r in d], 1e-300, None))
    y = np.arange(len(d))
    overlap = np.array([r["overlap"] for r in d], dtype=float)
    odds = np.array([r["odds_ratio"] for r in d], dtype=float)
    sig = np.array([r["adj_p_value"] < FDR for r in d])

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(d) + 2.2), facecolor=SURFACE)

    s = 50 + (overlap / max(overlap.max(), 1)) * 340
    sc = ax.scatter(x, y, s=s, c=odds, cmap=CMAP, edgecolors=INK,
                    linewidths=0.4, alpha=0.9)
    # ring the FDR-significant bubbles so significance is not color-alone
    ax.scatter(x[sig], y[sig], s=s[sig] + 40, facecolors="none",
               edgecolors=INK, linewidths=1.1, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel("-log10(adjusted p-value)", fontsize=10, color=MUTED)
    ax.axvline(-np.log10(FDR), color=MUTED, linestyle="--", linewidth=1.0)
    ax.text(-np.log10(FDR) + 0.12, len(d) - 0.4, "FDR 0.05", fontsize=8,
            color=MUTED, ha="left")
    ax.set_xlim(left=0)
    ax.set_xlim(right=max(x.max(), 2.0) * 1.15)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED)
    for xi, yi, ov in zip(x, y, overlap):
        ax.text(xi, yi, f"  {int(ov)}", va="center", ha="left",
                fontsize=7.5, color=MUTED)

    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Odds Ratio", fontsize=9, color=MUTED)
    cb.ax.tick_params(labelsize=8, colors=MUTED)

    ax.set_title(
        "Enrichment of the tumor/normal Transformer consensus genes (Enrichr)\n"
        "size = overlapping genes, ring = FDR-significant (adj p < 0.05)",
        fontsize=12, color=INK, weight="bold", loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[OK] bubble plot -> {out}")


if __name__ == "__main__":
    import sys
    import traceback

    try:
        config = load_config()
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
        run_enrichment(config, n_genes=n)
        print("\n[SUCCESS] enrichment complete")
    except Exception as e:  # noqa: BLE001
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
