"""
gene_mapping.py -- probe->gene mapping + gene-level matrix assembly

This module turns the raw probe-level GEO expression matrices into a clean,
gene-symbol-level, batch-annotated matrix WITHOUT doing any cross-sample
normalization/feature-selection (those happen inside each LODO fold later).

Design decisions (see project report / CLAUDE.md):
- All three datasets are Affymetrix GPL570 (HG-U133 Plus 2.0); the probe->gene
  map is parsed once from a local .soft.gz family file (no network needed).
- Multiple probes mapping to one gene are collapsed by mean (on log2 scale).
- Each dataset is put on a common log2 scale first: datasets whose VALUE column
  is still linear (max > 50) are log2(x+1)-transformed; already-log2 datasets
  are left as-is. This fixes the LUAD-is-linear / BRCA-COAD-are-log2 mismatch.
- The output matrix is the INTERSECTION of gene symbols across all datasets, so
  it contains no missing values by construction.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd


def parse_gpl570_gene_map(soft_file):
    """Stream-parse the GPL570 platform table from a .soft.gz family file.

    Returns
    -------
    dict[str, str]
        probe_id -> gene_symbol (first symbol; control probes with no symbol
        are dropped).
    """
    soft_file = Path(soft_file)
    mapping: dict[str, str] = {}

    with gzip.open(soft_file, "rt", encoding="utf-8", errors="ignore") as fh:
        in_table = False
        id_idx = sym_idx = None

        for line in fh:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if not in_table:
                continue
            if line.startswith("!platform_table_end"):
                break

            if line.startswith("ID\t"):  # header row
                cols = line.rstrip("\n").split("\t")
                id_idx = cols.index("ID")
                sym_idx = cols.index("Gene Symbol")
                continue

            if id_idx is None:
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(id_idx, sym_idx):
                continue

            probe = parts[id_idx]
            symbol = parts[sym_idx].strip()

            # Affymetrix control probes have no meaningful symbol.
            if not symbol or symbol in {"---", ""}:
                continue
            # "A /// B /// C" -> keep the first symbol.
            symbol = symbol.split(" /// ")[0].strip()
            if symbol:
                mapping[probe] = symbol

    if not mapping:
        raise ValueError(f"No probe->gene mappings parsed from {soft_file}")
    return mapping


def _log2_if_linear(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """log2(x+1) a dataset whose values look linear (not already log2)."""
    colmax = df.max(axis=0).max()
    if np.isfinite(colmax) and colmax > threshold:
        return np.log2(df.clip(lower=0.0) + 1.0)
    return df


def extract_tumor_normal(raw_dir, gse: str, cancer: str) -> dict[str, str]:
    """Extract per-sample tumor/normal status from a dataset's metadata CSV.

    Returns
    -------
    dict[str, str]
        sample GSM id -> "tumor" | "normal" | "exclude". "exclude" marks samples
        that are neither tumor nor normal tissue (e.g. cell lines), which must be
        dropped from a tumor-vs-normal task.

    The three datasets store the status in different fields (verified 2026-09-11):
        GSE45827 (BRCA): source_name_ch1 -> "Human {subtype} Tumor Sample",
                         "Human Normal", "Human CellLine" (14 cell lines).
        GSE31210 (LUAD): characteristics_ch1.0.tissue -> "primary lung tumor"
                         (226) / "normal lung" (20).
        GSE39582 (COAD): source_name_ch1 -> "primary colorectal Adenocarcinoma"
                         (566) / "non tumoral colorectal mucosa" (19). Its
                         geo_accession is a list-string, so use sample_id.
    """
    meta = pd.read_csv(Path(raw_dir) / f"{gse}_{cancer}_metadata.csv")
    out: dict[str, str] = {}

    if cancer == "BRCA":
        src = meta["source_name_ch1"].astype(str)
        for gsm, s in zip(meta["geo_accession"].astype(str), src):
            if "Normal" in s:
                out[gsm] = "normal"
            elif "CellLine" in s:
                out[gsm] = "exclude"
            else:
                out[gsm] = "tumor"
    elif cancer == "LUAD":
        tissue = meta["characteristics_ch1.0.tissue"].astype(str)
        for gsm, t in zip(meta["geo_accession"].astype(str), tissue):
            out[gsm] = "normal" if "normal" in t.lower() else "tumor"
    elif cancer == "COAD":
        src = meta["source_name_ch1"].astype(str)
        for gsm, s in zip(meta["sample_id"].astype(str), src):
            out[gsm] = "normal" if "non tumoral" in s.lower() else "tumor"
    else:
        raise ValueError(f"Unknown cancer type: {cancer}")

    return out


def assemble_gene_matrix(config):
    """Load raw probe matrices, map to genes, intersect, return gene-level data.

    Returns
    -------
    X : pd.DataFrame (n_samples x n_genes), log2 gene expression
    labels : list[str]  cancer type per sample (BRCA/LUAD/COAD)
    batches : list[str] GSE id per sample
    genes : list[str]   common gene symbols (columns)
    sample_ids : list[str] GSM sample ids (row index)
    tumor_normal : list[str] tumor/normal/exclude status per sample (aligned)
    """
    raw_dir = Path(config["data"]["raw_dir"])
    datasets = config["data"]["datasets"]  # {BRCA: GSE45827, ...}

    # Build the probe->gene map once from any local .soft.gz family file.
    soft_files = sorted(raw_dir.glob("*_family.soft.gz"))
    if not soft_files:
        raise FileNotFoundError(f"No *_family.soft.gz found in {raw_dir}")
    probe2gene = parse_gpl570_gene_map(soft_files[0])
    print(f"[OK] GPL570 probe->gene map: {len(probe2gene)} probes")

    per_dataset: dict[str, pd.DataFrame] = {}
    for cancer, gse in datasets.items():
        expr_file = raw_dir / f"{gse}_{cancer}_expression.csv"
        df = pd.read_csv(expr_file, index_col=0)  # probe x sample
        df = _log2_if_linear(df)                  # put on common log2 scale

        gene_col = df.index.map(probe2gene)
        keep = ~pd.isna(gene_col)
        mapped = df[keep].copy()
        mapped.index = gene_col[keep]
        gene_expr = mapped.groupby(level=0).mean()  # gene x sample (mean over probes)
        per_dataset[gse] = gene_expr
        print(f"[OK] {cancer} ({gse}): {df.shape[0]} probes -> "
              f"{gene_expr.shape[0]} genes x {gene_expr.shape[1]} samples")

    # Inner-join on gene symbols: only genes present in all three datasets.
    common_genes = sorted(
        set.intersection(*(set(g.index) for g in per_dataset.values()))
    )
    print(f"[OK] Common genes (intersection): {len(common_genes)}")

    frames, labels, batches, sample_ids, tumor_normal = [], [], [], [], []
    for cancer, gse in datasets.items():
        sub = per_dataset[gse].loc[common_genes].T  # sample x gene
        tn_map = extract_tumor_normal(raw_dir, gse, cancer)
        frames.append(sub)
        labels.extend([cancer] * sub.shape[0])
        batches.extend([gse] * sub.shape[0])
        sample_ids.extend(sub.index)
        tumor_normal.extend([tn_map.get(sid, "tumor") for sid in sub.index])

    X = pd.concat(frames, axis=0)
    assert not np.isnan(X.values).any(), "assembled matrix must have no NaN"
    return X, labels, batches, common_genes, sample_ids, tumor_normal
