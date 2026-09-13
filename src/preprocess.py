"""
preprocess.py -- assemble a clean, gene-level, batch-annotated matrix

This module does the DATA ASSEMBLY only: probe->gene mapping, per-dataset log2
scale normalization, gene intersection, and batch (GSE) tracking. It does NOT
do any cross-sample normalization or feature selection -- those must happen
inside each leave-one-dataset-out fold (see src/cross_dataset.py), fit on the
training split only, to avoid leakage.

Outputs (all in data/processed/):
    log2_expression.npy  : (n_samples x n_genes) log2 gene expression, no NaN
    genes.npy            : gene symbols (columns)
    labels.npy           : cancer type per sample (BRCA/LUAD/COAD)
    batch.npy            : GSE id per sample (required for LODO + ComBat)
    tumor_normal.npy     : tumor/normal/exclude status per sample (for binary task)
    sample_meta.csv      : sample_id, gse_id, cancer_type
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .gene_mapping import assemble_gene_matrix


def save_processed(config):
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    X, labels, batches, genes, sample_ids, tumor_normal = assemble_gene_matrix(config)

    np.save(processed_dir / "log2_expression.npy", X.values.astype(np.float32))
    np.save(processed_dir / "genes.npy", np.array(genes))
    np.save(processed_dir / "labels.npy", np.array(labels))
    np.save(processed_dir / "batch.npy", np.array(batches))
    np.save(processed_dir / "tumor_normal.npy", np.array(tumor_normal))

    meta = pd.DataFrame({
        "sample_id": sample_ids,
        "gse_id": batches,
        "cancer_type": labels,
        "tumor_normal": tumor_normal,
    })
    meta.to_csv(processed_dir / "sample_meta.csv", index=False)

    stats = {
        "total_samples": X.shape[0],
        "total_genes": X.shape[1],
        "classes": sorted(set(labels)),
        "batches": sorted(set(batches)),
        "samples_per_class": {c: labels.count(c) for c in sorted(set(labels))},
        "samples_per_batch": {b: batches.count(b) for b in sorted(set(batches))},
        "tumor_normal": {t: tumor_normal.count(t) for t in sorted(set(tumor_normal))},
        "has_nan": bool(np.isnan(X.values).any()),
        "scale": "log2 (per-dataset normalized)",
    }
    with open(processed_dir / "preprocessing_stats.yaml", "w", encoding="utf-8") as fh:
        yaml.dump(stats, fh, default_flow_style=False, allow_unicode=True)

    print("\n" + "=" * 60)
    print("Assembly complete")
    print("=" * 60)
    print(f"Matrix: {X.shape} (samples x genes)")
    for c in sorted(set(labels)):
        print(f"  {c}: {labels.count(c)} samples")
    for b in sorted(set(batches)):
        print(f"  batch {b}: {batches.count(b)} samples")
    for t in sorted(set(tumor_normal)):
        print(f"  tumor_normal '{t}': {tumor_normal.count(t)} samples")
    print(f"NaN present: {stats['has_nan']}")
    print(f"\nOutputs -> {processed_dir}/")
    print("  log2_expression.npy, genes.npy, labels.npy, batch.npy, tumor_normal.npy, sample_meta.csv")


if __name__ == "__main__":
    try:
        with open("config.yaml", "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        save_processed(config)
    except Exception as e:  # noqa: BLE001
        print(f"\n[ERROR] Preprocessing failed: {e}")
        traceback.print_exc()
        sys.exit(1)
