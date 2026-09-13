"""
cross_dataset.py -- leak-free leave-one-dataset-out (LODO) preparation

The honest evaluation protocol for this project. Because each GEO dataset is a
single cancer type, "batch" and "class" are perfectly confounded (1:1). A
random train/test split lets the model memorize the platform distribution
rather than learn biology. LODO holds out one whole GSE as the test set so the
held-out cancer type is completely unseen during training.

For each fold, every cross-sample statistic is fit on the TRAINING split only:
    log2 (already done at assembly, per-dataset)
 -> ComBat (batch-only, reference batch -- see combat_correct note)
 -> highly-variable-gene selection (train variance only)
 -> z-score (train mean/std only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .utils import load_config


def load_processed(config):
    """Load the assembled gene-level matrix + labels + batch ids."""
    processed_dir = Path(config["data"]["processed_dir"])
    X = np.load(processed_dir / "log2_expression.npy", allow_pickle=True)
    y = np.load(processed_dir / "labels.npy", allow_pickle=True)
    batch = np.load(processed_dir / "batch.npy", allow_pickle=True)
    genes = np.load(processed_dir / "genes.npy", allow_pickle=True)
    return X.astype(np.float64), np.asarray(y), np.asarray(batch), np.asarray(genes)


def encode_labels(y_str):
    """Encode string labels to ints; return (y_int, class_names)."""
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_int = le.fit_transform(y_str)
    return y_int, le.classes_


def select_hvg_train(X_train, n_genes):
    """Top-n genes by variance, computed on the TRAINING split only."""
    var = np.var(X_train, axis=0)
    idx = np.argsort(var)[::-1][:n_genes]
    return idx


def zscore_fit_transform(X_train, X_test):
    """z-score using mean/std fit on train only, applied to train and test."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-8] = 1.0  # guard zero-variance genes
    return (X_train - mu) / sd, (X_test - mu) / sd


def combat_correct(X_train, batch_train, X_test, batch_test):
    """Reference-batch ComBat (batch-only), fit on train only.

    NOTE: cancer type is intentionally NOT passed as a covariate. With the 1:1
    batch<->class confound the cancer covariate is perfectly collinear with the
    batch indicator, which makes the design matrix singular. Batch-only ComBat
    removes the batch effect; because batch == class here, this also removes the
    between-class signal -- a documented limitation, not a bug.

    If the test split's batches were all seen in training (random split), they
    are transformed with the fitted model. A novel held-out batch (LODO) has no
    estimable gamma/delta, so it is left unchanged (reference assumption
    gamma=0, delta=1) and mapped to the training scale by the later z-score.
    """
    if len(np.unique(batch_train)) < 2:
        return X_train, X_test

    try:
        from pycombat import Combat
    except ImportError:
        print("  [WARN] pycombat not available; skipping ComBat")
        return X_train, X_test

    combat = Combat()
    combat.fit(X_train, np.asarray(batch_train))  # X=None -> batch-only
    X_train_c = combat.transform(X_train, np.asarray(batch_train))

    test_batches = np.unique(batch_test)
    if len(test_batches) == len(combat.batches_) and all(
        b in combat.batches_ for b in test_batches
    ):
        X_test_c = combat.transform(X_test, np.asarray(batch_test))
    else:
        X_test_c = X_test  # novel held-out batch: leave as reference
    return X_train_c, X_test_c


def prepare_fold(X_train, X_test, batch_train, batch_test, config):
    """Run the full leak-free fold prep: ComBat -> HVG -> z-score."""
    pre = config.get("preprocess", {})

    if pre.get("batch_correction") == "combat":
        X_train, X_test = combat_correct(X_train, batch_train, X_test, batch_test)

    n_genes = config.get("model", {}).get("n_genes", 5000)
    idx = select_hvg_train(X_train, n_genes)
    X_train, X_test = X_train[:, idx], X_test[:, idx]

    X_train, X_test = zscore_fit_transform(X_train, X_test)
    return X_train, X_test, idx


def lodo_folds(config):
    """Yield (held_out, X_train, X_test, y_train, y_test, batch_train) per GSE."""
    X, y_str, batch, genes = load_processed(config)
    y, class_names = encode_labels(y_str)

    for held_out in np.unique(batch):
        test_mask = batch == held_out
        train_mask = ~test_mask

        X_train, X_test, idx = prepare_fold(
            X[train_mask], X[test_mask], batch[train_mask], batch[test_mask], config
        )
        yield {
            "held_out": held_out,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y[train_mask],
            "y_test": y[test_mask],
            "genes": genes[idx],
        }


if __name__ == "__main__":
    # Smoke test: print fold sizes and confirm no leakage (held-out class absent
    # from train).
    config = load_config()
    for fold in lodo_folds(config):
        held = fold["held_out"]
        train_classes = set(fold["y_train"])
        test_classes = set(fold["y_test"])
        print(
            f"held_out={held}: train={fold['X_train'].shape} "
            f"(classes {sorted(train_classes)}) "
            f"test={fold['X_test'].shape} (classes {sorted(test_classes)}) "
            f"genes={len(fold['genes'])}"
        )
        assert not train_classes & test_classes, "held-out class leaked into train!"
    print("[OK] no leakage across folds")
