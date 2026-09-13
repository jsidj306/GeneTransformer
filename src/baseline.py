"""
baseline.py -- RandomForest / SVM baselines under the leak-free LODO protocol

The primary evaluation is leave-one-dataset-out (LODO): each held-out GSE is a
completely unseen cancer type. We also run a stratified random-split control to
isolate the split as the cause of the performance difference: both use the same
train-only HVG/z-score preprocessing, only the split differs. The contrast
(random ~100% vs LODO ~0%) is the evidence that the earlier 100% was batch
leakage, not biology.
"""
from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from .utils import load_config, calculate_metrics
from .cross_dataset import load_processed, encode_labels, lodo_folds, prepare_fold


def make_models(seed=42, class_weight="balanced"):
    """Build the RF / SVM baselines.

    class_weight can be "balanced" (default, for the 3-class cancer task) or an
    explicit dict {label: weight} for cost-sensitive learning on the imbalanced
    binary tumor-vs-normal task (e.g. {0: 18.0, 1: 1.0}).
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed, class_weight=class_weight
        ),
        "SVM": SVC(kernel="rbf", random_state=seed, class_weight=class_weight),
    }


def fit_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred)


def _aggregate(metrics_list):
    """Mean +/- std over folds for accuracy and macro-F1."""
    out = {}
    for key in ("accuracy", "f1"):
        vals = [m[key] for m in metrics_list]
        out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "per_fold": vals}
    return out


def run_lodo_baselines(config):
    X, y_str, batch, genes = load_processed(config)
    seed = config.get("seed", 42)
    results = {"leave_one_dataset_out": {}, "random_split_control": {}}

    # ---- LODO: hold out one whole GSE (unseen cancer type) ----
    print("=" * 60)
    print("LODO evaluation (leave one dataset out)")
    print("=" * 60)
    lodo_fold_metrics = {m: [] for m in make_models(seed)}

    for fold in lodo_folds(config):
        held = fold["held_out"]
        print(f"\n--- held_out = {held} "
              f"(train {fold['X_train'].shape}, test {fold['X_test'].shape}) ---")
        for name, model in make_models(seed).items():
            m = fit_and_evaluate(
                model, fold["X_train"], fold["X_test"],
                fold["y_train"], fold["y_test"]
            )
            lodo_fold_metrics[name].append(m)
            print(f"  {name:<14} acc={m['accuracy']:.4f} macroF1={m['f1']:.4f}")

    for name in lodo_fold_metrics:
        agg = _aggregate(lodo_fold_metrics[name])
        results["leave_one_dataset_out"][name] = agg
        print(f"\n  {name}: macroF1 {agg['f1']['mean']:.4f} +/- {agg['f1']['std']:.4f}")

    # ---- Random-split control (same train-only scaling, random split) ----
    print("\n" + "=" * 60)
    print("Random-split control (leakage reference)")
    print("=" * 60)
    y, class_names = encode_labels(y_str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    ctrl_fold_metrics = {m: [] for m in make_models(seed)}

    for tr_idx, te_idx in skf.split(X, y):
        X_tr = X[tr_idx]; X_te = X[te_idx]
        y_tr = y[tr_idx]; y_te = y[te_idx]
        # same leak-free prep, but split is random (batch leaks)
        X_tr, X_te, _ = prepare_fold(X_tr, X_te, batch[tr_idx], batch[te_idx], config)
        for name, model in make_models(seed).items():
            m = fit_and_evaluate(model, X_tr, X_te, y_tr, y_te)
            ctrl_fold_metrics[name].append(m)

    for name in ctrl_fold_metrics:
        agg = _aggregate(ctrl_fold_metrics[name])
        results["random_split_control"][name] = agg
        print(f"\n  {name}: macroF1 {agg['f1']['mean']:.4f} +/- {agg['f1']['std']:.4f}")

    return results


def save_results(results, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "lodo_results.yaml", "w", encoding="utf-8") as fh:
        yaml.dump(results, fh, default_flow_style=False, allow_unicode=True)

    lines = ["Baseline LODO vs random-split (macro-F1)", "=" * 60, ""]
    for split_name in ("leave_one_dataset_out", "random_split_control"):
        lines.append(f"[{split_name}]")
        for model_name, agg in results[split_name].items():
            lines.append(
                f"  {model_name:<14} macroF1 "
                f"{agg['f1']['mean']:.4f} +/- {agg['f1']['std']:.4f}"
            )
        lines.append("")
    lines += [
        "Interpretation: LODO holds out a completely unseen cancer type (batch == ",
        "class confound), so a near-zero LODO macro-F1 vs ~1.0 random-split macro-F1 ",
        "demonstrates that the earlier 100% accuracy was batch leakage, not biology.",
    ]
    with open(out_dir / "lodo_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n[OK] Results -> {out_dir / 'lodo_results.yaml'}")


if __name__ == "__main__":
    import sys
    import traceback

    try:
        config = load_config()
        results = run_lodo_baselines(config)
        save_results(results)
        print("\n[SUCCESS] LODO baseline evaluation complete")
    except Exception as e:  # noqa: BLE001
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
