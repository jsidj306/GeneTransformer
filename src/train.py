"""
train.py -- training loop (early stopping + LR schedule) under LODO

The entry point now runs leave-one-dataset-out (LODO): each held-out GSE is a
completely unseen cancer type, so per-fold the model is trained on two classes
and evaluated on the third. This is the honest protocol -- the held-out class
is novel by design, so near-chance accuracy is the expected (and meaningful)
result that exposes the earlier 100% as batch leakage.
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import yaml
import time
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

from .model import GeneTransformer, create_model
from .utils import set_seed, load_config, get_device, format_time, plot_training_history, calculate_metrics


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_epoch(model, train_loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    total_samples = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = F.cross_entropy(outputs, y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_samples += len(y)

    return total_loss / total_samples


def evaluate(model, data_loader, device, class_weights=None):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = F.cross_entropy(outputs, y, weight=class_weights)
            total_loss += loss.item() * len(y)
            total_samples += len(y)
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(y.cpu())

    avg_loss = total_loss / total_samples
    return avg_loss, torch.cat(all_preds), torch.cat(all_labels)


def predict_scores(model, data_loader, device):
    """P(class=1) scores + true labels, for binary ROC/PR-AUC."""
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for X, y in data_loader:
            logits = model(X.to(device))
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu())
            labels.append(y.cpu())
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def predict_proba_positive(model, X, device, batch_size=256):
    """P(class=1) scores for a raw ndarray X (no labels needed)."""
    from torch.utils.data import DataLoader
    from .dataset import GeneDataset

    model.eval()
    loader = DataLoader(
        GeneDataset(X, np.zeros(len(X), dtype=int)),
        batch_size=batch_size, shuffle=False,
    )
    scores = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(scores).numpy()


def train_model(config, train_loader, val_loader, save_dir='results/checkpoints',
                class_weights=None, monitor='loss'):
    training_config = config.get('training', {})
    device = get_device()
    model = create_model(config).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.get('lr', 1e-3),
        weight_decay=training_config.get('weight_decay', 1e-4),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=training_config.get('lr_patience', 5),
    )

    # monitor 'loss' (minimize, 3-class) or 'auc' / 'pr_auc' (maximize, binary).
    binary = monitor in ('auc', 'pr_auc')
    patience = training_config.get('patience', 10)
    history = {'train_loss': [], 'val_loss': []}
    if binary:
        history['val_auc'] = []
        history['val_pr_auc'] = []
    else:
        history['val_accuracy'] = []
        history['val_f1'] = []

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    epochs = training_config.get('epochs', 100)
    best_score = float('-inf') if binary else float('inf')
    best_ckpt = Path(save_dir) / 'best_model.pt'
    no_improve = 0

    print(f"开始训练，共{epochs}个epoch... (monitor={monitor})")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, device, class_weights)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if binary:
            val_scores, _ = predict_scores(model, val_loader, device)
            val_auc = roc_auc_score(val_labels.numpy(), val_scores)
            val_pr_auc = average_precision_score(val_labels.numpy(), val_scores)
            history['val_auc'].append(float(val_auc))
            history['val_pr_auc'].append(float(val_pr_auc))
            score = val_auc if monitor == 'auc' else val_pr_auc
            improved = score > best_score + 1e-4
        else:
            val_metrics = calculate_metrics(val_labels.numpy(), val_preds.numpy())
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            score = val_loss
            improved = score < best_score - 1e-4

        scheduler.step(val_loss)

        if improved:
            best_score = score
            no_improve = 0
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(),
                 'val_loss': val_loss, 'monitor': monitor,
                 'best_score': float(best_score), 'config': config},
                best_ckpt,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"早停触发，停止训练 (epoch {epoch})")
                break

        if epoch % 5 == 0 or epoch == 1:
            if binary:
                print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | "
                      f"Val PR-AUC: {val_pr_auc:.4f}")
            else:
                print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"Val F1: {val_metrics['f1']:.4f}")

    # restore best weights (covers both monitor directions)
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device)['model_state_dict'])

    total_time = time.time() - start_time
    print(f"训练完成，总耗时: {format_time(total_time)}")
    return history


def run_lodo_transformer(config):
    from .cross_dataset import lodo_folds
    from .dataset import create_dataloaders
    from .model import count_parameters
    from sklearn.model_selection import train_test_split

    set_seed(config['seed'])
    device = get_device()
    seed = config['seed']
    batch_size = config['training'].get('batch_size', 32)

    fold_metrics = []
    for fold in lodo_folds(config):
        held = fold['held_out']
        print(f"\n{'='*60}\nLODO fold: held_out={held}\n{'='*60}")

        # within-train train/val split (stratified; held-out still unseen)
        X_tr, X_val, y_tr, y_val = train_test_split(
            fold['X_train'], fold['y_train'], test_size=0.2,
            stratify=fold['y_train'], random_state=seed,
        )
        train_loader, val_loader = create_dataloaders(
            X_tr, y_tr, X_val, y_val, batch_size=batch_size,
        )

        config['model']['n_genes'] = fold['X_train'].shape[1]
        save_dir = f'results/checkpoints/lodo_{held}'
        train_model(config, train_loader, val_loader, save_dir)

        model = create_model(config).to(device)
        ckpt = Path(save_dir) / 'best_model.pt'
        model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])

        _, test_loader = create_dataloaders(
            fold['X_train'], fold['y_train'], fold['X_test'], fold['y_test'],
            batch_size=batch_size,
        )
        _, test_preds, test_labels = evaluate(model, test_loader, device)
        m = calculate_metrics(test_labels.numpy(), test_preds.numpy())
        fold_metrics.append(m)
        print(f"  {held}: acc={m['accuracy']:.4f} macroF1={m['f1']:.4f}")

    accs = [m['accuracy'] for m in fold_metrics]
    f1s = [m['f1'] for m in fold_metrics]
    print(f"\nTransformer LODO: acc {np.mean(accs):.4f} +/- {np.std(accs):.4f}, "
          f"macroF1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    return {
        'accuracy': {'mean': float(np.mean(accs)), 'std': float(np.std(accs)), 'per_fold': accs},
        'f1': {'mean': float(np.mean(f1s)), 'std': float(np.std(f1s)), 'per_fold': f1s},
    }


def run_tumor_normal_transformer(config):
    """Binary tumor-vs-normal Transformer under LODO, cost-sensitive.

    Uses class_weight (normal upweighted, default 18) and selects the best
    checkpoint by a threshold-free metric (PR-AUC / AUC) instead of accuracy/F1.
    Hard labels (normal_recall / tumor_recall) come from quantile alignment on
    the test score distribution, never a train-fixed threshold.

    Resumable: each fold's metrics are written to
    results/tn_transformer_folds.yaml as soon as the fold finishes, and folds
    already present there are skipped on restart. Colab free-tier GPU sessions
    are often killed mid-run (usage quota / preemption), so this avoids redoing
    completed folds after a disconnect.
    """
    import copy
    from .cross_dataset import prepare_fold
    from .dataset import create_dataloaders
    from .tumor_normal import (
        load_binary, class_weight_dict, quantile_threshold, binary_metrics,
    )
    from sklearn.model_selection import train_test_split

    set_seed(config['seed'])
    device = get_device()
    seed = config['seed']
    batch_size = config['training'].get('batch_size', 32)

    cfg = copy.deepcopy(config)
    cfg['model']['n_classes'] = 2  # binary: 0=normal, 1=tumor

    cw = class_weight_dict(cfg)
    weights = torch.tensor([cw[0], cw[1]], dtype=torch.float32).to(device)
    monitor = cfg.get('tumor_normal', {}).get('primary_metric', 'pr_auc')

    X, y, batch, genes = load_binary(cfg)

    print("=" * 60)
    print("Tumor vs normal -- Transformer (LODO, cost-sensitive)")
    print("=" * 60)
    print(f"class_weight = {cw} | monitor = {monitor} | "
          f"class counts = {np.bincount(y).tolist()} (0=normal, 1=tumor)")

    # --- resume support: reload completed folds, skip them on restart ---
    folds_path = Path('results/tn_transformer_folds.yaml')
    fold_records = []
    done = set()
    if folds_path.exists():
        try:
            saved = yaml.safe_load(folds_path.read_text(encoding='utf-8')) or []
            for rec in saved:
                h = str(rec.get('held_out'))
                if h and h not in done:
                    fold_records.append(rec)
                    done.add(h)
            if done:
                print(f"[resume] 已完成折: {sorted(done)}，将跳过")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 无法读取 {folds_path} ({e})，从头开始")
            fold_records, done = [], set()

    for held_out in np.unique(batch):
        held = str(held_out)
        if held in done:
            print(f"[resume] 跳过已完成的折 {held}")
            continue
        te = batch == held_out
        tr = ~te
        X_tr, X_te, _ = prepare_fold(X[tr], X[te], batch[tr], batch[te], cfg)
        y_tr, y_te = y[tr], y[te]
        cfg['model']['n_genes'] = X_tr.shape[1]

        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed,
        )
        train_loader, val_loader = create_dataloaders(
            X_tr2, y_tr2, X_val, y_val, batch_size=batch_size,
        )
        save_dir = f'results/checkpoints/tn_lodo_{held}'
        print(f"\n{'='*60}\nLODO fold: held_out={held}\n{'='*60}")
        train_model(cfg, train_loader, val_loader, save_dir,
                    class_weights=weights, monitor=monitor)

        model = create_model(cfg).to(device)
        ckpt = Path(save_dir) / 'best_model.pt'
        model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])

        s_te = predict_proba_positive(model, X_te, device)
        normal_prev = float((y_tr == 0).mean())
        thr = quantile_threshold(s_te, normal_prev)
        pred = (s_te > thr).astype(int)  # 0=normal (low), 1=tumor (high)

        m = binary_metrics(y_te, pred, s_te)
        m['threshold'] = thr
        m['normal_prev'] = normal_prev
        m['held_out'] = held
        fold_records.append(m)
        # save immediately so a mid-run disconnect does not lose this fold
        folds_path.parent.mkdir(parents=True, exist_ok=True)
        folds_path.write_text(
            yaml.dump(fold_records, default_flow_style=False, allow_unicode=True),
            encoding='utf-8',
        )
        print(f"  {held}: PR-AUC={m['pr_auc']:.3f} AUC={m['auc']:.3f} "
              f"normal_recall={m['normal_recall']:.3f} "
              f"tumor_recall={m['tumor_recall']:.3f}")
        print(f"  [saved] 该折已写入 {folds_path}")

    keys = ['pr_auc', 'auc', 'normal_recall', 'tumor_recall']
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_records]
        agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                  'per_fold': vals}
    print("\nTransformer tumor/normal LODO (PRIMARY = PR-AUC / AUC):")
    print(f"  PR-AUC {agg['pr_auc']['mean']:.3f} +/- {agg['pr_auc']['std']:.3f}")
    print(f"  AUC    {agg['auc']['mean']:.3f} +/- {agg['auc']['std']:.3f}")
    print(f"  normal_recall {agg['normal_recall']['mean']:.3f} | "
          f"tumor_recall {agg['tumor_recall']['mean']:.3f}")
    return {'tumor_normal_lodo': agg}


if __name__ == "__main__":
    import sys
    import traceback

    mode = sys.argv[1] if len(sys.argv) > 1 else "cancer"

    try:
        config = load_config()
        if mode in ("tumor_normal", "tn"):
            results = run_tumor_normal_transformer(config)
            out = Path('results/tn_transformer_results.yaml')
        else:
            results = run_lodo_transformer(config)
            out = Path('results/lodo_transformer_results.yaml')
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as fh:
            yaml.dump(results, fh, default_flow_style=False, allow_unicode=True)
        print(f"\n[SUCCESS] saved -> {out}")
    except Exception as e:  # noqa: BLE001
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
