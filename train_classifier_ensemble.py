"""
FGI-PointTransformer-DL-3D reproduction — HeliALS 16D

Paper-aligned training + evaluation
-----------------------------------
Follows the FGI-PointTransformer-DL-3D setup in the benchmark paper:

  - 95-5 train-validation split (set in the dataset config)
  - Five randomly initialized Point Transformer models trained from scratch
  - Cross-entropy loss (or weighted cross-entropy via --weighted-loss for the
    FGI-PointTransformerWeighted-DL-3D variant)
  - Paper augmentations on the train set:
        small random translation, scaling, and jittering of coordinates
        small random scaling and jittering of other attributes
        (return_number is kept raw and not augmented)
  - At inference: majority voting across the five trained models on the test
    set, exactly as the paper reports in Table 3.

Modern training practice (paper is silent on epochs/LR/optimizer):
  - AdamW
  - Linear warmup + cosine annealing LR schedule
  - Gradient clipping
  - Optional mixed precision (--use-amp)

Metrics reported at end:
  - Overall accuracy (OA)
  - Macro-average accuracy (= macro recall)
  - Per-species precision / recall / F1
  - Confusion matrix
  - JSON dump of all per-seed and ensemble predictions

Usage
-----
  python train_classifier_ensemble.py \
      --data-path data/HeliALS_voxelagg_8192_16D \
      --output-dir runs/classifiers/PointTransformer \
      --epochs 200 --batch-size 8 --lr 1e-3 --num-models 5

Add --weighted-loss to reproduce FGI-PointTransformerWeighted-DL-3D.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

from TreeSpeciesDatasetHELIALS import (
    TreeSpeciesDatasetHELIALS,
    SPECIES_NAMES,
    FEATURE_NAMES_16D,
)
from helials_classifier import HeliALSClassifier


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data: collate for the dataloader's ('HeliALS', 'sample', (pc, label)) format
# ---------------------------------------------------------------------------

def collate_helials(batch):
    pcs = torch.stack([torch.as_tensor(b[2][0], dtype=torch.float32) for b in batch])
    labels = torch.as_tensor([int(b[2][1]) for b in batch], dtype=torch.long)
    return pcs, labels


def build_loader(data_path: str, subset: str, num_points: int, batch_size: int,
                 val_ratio: float, seed: int,
                 csv_path: str | None, final_csv_path: str | None,
                 num_workers: int, shuffle: bool) -> DataLoader:
    cfg = SimpleNamespace(
        N_POINTS=num_points,
        subset=subset,
        DATA_PATH=data_path,
        val_ratio=val_ratio,
        seed=seed,
    )
    if csv_path is not None:
        cfg.CSV_PATH = csv_path
    if final_csv_path is not None:
        cfg.FINAL_CSV_PATH = final_csv_path

    ds = TreeSpeciesDatasetHELIALS(cfg)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,          # drop last on train only
        collate_fn=collate_helials,
    )
    return loader, ds


# ---------------------------------------------------------------------------
# Paper augmentations (applied on-GPU per batch during training)
# ---------------------------------------------------------------------------

class PaperAugmenter:
    """
    Paper (FGI-PointTransformer-DL-3D, Table 3):
        "small random translation, scaling, and jittering of coordinates,
         and small random scaling and jittering of other attributes"

    `return_number` is kept raw — the paper explicitly excludes it from
    attribute normalization, and it encodes discrete echo ordering that
    would be meaningless after scaling/jitter.
    """

    def __init__(self,
                 coord_translate_range: float = 0.03,
                 coord_scale_range: tuple[float, float] = (0.9, 1.1),
                 coord_jitter_std: float = 0.005,
                 coord_jitter_clip: float = 0.02,
                 attr_scale_range: tuple[float, float] = (0.95, 1.05),
                 attr_jitter_std: float = 0.005,
                 attr_jitter_clip: float = 0.02,
                 return_number_index: int = 15):
        self.ct = coord_translate_range
        self.cs = coord_scale_range
        self.cj_std = coord_jitter_std
        self.cj_clip = coord_jitter_clip
        self.as_ = attr_scale_range
        self.aj_std = attr_jitter_std
        self.aj_clip = attr_jitter_clip
        self.rn_idx = return_number_index

    def __call__(self, pc: torch.Tensor) -> torch.Tensor:
        """
        pc: (B, N, 16) on CUDA or CPU. Returns augmented tensor of same shape.
        Convention: [0:3]=xyz, [3:15]=12 per-scanner attrs, [15]=return_number.
        """
        B, N, C = pc.shape
        assert C == 16, f"expected C=16, got {C}"
        device = pc.device

        xyz = pc[..., :3]
        attrs = pc[..., 3:self.rn_idx]     # (B, N, 12)
        ret_n = pc[..., self.rn_idx:self.rn_idx + 1]   # (B, N, 1), kept raw

        # --- coordinate augmentations (per-sample, not per-point) ---
        # isotropic scale
        s_lo, s_hi = self.cs
        scale = torch.empty(B, 1, 1, device=device).uniform_(s_lo, s_hi)
        xyz = xyz * scale
        # translation
        if self.ct > 0:
            shift = torch.empty(B, 1, 3, device=device).uniform_(-self.ct, self.ct)
            xyz = xyz + shift
        # per-point jitter
        if self.cj_std > 0:
            jit = torch.randn_like(xyz) * self.cj_std
            jit.clamp_(-self.cj_clip, self.cj_clip)
            xyz = xyz + jit

        # --- attribute augmentations (per-sample scale, per-point jitter) ---
        a_lo, a_hi = self.as_
        a_scale = torch.empty(B, 1, attrs.shape[-1], device=device).uniform_(a_lo, a_hi)
        attrs = attrs * a_scale
        if self.aj_std > 0:
            a_jit = torch.randn_like(attrs) * self.aj_std
            a_jit.clamp_(-self.aj_clip, self.aj_clip)
            attrs = attrs + a_jit

        return torch.cat([xyz, attrs, ret_n], dim=-1)


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    # Standard practice: no weight decay on biases and BatchNorm/LayerNorm params
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return AdamW(
        [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


def build_lr_scheduler(optimizer, epochs: int, warmup_epochs: int,
                       min_lr_ratio: float = 0.01) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


def compute_class_weights(train_labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized so sum == num_classes."""
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv * num_classes / inv.sum()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Train / evaluate one model
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits = []
    all_labels = []
    for pc, y in loader:
        pc = pc.to(device, non_blocking=True)
        logits = model(pc)
        all_logits.append(logits.float().cpu())
        all_labels.append(y)
    return torch.cat(all_logits, 0), torch.cat(all_labels, 0)


def train_one_model(seed: int, args, device: torch.device, weights_path: Path):
    """Train one PointTransformer with `seed`. Saves best checkpoint. Returns
    the path to that checkpoint."""
    set_seed(seed)

    train_loader, train_ds = build_loader(
        args.data_path, 'train',
        num_points=args.num_points, batch_size=args.batch_size,
        val_ratio=args.val_ratio, seed=args.split_seed,
        csv_path=args.csv_path, final_csv_path=args.final_csv_path,
        num_workers=args.num_workers, shuffle=True,
    )
    val_loader, _ = build_loader(
        args.data_path, 'val',
        num_points=args.num_points, batch_size=args.batch_size,
        val_ratio=args.val_ratio, seed=args.split_seed,
        csv_path=args.csv_path, final_csv_path=args.final_csv_path,
        num_workers=args.num_workers, shuffle=False,
    )

    num_classes = len(train_ds.classes)
    model_cfg = SimpleNamespace(
        num_classes=num_classes,
        input_dim=args.input_dim,
        smooth=args.label_smoothing,
        backbone=args.backbone,
    )
    model = HeliALSClassifier(model_cfg).to(device)

    if args.weighted_loss:
        cw = compute_class_weights(train_ds.label.astype(np.int64), num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
        print(f"  [seed {seed}] using weighted CE with weights = {cw.cpu().numpy().round(3).tolist()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    augmenter = PaperAugmenter()

    best_val_oa = -1.0
    best_epoch = -1
    history = []
    patience_counter = 0
    early_stopped = False

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for pc, y in train_loader:
            pc = pc.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pc = augmenter(pc)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                logits = model(pc)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(-1) == y).sum().item()
            total_seen += y.size(0)

        scheduler.step()
        train_oa = total_correct / max(1, total_seen)
        train_loss = total_loss / max(1, total_seen)

        val_logits, val_y = evaluate(model, val_loader, device)
        val_pred = val_logits.argmax(-1).numpy()
        val_oa = accuracy_score(val_y.numpy(), val_pred)
        _, _, _, _ = precision_recall_fscore_support(
            val_y.numpy(), val_pred, average='macro', zero_division=0
        )
        val_macro = precision_recall_fscore_support(
            val_y.numpy(), val_pred, average='macro', zero_division=0
        )[1]

        lr_now = optimizer.param_groups[0]['lr']
        dt = time.time() - t0
        history.append({
            'epoch': epoch, 'lr': lr_now,
            'train_loss': train_loss, 'train_oa': train_oa,
            'val_oa': val_oa, 'val_macro_recall': val_macro,
            'time_s': dt,
        })
        print(f"  [seed {seed}] epoch {epoch+1:3d}/{args.epochs} "
              f"lr={lr_now:.2e} loss={train_loss:.4f} "
              f"train_oa={train_oa*100:.2f} val_oa={val_oa*100:.2f} "
              f"val_macro={val_macro*100:.2f} ({dt:.1f}s)")

        # best-by-val checkpoint + early stopping on val_oa
        if val_oa > best_val_oa:
            best_val_oa = val_oa
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'seed': seed,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_oa': val_oa,
                'val_macro': val_macro,
                'classes': train_ds.classes,
                'args': vars(args),
            }, weights_path)
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"  [seed {seed}] early stop @ epoch {epoch+1}: "
                      f"no val_oa improvement for {args.patience} epochs "
                      f"(best val_oa={best_val_oa*100:.2f}% @ epoch {best_epoch+1})")
                early_stopped = True
                break

    tag = "early-stopped" if early_stopped else "finished"
    print(f"  [seed {seed}] {tag}: best val_oa={best_val_oa*100:.2f}% @ epoch {best_epoch+1}")
    return weights_path, history, best_val_oa


# ---------------------------------------------------------------------------
# Ensemble on test set
# ---------------------------------------------------------------------------

def majority_vote(preds_matrix: np.ndarray) -> np.ndarray:
    """preds_matrix: (num_models, num_samples) int. Returns (num_samples,) majority class.
    Ties broken by the class that appears first across model order (i.e., lowest index)."""
    num_samples = preds_matrix.shape[1]
    out = np.empty(num_samples, dtype=np.int64)
    for i in range(num_samples):
        vals, counts = np.unique(preds_matrix[:, i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def report_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    oa = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    macro_avg_recall = recall.mean()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    return {
        'overall_accuracy': float(oa),
        'macro_average_accuracy': float(macro_avg_recall),   # paper's metric
        'per_class': [
            {
                'class': classes[i],
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
            }
            for i in range(len(classes))
        ],
        'confusion_matrix': cm.tolist(),
    }


def print_metrics(tag: str, metrics: dict, classes: list[str]) -> None:
    print(f"\n{'=' * 72}\n{tag}\n{'=' * 72}")
    print(f"  Overall accuracy       : {metrics['overall_accuracy']*100:.2f}%")
    print(f"  Macro-average accuracy : {metrics['macro_average_accuracy']*100:.2f}%")
    print(f"\n  {'Species':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    for row in metrics['per_class']:
        print(f"  {row['class']:<12} "
              f"{row['precision']*100:>9.2f}% {row['recall']*100:>9.2f}% "
              f"{row['f1']*100:>9.2f}% {row['support']:>8d}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = "  " + " " * 12 + " ".join(f"{c[:5]:>6}" for c in classes)
    print(header)
    for i, row in enumerate(metrics['confusion_matrix']):
        print(f"  {classes[i]:<12}" + " ".join(f"{v:>6d}" for v in row))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / 'data'
_DATASET_CHOICES = (sorted(d.name for d in _DATA_DIR.iterdir() if d.is_dir())
                    if _DATA_DIR.is_dir() else [])


def _resolve_data_path(args):
    if args.dataset:
        args.data_path = str(_DATA_DIR / args.dataset)
    if not args.data_path:
        raise SystemExit("Provide --dataset <name> or --data-path <path>")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument('--dataset', type=str, default=None,
                   choices=_DATASET_CHOICES or None,
                   help='Dataset directory name under data/ '
                        f'(available: {", ".join(_DATASET_CHOICES) or "<none>"}). '
                        'Shorthand for --data-path data/<name>.')
    p.add_argument('--data-path', type=str, default=None,
                   help='Explicit path to a preprocessed dataset directory '
                        '(e.g. data/HeliALS_voxelagg_8192_16D); overrides '
                        '--dataset if both are set.')
    p.add_argument('--csv-path', type=str, default=None,
                   help='path to training-and-test-segments-with-species.csv. '
                        'Default: <parent-of-data-path>/training-and-test-segments-with-species.csv')
    p.add_argument('--final-csv-path', type=str, default=None,
                   help='optional path to final-segments-with-species.csv (quality filter)')
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--num-points', type=int, default=8192, help='paper: 8192')
    p.add_argument('--input-dim', type=int, default=16)
    p.add_argument('--backbone', type=str, default='PointTransformer',
                   help='Classifier backbone name; must be registered in '
                        'models.build.MODELS (e.g. PointTransformer, DGCNN, '
                        'PointNet, PCT, PointMLP, CurveNet, DeepGCN, ...).')
    p.add_argument('--val-ratio', type=float, default=0.05, help='paper: 95-5 split -> 0.05')
    p.add_argument('--split-seed', type=int, default=42,
                   help='seed controlling the train/val split; same for all 5 models so they vote on the same held-out data')
    # training
    p.add_argument('--epochs', type=int, default=200,
                   help='maximum training epochs (early stopping may cut short)')
    p.add_argument('--warmup-epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=30,
                   help='early-stop if val_oa does not improve for this many epochs. '
                        'Set <=0 to disable (train full --epochs).')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=5e-2)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--weighted-loss', action='store_true',
                   help='use class-weighted cross-entropy (FGI-PointTransformerWeighted-DL-3D)')
    p.add_argument('--use-amp', action='store_true', help='mixed-precision training')
    p.add_argument('--num-workers', type=int, default=4)
    # ensemble
    p.add_argument('--num-models', type=int, default=5,
                   help='paper: 5 randomly initialized models')
    p.add_argument('--model-seeds', type=str, default=None,
                   help='comma-separated seeds for the ensemble; default: 0..num-models-1')
    args = p.parse_args()
    _resolve_data_path(args)
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.model_seeds:
        seeds = [int(s) for s in args.model_seeds.split(',')]
        if len(seeds) != args.num_models:
            raise ValueError(f"--model-seeds has {len(seeds)} values, --num-models={args.num_models}")
    else:
        seeds = list(range(args.num_models))

    # Dump run config upfront
    with open(output_dir / 'run_config.json', 'w') as f:
        json.dump(vars(args) | {'seeds': seeds}, f, indent=2, default=str)

    # Train N models
    per_seed_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'#' * 72}\n# Training model {i+1}/{len(seeds)}  (seed={seed})\n{'#' * 72}")
        ckpt = output_dir / f'model_seed{seed}.pt'
        _, history, best_val = train_one_model(seed, args, device, ckpt)
        with open(output_dir / f'history_seed{seed}.json', 'w') as f:
            json.dump(history, f, indent=2)
        per_seed_results.append({'seed': seed, 'ckpt': str(ckpt), 'best_val_oa': best_val})

    # Test loader once; same data for every model
    test_loader, test_ds = build_loader(
        args.data_path, 'test',
        num_points=args.num_points, batch_size=args.batch_size,
        val_ratio=args.val_ratio, seed=args.split_seed,
        csv_path=args.csv_path, final_csv_path=args.final_csv_path,
        num_workers=args.num_workers, shuffle=False,
    )
    classes = list(test_ds.classes)
    num_classes = len(classes)
    test_labels_np = None

    # Collect test predictions from each best checkpoint
    all_test_preds = []        # list of (num_test,) int arrays
    all_test_logits = []       # list of (num_test, C) float arrays (for soft voting info)
    per_model_metrics = []

    for res in per_seed_results:
        ckpt = res['ckpt']
        print(f"\nInference with {ckpt}")
        model_cfg = SimpleNamespace(
            num_classes=num_classes,
            input_dim=args.input_dim,
            smooth=args.label_smoothing,
            backbone=args.backbone,
        )
        model = HeliALSClassifier(model_cfg).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state['model_state'])

        logits, y = evaluate(model, test_loader, device)
        preds = logits.argmax(-1).numpy()
        y_np = y.numpy()
        if test_labels_np is None:
            test_labels_np = y_np
        all_test_preds.append(preds)
        all_test_logits.append(logits.numpy())

        metrics = report_metrics(y_np, preds, classes)
        print_metrics(f"Per-seed test metrics  (seed={res['seed']})", metrics, classes)
        per_model_metrics.append({'seed': res['seed'], **metrics})

    # Majority vote — paper's inference protocol
    preds_matrix = np.stack(all_test_preds, axis=0)   # (num_models, num_test)
    voted = majority_vote(preds_matrix)
    ensemble_metrics = report_metrics(test_labels_np, voted, classes)
    print_metrics(
        f"ENSEMBLE (majority vote of {len(seeds)} models) — paper's protocol",
        ensemble_metrics, classes,
    )

    # Also compute soft-vote (mean of logits) as a bonus reference
    mean_logits = np.mean(np.stack(all_test_logits, 0), axis=0)
    soft_preds = mean_logits.argmax(-1)
    soft_metrics = report_metrics(test_labels_np, soft_preds, classes)
    print_metrics(
        f"Soft vote (mean logits)  — bonus, NOT paper's protocol",
        soft_metrics, classes,
    )

    # Save everything
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'seeds': seeds,
            'per_model_metrics': per_model_metrics,
            'ensemble_majority_vote': ensemble_metrics,
            'ensemble_soft_vote': soft_metrics,
            'test_labels': test_labels_np.tolist(),
            'per_model_predictions': [p.tolist() for p in all_test_preds],
            'ensemble_predictions': voted.tolist(),
            'classes': classes,
        }, f, indent=2)

    np.save(output_dir / 'test_logits_per_model.npy', np.stack(all_test_logits, 0))

    print(f"\nAll artifacts saved to: {output_dir}")
    print(f"  - per-seed checkpoints:  model_seed<N>.pt")
    print(f"  - per-seed history:      history_seed<N>.json")
    print(f"  - final results JSON:    results.json")
    print(f"  - test logits tensor:    test_logits_per_model.npy")


if __name__ == '__main__':
    main()
