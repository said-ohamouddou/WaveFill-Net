#!/usr/bin/env python3
"""
WaveFill-Net main pipeline: for a given classifier backbone, train our
two-loss WaveFill-Net imputer and compare it against the standard
missing-wavelength imputation baselines.

Method (decided by component + hyperparameter ablations):

    L_total = lambda_recon * L1(pred, GT)
            + lambda_cls   * CE(classifier(P_tilde), y)

    Architecture: per-point MLP conditioned on
                  (xyz, visible_8ch, wavelength_id_embedding)
                  hidden = 256, deterministic (no latent noise z),
                  moment-matching loss dropped.
    Weights     : lambda_recon = 10, lambda_cls = 0.5

Baselines compared at eval time:
    full        : 16D reference (classifier on un-masked input)
    zero        : naive lower bound
    mean        : global per-channel mean from training set
    linear      : per-wavelength linear regression
    knn         : K-NN over (xyz + visible 8 channels) in training set
    missforest  : missForest-style conditional random forest (single-pass;
                  the missingness is monotone, so the iterative refinement
                  of missForest/missRanger reduces to one conditional fit)

To keep wall-clock manageable across the 10 backbones x 5 seeds = 50 runs,
the backbone-agnostic baselines (zero, mean, linear, knn, missforest) are
computed and
*timed* exactly once per dataset. Both the imputed test arrays AND the per-
wavelength fill timing are cached to
    data/<dataset>/imputed_test_cache/<method>.npz   (the imputed channels)
    data/<dataset>/imputed_test_cache/<method>.json  (timing + signature)
and reused verbatim by every subsequent (backbone, seed) eval. WaveFill-Net
is the only method recomputed per-seed (it depends on the trained imputer).

Setup expected on disk:
    runs/all_16d_backbones_1024pts_5seed/<BACKBONE>/classifier/
        model_seed{0..4}.pt   (frozen ensemble used for L_cls + paired eval)

Outputs (under --output-dir, default runs/method/<BACKBONE>/):
    seed<S>/wavefill_best.pt        imputer checkpoint (MLP variant)
    seed<S>/wavefill_relukan_best.pt  ditto for --imputer relukan
    seed<S>/train_history.json      per-epoch losses + val RMSE
    seed<S>/eval_results.json       OA / macro-F1 for all methods
    summary.json                    mean +- std across seeds
    summary.csv                     flat per-seed CSV for plots
    comparison.tex                  ready-to-compile LaTeX table
    convergence_val_rmse.png        per-wavelength val RMSE vs epoch
    convergence_train_losses.png    train recon + cls losses vs epoch

Usage
-----
    # Single backbone, 5 seeds, default hyperparameters
    python main.py --backbone PointTransformer

    # Loop across all 10 backbones
    for BB in PointTransformer PointTransformerV2 PCT PointMLP DGCNN \
              DeepGCN GDAN PointNet PointNet2_MSG KANDGCNN; do
        python main.py --backbone "$BB"
    done

    # Quick dev pass (single seed, short training)
    python main.py --backbone DGCNN --seeds 0 --epochs 60

    # Skip the slow KNN baseline
    python main.py --backbone DGCNN --no-knn

    # Both WaveFill-Net variants into the same output dir (rows accumulate)
    python main.py --backbone PointTransformer --imputer mlp
    python main.py --backbone PointTransformer --imputer relukan

    # Only build the shared KNN cache (no training):
    python main.py --backbone PointTransformer --build-caches-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_recall_fscore_support)
from sklearn.neighbors import NearestNeighbors
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from TreeSpeciesDatasetHELIALS import TreeSpeciesDatasetHELIALS, SPECIES_NAMES
from helials_classifier import HeliALSClassifier

try:
    from torch_relu_kan import ReLUKANLayer
except ImportError:                                   # optional dependency
    ReLUKANLayer = None

try:
    from torch_fast_kan import FastKANLayer
except ImportError:                                   # optional dependency
    FastKANLayer = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CLASSIFIER_ROOT_DEFAULT = ROOT / "runs/all_16d_backbones_1024pts_5seed"
DATASET = "HeliALS_voxelagg_1024_16D"
NUM_POINTS = 1024
INPUT_DIM = 16
NUM_CLASSES = 9
LABEL_SMOOTHING = 0.1

# Wavelength id -> the 4 spectral channels for that wavelength in the 16D layout.
WAVELENGTH_CHANNELS = {
    0: [3, 6, 9, 12],   # 1550 nm  (intensity, amplitude, reflectance, deviation)
    1: [4, 7, 10, 13],  #  905 nm
    2: [5, 8, 11, 14],  #  532 nm
}
NUM_WAVELENGTHS = 3
SPECTRAL_DIMS = list(range(3, 15))   # the 12 spectral channels
WL_NM = {0: "1550nm", 1: "905nm", 2: "532nm"}   # display labels

CLASSES = [SPECIES_NAMES[i] for i in range(1, 10)]

# Default hyperparameters, fixed A PRIORI -- they are NOT the argmax of any
# sweep. The sweeps in ablation_hparam_kan.py are descriptive: they characterise
# how performance varies around this pre-set configuration, and do not feed
# back into it. Nothing here was selected on the test set.
LAMBDA_RECON = 10.0
LAMBDA_CLS   = 0.5
HIDDEN       = 256
EMB_DIM      = 16
NUM_CLASSIFIERS_FOR_TRAIN = 5   # frozen ensemble size used in L_cls

# ReLU-KAN imputer variant (same depth/width as the MLP; only the layer
# primitive differs). g = grid size, k = phase overlap.
KAN_GRID = 5
KAN_K    = 3
# ReLU-KAN width, chosen so the two variants are PARAMETER-MATCHED: a
# ReLUKANLayer's Conv2d kernel spans (g+k, in_features), i.e. (g+k)=8 times a
# dense layer's weights, so a KAN at the MLP's width would carry 8.05x its
# parameters (1,125,604 vs 139,828) and any gain would be capacity, not
# architecture. At h=88 the KAN has 150,700 params -- 1.08x the MLP -- so the
# comparison isolates the layer primitive.
KAN_HIDDEN = 88

# FastKAN (radial-basis) variant. Width chosen the same way: at h=84 with 8
# grids it has 151,338 parameters, i.e. 1.00x the ReLU-KAN and 1.08x the MLP,
# so all three variants are parameter-matched and the comparison isolates the
# layer primitive rather than capacity.
FASTKAN_HIDDEN = 84
FASTKAN_GRIDS  = 8

# Cache parameters for the heavy baselines (shared across backbones).
KNN_K = 10

# missForest baseline (single-pass conditional random forest). With a single
# missing wavelength block the missingness is monotone -- the 8 visible
# spectral channels are always fully observed -- so missForest's iterative
# refinement converges after one conditional fit and reduces exactly to this.
MF_N_ESTIMATORS     = 100
MF_MIN_SAMPLES_LEAF = 2
MF_MAX_TRAIN_POINTS = 1_085_440   # full HeliALS training cloud (1060 x 1024)


# ---------------------------------------------------------------------------
# Channel split / re-assembly
# ---------------------------------------------------------------------------

def get_visible_missing_indices(wl_id: int) -> tuple[list[int], list[int]]:
    missing = WAVELENGTH_CHANNELS[wl_id]
    visible = [c for c in SPECTRAL_DIMS if c not in missing]
    return visible, missing


def split_pc(pc: torch.Tensor, wl_id: int):
    """pc: (B, N, 16) -> (xyz, visible_8ch, missing_4ch_GT, return_number)."""
    visible_idx, missing_idx = get_visible_missing_indices(wl_id)
    return (pc[..., :3],
            pc[..., visible_idx],
            pc[..., missing_idx],
            pc[..., 15:16])


def assemble_pc16(xyz, visible, generated, return_number, wl_id) -> torch.Tensor:
    visible_idx, missing_idx = get_visible_missing_indices(wl_id)
    B, N = xyz.shape[:2]
    out = torch.empty(B, N, 15, device=xyz.device, dtype=xyz.dtype)
    out[..., :3] = xyz
    for i, c in enumerate(visible_idx):
        out[..., c] = visible[..., i]
    for i, c in enumerate(missing_idx):
        out[..., c] = generated[..., i]
    return torch.cat([out, return_number], dim=-1)


# ---------------------------------------------------------------------------
# WaveFill-Net imputer
# ---------------------------------------------------------------------------

class WaveFillImputer(nn.Module):
    """Per-point MLP conditioned on (xyz, visible 8ch, wavelength_id_embedding).
    Deterministic; sigmoid output in [0, 1]."""

    def __init__(self, hidden: int = HIDDEN,
                 emb_dim: int = EMB_DIM,
                 num_wavelengths: int = NUM_WAVELENGTHS):
        super().__init__()
        self.wl_embed = nn.Embedding(num_wavelengths, emb_dim)
        self.hidden_width = hidden
        in_dim = 3 + 8 + emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),  nn.GELU(),
            nn.Linear(hidden, hidden),  nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, xyz, visible, wl_id):
        _, N, _ = xyz.shape
        emb = self.wl_embed(wl_id).unsqueeze(1).expand(-1, N, -1)
        h = torch.cat([xyz, visible, emb], dim=-1)
        return torch.sigmoid(self.net(h))


class WaveFillReLUKAN(nn.Module):
    """ReLU-KAN variant of WaveFill-Net: every nn.Linear of WaveFillImputer
    replaced 1:1 by a ReLUKANLayer, at the same widths and the same depth.

    The GELU activations are dropped on purpose -- a ReLU-KAN layer learns a
    nonlinear univariate basis per input coordinate (squared products of two
    ReLU phases), so it is already nonlinear. Everything else (conditioning
    inputs, depth, widths, sigmoid output, loss, optimizer) is identical to
    WaveFillImputer, so the two differ only in the layer primitive.

    Shape handling: ReLUKANLayer consumes (batch, in_features, 1) and returns
    (batch, out_features, 1), i.e. it indexes points along a single flat batch
    axis, so we flatten B*N -> batch on the way in and restore (B, N, 4) out.
    """

    def __init__(self, hidden: int = HIDDEN,
                 emb_dim: int = EMB_DIM,
                 num_wavelengths: int = NUM_WAVELENGTHS,
                 grid: int = KAN_GRID, k: int = KAN_K,
                 train_ab: bool = True):
        super().__init__()
        if ReLUKANLayer is None:
            raise ImportError(
                "torch_relu_kan.ReLUKANLayer is unavailable; cannot build the "
                "ReLU-KAN imputer (needed for --imputer relukan).")
        self.wl_embed = nn.Embedding(num_wavelengths, emb_dim)
        self.hidden_width = hidden
        in_dim = 3 + 8 + emb_dim
        self.grid, self.k = grid, k
        self.layers = nn.ModuleList([
            ReLUKANLayer(in_dim, grid, k, hidden, train_ab=train_ab),
            ReLUKANLayer(hidden, grid, k, hidden, train_ab=train_ab),
            ReLUKANLayer(hidden, grid, k, hidden, train_ab=train_ab),
            ReLUKANLayer(hidden, grid, k, 4,      train_ab=train_ab),
        ])

    def forward(self, xyz, visible, wl_id):
        B, N, _ = xyz.shape
        emb = self.wl_embed(wl_id).unsqueeze(1).expand(-1, N, -1)
        h = torch.cat([xyz, visible, emb], dim=-1)
        h = h.reshape(B * N, h.size(-1), 1)
        for layer in self.layers:
            h = layer(h)
        return torch.sigmoid(h.reshape(B, N, 4))


class WaveFillFastKAN(nn.Module):
    """FastKAN variant: the same per-point imputer with radial-basis-function
    KAN layers (kans/layers.py:FastKANLayer) instead of ReLU-KAN ones.

    Included because at matched capacity it reaches essentially the same
    accuracy as the ReLU-KAN variant at a fraction of the cost (~12x faster
    per training step in our measurements), so it is the practical choice
    whenever throughput matters. Conditioning, depth, widths and the sigmoid
    output are identical to the other variants -- only the layer primitive
    differs.

    Like ReLUKANLayer, FastKANLayer consumes a 2-D (batch, features) tensor,
    so (B, N, F) is flattened to B*N and restored.
    """

    def __init__(self, hidden: int = None,
                 emb_dim: int = EMB_DIM,
                 num_wavelengths: int = NUM_WAVELENGTHS,
                 num_grids: int = FASTKAN_GRIDS):
        super().__init__()
        if FastKANLayer is None:
            raise ImportError(
                "kans.layers.FastKANLayer is unavailable; cannot build the "
                "FastKAN imputer (needed for --imputer fastkan).")
        hidden = FASTKAN_HIDDEN if hidden is None else hidden
        self.wl_embed = nn.Embedding(num_wavelengths, emb_dim)
        self.hidden_width = hidden
        self.num_grids = num_grids
        in_dim = 3 + 8 + emb_dim
        mk = lambda i, o: FastKANLayer(i, o, num_grids=num_grids)  # noqa: E731
        self.layers = nn.ModuleList([mk(in_dim, hidden), mk(hidden, hidden),
                                     mk(hidden, hidden), mk(hidden, 4)])

    def forward(self, xyz, visible, wl_id):
        B, N, _ = xyz.shape
        emb = self.wl_embed(wl_id).unsqueeze(1).expand(-1, N, -1)
        h = torch.cat([xyz, visible, emb], dim=-1).reshape(B * N, -1)
        for layer in self.layers:
            h = layer(h)
        return torch.sigmoid(h.reshape(B, N, 4))


# Imputer registry: --imputer selects which one the run trains + evaluates.
IMPUTERS = {"mlp": WaveFillImputer, "relukan": WaveFillReLUKAN,
            "fastkan": WaveFillFastKAN}

# The MLP keeps the original, unsuffixed names and the "wavefill" result key so
# that runs produced before the ReLU-KAN variant existed remain valid; every
# other variant is suffixed and so trains/evaluates into its own slot.

def imputer_ckpt_name(kind: str, grid: int | None = None,
                      k: int | None = None) -> str:
    """Checkpoint filename for one imputer variant.

    The ReLU-KAN name encodes grid/k: without them, re-running with a
    different --kan-grid reuses the existing checkpoint (the skip-train guard
    only tests the path), while eval rebuilds the architecture from the
    values stored INSIDE that checkpoint. The run would then be labelled with
    the new grid but actually evaluate the old model -- silently wrong, and
    exactly what a grid/k sensitivity sweep would trigger.
    """
    if kind == "mlp":
        return "wavefill_best.pt"
    if kind == "relukan":
        g = KAN_GRID if grid is None else grid
        kk = KAN_K if k is None else k
        # The default g/k keep the original unsuffixed name so checkpoints
        # produced before this change stay valid.
        if (g, kk) == (KAN_GRID, KAN_K):
            return "wavefill_relukan_best.pt"
        return f"wavefill_relukan_g{g}_k{kk}_best.pt"
    return f"wavefill_{kind}_best.pt"


def imputer_history_name(kind: str) -> str:
    return ("train_history.json" if kind == "mlp"
            else f"train_history_{kind}.json")


def imputer_method_key(kind: str) -> str:
    return "wavefill" if kind == "mlp" else f"wavefill_{kind}"


def build_imputer(kind: str, hidden: int | None = None, **kw) -> nn.Module:
    """Build one imputer variant.

    `hidden=None` selects each variant's own default width: HIDDEN (256) for
    the MLP, KAN_HIDDEN (88) for the ReLU-KAN, FASTKAN_HIDDEN (84) for
    FastKAN -- which is what keeps all three parameter-matched. Pass an explicit width to override (e.g. the
    capacity-sweep arms).
    """
    if kind not in IMPUTERS:
        raise ValueError(f"unknown imputer '{kind}'; choose from "
                         f"{sorted(IMPUTERS)}")
    if kind == "relukan":
        return WaveFillReLUKAN(hidden=KAN_HIDDEN if hidden is None else hidden,
                               grid=kw.get("grid", KAN_GRID),
                               k=kw.get("k", KAN_K))
    if kind == "fastkan":
        return WaveFillFastKAN(
            hidden=FASTKAN_HIDDEN if hidden is None else hidden,
            num_grids=kw.get("num_grids", FASTKAN_GRIDS))
    return WaveFillImputer(hidden=HIDDEN if hidden is None else hidden)


def reconstruction_loss(pred, target):
    return F.l1_loss(pred, target)


def ensemble_logits(classifiers, pc):
    """Average logits across the frozen ensemble; gradients flow through
    pc -> imputer, classifier weights stay frozen."""
    out = None
    for m in classifiers:
        l = m(pc)
        out = l if out is None else out + l
    return out / len(classifiers)


def load_classifier_ensemble(classifier_dir: Path, backbone: str,
                              device: torch.device,
                              n: int = NUM_CLASSIFIERS_FOR_TRAIN
                              ) -> list[nn.Module]:
    classifiers = []
    for k in range(n):
        ckpt = classifier_dir / f"model_seed{k}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing classifier checkpoint: {ckpt}")
        cfg = SimpleNamespace(num_classes=NUM_CLASSES, input_dim=INPUT_DIM,
                              smooth=LABEL_SMOOTHING, backbone=backbone)
        m = HeliALSClassifier(cfg).to(device)
        state = torch.load(ckpt, map_location=device)
        m.load_state_dict(state["model_state"])
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        classifiers.append(m)
    return classifiers


def load_paired_classifier(classifier_dir: Path, backbone: str, seed: int,
                            device: torch.device) -> nn.Module:
    cls_ckpt = classifier_dir / f"model_seed{seed}.pt"
    if not cls_ckpt.exists():
        raise FileNotFoundError(f"missing classifier checkpoint: {cls_ckpt}")
    cfg = SimpleNamespace(num_classes=NUM_CLASSES, input_dim=INPUT_DIM,
                          smooth=LABEL_SMOOTHING, backbone=backbone)
    m = HeliALSClassifier(cfg).to(device)
    state = torch.load(cls_ckpt, map_location=device)
    m.load_state_dict(state["model_state"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def zero_fill(pc, wl_id):
    out = pc.clone()
    out[..., WAVELENGTH_CHANNELS[wl_id]] = 0.0
    return out


def compute_train_per_channel_mean(ds_train) -> torch.Tensor:
    """Global per-channel mean over training points, on the 12 spectral channels."""
    arr = ds_train.data[:, :, 3:15].astype(np.float32)
    return torch.as_tensor(arr.reshape(-1, 12).mean(axis=0), dtype=torch.float32)


def mean_fill(pc, wl_id, train_means_12):
    """Replace the 4 channels of wl_id with their training-set mean."""
    out = pc.clone()
    chs = WAVELENGTH_CHANNELS[wl_id]
    means = train_means_12[[c - 3 for c in chs]].to(pc.device, pc.dtype)
    out[..., chs] = means.view(1, 1, 4)
    return out


class LinearImputer:
    """Per-wavelength linear regression: (xyz + visible 8 channels) -> 4
    missing channels. Fit on the full training point cloud (or a subsample
    if too large). Very fast, no caching needed."""

    def __init__(self, ds_train, max_train_points: int = 200_000):
        flat = ds_train.data.reshape(-1, 16).astype(np.float32)
        if flat.shape[0] > max_train_points:
            rng = np.random.default_rng(0)
            sel = rng.choice(flat.shape[0], max_train_points, replace=False)
            flat = flat[sel]
        self._models: dict[int, LinearRegression] = {}
        print(f"  Fitting Linear baseline over {flat.shape[0]:,} points...")
        for wl in range(NUM_WAVELENGTHS):
            visible_idx, missing_idx = get_visible_missing_indices(wl)
            X = flat[:, [0, 1, 2] + visible_idx]
            Y = flat[:, missing_idx]
            self._models[wl] = LinearRegression(n_jobs=-1).fit(X, Y)

    def fill(self, pc, wl_id):
        visible_idx, missing_idx = get_visible_missing_indices(wl_id)
        reg = self._models[wl_id]
        B, N, _ = pc.shape
        feats = torch.cat([pc[..., :3], pc[..., visible_idx]], dim=-1)
        feats_np = feats.detach().cpu().numpy().reshape(-1, feats.size(-1))
        pred = reg.predict(feats_np).astype(np.float32).reshape(B, N, 4)
        out = pc.clone()
        out[..., missing_idx] = torch.from_numpy(pred).to(pc.device, pc.dtype)
        return out



def _train_fingerprint(ds_train) -> str | None:
    """Cheap fingerprint of the training cloud: shape + dtype + content hash.

    Used to reject a cached KNN/missForest model that was fitted on a
    different train split. Returns None when the dataset is not available
    (cache-only construction), in which case the caller cannot validate.
    """
    if ds_train is None:
        return None
    arr = np.ascontiguousarray(ds_train.data)
    h = hashlib.sha256()
    h.update(np.array(arr.shape, dtype=np.int64).tobytes())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _cache_matches(payload, ds_train, **params) -> bool:
    """True when a loaded pickle carries a fingerprint+params matching now.

    Older caches are plain dicts with no metadata; those cannot be validated,
    so they are accepted only when there is no train set to check against
    (and rejected otherwise, since silently fitting on the wrong split is the
    failure we are guarding against).
    """
    if not isinstance(payload, dict) or "__meta__" not in payload:
        return ds_train is None          # unverifiable: trust only if nothing to check
    meta = payload["__meta__"]
    fp = _train_fingerprint(ds_train)
    if fp is not None and meta.get("train_fingerprint") != fp:
        return False
    return all(meta.get(k) == v for k, v in params.items())

def knn_cache_path() -> Path:
    return ROOT / "data" / DATASET / f"knn_cache_k{KNN_K}.pkl"


class KNNImputer:
    """K-NN over (xyz + visible 8 channels) in the training-point database,
    averaging the K neighbours' missing-wavelength channels. Pickle-cached
    so it can be shared across all backbones (depends only on the dataset).
    """

    def __init__(self, ds_train=None, k: int = KNN_K,
                 cache_path: Path | str | None = None):
        self.k = k
        if cache_path is None:
            cache_path = knn_cache_path()
        cache_path = Path(cache_path)

        if cache_path.exists():
            t0 = time.time()
            print(f"  Loading KNN cache from {cache_path}...")
            try:
                with open(cache_path, "rb") as f:
                    self._indexes = pickle.load(f)
            except Exception as e:                    # noqa: BLE001
                # A pickle written under a different numpy/sklearn raises
                # ModuleNotFoundError('numpy._core') here. Rebuilding is
                # cheap (~25 s), so fall through rather than abort.
                print(f"  [cache unreadable: {type(e).__name__}] rebuilding")
            else:
                if _cache_matches(self._indexes, ds_train, k=k):
                    print(f"  Loaded in {time.time() - t0:.1f}s")
                    return
                print("  [cache stale: train set or k differs] rebuilding")

        if ds_train is None:
            raise ValueError(
                f"KNN cache not found at {cache_path} and no training "
                "dataset provided to build it.")

        flat = ds_train.data.reshape(-1, 16).astype(np.float32)
        self._indexes: dict[int, tuple] = {}
        print(f"  Building KNN indexes (k={k}) over {flat.shape[0]:,} points...")
        t0 = time.time()
        for wl in range(NUM_WAVELENGTHS):
            visible_idx, missing_idx = get_visible_missing_indices(wl)
            X = flat[:, [0, 1, 2] + visible_idx]
            Y = flat[:, missing_idx]
            idx = NearestNeighbors(n_neighbors=k, algorithm="auto",
                                   n_jobs=-1).fit(X)
            self._indexes[wl] = (idx, Y)
        print(f"  Built in {time.time() - t0:.1f}s")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Saving KNN cache to {cache_path}...")
        self._indexes["__meta__"] = {
            "train_fingerprint": _train_fingerprint(ds_train), "k": k}
        with open(cache_path, "wb") as f:
            pickle.dump(self._indexes, f, protocol=pickle.HIGHEST_PROTOCOL)

    def fill(self, pc, wl_id):
        visible_idx, missing_idx = get_visible_missing_indices(wl_id)
        idx, Y = self._indexes[wl_id]
        B, N, _ = pc.shape
        feats = torch.cat([pc[..., :3], pc[..., visible_idx]], dim=-1)
        feats_np = feats.detach().cpu().numpy().reshape(-1, feats.size(-1))
        _, knn_idx = idx.kneighbors(feats_np)
        gathered = Y[knn_idx].mean(axis=1).reshape(B, N, 4).astype(np.float32)
        out = pc.clone()
        out[..., missing_idx] = torch.from_numpy(gathered).to(pc.device, pc.dtype)
        return out


def missforest_cache_path() -> Path:
    return (ROOT / "data" / DATASET /
            f"missforest_cache_n{MF_N_ESTIMATORS}.pkl")


class MissForestImputer:
    """missForest-style imputation (Stekhoven & Buhlmann, 2012): a random
    forest predicting the missing channels from the observed ones.

    In the missing-wavelength setting the missingness is *monotone* -- exactly
    the 4 channels of w* are absent and the 8 remaining spectral channels are
    fully observed at every point -- so there are no mutually-missing columns
    to cycle over. missForest's iterative refinement therefore converges after
    a single conditional fit, and this single-pass forest is that fixed point.
    (missRanger differs only in its tree backend; in our setting the two agree
    to within 1e-4 RMSE, so we report one row.)

    Per-wavelength model: (xyz + visible 8 channels) -> 4 missing channels.
    Pickle-cached so it is shared across all backbones (depends only on the
    dataset), matching the KNN baseline's caching contract.
    """

    def __init__(self, ds_train=None, cache_path: Path | str | None = None,
                 n_estimators: int = MF_N_ESTIMATORS,
                 max_train_points: int = MF_MAX_TRAIN_POINTS):
        if cache_path is None:
            cache_path = missforest_cache_path()
        cache_path = Path(cache_path)

        if cache_path.exists():
            t0 = time.time()
            print(f"  Loading missForest cache from {cache_path}...")
            try:
                with open(cache_path, "rb") as f:
                    self._models = pickle.load(f)
            except Exception as e:                    # noqa: BLE001
                print(f"  [cache unreadable: {type(e).__name__}] rebuilding")
            else:
                if _cache_matches(self._models, ds_train,
                                  n_estimators=n_estimators,
                                  min_samples_leaf=MF_MIN_SAMPLES_LEAF):
                    print(f"  Loaded in {time.time() - t0:.1f}s")
                    return
                print("  [cache stale: train set or params differ] rebuilding")

        if ds_train is None:
            raise ValueError(
                f"missForest cache not found at {cache_path} and no training "
                "dataset provided to build it.")

        flat = ds_train.data.reshape(-1, 16).astype(np.float32)
        if flat.shape[0] > max_train_points:
            rng = np.random.default_rng(0)
            sel = rng.choice(flat.shape[0], max_train_points, replace=False)
            flat = flat[sel]
        self._models: dict[int, RandomForestRegressor] = {}
        print(f"  Fitting missForest ({n_estimators} trees) over "
              f"{flat.shape[0]:,} points...")
        t0 = time.time()
        for wl in range(NUM_WAVELENGTHS):
            visible_idx, missing_idx = get_visible_missing_indices(wl)
            X = flat[:, [0, 1, 2] + visible_idx]
            Y = flat[:, missing_idx]
            self._models[wl] = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=MF_MIN_SAMPLES_LEAF,
                n_jobs=-1, random_state=0,
            ).fit(X, Y)
            print(f"    wl={wl} fitted ({time.time() - t0:.1f}s elapsed)")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Saving missForest cache to {cache_path}...")
        self._models["__meta__"] = {
            "train_fingerprint": _train_fingerprint(ds_train),
            "n_estimators": n_estimators,
            "min_samples_leaf": MF_MIN_SAMPLES_LEAF}
        with open(cache_path, "wb") as f:
            pickle.dump(self._models, f, protocol=pickle.HIGHEST_PROTOCOL)

    def fill(self, pc, wl_id):
        visible_idx, missing_idx = get_visible_missing_indices(wl_id)
        reg = self._models[wl_id]
        B, N, _ = pc.shape
        feats = torch.cat([pc[..., :3], pc[..., visible_idx]], dim=-1)
        feats_np = feats.detach().cpu().numpy().reshape(-1, feats.size(-1))
        pred = reg.predict(feats_np).astype(np.float32).reshape(B, N, 4)
        out = pc.clone()
        out[..., missing_idx] = torch.from_numpy(pred).to(pc.device, pc.dtype)
        return out


# ---------------------------------------------------------------------------
# Imputed-test cache: baselines are functions of (dataset, K) only, so we
# compute + time them once and reuse across all (backbone, seed) runs.
# ---------------------------------------------------------------------------

def imputed_cache_dir() -> Path:
    return ROOT / "data" / DATASET / "imputed_test_cache"


def imputed_cache_paths(method: str, test_sig: str | None = None
                         ) -> tuple[Path, Path]:
    if method == "knn":
        tag = f"{method}_k{KNN_K}"
    elif method == "missforest":
        tag = f"{method}_n{MF_N_ESTIMATORS}"
    else:
        tag = method
    if test_sig:
        tag = f"{tag}_sig{test_sig[:8]}"
    d = imputed_cache_dir()
    return d / f"{tag}.npz", d / f"{tag}.json"


def _method_cache_params(method: str) -> dict:
    """Everything a cached imputation depends on besides the test set.

    The .npz holds imputed channels produced by a model FITTED ON TRAIN, so
    validating only the test signature let a changed train split (or changed
    hyperparameters) silently serve stale predictions.
    """
    if method == "knn":
        return {"k": KNN_K}
    if method == "missforest":
        return {"n_estimators": MF_N_ESTIMATORS,
                "min_samples_leaf": MF_MIN_SAMPLES_LEAF,
                "max_train_points": MF_MAX_TRAIN_POINTS}
    if method == "linear":
        return {"max_train_points": 200_000}
    return {}                      # zero/mean: no fitted hyperparameters


def _test_set_signature(all_pc_cpu: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.array(all_pc_cpu.shape, dtype=np.int64).tobytes())
    h.update(str(all_pc_cpu.dtype).encode())
    h.update(np.ascontiguousarray(all_pc_cpu).tobytes())
    return h.hexdigest()[:16]


def _build_imputed_test(method: str, make_fill_fn, all_pc: torch.Tensor,
                         batch_size: int, device: torch.device,
                         test_sig: str, train_fp: str | None = None,
                         ) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Run the baseline over the test set, time it, persist the result."""
    npz_path, meta_path = imputed_cache_paths(method, test_sig)
    cuda = (device.type == "cuda")
    n_test = all_pc.size(0)
    n_points = all_pc.size(1)

    print(f"  [build] imputed-test cache for '{method}' over "
          f"{n_test} trees x {n_points} points x 3 wavelengths "
          f"(batch_size={batch_size})...")
    imputed: dict[int, np.ndarray] = {}
    timings: dict[int, float] = {}
    for wl in range(NUM_WAVELENGTHS):
        _, missing_idx = get_visible_missing_indices(wl)
        fill_fn = make_fill_fn(wl)
        out = np.empty((n_test, n_points, 4), dtype=np.float32)
        t_total = 0.0
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                chunk = all_pc[i:i + batch_size]
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                filled = fill_fn(chunk)
                if cuda:
                    torch.cuda.synchronize()
                t_total += time.perf_counter() - t0
                out[i:i + batch_size] = (
                    filled[..., missing_idx].detach().cpu().numpy()
                )
        imputed[wl] = out
        timings[wl] = 1000.0 * t_total / max(1, n_test)
        print(f"    wl={wl}  fill = {timings[wl]:7.3f} ms/tree  "
              f"(total {t_total:.1f}s)")

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path, **{f"wl{wl}": imputed[wl] for wl in range(NUM_WAVELENGTHS)}
    )
    meta_path.write_text(json.dumps({
        "method":            method,
        "k":                 KNN_K if method == "knn" else None,
        "n_estimators":      (MF_N_ESTIMATORS if method == "missforest"
                              else None),
        "fill_ms_per_tree":  {str(wl): timings[wl]
                              for wl in range(NUM_WAVELENGTHS)},
        "test_signature":    test_sig,
        "train_fingerprint": train_fp,
        "params":            _method_cache_params(method),
        "n_test":            int(n_test),
        "n_points":          int(n_points),
        "batch_size":        int(batch_size),
        "device":            str(device),
        "built_at":          time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"  [saved] {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)")
    return imputed, timings


def load_or_build_imputed_test(method: str, deferred_fill_factory,
                                all_pc: torch.Tensor, batch_size: int,
                                device: torch.device, test_sig: str,
                                train_fp: str | None = None,
                                ) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Load the cache if present; otherwise build it. The fill factory is
    deferred so expensive imputers (e.g. KNN fit) are skipped on cache hit."""
    npz_path, meta_path = imputed_cache_paths(method, test_sig)
    if npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        # Validate the TRAIN fingerprint and the fitting hyperparameters too:
        # these predictions came from a model fitted on train, so a changed
        # train split must invalidate them even though the test set is
        # unchanged. Only zero filling is independent of training data;
        # mean filling needs the fingerprint too. Older caches without a
        # fingerprint are rebuilt when the caller supplies the current one.
        want_params = _method_cache_params(method)
        ok = (meta.get("test_signature") == test_sig
              and meta.get("train_fingerprint") == train_fp
              and all(meta.get("params", {}).get(k) == v
                      for k, v in want_params.items()))
        if ok:
            print(f"  [cache hit ] '{method}' -> {npz_path.name}")
            data = np.load(npz_path)
            imputed = {wl: data[f"wl{wl}"] for wl in range(NUM_WAVELENGTHS)}
            timings = {int(k): float(v)
                        for k, v in meta["fill_ms_per_tree"].items()}
            return imputed, timings
        why = ("test set" if meta.get("test_signature") != test_sig
               else "train set" if meta.get("train_fingerprint") != train_fp
               else "hyperparameters")
        print(f"  [cache stale] '{method}' ({why} differs); rebuilding")
    else:
        print(f"  [cache miss] '{method}'; will build")
    make_fill_fn = deferred_fill_factory()
    return _build_imputed_test(method, make_fill_fn, all_pc, batch_size,
                                device, test_sig, train_fp)


def apply_cached_imputation(all_pc: torch.Tensor, imputed_4ch: np.ndarray,
                             wl_id: int) -> torch.Tensor:
    _, missing_idx = get_visible_missing_indices(wl_id)
    out = all_pc.clone()
    out[..., missing_idx] = (
        torch.from_numpy(imputed_4ch).to(all_pc.device, all_pc.dtype)
    )
    return out


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

def build_dataset(subset: str):
    data_path = ROOT / "data" / DATASET
    cfg = SimpleNamespace(
        N_POINTS=NUM_POINTS, subset=subset,
        DATA_PATH=str(data_path),
        val_ratio=0.05, seed=42,
    )
    return TreeSpeciesDatasetHELIALS(cfg)


def collate(batch):
    pcs = torch.stack([torch.as_tensor(b[2][0], dtype=torch.float32)
                       for b in batch])
    labels = torch.as_tensor([int(b[2][1]) for b in batch], dtype=torch.long)
    return pcs, labels


# ---------------------------------------------------------------------------
# Training (paired-eval: WaveFill seed S uses classifier ensemble for L_cls,
# then is evaluated against classifier seed S at test time)
# ---------------------------------------------------------------------------

def cosine_warmup(opt, epochs, warmup=10, min_ratio=0.01):
    def f(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        t = (ep - warmup) / max(1, epochs - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return LambdaLR(opt, f)


def train_one_seed(seed: int, args, backbone: str,
                   classifier_dir: Path, device: torch.device,
                   out_dir: Path) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)

    seed_dir = out_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = seed_dir / imputer_ckpt_name(
        args.imputer, args.kan_grid, args.kan_k)
    history_path = seed_dir / imputer_history_name(args.imputer)

    if best_ckpt.exists() and not args.force_train:
        prev = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        planned, done = prev.get("epochs_planned"), prev.get("completed")
        if planned is None or done is None:
            raise SystemExit(
                f"\n{best_ckpt} predates the completion marker, so its epoch "
                "budget is unknown; reusing it could mix budgets in one "
                "table.\n  Re-run with --force-train, or delete that seed "
                "directory.\n")
        if planned != args.epochs or not done:
            raise SystemExit(
                f"\n{best_ckpt} was trained for {prev.get('epoch', -1) + 1}/"
                f"{planned} epochs (completed={done}), but this run asks for "
                f"{args.epochs}.\n  Mixing budgets in one table is not "
                "comparable. Re-run with --force-train, or delete that seed "
                "directory.\n")
        print(f"  [skip train] seed={seed} -> {best_ckpt.name} "
              f"({planned} ep, complete)")
        history = (json.load(open(history_path))
                   if history_path.exists() else [])
        return {"ckpt": best_ckpt, "history": history}

    train_ds = build_dataset("train")
    val_ds   = build_dataset("val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True,
                            collate_fn=collate)

    model = build_imputer(args.imputer, hidden=args.hidden,
                          grid=args.kan_grid, k=args.kan_k).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [train] seed={seed}  imputer={args.imputer}  "
          f"params={n_params:,}  "
          f"lambda_recon={LAMBDA_RECON}  lambda_cls={LAMBDA_CLS}")

    # Single paired classifier used for both the L_cls supervision signal at
    # training time AND val-OA-based best-checkpoint selection. WaveFill seed s
    # is paired with classifier seed s end-to-end.
    print(f"  [train] loading paired classifier (seed={seed}) for L_cls "
          f"+ val OA from {classifier_dir}...")
    paired_cls = load_paired_classifier(classifier_dir, backbone, seed, device)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = cosine_warmup(opt, args.epochs,
                          warmup=min(10, args.epochs // 5))

    history: list[dict] = []
    best_val_oa = -1.0
    for epoch in range(args.epochs):
        model.train()
        rec_tot, cls_tot, n_batches = 0.0, 0.0, 0
        for pc, y in train_loader:
            pc = pc.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B = pc.size(0)
            wl_id = int(torch.randint(0, NUM_WAVELENGTHS, (1,)).item())
            wl_id_t = torch.full((B,), wl_id, dtype=torch.long, device=device)
            xyz, visible, real, ret_n = split_pc(pc, wl_id)
            pred = model(xyz, visible, wl_id_t)

            L_rec = reconstruction_loss(pred, real)
            pc_recon = assemble_pc16(xyz, visible, pred, ret_n, wl_id)
            logits = paired_cls(pc_recon)
            L_cls = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
            L = LAMBDA_RECON * L_rec + LAMBDA_CLS * L_cls

            opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            rec_tot += float(L_rec); cls_tot += float(L_cls); n_batches += 1

        # Per-wavelength validation: reconstruction RMSE + paired-classifier OA.
        model.eval()
        rmse_per_wl = {wl: [] for wl in range(NUM_WAVELENGTHS)}
        correct_per_wl = {wl: 0 for wl in range(NUM_WAVELENGTHS)}
        n_val = 0
        with torch.no_grad():
            for pc, y in val_loader:
                pc = pc.to(device, non_blocking=True)
                y_dev = y.to(device, non_blocking=True)
                n_val += int(y.size(0))
                for wl in range(NUM_WAVELENGTHS):
                    B = pc.size(0)
                    wl_id_t = torch.full((B,), wl, dtype=torch.long, device=device)
                    xyz, visible, real, ret_n = split_pc(pc, wl)
                    pred_4ch = model(xyz, visible, wl_id_t)
                    rmse_per_wl[wl].append(
                        (pred_4ch - real).pow(2).mean(dim=(1, 2)).sqrt().cpu().numpy()
                    )
                    pc_recon = assemble_pc16(xyz, visible, pred_4ch, ret_n, wl)
                    logits = paired_cls(pc_recon)
                    correct_per_wl[wl] += int(
                        (logits.argmax(-1) == y_dev).sum().item()
                    )
        val_rmse = {f"rmse_wl{wl}": float(np.concatenate(rmse_per_wl[wl]).mean())
                    for wl in range(NUM_WAVELENGTHS)}
        val_rmse["rmse_avg"] = float(np.mean(list(val_rmse.values())))
        val_oa = {f"oa_wl{wl}": correct_per_wl[wl] / max(1, n_val)
                   for wl in range(NUM_WAVELENGTHS)}
        val_oa["oa_avg"] = float(np.mean(list(val_oa.values())))

        sched.step()
        rec_avg, cls_avg = rec_tot / n_batches, cls_tot / n_batches
        history.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"],
                        "train_recon": rec_avg, "train_cls": cls_avg,
                        "val": val_rmse, "val_oa": val_oa})

        if val_oa["oa_avg"] > best_val_oa:
            best_val_oa = val_oa["oa_avg"]
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_oa_avg":   best_val_oa,
                        "val_rmse_avg": val_rmse["rmse_avg"],
                        "hidden": getattr(model, "hidden_width",
                                            args.hidden), "backbone": backbone,
                        "imputer": args.imputer,
                        "kan_grid": args.kan_grid, "kan_k": args.kan_k,
                        "lambda_recon": LAMBDA_RECON,
                        "lambda_cls":   LAMBDA_CLS,
                        # Budget + completion marker: without them a run
                        # stopped early is indistinguishable from a finished
                        # one and the skip-train guard reuses it silently.
                        "epochs_planned": args.epochs,
                        "completed": False}, best_ckpt)

        print(f"  [seed {seed}] ep {epoch+1:3d}/{args.epochs}  "
              f"recon={rec_avg:.4f}  cls={cls_avg:.4f}  "
              f"val_oa[0/1/2/avg]={val_oa['oa_wl0']*100:5.2f}/"
              f"{val_oa['oa_wl1']*100:5.2f}/{val_oa['oa_wl2']*100:5.2f}/"
              f"{val_oa['oa_avg']*100:5.2f}  "
              f"val_rmse_avg={val_rmse['rmse_avg']:.4f}  "
              f"best_oa={best_val_oa*100:5.2f}", flush=True)

    # The in-loop save cannot know whether training will finish, and its `ep`
    # is the BEST epoch, not the last one. Stamp completion here instead:
    # reaching this line means the epoch loop ran to the end.
    _done = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    _done["completed"] = True
    torch.save(_done, best_ckpt)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    return {"ckpt": best_ckpt, "history": history}


# ---------------------------------------------------------------------------
# Eval (paired classifier; all configured baselines; full reference)
# ---------------------------------------------------------------------------

def report_metrics(y, pred):
    oa = accuracy_score(y, pred)
    _, r, f1, _ = precision_recall_fscore_support(
        y, pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    cm = confusion_matrix(y, pred, labels=list(range(NUM_CLASSES)))
    return {"overall_accuracy": float(oa),
            "macro_f1": float(f1.mean()),
            "macro_recall": float(r.mean()),
            "per_class_f1": {CLASSES[i]: float(f1[i]) for i in range(NUM_CLASSES)},
            "confusion_matrix": cm.astype(int).tolist()}


def eval_one_seed(seed: int, args, backbone: str,
                  classifier_dir: Path, device: torch.device,
                  out_dir: Path,
                  all_pc: torch.Tensor, all_y: np.ndarray,
                  baseline_imputed: dict[str, dict[int, np.ndarray]],
                  baseline_timings: dict[str, dict[int, float]]) -> dict:
    """Evaluate WaveFill seed S against the paired classifier (cls seed S),
    reusing the shared baseline imputations."""
    seed_dir = out_dir / f"seed{seed}"
    eval_json = seed_dir / "eval_results.json"

    def _load_eval(path: Path) -> dict:
        with open(path) as f:
            r = json.load(f)
        for info in r.get("by_method", {}).values():
            info["per_wl"] = {
                (int(k) if k.isdigit() else k): v
                for k, v in info["per_wl"].items()
            }
        return r

    # Results from other imputer variants already evaluated for this seed are
    # merged in, so running --imputer mlp and --imputer relukan into the same
    # --output-dir accumulates both rows instead of overwriting one another.
    prev = _load_eval(eval_json) if eval_json.exists() else None
    # A retrained checkpoint invalidates any cached evaluation of it, so
    # --force-train must not return stale metrics.
    if prev is not None and not args.force_eval and not args.force_train:
        if imputer_method_key(args.imputer) in prev.get("by_method", {}):
            print(f"  [skip eval ] seed={seed} -> {eval_json}")
            return prev

    ckpt_path = seed_dir / imputer_ckpt_name(
        args.imputer, args.kan_grid, args.kan_k)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing imputer checkpoint: {ckpt_path}")

    classifier = load_paired_classifier(classifier_dir, backbone, seed, device)

    state = torch.load(ckpt_path, map_location=device)
    # Fail loudly rather than evaluate a model whose architecture differs from
    # what this run claims to be testing.
    for field, want in (("imputer", args.imputer), ("kan_grid", args.kan_grid),
                        ("kan_k", args.kan_k)):
        got = state.get(field)
        if got is not None and got != want and (
                args.imputer == "relukan" or field == "imputer"):
            raise RuntimeError(
                f"checkpoint/config mismatch in {ckpt_path.name}: "
                f"{field}={got!r} in the checkpoint but {want!r} requested. "
                "Use --force-train or a different --output-dir.")
    imputer = build_imputer(state.get("imputer", args.imputer),
                            hidden=state.get("hidden"),
                            grid=state.get("kan_grid", args.kan_grid),
                            k=state.get("kan_k", args.kan_k)).to(device)
    imputer.load_state_dict(state["model_state"])
    imputer.eval()

    n_test = all_pc.size(0)
    cuda = (device.type == "cuda")

    @torch.no_grad()
    def classify_chunked(test_pc16: torch.Tensor) -> np.ndarray:
        logits = []
        for i in range(0, n_test, args.batch_size):
            logits.append(classifier(test_pc16[i:i + args.batch_size]).cpu())
        return torch.cat(logits, 0).argmax(-1).numpy()

    @torch.no_grad()
    def predict_wavefill(wl: int) -> tuple[np.ndarray, float]:
        logits = []
        fill_time = 0.0
        for i in range(0, n_test, args.batch_size):
            chunk = all_pc[i:i + args.batch_size]
            B = chunk.size(0)
            wl_id_t = torch.full((B,), wl, dtype=torch.long, device=device)
            xyz, visible, _, ret_n = split_pc(chunk, wl)
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred_4ch = imputer(xyz, visible, wl_id_t)
            filled = assemble_pc16(xyz, visible, pred_4ch, ret_n, wl)
            if cuda:
                torch.cuda.synchronize()
            fill_time += time.perf_counter() - t0
            logits.append(classifier(filled).cpu())
        pred = torch.cat(logits, 0).argmax(-1).numpy()
        return pred, fill_time

    n_params_imputer    = sum(p.numel() for p in imputer.parameters())
    n_params_classifier = sum(p.numel() for p in classifier.parameters())

    results = {
        "seed": seed,
        "n_test": int(n_test),
        "backbone": backbone,
        "model_params": {
            imputer_method_key(args.imputer): int(n_params_imputer),
            "paired_classifier":   int(n_params_classifier),
        },
        "by_method": {},
    }

    print(f"  [eval seed={seed}] full reference...", flush=True)
    pred = classify_chunked(all_pc)
    full_metrics = report_metrics(all_y, pred)
    full_metrics["fill_time_ms_per_tree"] = 0.0
    results["by_method"]["full"] = {
        "per_wl": {0: dict(full_metrics),
                   1: dict(full_metrics),
                   2: dict(full_metrics)}
    }

    for tag in ("zero", "mean", "linear", "knn", "missforest"):
        if tag not in baseline_imputed:
            continue
        per_wl_imp = baseline_imputed[tag]
        per_wl_time = baseline_timings[tag]
        results["by_method"][tag] = {"per_wl": {}}
        for wl in range(NUM_WAVELENGTHS):
            test_pc16 = apply_cached_imputation(all_pc, per_wl_imp[wl], wl)
            pred = classify_chunked(test_pc16)
            m = report_metrics(all_y, pred)
            m["fill_time_ms_per_tree"] = per_wl_time[wl]
            results["by_method"][tag]["per_wl"][wl] = m
            print(f"    {tag:<12} wl={wl}  OA={m['overall_accuracy']*100:5.2f}  "
                  f"F1={m['macro_f1']*100:5.2f}  "
                  f"fill={m['fill_time_ms_per_tree']:.3f}ms (cached)",
                  flush=True)

    wf_key = imputer_method_key(args.imputer)
    results["by_method"][wf_key] = {"per_wl": {}}
    for wl in range(NUM_WAVELENGTHS):
        pred, fill_time = predict_wavefill(wl)
        m = report_metrics(all_y, pred)
        m["fill_time_ms_per_tree"] = 1000.0 * fill_time / max(1, n_test)
        results["by_method"][wf_key]["per_wl"][wl] = m
        print(f"    {wf_key:<12} wl={wl}  OA={m['overall_accuracy']*100:5.2f}  "
              f"F1={m['macro_f1']*100:5.2f}  "
              f"fill={m['fill_time_ms_per_tree']:.3f}ms", flush=True)

    # Carry forward rows produced by other imputer variants for this seed.
    if prev is not None:
        for key, info in prev.get("by_method", {}).items():
            results["by_method"].setdefault(key, info)
        for key, val in prev.get("model_params", {}).items():
            results.setdefault("model_params", {}).setdefault(key, val)

    with open(eval_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ---------------------------------------------------------------------------
# Aggregation + outputs
# ---------------------------------------------------------------------------

METHOD_ORDER = ["full", "zero", "mean", "linear", "knn", "missforest",
                "wavefill", "wavefill_relukan", "wavefill_fastkan"]
METHOD_LABEL_PRETTY = {
    "full":             "Full 16D (ref.)",
    "zero":             "Zero-fill",
    "mean":             "Mean-fill",
    "linear":           "Linear",
    "knn":              "KNN",
    "missforest":       "missForest (RF)",
    "wavefill":         "WaveFill-Net (ours)",
    "wavefill_relukan": "WaveFill-Net (ReLU-KAN)",
    "wavefill_fastkan": "WaveFill-Net (FastKAN)",
}

# Our own methods -- bolded in the LaTeX tables, and excluded from the
# "strongest baseline" comparison rows.
OURS_METHODS = {"wavefill", "wavefill_relukan", "wavefill_fastkan"}


def _methods_across_seeds(seed_results) -> list[str]:
    """Union of methods over ALL seeds, in first-seen order.

    Taking the list from seed 0 alone (the previous behaviour) was unsafe in
    both directions: a method missing from a later seed raised KeyError and
    killed the aggregate after every run had already finished, while a method
    missing from seed 0 was silently dropped from every table even though
    other seeds had it. Methods are aggregated over whichever seeds have
    them, and the seed count is reported per method.
    """
    seen: dict[str, None] = {}
    for r in seed_results:
        for m in r.get("by_method", {}):
            seen.setdefault(m, None)
    return list(seen)


def aggregate(seed_results, scale=100.0):
    """Mean +- std across seeds for accuracy; mean only for fill timing.

    A method present in only a subset of seeds is aggregated over that subset;
    `n_seeds` records how many actually contributed.
    """
    methods = _methods_across_seeds(seed_results)
    out = {}
    for m in methods:
        per_wl = {}
        seed_results_m = [r for r in seed_results if m in r.get("by_method", {})]
        for wl in range(NUM_WAVELENGTHS):
            oas, f1s, times = [], [], []
            for r in seed_results_m:
                cell = r["by_method"][m]["per_wl"][wl]
                oas.append(cell["overall_accuracy"] * scale)
                f1s.append(cell["macro_f1"] * scale)
                if "fill_time_ms_per_tree" in cell:
                    times.append(cell["fill_time_ms_per_tree"])
            per_wl[wl] = {
                "oa_mean":   statistics.mean(oas),
                "oa_std":    statistics.stdev(oas) if len(oas) > 1 else 0.0,
                "f1_mean":   statistics.mean(f1s),
                "f1_std":    statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
                "time_mean_ms_per_tree": (statistics.mean(times)
                                            if times else None),
                # Per-wavelength seed count too: a method present in only one
                # seed was previously reported without any count here, so an
                # export could imply the run's nominal seed count instead of
                # the method's actual one.
                "n_seeds": len(oas),
                "oa_values": oas,
                "f1_values": f1s,
            }
        per_seed_oa, per_seed_f1, per_seed_time = [], [], []
        for r in seed_results_m:
            cells = r["by_method"][m]["per_wl"]
            per_seed_oa.append(statistics.mean(cells[wl]["overall_accuracy"] * scale
                                                for wl in range(NUM_WAVELENGTHS)))
            per_seed_f1.append(statistics.mean(cells[wl]["macro_f1"] * scale
                                                for wl in range(NUM_WAVELENGTHS)))
            seed_times = [cells[wl].get("fill_time_ms_per_tree")
                           for wl in range(NUM_WAVELENGTHS)]
            if all(t is not None for t in seed_times):
                per_seed_time.append(statistics.mean(seed_times))
        per_wl["avg"] = {
            "oa_mean": statistics.mean(per_seed_oa),
            "oa_std":  statistics.stdev(per_seed_oa) if len(per_seed_oa) > 1 else 0.0,
            "f1_mean": statistics.mean(per_seed_f1),
            "f1_std":  statistics.stdev(per_seed_f1) if len(per_seed_f1) > 1 else 0.0,
            "time_mean_ms_per_tree": (statistics.mean(per_seed_time)
                                       if per_seed_time else None),
            "oa_values": per_seed_oa,
            "f1_values": per_seed_f1,
            # How many seeds actually contributed to this method's numbers.
            "n_seeds": len(seed_results_m),
        }
        out[m] = per_wl
    return out


def aggregate_confusion_matrices(seed_results) -> dict:
    """Sum + per-seed list of confusion matrices per (method, wavelength)."""
    out: dict = {}
    methods = _methods_across_seeds(seed_results)
    for m in methods:
        out[m] = {}
        for wl in range(NUM_WAVELENGTHS):
            per_seed = []
            for r in seed_results:
                if m not in r.get("by_method", {}):
                    continue        # method absent from this seed
                cell = r["by_method"][m]["per_wl"][wl]
                cm = cell.get("confusion_matrix")
                if cm is not None:
                    per_seed.append(np.asarray(cm, dtype=np.int64))
            if not per_seed:
                continue
            summed = np.sum(np.stack(per_seed, 0), axis=0)
            out[m][str(wl)] = {
                "sum":      summed.astype(int).tolist(),
                "per_seed": [cm.astype(int).tolist() for cm in per_seed],
            }
        if all(str(wl) in out[m] for wl in range(NUM_WAVELENGTHS)):
            per_seed_avg = []
            # Index over the seeds that actually contributed to THIS method,
            # not over every seed in the run -- a method missing from some
            # seed has a shorter per_seed list.
            n_contrib = min(len(out[m][str(wl)]["per_seed"])
                            for wl in range(NUM_WAVELENGTHS))
            for s_idx in range(n_contrib):
                acc = None
                for wl in range(NUM_WAVELENGTHS):
                    cm = np.asarray(
                        out[m][str(wl)]["per_seed"][s_idx], dtype=np.int64
                    )
                    acc = cm if acc is None else acc + cm
                per_seed_avg.append(acc)
            out[m]["avg"] = {
                "sum":      np.sum(np.stack(per_seed_avg, 0), 0).astype(int).tolist(),
                "per_seed": [cm.astype(int).tolist() for cm in per_seed_avg],
            }
    return {
        "classes": CLASSES,
        "by_method": out,
    }


def write_csv(seed_results, path: Path):
    cols = ["method", "seed", "wavelength", "oa", "macro_f1",
            "fill_time_ms_per_tree"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in seed_results:
            for m, info in r["by_method"].items():
                for wl, cell in info["per_wl"].items():
                    w.writerow({
                        "method": m, "seed": r["seed"], "wavelength": wl,
                        "oa": cell["overall_accuracy"] * 100,
                        "macro_f1": cell["macro_f1"] * 100,
                        "fill_time_ms_per_tree": cell.get("fill_time_ms_per_tree", ""),
                    })


def write_latex(agg: dict, n_seeds: int, backbone: str, path: Path):
    methods = [m for m in METHOD_ORDER if m in agg]
    lines = [
        "\\documentclass[11pt]{article}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage[a4paper,margin=2cm]{geometry}",
        "\\usepackage{booktabs}",
        "\\usepackage{amsmath, bm}",
        "\\usepackage{caption}",
        "\\usepackage[table]{xcolor}",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        f"\\title{{WaveFill-Net vs.~baselines ({backbone} classifier)}}",
        "\\author{Auto-generated by \\texttt{main.py}}",
        "\\date{\\today}",
        "\\begin{document}",
        "\\maketitle",
        "\\noindent\\textit{Two-loss WaveFill-Net (L1 reconstruction + "
        "task-coupled CE; moment matching and latent noise $\\mathbf{z}$ "
        "dropped per ablation). Hyperparameters: "
        f"$\\lambda_{{\\mathrm{{recon}}}}{{=}}{LAMBDA_RECON}$, "
        f"$\\lambda_{{\\mathrm{{cls}}}}{{=}}{LAMBDA_CLS}$, "
        f"hidden $H{{=}}{HIDDEN}$. "
        f"Paired-eval against {backbone} classifier, {n_seeds} seeds. "
        "Each cell: OA / macro-F1 (\\%), mean $\\pm$ std.}",
        "\\bigskip",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Missing-wavelength imputation results, "
        f"{backbone} classifier.}}",
        f"\\label{{tab:method_{backbone}}}",
        "\\footnotesize",
        "\\begin{tabular}{l c c c c}",
        "\\toprule",
        "\\textbf{Method} & "
        "\\textbf{WL=0 (1550\\,nm)} & \\textbf{WL=1 (905\\,nm)} & "
        "\\textbf{WL=2 (532\\,nm)} & \\textbf{Avg over $\\lambda$} \\\\",
        "\\midrule",
    ]

    def _fmt(d):
        return (f"{d['oa_mean']:.2f}\\,$\\pm$\\,{d['oa_std']:.2f} / "
                f"{d['f1_mean']:.2f}\\,$\\pm$\\,{d['f1_std']:.2f}")

    for m in methods:
        lbl = METHOD_LABEL_PRETTY.get(m, m)
        if m in OURS_METHODS:
            lbl = "\\textbf{" + lbl + "}"
        cells = [_fmt(agg[m][0]), _fmt(agg[m][1]),
                 _fmt(agg[m][2]), _fmt(agg[m]["avg"])]
        lines.append(f" {lbl} & " + " & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    time_methods = [m for m in methods
                     if agg[m]["avg"].get("time_mean_ms_per_tree") is not None]
    if time_methods:
        lines += [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Fill (imputation) time, mean ms per tree. "
            f"{backbone} classifier, batch size used during the imputed-test "
            f"cache build. Backbone-agnostic baselines (Zero/Mean/Linear/KNN) "
            f"are measured once at cache build; WaveFill-Net is averaged "
            f"across the {n_seeds} seed(s).}}",
            f"\\label{{tab:time_{backbone}}}",
            "\\footnotesize",
            "\\begin{tabular}{l r r r r}",
            "\\toprule",
            "\\textbf{Method} & "
            "\\textbf{WL=0} & \\textbf{WL=1} & \\textbf{WL=2} & "
            "\\textbf{Avg} \\\\",
            "\\midrule",
        ]
        for m in time_methods:
            lbl = METHOD_LABEL_PRETTY.get(m, m)
            if m in OURS_METHODS:
                lbl = "\\textbf{" + lbl + "}"
            row_cells = []
            for k in (0, 1, 2, "avg"):
                t = agg[m][k].get("time_mean_ms_per_tree")
                row_cells.append(f"{t:.3f}" if t is not None else "--")
            lines.append(f" {lbl} & " + " & ".join(row_cells) + " \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]

    lines += [
        "\\end{document}",
        "",
    ]
    path.write_text("\n".join(lines))


def print_seed_table(result: dict, backbone: str, seed: int):
    """One comparison table for a single (backbone, seed), printed as soon as
    that seed is evaluated -- so progress is visible without waiting for the
    whole sweep. No std: a single seed has no spread."""
    by_method = result.get("by_method", {})
    methods = [m for m in METHOD_ORDER if m in by_method]
    if not methods:
        return
    wls = list(range(NUM_WAVELENGTHS))

    def cell(m, wl):
        c = by_method[m]["per_wl"][wl]
        return c["overall_accuracy"] * 100, c["macro_f1"] * 100

    print("\n" + "-" * 84)
    print(f"  seed {seed}  --  backbone: {backbone}   [OA / macro-F1 %]")
    print("-" * 84)
    print(f"  {'method':<24}" + "".join(f"{f'wl={w} ({WL_NM[w]})':>16}"
                                        for w in wls) + f"{'avg':>14}")
    print("  " + "-" * 80)
    for m in methods:
        oas, f1s, cells = [], [], []
        for wl in wls:
            oa, f1 = cell(m, wl)
            oas.append(oa); f1s.append(f1)
            cells.append(f"{oa:6.2f}/{f1:5.2f}")
        lbl = METHOD_LABEL_PRETTY.get(m, m) + (" *" if m in OURS_METHODS else "")
        print(f"  {lbl:<24}" + "".join(f"{c:>16}" for c in cells)
              + f"{statistics.mean(oas):7.2f}/{statistics.mean(f1s):5.2f}")

    # ours vs the strongest baseline, on this seed
    base = [m for m in methods if m not in OURS_METHODS and m != "full"]
    ours = [m for m in methods if m in OURS_METHODS]
    if base and ours:
        def wl_avg(m):
            return statistics.mean(cell(m, wl)[0] for wl in wls)
        best = max(base, key=wl_avg)
        print("  " + "-" * 80)
        print(f"  strongest baseline: {METHOD_LABEL_PRETTY.get(best, best)}"
              f" ({wl_avg(best):.2f} OA avg)")
        for m in ours:
            print(f"    {METHOD_LABEL_PRETTY.get(m, m):<24} "
                  f"{wl_avg(m) - wl_avg(best):+6.2f} pp OA")
    print("-" * 84, flush=True)


def print_summary(agg, n_seeds, backbone):
    methods = [m for m in METHOD_ORDER if m in agg]
    print("\n" + "=" * 80)
    print(f"  WaveFill-Net + baselines  -- backbone: {backbone}")
    print(f"  Results over {n_seeds} seed(s) (paired-eval)")
    print("=" * 80)
    head = f"  {'method':<22} {'wl=0':>14}  {'wl=1':>14}  {'wl=2':>14}  {'avg':>14}"
    print(head + "    [OA mean±std]")
    for m in methods:
        cells = []
        for k in (0, 1, 2, "avg"):
            d = agg[m][k]
            cells.append(f"{d['oa_mean']:5.2f}±{d['oa_std']:4.2f}")
        print(f"  {METHOD_LABEL_PRETTY.get(m, m):<22} "
              + "  ".join(f"{c:>14}" for c in cells))

    print()
    print(head + "    [fill ms/tree, mean]")
    for m in methods:
        cells = []
        for k in (0, 1, 2, "avg"):
            t = agg[m][k].get("time_mean_ms_per_tree")
            cells.append(f"{t:>10.3f} ms" if t is not None else f"{'-':>13}")
        print(f"  {METHOD_LABEL_PRETTY.get(m, m):<22} "
              + "  ".join(f"{c:>14}" for c in cells))
    print("=" * 80)

    ours = [m for m in METHOD_ORDER if m in OURS_METHODS and m in agg]
    if not ours:
        return
    for mine in ours:
        for ref in ("zero", "mean", "linear", "knn", "missforest", "full"):
            if ref not in agg:
                continue
            print(f"\n  {METHOD_LABEL_PRETTY.get(mine, mine)} vs "
                  f"{ref.upper():<10}  (OA delta, mean):")
            for k in (0, 1, 2, "avg"):
                d = agg[mine][k]["oa_mean"] - agg[ref][k]["oa_mean"]
                label_k = f"wl={k}" if k != "avg" else "avg"
                print(f"    {label_k:<8}: {d:+5.2f} pp")
    # Head-to-head between our two imputer variants, when both are present.
    if len(ours) > 1:
        a, b = ours[0], ours[1]
        print(f"\n  {METHOD_LABEL_PRETTY.get(b, b)} vs "
              f"{METHOD_LABEL_PRETTY.get(a, a)}  (OA delta, mean):")
        for k in (0, 1, 2, "avg"):
            d = agg[b][k]["oa_mean"] - agg[a][k]["oa_mean"]
            label_k = f"wl={k}" if k != "avg" else "avg"
            print(f"    {label_k:<8}: {d:+5.2f} pp")
    print()


# ---------------------------------------------------------------------------
# Convergence plots
# ---------------------------------------------------------------------------

def make_convergence_plots(seeds: list[int], out_dir: Path, backbone: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plots] matplotlib not installed")
        return

    histories = []
    for s in seeds:
        path = out_dir / f"seed{s}" / "train_history.json"
        if path.exists():
            with open(path) as f:
                h = json.load(f)
            if h:
                histories.append(h)
    if not histories:
        print("  [skip plots] no training histories found in", out_dir)
        return

    min_epochs = min(len(h) for h in histories)
    epochs = np.arange(min_epochs)

    def _stack(keys):
        arrs = []
        for h in histories:
            series = []
            for ep in h[:min_epochs]:
                v = ep
                for k in keys:
                    v = v[k]
                series.append(v)
            arrs.append(series)
        return np.array(arrs)

    def _band(ax, x, mean, std, color, label):
        ax.plot(x, mean, color=color, label=label, linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)

    # val RMSE per wavelength + avg
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [(0, "wl=0 (1550 nm)"), (1, "wl=1 (905 nm)"),
              (2, "wl=2 (532 nm)"), ("avg", "Average over $\\lambda$")]
    for ax, (wl, title) in zip(axes.flat, panels):
        key = f"rmse_wl{wl}" if wl != "avg" else "rmse_avg"
        arr = _stack(("val", key))
        _band(ax, epochs, arr.mean(0), arr.std(0),
              color="tab:blue", label="WaveFill-Net")
        ax.set_title(title); ax.set_xlabel("epoch")
        ax.set_ylabel("val RMSE"); ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"Validation RMSE convergence ({backbone}; "
                 f"{len(histories)} seeds, mean $\\pm$ std)", fontsize=12)
    fig.tight_layout()
    p1 = out_dir / "convergence_val_rmse.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # training losses
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [
        (axes[0], "train_recon",
            "Training $\\mathcal{L}_{\\mathrm{recon}}$ (L1)"),
        (axes[1], "train_cls",
            "Training $\\mathcal{L}_{\\mathrm{cls}}$ (task-coupled CE)"),
    ]:
        arr = _stack((key,))
        _band(ax, epochs, arr.mean(0), arr.std(0),
              color="tab:blue", label="WaveFill-Net")
        ax.set_title(title); ax.set_xlabel("epoch")
        ax.set_ylabel("loss"); ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"Training-loss convergence ({backbone}; "
                 f"{len(histories)} seeds, mean $\\pm$ std)", fontsize=12)
    fig.tight_layout()
    p2 = out_dir / "convergence_train_losses.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"  convergence_val_rmse.png        : {p1}")
    print(f"  convergence_train_losses.png    : {p2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True,
                   help="Classifier backbone name; must match a directory "
                        "under <classifier-root>/<BACKBONE>/classifier/ "
                        "containing model_seed{0..4}.pt.")
    p.add_argument("--classifier-root",
                   default=str(CLASSIFIER_ROOT_DEFAULT),
                   help="Where the per-backbone classifier ensembles live. "
                        f"Default: {CLASSIFIER_ROOT_DEFAULT}")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output-dir", default=None,
                   help="Where to store the outputs. Default: "
                        "runs/method/<BACKBONE>/")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training and only re-evaluate / re-aggregate "
                        "existing runs.")
    p.add_argument("--force-train", action="store_true",
                   help="Retrain even if checkpoint exists.")
    p.add_argument("--force-eval", action="store_true",
                   help="Re-evaluate even if eval_results.json exists.")
    p.add_argument("--no-mean", action="store_true",
                   help="Skip the mean-fill baseline.")
    p.add_argument("--no-linear", action="store_true",
                   help="Skip the linear-regression baseline.")
    p.add_argument("--imputer", default="mlp", choices=sorted(IMPUTERS),
                   help="which WaveFill-Net variant to train/evaluate: "
                        "'mlp' (published) or 'relukan' (ReLU-KAN layers). "
                        "Each writes its own checkpoint and result key, so "
                        "both can be run into the same --output-dir.")
    p.add_argument("--hidden", type=int, default=None,
                   help="imputer hidden width; default is per-variant "
                        f"(MLP {HIDDEN}, ReLU-KAN {KAN_HIDDEN}) so the two "
                        "stay parameter-matched")
    p.add_argument("--kan-grid", type=int, default=KAN_GRID,
                   help=f"ReLU-KAN grid size g (default: {KAN_GRID})")
    p.add_argument("--kan-k", type=int, default=KAN_K,
                   help=f"ReLU-KAN phase overlap k (default: {KAN_K})")
    p.add_argument("--no-missforest", action="store_true",
                   help="skip the missForest (random-forest) baseline")
    p.add_argument("--no-knn", action="store_true",
                   help="Skip the KNN baseline.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip the convergence plots (matplotlib).")
    p.add_argument("--build-caches-only", action="store_true",
                   help="Only construct the KNN-index pickle AND the imputed-"
                        "test caches (zero/mean/linear/knn/missforest), then "
                        "exit. No "
                        "training, no eval. Useful to run once before "
                        "launching the per-backbone runs in parallel.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan only; do not train or eval.")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        sys.exit("ERROR: --seeds must contain at least one integer")

    backbone = args.backbone
    classifier_root = Path(args.classifier_root).resolve()
    classifier_dir = classifier_root / backbone / "classifier"
    out_dir = (Path(args.output_dir).resolve() if args.output_dir
               else (ROOT / "runs/method" / backbone).resolve())

    if not (ROOT / "data" / DATASET).exists():
        sys.exit(f"ERROR: dataset directory missing: data/{DATASET}")

    # Skip classifier check when only building caches; baselines don't need it.
    if not args.build_caches_only:
        for s in seeds:
            if not (classifier_dir / f"model_seed{s}.pt").exists():
                sys.exit(f"ERROR: missing classifier checkpoint for "
                         f"backbone={backbone} seed={s}: "
                         f"{classifier_dir / f'model_seed{s}.pt'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("  WaveFill-Net main pipeline")
    print("=" * 80)
    print(f"  Backbone         : {backbone}")
    print(f"  Classifier dir   : {classifier_dir}")
    print(f"  Device           : {device}")
    print(f"  Seeds            : {seeds}")
    print(f"  Epochs / seed    : {args.epochs}")
    print(f"  Hidden           : {HIDDEN}")
    print(f"  Imputer          : {args.imputer}"
          + (f"  (grid={args.kan_grid}, k={args.kan_k})"
             if args.imputer == "relukan" else ""))
    print(f"  z_dim            : 0  (dropped)")
    print(f"  lambda_recon     : {LAMBDA_RECON}")
    print(f"  lambda_cls       : {LAMBDA_CLS}")
    print(f"  L_cls teacher    : paired classifier (single, seed-matched)")
    enabled = []
    enabled.append("zero")
    if not args.no_mean:   enabled.append("mean")
    if not args.no_linear: enabled.append("linear")
    if not args.no_knn:    enabled.append(f"knn (k={KNN_K})")
    if not args.no_missforest:
        enabled.append(f"missforest (n={MF_N_ESTIMATORS})")
    print(f"  Baselines        : {', '.join(enabled)}")
    print(f"  Ours             : {imputer_method_key(args.imputer)}")
    print(f"  Reference        : full (16D upper bound)")
    print(f"  Output dir       : {out_dir}")
    print("=" * 80)

    if args.dry_run:
        for s in seeds:
            print(f"  [plan] seed={s}: would train + eval "
                  f"(paired vs cls seed {s})")
        return

    # A seed needs evaluating when its file is missing, when --force-eval or
    # --force-train is given, OR when the requested imputer variant is not yet
    # among its recorded methods. Testing mere file existence meant that adding
    # a variant (e.g. --imputer fastkan next to an mlp-only file) left
    # need_eval False, so the test set was never loaded and eval_one_seed
    # crashed on all_pc=None.
    def _seed_needs_eval(s: int) -> bool:
        f = out_dir / f"seed{s}" / "eval_results.json"
        if not f.exists() or args.force_eval or args.force_train:
            return True
        try:
            return imputer_method_key(args.imputer) not in \
                json.loads(f.read_text()).get("by_method", {})
        except (json.JSONDecodeError, OSError):
            return True                     # unreadable -> redo it

    any_eval_pending = any(_seed_needs_eval(s) for s in seeds)
    need_eval = args.build_caches_only or any_eval_pending

    all_pc = None
    all_y = None
    test_sig = None
    baseline_imputed: dict[str, dict[int, np.ndarray]] = {}
    baseline_timings: dict[str, dict[int, float]] = {}
    if need_eval:
        print("\n--- Test set + imputed-test cache ---")
        test_ds = build_dataset("test")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4,
                                 pin_memory=True, collate_fn=collate)
        pcs, ys = [], []
        for pc, y in test_loader:
            pcs.append(pc.to(device))
            ys.append(y)
        all_pc = torch.cat(pcs, 0)
        all_y = torch.cat(ys, 0).numpy()
        all_pc_cpu = all_pc.detach().cpu().numpy()
        test_sig = _test_set_signature(all_pc_cpu)
        print(f"  test set      : {all_pc.shape[0]} trees x "
              f"{all_pc.shape[1]} points x {all_pc.shape[2]} channels")
        print(f"  test signature: {test_sig}")

        def _zero_factory():
            return lambda wl: (lambda pc: zero_fill(pc, wl))

        def _mean_factory():
            tm = compute_train_per_channel_mean(train_ds)
            return lambda wl: (lambda pc: mean_fill(pc, wl, tm))

        def _linear_factory():
            lin = LinearImputer(train_ds)
            return lambda wl: (lambda pc: lin.fill(pc, wl))

        def _knn_factory():
            path = knn_cache_path()
            # Always pass the train set: the cache validator needs it to
            # check the fingerprint, and the imputer needs it to rebuild when
            # the cache is stale or unreadable. Passing None when a cache file
            # existed made validation unreachable (it can only "trust" an
            # unverifiable cache) -- the exact hole being guarded against.
            knn = KNNImputer(train_ds, k=KNN_K,
                             cache_path=path)
            return lambda wl: (lambda pc: knn.fill(pc, wl))

        def _missforest_factory():
            path = missforest_cache_path()
            # Always pass the train set: the cache validator needs it to
            # check the fingerprint, and the imputer needs it to rebuild when
            # the cache is stale or unreadable. Passing None when a cache file
            # existed made validation unreachable (it can only "trust" an
            # unverifiable cache) -- the exact hole being guarded against.
            mf = MissForestImputer(train_ds,
                                   cache_path=path)
            return lambda wl: (lambda pc: mf.fill(pc, wl))

        plan = [("zero", _zero_factory)]
        if not args.no_mean:   plan.append(("mean",   _mean_factory))
        if not args.no_linear: plan.append(("linear", _linear_factory))
        if not args.no_knn:    plan.append(("knn",    _knn_factory))
        if not args.no_missforest:
            plan.append(("missforest", _missforest_factory))

        # Fingerprint the same training data used by the deferred factories,
        # including on prediction-cache hits. Load it only once per run.
        train_ds = build_dataset("train") if len(plan) > 1 else None
        train_fp = _train_fingerprint(train_ds)
        for name, factory in plan:
            imp, tim = load_or_build_imputed_test(
                name, factory, all_pc, args.batch_size, device, test_sig,
                train_fp=train_fp if name != "zero" else None,
            )
            baseline_imputed[name] = imp
            baseline_timings[name] = tim

    if args.build_caches_only:
        print("\n[--build-caches-only] cache build complete; exiting.")
        return

    seed_results: list[dict] = []
    for s in seeds:
        print(f"\n----- seed {s} -----")
        if not args.skip_train:
            train_one_seed(s, args, backbone, classifier_dir, device, out_dir)
        result = eval_one_seed(s, args, backbone, classifier_dir, device,
                                out_dir,
                                all_pc=all_pc, all_y=all_y,
                                baseline_imputed=baseline_imputed,
                                baseline_timings=baseline_timings)
        seed_results.append(result)
        print_seed_table(result, backbone, s)

    # ---- Aggregate ----
    agg = aggregate(seed_results)
    n_seeds = len(seed_results)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "backbone":   backbone,
            "n_seeds":    n_seeds,
            "hyperparameters": {
                "lambda_recon": LAMBDA_RECON,
                "lambda_cls":   LAMBDA_CLS,
                "z_dim":        0,
                "hidden":       HIDDEN,
                "l_cls_teacher": "paired_classifier_single",
            },
            "aggregated": agg,
        }, f, indent=2, default=str)
    write_csv(seed_results, out_dir / "summary.csv")
    write_latex(agg, n_seeds, backbone, out_dir / "comparison.tex")

    cm_data = aggregate_confusion_matrices(seed_results)
    cm_data["backbone"] = backbone
    cm_data["n_seeds"]  = len(seed_results)
    with open(out_dir / "confusion_matrices.json", "w") as f:
        json.dump(cm_data, f, indent=2)

    print_summary(agg, n_seeds, backbone)

    if not args.no_plots:
        print("\n--- Convergence plots ---")
        make_convergence_plots(seeds, out_dir, backbone)

    print(f"\n  summary.json              : {out_dir / 'summary.json'}")
    print(f"  summary.csv               : {out_dir / 'summary.csv'}")
    print(f"  comparison.tex            : {out_dir / 'comparison.tex'}")
    print(f"  confusion_matrices.json   : {out_dir / 'confusion_matrices.json'}")
    print("  Compile with: pdflatex -interaction=nonstopmode "
          f"-output-directory={out_dir} {out_dir / 'comparison.tex'}")


if __name__ == "__main__":
    main()
