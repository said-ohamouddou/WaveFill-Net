#!/usr/bin/env python3
"""
Component ablation for the FastKAN WaveFill-Net (Experiment 2).

Mirrors ablation_ptv2.py's component study, but the imputer's nn.Linear layers
are replaced 1:1 by FastKANLayer (radial-basis KAN). Four variants, one design choice toggled at
a time against the full method:

    * WaveFill-Net KAN (full)     the adopted recipe (lambda_recon=10, lambda_cls=0.5)
    * w/o L_cls (recon only)      drop the task-coupled CE loss   (lambda_cls=0)
    * w/o L_recon (cls only)      drop the reconstruction loss    (lambda_recon=0)
    * w/o wavelength embedding    the imputer no longer knows which wl is missing

Cross-classifier protocol
-------------------------
Each variant is trained ONCE, with the PointTransformerV2 classifier as the
frozen teacher supervising L_cls. The resulting imputer is then scored against
EVERY backbone's classifier (seed-matched: imputer seed s vs classifier seed s),
so the table separates two things:

    * teacher column   -- in-distribution; the classifier that trained it
    * held-out columns -- classifiers the imputer never saw

This is what breaks the circularity of the paired protocol: a variant that
merely learns to please its own teacher scores well in the teacher column and
collapses on the held-out ones. Evaluators without a trained classifier on
disk are skipped with a warning.

Everything else matches main.py: per-point imputation of the 4 missing
channels, best checkpoint chosen on validation OA averaged over the 3
wavelengths, then the held-out test set scored per wavelength.

Defaults: 5 seeds x 100 epochs -- same budget as the main experiment.

Usage
-----
    python ablation_components_kan.py                        # 5 seeds, 100 epochs
    python ablation_components_kan.py --seeds 0 --epochs 25   # quick pass
    python ablation_components_kan.py --teacher DGCNN --evaluators DGCNN,PointNet
    python ablation_components_kan.py --force-train
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def _require_cuda_extensions() -> None:
    """Fail early and legibly if the wrong conda env is active.

    The CUDA extensions (pointops, pointnet2_ops) live only in `torch311`.
    Run from `base` and the backbone factory raises a confusing
    "Backbone 'X' is not available" deep inside model construction, long
    after data loading has burned time. This turns that into one clear line.
    """
    import importlib
    missing = [m for m in ("pointops", "pointnet2_ops")
               if not importlib.util.find_spec(m)]
    if missing:
        import sys as _s
        raise SystemExit(
            f"\nERROR: missing CUDA extension(s): {', '.join(missing)}\n"
            f"  running: python {_s.version.split()[0]} at {_s.executable}\n"
            "  These live in the 'torch311' env. Activate it first:\n"
            "      conda activate torch311\n"
            "  (or launch via `bash run_main_experiment.sh`, which does it "
            "for you).\n")


_require_cuda_extensions()

import main as M  # noqa: E402
from torch_fast_kan import FastKANLayer  # noqa: E402


# ---------------------------------------------------------------------------
# Variant config
# ---------------------------------------------------------------------------

@dataclass
class Cfg:
    name:          str
    lambda_recon:  float = M.LAMBDA_RECON     # 10.0
    lambda_cls:    float = M.LAMBDA_CLS       # 0.5
    hidden:        int   = M.FASTKAN_HIDDEN   # 84, parameter-matched to the MLP
    emb_dim:       int   = M.EMB_DIM          # 16
    use_embedding: bool  = True
    num_grids:     int   = M.FASTKAN_GRIDS    # 8  (FastKAN RBF grid points)
    lr:            float = 2e-4

    def slug(self) -> str:
        import re
        return re.sub(r"[^0-9a-zA-Z]+", "_", self.name).strip("_").lower()


VARIANTS = [
    Cfg("WaveFill-Net FastKAN (full)"),
    Cfg("w/o L_cls (recon only)",   lambda_cls=0.0),
    Cfg("w/o L_recon (cls only)",   lambda_recon=0.0),
    Cfg("w/o wavelength embedding", use_embedding=False),
]


# ---------------------------------------------------------------------------
# Ablatable ReLU-KAN imputer
# ---------------------------------------------------------------------------

class AblKANImputer(nn.Module):
    """Per-point FastKAN imputer with the wavelength-embedding switch exposed.

    Mirrors main.WaveFillFastKAN so the ablation's "full" row is directly
    comparable to the main experiment's FastKAN row. FastKANLayer consumes a
    2-D (batch, features) tensor, so (B, N, F) is flattened to B*N on the way
    in and restored on the way out. No GELU between layers: a KAN layer is
    already nonlinear (a radial-basis expansion plus a SiLU base branch).
    """

    def __init__(self, cfg: Cfg, num_wavelengths: int = M.NUM_WAVELENGTHS):
        super().__init__()
        self.use_embedding = cfg.use_embedding
        emb = cfg.emb_dim if cfg.use_embedding else 0
        if cfg.use_embedding:
            self.wl_embed = nn.Embedding(num_wavelengths, cfg.emb_dim)
        in_dim = 3 + 8 + emb
        h = cfg.hidden
        mk = lambda i, o: FastKANLayer(i, o,  # noqa: E731
                                       num_grids=cfg.num_grids)
        self.layers = nn.ModuleList([mk(in_dim, h), mk(h, h),
                                     mk(h, h), mk(h, 4)])

    def forward(self, xyz, visible, wl_id):
        B, N, _ = xyz.shape
        parts = [xyz, visible]
        if self.use_embedding:
            parts.append(self.wl_embed(wl_id).unsqueeze(1).expand(-1, N, -1))
        h = torch.cat(parts, dim=-1).reshape(B * N, -1)
        for layer in self.layers:
            h = layer(h)
        return torch.sigmoid(h.reshape(B, N, 4))


# ---------------------------------------------------------------------------
# Train one (cfg, seed) -- mirrors main.train_one_seed
# ---------------------------------------------------------------------------

def train_seed(cfg: Cfg, seed: int, classifier, device, seed_dir: Path,
               epochs: int, batch_size: int, force: bool) -> Path:
    seed_dir.mkdir(parents=True, exist_ok=True)
    ckpt = seed_dir / "best.pt"
    if ckpt.exists() and not force:
        prev = torch.load(ckpt, map_location="cpu", weights_only=False)
        planned = prev.get("epochs_planned")
        done = prev.get("completed")
        if planned is None or done is None:
            raise SystemExit(
                f"\n{ckpt} predates the completion marker, so its epoch budget "
                "is unknown and reusing it could mix budgets in one table.\n"
                "  Re-run with --force-train, or delete that seed directory.\n")
        if planned != epochs or not done:
            raise SystemExit(
                f"\n{ckpt} was trained for {prev.get('epoch', -1) + 1}/{planned} "
                f"epochs (completed={done}), but this run asks for {epochs}.\n"
                "  Mixing budgets in one table is not comparable. Re-run with "
                "--force-train, or delete that seed directory.\n")
        print(f"    [skip train] {cfg.name} seed={seed} -> {ckpt.name} "
              f"({planned} ep, complete)")
        return ckpt

    torch.manual_seed(seed)
    np.random.seed(seed)

    tr = DataLoader(M.build_dataset("train"), batch_size=batch_size,
                    shuffle=True, num_workers=4, pin_memory=True,
                    drop_last=True, collate_fn=M.collate)
    va = DataLoader(M.build_dataset("val"), batch_size=batch_size,
                    shuffle=False, num_workers=2, pin_memory=True,
                    collate_fn=M.collate)

    model = AblKANImputer(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"    [train] {cfg.name}  seed={seed}  params={n_par:,}  "
          f"(l_rec={cfg.lambda_recon} l_cls={cfg.lambda_cls} "
          f"emb={cfg.use_embedding} num_grids={cfg.num_grids})", flush=True)

    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = M.cosine_warmup(opt, epochs, warmup=min(10, max(1, epochs // 5)))

    best_oa = -1.0
    history: list[dict] = []
    for ep in range(epochs):
        model.train()
        rec_t, cls_t, nb = 0.0, 0.0, 0
        for pc, y in tr:
            pc = pc.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B = pc.size(0)
            wl = int(torch.randint(0, M.NUM_WAVELENGTHS, (1,)).item())
            wl_t = torch.full((B,), wl, dtype=torch.long, device=device)
            xyz, vis, real, ret = M.split_pc(pc, wl)

            pred = model(xyz, vis, wl_t)
            L = torch.zeros((), device=device)
            L_rec = F.l1_loss(pred, real)
            if cfg.lambda_recon > 0:
                L = L + cfg.lambda_recon * L_rec
            L_cls = torch.zeros((), device=device)
            if cfg.lambda_cls > 0:
                logits = classifier(M.assemble_pc16(xyz, vis, pred, ret, wl))
                L_cls = F.cross_entropy(logits, y,
                                        label_smoothing=M.LABEL_SMOOTHING)
                L = L + cfg.lambda_cls * L_cls

            opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            rec_t += float(L_rec)
            cls_t += float(L_cls)
            nb += 1

        # validation: paired-classifier OA averaged over the 3 wavelengths
        model.eval()
        correct = {wl: 0 for wl in range(M.NUM_WAVELENGTHS)}
        rmse = {wl: [] for wl in range(M.NUM_WAVELENGTHS)}
        n_val = 0
        with torch.no_grad():
            for pc, y in va:
                pc = pc.to(device, non_blocking=True)
                y_dev = y.to(device, non_blocking=True)
                n_val += int(y.size(0))
                for wl in range(M.NUM_WAVELENGTHS):
                    B = pc.size(0)
                    wl_t = torch.full((B,), wl, dtype=torch.long, device=device)
                    xyz, vis, real, ret = M.split_pc(pc, wl)
                    pred = model(xyz, vis, wl_t)
                    rmse[wl].append(
                        (pred - real).pow(2).mean(dim=(1, 2)).sqrt().cpu().numpy())
                    logits = classifier(M.assemble_pc16(xyz, vis, pred, ret, wl))
                    correct[wl] += int((logits.argmax(-1) == y_dev).sum())
        oa_avg = statistics.mean(correct[wl] / max(1, n_val)
                                 for wl in range(M.NUM_WAVELENGTHS))
        rmse_avg = statistics.mean(float(np.concatenate(rmse[wl]).mean())
                                   for wl in range(M.NUM_WAVELENGTHS))
        sched.step()
        if oa_avg > best_oa:
            best_oa = oa_avg
            torch.save({"model_state": model.state_dict(), "epoch": ep,
                        "val_oa_avg": best_oa, "cfg": cfg.__dict__,
                        # Budget this run was launched with, and whether it
                        # reached the end. Without these a checkpoint stopped
                        # at epoch 12/100 is indistinguishable from a finished
                        # one, and the skip-train guard silently reuses it --
                        # which is how a table once mixed 10- and 50-epoch rows.
                        "epochs_planned": epochs,
                        "completed": False}, ckpt)
        history.append({"epoch": ep, "lr": opt.param_groups[0]["lr"],
                        "train_recon": rec_t / max(1, nb),
                        "train_cls": cls_t / max(1, nb),
                        "val_rmse_avg": rmse_avg, "val_oa_avg": oa_avg})
        print(f"      ep {ep+1:3d}/{epochs}  rec={rec_t/max(1,nb):.4f} "
              f"cls={cls_t/max(1,nb):.4f}  val_rmse={rmse_avg:.4f} "
              f"val_oa={oa_avg*100:5.2f}  (best {best_oa*100:5.2f})", flush=True)
    # Per-epoch history, so convergence curves can be re-plotted without
    # retraining (the checkpoint alone keeps only the best epoch).
    # The in-loop save cannot know whether training will finish, and its `ep`
    # is the BEST epoch, not the last one. Stamp completion here instead:
    # reaching this line means the epoch loop ran to the end.
    _done = torch.load(ckpt, map_location="cpu", weights_only=False)
    _done["completed"] = True
    torch.save(_done, ckpt)
    (seed_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    return ckpt


@torch.no_grad()
def eval_seed(cfg: Cfg, ckpt: Path, classifier, all_pc, all_y, device,
              batch_size: int) -> dict:
    """Per-wavelength test metrics + reconstruction RMSE."""
    model = AblKANImputer(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    model.eval()
    n = all_pc.size(0)
    out = {}
    for wl in range(M.NUM_WAVELENGTHS):
        preds, rmses = [], []
        for i in range(0, n, batch_size):
            chunk = all_pc[i:i + batch_size].to(device, non_blocking=True)
            B = chunk.size(0)
            wl_t = torch.full((B,), wl, dtype=torch.long, device=device)
            xyz, vis, real, ret = M.split_pc(chunk, wl)
            pred = model(xyz, vis, wl_t)
            rmses.append((pred - real).pow(2).mean(dim=(1, 2)).sqrt().cpu().numpy())
            logits = classifier(M.assemble_pc16(xyz, vis, pred, ret, wl))
            preds.append(logits.argmax(-1).cpu().numpy())
        m = M.report_metrics(all_y, np.concatenate(preds))
        m["rmse"] = float(np.concatenate(rmses).mean())
        out[wl] = m
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _sd(vals) -> float:
    """Sample std (n-1); 0.0 for a single seed."""
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def print_table(results: dict[str, dict], seeds: list[int], backbone: str,
                epochs: int):
    wls = list(range(M.NUM_WAVELENGTHS))
    print("\n" + "=" * 122)
    print(f"  COMPONENT ABLATION -- FastKAN WaveFill-Net   "
          f"(backbone {backbone}, {len(seeds)} seed(s), {epochs} epochs)")
    print("=" * 122)
    print(f"  {'variant':<30}" + "".join(f"{f'wl={w} ({M.WL_NM[w]})':>22}"
                                         for w in wls)
          + f"{'avg':>22}{'RMSE':>9}")
    print("  " + "-" * 122)
    for name, per_seed in results.items():
        cells = []
        for wl in wls:
            oas = [100 * s[wl]["overall_accuracy"] for s in per_seed]
            f1s = [100 * s[wl]["macro_f1"] for s in per_seed]
            cells.append(f"{statistics.mean(oas):6.2f}±{_sd(oas):4.2f}/"
                         f"{statistics.mean(f1s):5.2f}")
        # avg: mean over wavelengths PER SEED first, then across seeds, so the
        # std reflects seed variation rather than wavelength spread.
        avg_oa = [100 * statistics.mean(s[w]["overall_accuracy"] for w in wls)
                  for s in per_seed]
        avg_f1 = [100 * statistics.mean(s[w]["macro_f1"] for w in wls)
                  for s in per_seed]
        rms = statistics.mean(s[w]["rmse"] for s in per_seed for w in wls)
        print(f"  {name:<30}" + "".join(f"{c:>22}" for c in cells)
              + f"{statistics.mean(avg_oa):9.2f}±{_sd(avg_oa):4.2f}/"
                f"{statistics.mean(avg_f1):5.2f}"
              + f"{rms:9.4f}")
    print("=" * 122)

    # deltas vs the full method
    full_name = VARIANTS[0].name
    if full_name in results:
        def avg_oa(name):
            return 100 * statistics.mean(
                s[wl]["overall_accuracy"]
                for s in results[name] for wl in wls)
        base = avg_oa(full_name)
        print(f"  OA delta vs '{full_name}' ({base:.2f} avg):")
        for name in results:
            if name == full_name:
                continue
            print(f"    {name:<30} {avg_oa(name) - base:+6.2f} pp OA")
    print(f"\n  OA / macro-F1 in %, RMSE on the 4 imputed channels."
          f"  {len(seeds)} seed(s) -- indicative only.\n")


# ---------------------------------------------------------------------------

SHORT = {"PointTransformerV2": "PTv2", "DGCNN": "DGCNN", "DeepGCN": "DeepGCN",
         "PCT": "PCT", "PointNet": "PointNet", "PointNet2_MSG": "PointNet++"}


def print_crossclf_table(results: dict, evaluators: list[str], teacher: str,
                         seeds: list[int]):
    """variants x evaluator-backbones: does the gain survive a classifier the
    imputer was never trained against?"""
    wls = list(range(M.NUM_WAVELENGTHS))
    LBLW = 32
    width = LBLW + 14 * len(evaluators) + 16
    print("\n" + "=" * width)
    print(f"  CROSS-CLASSIFIER  --  imputer trained ONCE with the {teacher} "
          f"teacher, scored by every backbone")
    print("=" * width)
    hdr = f"  {'variant':<{LBLW-2}}"
    for bb in evaluators:
        mark = "*" if bb == teacher else ""
        hdr += f"{SHORT.get(bb, bb) + mark:>14}"
    print(hdr + f"{'mean held-out':>16}")
    print("  " + "-" * (width - 4))

    for cfg in VARIANTS:
        cells, held = [], []
        for bb in evaluators:
            runs = results[bb][cfg.name]
            if not runs:
                cells.append(f"{'--':>14}")
                continue
            per_seed = [100 * statistics.mean(s[w]["overall_accuracy"]
                                              for w in wls) for s in runs]
            oa = statistics.mean(per_seed)
            cells.append(f"{oa:9.2f}±{_sd(per_seed):<4.2f}")
            if bb != teacher:
                held.append(oa)
        mh = f"{statistics.mean(held):16.2f}" if held else f"{'--':>16}"
        print(f"  {cfg.name:<{LBLW-2}}" + "".join(cells) + mh)

    # delta vs the full method, on held-out classifiers only
    full = VARIANTS[0].name
    held_bb = [b for b in evaluators if b != teacher]
    if held_bb:
        def held_oa(name):
            return statistics.mean(
                100 * statistics.mean(s[w]["overall_accuracy"]
                                      for s in results[b][name] for w in wls)
                for b in held_bb if results[b][name])
        base = held_oa(full)
        print("  " + "-" * (width - 4))
        print(f"  held-out OA delta vs '{full}' ({base:.2f}):")
        for cfg in VARIANTS[1:]:
            print(f"    {cfg.name:<32} {held_oa(cfg.name) - base:+6.2f} pp OA")
    print("=" * width)
    print(f"  * = teacher (in-distribution).  OA % averaged over "
          f"{M.NUM_WAVELENGTHS} wavelengths, {len(seeds)} seed(s).\n", flush=True)


def write_tidy_csv(path, rows: list[dict]) -> None:
    """Long-format CSV: one row per (arm/variant, seed, wavelength, metric set).

    JSON keeps the full nested record; this is the flat table to re-plot from
    without parsing it.
    """
    import csv as _csv
    if not rows:
        return
    cols = list(rows[0])
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def aggregated_rows(rows: list[dict], group_keys: list[str],
                    metrics=("oa", "macro_f1", "rmse", "fill_ms_per_tree")
                    ) -> list[dict]:
    """Collapse per-seed rows into mean/std per group, for direct re-plotting.

    Two aggregations are emitted per group:
      * one row per wavelength  (wavelength = "0"/"1"/"2")
      * one row with wavelength = "avg", where each seed is first averaged
        over wavelengths and the mean/std are taken over those per-seed
        values -- the same order main.aggregate() uses, so the std reflects
        seed variation rather than wavelength spread.
    Std is the SAMPLE std (n-1), matching "mean +- std over seeds".
    """
    import statistics as _st
    from collections import defaultdict

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # group -> wavelength -> seed -> {metric: value}
    tree: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        tree[key][str(r["wavelength"])][r["seed"]] = r

    out: list[dict] = []
    for key, per_wl in tree.items():
        base = dict(zip(group_keys, key))
        wls = sorted(w for w in per_wl if w != "avg")

        for wl in wls + ["avg"]:
            row = dict(base)
            row["wavelength"] = wl
            n = 0
            for met in metrics:
                if wl == "avg":
                    # per seed: mean over wavelengths, then over seeds
                    seeds = set().union(*(set(per_wl[w]) for w in wls)) if wls else set()
                    vals = []
                    for s in sorted(seeds):
                        per_seed = [_num(per_wl[w][s].get(met))
                                    for w in wls if s in per_wl[w]]
                        per_seed = [v for v in per_seed if v is not None]
                        if per_seed:
                            vals.append(_st.mean(per_seed))
                else:
                    vals = [v for v in (_num(per_wl[wl][s].get(met))
                                        for s in sorted(per_wl[wl]))
                            if v is not None]
                if not vals:
                    row[f"{met}_mean"] = ""
                    row[f"{met}_std"] = ""
                    continue
                row[f"{met}_mean"] = _st.mean(vals)
                row[f"{met}_std"] = _st.stdev(vals) if len(vals) > 1 else 0.0
                n = max(n, len(vals))
            row["n_seeds"] = n
            out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="PointTransformerV2",
                    help="backbone whose frozen classifier supervises L_cls "
                         "during training (the imputer is trained ONCE, here)")
    ap.add_argument("--evaluators",
                    default="PointTransformerV2,DGCNN,DeepGCN,PCT,PointNet",
                    help="backbones whose classifiers score the trained "
                         "imputer (seed-matched); missing ones are skipped")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--classifier-root",
                    default="runs/all_16d_backbones_1024pts_5seed")
    ap.add_argument("--out-dir", default="runs/ablation_components_kan")
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    out_root = ROOT / args.out_dir
    cls_root = ROOT / args.classifier_root

    teacher_dir = cls_root / args.teacher / "classifier"
    if not (teacher_dir / f"model_seed{seeds[0]}.pt").exists():
        raise SystemExit(f"no teacher classifier at {teacher_dir} "
                         "-- run train_classifiers.sh")

    # Evaluator backbones: keep only those with a seed-matched checkpoint.
    evaluators, skipped = [], []
    for bb in args.evaluators.replace(",", " ").split():
        if all((cls_root / bb / "classifier" / f"model_seed{s}.pt").exists()
               for s in seeds):
            evaluators.append(bb)
        else:
            skipped.append(bb)

    print("=" * 96)
    print("  ReLU-KAN component ablation  --  cross-classifier")
    print("=" * 96)
    print(f"  Teacher (trains L_cls) : {args.teacher}")
    print(f"  Evaluators             : {', '.join(evaluators)}")
    if skipped:
        print(f"  Skipped (no classifier): {', '.join(skipped)}")
    print(f"  Seeds    : {seeds}   Epochs: {args.epochs}   Device: {device}")
    print(f"  Variants : {len(VARIANTS)}")
    for c in VARIANTS:
        print(f"    - {c.name}")
    print("=" * 96, flush=True)

    print("\n--- Loading test set ---", flush=True)
    loader = DataLoader(M.build_dataset("test"), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, collate_fn=M.collate)
    pcs, ys = [], []
    for pc, y in loader:
        pcs.append(pc)          # keep on CPU; batches move to GPU in eval_seed
        ys.append(y)
    all_pc = torch.cat(pcs, 0)
    all_y = torch.cat(ys, 0).numpy()
    print(f"  test: {tuple(all_pc.shape)} (held in RAM)", flush=True)

    # results[evaluator][variant] -> list over seeds of {wl: metrics}
    results: dict[str, dict[str, list]] = {
        bb: {c.name: [] for c in VARIANTS} for bb in evaluators}
    t0 = time.time()
    for seed in seeds:
        print(f"\n{'#'*96}\n#  seed {seed}\n{'#'*96}", flush=True)
        teacher = M.load_paired_classifier(teacher_dir, args.teacher,
                                           seed, device)
        # 1) train each variant ONCE, supervised by the teacher
        ckpts = {}
        for cfg in VARIANTS:
            seed_dir = out_root / cfg.slug() / f"seed{seed}"
            ckpts[cfg.name] = train_seed(cfg, seed, teacher, device, seed_dir,
                                         args.epochs, args.batch_size,
                                         args.force_train)
        del teacher
        torch.cuda.empty_cache()

        # 2) score every variant against EVERY backbone's classifier
        for bb in evaluators:
            clf = M.load_paired_classifier(cls_root / bb / "classifier",
                                           bb, seed, device)
            tag = "teacher" if bb == args.teacher else "held-out"
            print(f"\n  --- evaluator: {bb} ({tag}), seed {seed} ---",
                  flush=True)
            for cfg in VARIANTS:
                per_wl = eval_seed(cfg, ckpts[cfg.name], clf, all_pc, all_y,
                                   device, args.batch_size)
                results[bb][cfg.name].append(per_wl)
                oa = 100 * statistics.mean(per_wl[w]["overall_accuracy"]
                                           for w in range(M.NUM_WAVELENGTHS))
                print(f"    {cfg.name:<30} OA avg = {oa:5.2f}", flush=True)
            del clf
            torch.cuda.empty_cache()

    for bb in evaluators:
        print_table(results[bb], seeds,
                    f"{bb} ({'teacher' if bb == args.teacher else 'held-out'})",
                    args.epochs)
    print_crossclf_table(results, evaluators, args.teacher, seeds)

    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for bb, per_bb in results.items():
        for vname, runs in per_bb.items():
            for si, per_wl in zip(seeds, runs):
                for wl, m in per_wl.items():
                    rows.append({
                        "evaluator": bb, "is_teacher": bb == args.teacher,
                        "variant": vname, "seed": si, "wavelength": wl,
                        "oa": m["overall_accuracy"] * 100,
                        "macro_f1": m["macro_f1"] * 100,
                        "rmse": m.get("rmse", ""),
                    })
    write_tidy_csv(out_root / "results.csv", rows)          # per seed
    write_tidy_csv(out_root / "results_aggregated.csv",
                   aggregated_rows(rows, ["evaluator", "is_teacher", "variant"]))   # mean +- std
    (out_root / "results.json").write_text(json.dumps(
        {"teacher": args.teacher, "evaluators": evaluators,
         "seeds": seeds, "epochs": args.epochs,
         "results": {bb: {k: [{str(w): v for w, v in s.items()} for s in vs]
                          for k, vs in per_bb.items()}
                     for bb, per_bb in results.items()}},
        indent=2, default=str))
    print(f"  wrote {out_root/'results.json'}   "
          f"(total {(time.time()-t0)/60:.1f} min)\n")


if __name__ == "__main__":
    main()
