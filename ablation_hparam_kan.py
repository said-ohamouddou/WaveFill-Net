#!/usr/bin/env python3
"""
Hyperparameter ablation for the FastKAN WaveFill-Net (Experiment 3).

Backbone: PointTransformerV2 (teacher + paired evaluator, seed-matched).

PURPOSE: this sweep is DESCRIPTIVE, not a model-selection procedure. The
defaults in main.py were fixed a priori and are NOT the argmax of this table;
no hyperparameter is chosen from these numbers, and nothing here feeds back
into the main experiment. That is why the test split is used: the question
asked is "how does each hyperparameter affect GENERALISATION?", which is a
property of held-out data. Report it as a sensitivity analysis, and do not
describe any value below as a "winner".

Four hyperparameters are swept ONE AT A TIME around that a-priori default,
every other setting held fixed. Seven values each:

    lambda_recon   0, 1, 2, 5, 10*, 20, 50            reconstruction-loss weight
    lambda_cls     0, 0.1, 0.25, 0.5*, 1, 2, 5        task-coupled CE weight
    num_grids      2, 4, 6, 8*, 12, 16, 24            FastKAN RBF resolution
    hidden         32, 48, 64, 84*, 112, 144, 192     layer width

(* = default, trained once and shared by all four groups.)

Because num_grids and hidden both change the parameter count, the table
reports it per config -- a gain that tracks parameters is capacity, not basis
resolution.

Total: 4 groups x 7 values = 28 configs, minus 3 duplicate defaults = 25 runs
per seed.

Everything (model, training loop, evaluation) is imported from
ablation_components_kan.py, so the protocol is identical to the component ablation.

Usage
-----
    python ablation_hparam_kan.py                          # 1 seed, 100 epochs
    python ablation_hparam_kan.py --seeds 0,1,2            # multi-seed (75 runs)
    python ablation_hparam_kan.py --params num_grids,hidden  # basis/width only
    python ablation_hparam_kan.py --dry-run                # list configs, train nothing
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
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
from ablation_components_kan import (Cfg, AblKANImputer, train_seed,  # noqa: E402
                              eval_seed)

DEFAULT_BACKBONE = "PointTransformerV2"

# field -> (default value, 7 swept values)
SWEEPS: dict[str, tuple] = {
    "lambda_recon": (M.LAMBDA_RECON, [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]),
    "lambda_cls":   (M.LAMBDA_CLS,   [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]),
    # FastKAN's own basis hyperparameter: how many RBF grid points span
    # [grid_min, grid_max]. More grids = finer basis = more parameters, so the
    # table reports the count alongside (a gain that tracks parameters is
    # capacity, not resolution).
    "num_grids":    (M.FASTKAN_GRIDS, [2, 4, 6, 8, 12, 16, 24]),
    # Width, swept to separate "more basis" from "more capacity": the same
    # parameter budget can be spent either way.
    "hidden":       (M.FASTKAN_HIDDEN, [32, 48, 64, 84, 112, 144, 192]),
}

LABEL = {"lambda_recon": "lambda_recon", "lambda_cls": "lambda_cls",
         "num_grids": "RBF grid points", "hidden": "hidden width"}


DEFAULT_NAME = "default (a-priori config)"


def make_cfg(field: str, value) -> Cfg:
    """Default config with exactly one field overridden. The unmodified
    default gets a neutral name, since it is shared by all four groups."""
    is_default = value == SWEEPS[field][0]
    name = DEFAULT_NAME if is_default else f"{LABEL[field]}={value}"
    cfg = Cfg(name=name, **{field: value})
    # lambda_recon=0 AND lambda_cls=0 would leave no loss at all.
    if cfg.lambda_recon == 0 and cfg.lambda_cls == 0:
        raise ValueError("empty loss")
    return cfg


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
    ap.add_argument("--teacher", "--backbone", dest="backbone",
                    default=DEFAULT_BACKBONE,
                    help="backbone whose frozen classifier trains AND scores "
                         f"the imputer (default: {DEFAULT_BACKBONE})")
    ap.add_argument("--params", default=",".join(SWEEPS),
                    help=f"subset of {{{', '.join(SWEEPS)}}}")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--classifier-root",
                    default="runs/all_16d_backbones_1024pts_5seed")
    ap.add_argument("--out-dir", default="runs/ablation_hparam_kan")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fields = [f.strip() for f in args.params.replace(",", " ").split()]
    for f in fields:
        if f not in SWEEPS:
            raise SystemExit(f"unknown --params '{f}'; choose from {list(SWEEPS)}")

    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = ROOT / args.out_dir
    cls_dir = ROOT / args.classifier_root / args.backbone / "classifier"
    if not (cls_dir / f"model_seed{seeds[0]}.pt").exists():
        raise SystemExit(f"no classifier at {cls_dir} -- run train_classifiers.sh")

    # Build the config list; the shared default is trained only once.
    groups: dict[str, list[Cfg]] = {}
    seen: dict[str, Cfg] = {}
    for f in fields:
        default, values = SWEEPS[f]
        cfgs = []
        for v in values:
            try:
                c = make_cfg(f, v)
            except ValueError:
                print(f"  [skip] {LABEL[f]}={v}: would leave an empty loss")
                continue
            key = c.slug() if v != default else "__default__"
            if key in seen:
                cfgs.append(seen[key])          # reuse the shared default
            else:
                seen[key] = c
                cfgs.append(c)
            cfgs[-1] = seen[key]
        groups[f] = cfgs
    unique = {c.slug(): c for cs in groups.values() for c in cs}

    print("=" * 92)
    print(f"  FastKAN hyperparameter ablation -- backbone {args.backbone}")
    print("=" * 92)
    print(f"  Seeds    : {seeds}   Epochs: {args.epochs}   Device: {device}")
    print(f"  Swept    : {', '.join(LABEL[f] for f in fields)}")
    for f in fields:
        d, vals = SWEEPS[f]
        print(f"    {LABEL[f]:<18} {vals}   (default {d})")
    print(f"  Unique configs : {len(unique)}   "
          f"total runs: {len(unique) * len(seeds)}")
    print("=" * 92, flush=True)

    if args.dry_run:
        for f in fields:
            print(f"\n  [{LABEL[f]}]")
            for c in groups[f]:
                n = sum(p.numel() for p in AblKANImputer(c).parameters())
                mark = " (default)" if getattr(c, f) == SWEEPS[f][0] else ""
                print(f"    {c.name:<26} params={n:>9,}  -> {c.slug()}{mark}")
        return

    print("\n--- Loading test set ---", flush=True)
    loader = DataLoader(M.build_dataset("test"), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, collate_fn=M.collate)
    pcs, ys = [], []
    for pc, y in loader:
        pcs.append(pc)
        ys.append(y)
    all_pc = torch.cat(pcs, 0)
    all_y = torch.cat(ys, 0).numpy()
    print(f"  test: {tuple(all_pc.shape)} (held in RAM)", flush=True)

    # results[slug] -> list over seeds of {wl: metrics}
    results: dict[str, list] = {s: [] for s in unique}
    t0 = time.time()
    for seed in seeds:
        print(f"\n{'#'*92}\n#  seed {seed}\n{'#'*92}", flush=True)
        clf = M.load_paired_classifier(cls_dir, args.backbone, seed, device)
        for slug, cfg in unique.items():
            seed_dir = out_root / slug / f"seed{seed}"
            ckpt = train_seed(cfg, seed, clf, device, seed_dir,
                              args.epochs, args.batch_size, args.force_train)
            per_wl = eval_seed(cfg, ckpt, clf, all_pc, all_y,
                               device, args.batch_size)
            results[slug].append(per_wl)
            oa = 100 * statistics.mean(per_wl[w]["overall_accuracy"]
                                       for w in range(M.NUM_WAVELENGTHS))
            print(f"    [eval ] {cfg.name:<26} OA avg = {oa:5.2f}", flush=True)
        del clf
        torch.cuda.empty_cache()

    # ---- one table per swept hyperparameter -------------------------------
    wls = list(range(M.NUM_WAVELENGTHS))

    def _sd(vals):
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    def stats(slug):
        """(oa_mean, oa_std, f1_mean, rmse) -- std over seeds, per-seed
        wavelength means first so it reflects seed variation."""
        runs = results[slug]
        per_seed = [100 * statistics.mean(s[w]["overall_accuracy"] for w in wls)
                    for s in runs]
        f1 = 100 * statistics.mean(s[w]["macro_f1"] for s in runs for w in wls)
        rm = statistics.mean(s[w]["rmse"] for s in runs for w in wls)
        return statistics.mean(per_seed), _sd(per_seed), f1, rm

    for f in fields:
        default, _ = SWEEPS[f]
        print("\n" + "=" * 92)
        print(f"  SWEEP: {LABEL[f]}   ({args.backbone}, {len(seeds)} seed(s), "
              f"{args.epochs} epochs)")
        print("=" * 92)
        print(f"  {'value':<14}{'params':>12}{'OA (mean±std)':>18}"
              f"{'macro-F1':>11}{'RMSE':>10}{'dOA vs def':>13}")
        print("  " + "-" * 88)
        base_oa = stats(groups[f][SWEEPS[f][1].index(default)].slug())[0]
        for c, v in zip(groups[f], SWEEPS[f][1]):
            if c.slug() not in results or not results[c.slug()]:
                continue
            oa, sd, f1, rm = stats(c.slug())
            npar = sum(p.numel() for p in AblKANImputer(c).parameters())
            mark = " *" if v == default else ""
            print(f"  {str(v) + mark:<14}{npar:>12,}{oa:>11.2f}±{sd:<6.2f}"
                  f"{f1:>11.2f}{rm:>10.4f}{oa - base_oa:>+13.2f}")
        print("=" * 92)
        print(f"  * = default.  OA/F1 in %, averaged over {len(wls)} "
              f"wavelengths and {len(seeds)} seed(s).", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in fields:
        for cfg_, val in zip(groups[f], SWEEPS[f][1]):
            for si, per_wl in zip(seeds, results.get(cfg_.slug(), [])):
                npar = sum(p.numel() for p in AblKANImputer(cfg_).parameters())
                for wl, m in per_wl.items():
                    rows.append({
                        "param": f, "value": val, "is_default": val == SWEEPS[f][0],
                        "config": cfg_.slug(), "n_params": npar,
                        "seed": si, "wavelength": wl,
                        "oa": m["overall_accuracy"] * 100,
                        "macro_f1": m["macro_f1"] * 100,
                        "rmse": m.get("rmse", ""),
                    })
    write_tidy_csv(out_root / "results.csv", rows)          # per seed
    write_tidy_csv(out_root / "results_aggregated.csv",
                   aggregated_rows(rows, ["param", "value", "is_default", "config", "n_params"]))   # mean +- std
    (out_root / "results.json").write_text(json.dumps(
        {"backbone": args.backbone, "seeds": seeds, "epochs": args.epochs,
         "sweeps": {f: SWEEPS[f][1] for f in fields},
         "results": {s: [{str(w): v for w, v in run.items()} for run in runs]
                     for s, runs in results.items()}},
        indent=2, default=str))
    print(f"\n  wrote {out_root/'results.json'}   "
          f"(total {(time.time()-t0)/60:.1f} min)\n")


if __name__ == "__main__":
    main()
