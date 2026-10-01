#!/usr/bin/env python3
"""
One-shot imputation experiment: every baseline + BOTH WaveFill-Net variants,
for every backbone, with a comparison table printed after EACH seed and one
consolidated table at the very end.

Why this exists
---------------
main.py trains one imputer variant per invocation, so running the MLP and the
ReLU-KAN means two separate passes -- and the per-seed tables of the first pass
cannot show the second variant. This driver interleaves them instead:

    for each backbone:
        for each seed:
            train MLP        (if needed)
            train ReLU-KAN   (if needed)
            evaluate baselines + both variants
            >>> print the table for this seed <<<
        >>> print the aggregate table for this backbone <<<
    >>> print one consolidated table over all backbones <<<

Everything is reused from main.py -- same data, same losses, same paired-eval
protocol, same metrics and aggregation. Nothing is re-implemented here.

Prerequisite: the frozen classifier ensembles must already exist under
--classifier-root (train them with train_classifiers.sh).

Resumable: existing imputer checkpoints are reused unless --force-train is
given, and per-seed evaluations are recomputed every run (they are cheap
relative to training) so the tables always reflect every variant on disk.

Usage
-----
    python run_imputation_experiments.py
    python run_imputation_experiments.py --epochs 100 --seeds 0,1,2,3,4
    python run_imputation_experiments.py --backbones PointNet --seeds 0
    python run_imputation_experiments.py --imputers mlp        # one variant
    python run_imputation_experiments.py --force-train         # retrain all
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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

DEFAULT_BACKBONES = ["PointTransformerV2", "DGCNN", "DeepGCN",
                     "PCT", "PointNet"]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES),
                   help="comma- or space-separated backbone list")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--imputers", default="mlp,relukan,fastkan",
                   help="which WaveFill-Net variants to train/evaluate")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden", type=int, default=None,
                   help="imputer hidden width; default is per-variant "
                        "(MLP 256, ReLU-KAN 88) so they stay param-matched")
    p.add_argument("--kan-grid", type=int, default=M.KAN_GRID)
    p.add_argument("--kan-k", type=int, default=M.KAN_K)
    p.add_argument("--classifier-root",
                   default="runs/all_16d_backbones_1024pts_5seed")
    p.add_argument("--output-root", default="runs/method")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--no-knn", action="store_true")
    p.add_argument("--no-missforest", action="store_true")
    p.add_argument("--no-mean", action="store_true")
    p.add_argument("--no-linear", action="store_true")
    p.add_argument("--skip-train", action="store_true",
                   help="evaluate existing checkpoints only")
    return p.parse_args()


def _split(s: str) -> list[str]:
    return [x for x in s.replace(",", " ").split() if x]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

WLN = M.WL_NM


def _fmt_pair(oa, f1):
    return f"{oa:6.2f}/{f1:5.2f}"


def print_seed_table(by_method: dict, backbone: str, seed: int):
    """Comparison table for one (backbone, seed). No std: single seed."""
    methods = [m for m in M.METHOD_ORDER if m in by_method]
    wls = list(range(M.NUM_WAVELENGTHS))

    def cell(m, wl):
        c = by_method[m]["per_wl"][wl]
        return c["overall_accuracy"] * 100, c["macro_f1"] * 100

    print("\n" + "-" * 88)
    print(f"  {backbone}  --  seed {seed}          [OA / macro-F1 %]")
    print("-" * 88)
    print(f"  {'method':<26}" + "".join(f"{f'wl={w} ({WLN[w]})':>16}"
                                        for w in wls) + f"{'avg':>14}")
    print("  " + "-" * 84)
    for m in methods:
        oas, f1s, cells = [], [], []
        for wl in wls:
            oa, f1 = cell(m, wl)
            oas.append(oa); f1s.append(f1)
            cells.append(_fmt_pair(oa, f1))
        lbl = M.METHOD_LABEL_PRETTY.get(m, m) + (" *" if m in M.OURS_METHODS else "")
        print(f"  {lbl:<26}" + "".join(f"{c:>16}" for c in cells)
              + f"{statistics.mean(oas):7.2f}/{statistics.mean(f1s):5.2f}")

    base = [m for m in methods if m not in M.OURS_METHODS and m != "full"]
    ours = [m for m in methods if m in M.OURS_METHODS]
    if base and ours:
        def wl_avg(m):
            return statistics.mean(cell(m, wl)[0] for wl in wls)
        best = max(base, key=wl_avg)
        print("  " + "-" * 84)
        print(f"  strongest baseline: {M.METHOD_LABEL_PRETTY.get(best, best)} "
              f"({wl_avg(best):.2f} OA avg)")
        for m in ours:
            print(f"    {M.METHOD_LABEL_PRETTY.get(m, m):<26} "
                  f"{wl_avg(m) - wl_avg(best):+6.2f} pp OA")
        if len(ours) > 1:
            a, b = ours[0], ours[1]
            print(f"    {M.METHOD_LABEL_PRETTY.get(b, b)} vs "
                  f"{M.METHOD_LABEL_PRETTY.get(a, a)}: "
                  f"{wl_avg(b) - wl_avg(a):+6.2f} pp OA")
    print("-" * 88, flush=True)


def print_backbone_table(agg: dict, backbone: str, n_seeds: int):
    """Aggregate over seeds for one backbone."""
    methods = [m for m in M.METHOD_ORDER if m in agg]
    print("\n" + "=" * 128)
    print(f"  {backbone}  --  aggregate over {n_seeds} seed(s)"
          f"      [OA mean±std / macro-F1 mean±std, %]")
    print("=" * 128)
    print(f"  {'method':<26}" + "".join(f"{f'wl={w}':>25}"
                                        for w in range(M.NUM_WAVELENGTHS))
          + f"{'avg':>25}")
    print("  " + "-" * 124)
    for m in methods:
        cells = []
        for k in (0, 1, 2, "avg"):
            d = agg[m][k]
            # Both stds are printed: macro-F1 varies far more across seeds
            # than OA, so a single +- would misrepresent the uncertainty.
            cells.append(f"{d['oa_mean']:5.2f}±{d['oa_std']:4.2f}/"
                         f"{d['f1_mean']:5.2f}±{d['f1_std']:4.2f}")
        lbl = M.METHOD_LABEL_PRETTY.get(m, m) + (" *" if m in M.OURS_METHODS else "")
        print(f"  {lbl:<26}" + "".join(f"{c:>25}" for c in cells))
    print("=" * 128, flush=True)


SHORT = {"PointTransformerV2": "PTv2", "PointTransformer": "PT",
         "PointNet2_MSG": "PointNet++", "PointNet": "PointNet",
         "DGCNN": "DGCNN", "DeepGCN": "DeepGCN", "PCT": "PCT",
         "PointMLP": "PointMLP", "GDAN": "GDAN", "KANDGCNN": "KAN-DGCNN"}


def print_final_table(all_agg: dict[str, dict], n_seeds: int):
    """One consolidated table: methods x backbones, averaged over wavelengths."""
    backbones = list(all_agg)
    methods = [m for m in M.METHOD_ORDER
               if any(m in all_agg[b] for b in backbones)]
    if not backbones or not methods:
        return
    LBLW = 26
    width = LBLW + 17 * len(backbones) + 16

    print("\n" + "=" * width)
    print("  FINAL  --  OA mean±std (%), averaged over the 3 wavelengths")
    print("=" * width)
    print(f"  {'method':<{LBLW - 2}}"
          + "".join(f"{SHORT.get(b, b):>17}" for b in backbones)
          + f"{'mean over BB':>16}")
    print("  " + "-" * (width - 4))
    for m in methods:
        cells, means = [], []
        for b in backbones:
            c = all_agg[b].get(m)
            if c is None:
                cells.append(f"{'--':>17}")
            else:
                cells.append(f"{c['avg']['oa_mean']:10.2f}±{c['avg']['oa_std']:4.2f}")
                means.append(c["avg"]["oa_mean"])
        lbl = M.METHOD_LABEL_PRETTY.get(m, m) + (" *" if m in M.OURS_METHODS else "")
        avg = f"{statistics.mean(means):16.2f}" if means else f"{'--':>16}"
        print(f"  {lbl:<{LBLW - 2}}" + "".join(cells) + avg)

    base = [m for m in methods if m not in M.OURS_METHODS and m != "full"]
    ours = [m for m in methods if m in M.OURS_METHODS]
    if base and ours:
        print("  " + "-" * (width - 4))
        for b in backbones:
            avail = [m for m in base if m in all_agg[b]]
            if not avail:
                continue
            best = max(avail, key=lambda m: all_agg[b][m]["avg"]["oa_mean"])
            parts = [
                f"{M.METHOD_LABEL_PRETTY.get(m, m)} "
                f"{all_agg[b][m]['avg']['oa_mean'] - all_agg[b][best]['avg']['oa_mean']:+.2f}"
                for m in ours if m in all_agg[b]
            ]
            if parts:
                print(f"  {SHORT.get(b, b):<24} vs best baseline "
                      f"({M.METHOD_LABEL_PRETTY.get(best, best)}): "
                      + ",  ".join(parts) + " pp OA")
    print("=" * width)
    print(f"  * = our method.  {n_seeds} seed(s).  "
          f"'mean over BB' = unweighted mean across backbones.\n", flush=True)


def _write_rows(path, rows: list[dict]) -> None:
    """Write a list of flat dicts as CSV (no-op on empty input)."""
    import csv as _csv
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def agg_rows(agg: dict, backbone: str, n_seeds: int) -> list[dict]:
    """Flatten main.aggregate() output into long-format rows.

    One row per (backbone, method, wavelength), wavelength "avg" included --
    that row is the per-seed wavelength mean aggregated over seeds, so its
    std reflects seed variation, matching the printed tables and the LaTeX.
    """
    rows = []
    for m in M.METHOD_ORDER:
        if m not in agg:
            continue
        for wl in (0, 1, 2, "avg"):
            c = agg[m][wl]
            rows.append({
                "backbone": backbone,
                "method": m,
                "method_label": M.METHOD_LABEL_PRETTY.get(m, m),
                "is_ours": m in M.OURS_METHODS,
                "wavelength": wl,
                "oa_mean": c["oa_mean"], "oa_std": c["oa_std"],
                "macro_f1_mean": c["f1_mean"], "macro_f1_std": c["f1_std"],
                "fill_ms_per_tree": c.get("time_mean_ms_per_tree", ""),
                "n_seeds": c.get("n_seeds", n_seeds),
            })
    return rows


def write_agg_csv(agg: dict, backbone: str, n_seeds: int, path) -> None:
    _write_rows(path, agg_rows(agg, backbone, n_seeds))


# ---------------------------------------------------------------------------
# Per-seed evaluation (baselines + every trained variant)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_seed(backbone, seed, args, device, seed_dir, imputers,
              all_pc, all_y, baseline_imputed, baseline_timings,
              classifier_dir) -> dict:
    classifier = M.load_paired_classifier(classifier_dir, backbone, seed, device)
    n_test = all_pc.size(0)
    cuda = device.type == "cuda"

    def classify(pc16):
        out = []
        for i in range(0, n_test, args.batch_size):
            out.append(classifier(pc16[i:i + args.batch_size]).argmax(-1).cpu().numpy())
        return np.concatenate(out)

    by_method: dict = {}

    full = M.report_metrics(all_y, classify(all_pc))
    full["fill_time_ms_per_tree"] = 0.0
    by_method["full"] = {"per_wl": {wl: dict(full)
                                    for wl in range(M.NUM_WAVELENGTHS)}}

    for tag, per_wl_imp in baseline_imputed.items():
        by_method[tag] = {"per_wl": {}}
        for wl in range(M.NUM_WAVELENGTHS):
            pc16 = M.apply_cached_imputation(all_pc, per_wl_imp[wl], wl)
            m = M.report_metrics(all_y, classify(pc16))
            m["fill_time_ms_per_tree"] = baseline_timings[tag][wl]
            by_method[tag]["per_wl"][wl] = m

    for kind in imputers:
        ckpt = seed_dir / M.imputer_ckpt_name(kind, args.kan_grid, args.kan_k)
        if not ckpt.exists():
            continue
        state = torch.load(ckpt, map_location=device)
        imp = M.build_imputer(state.get("imputer", kind),
                              hidden=state.get("hidden"),
                              grid=state.get("kan_grid", args.kan_grid),
                              k=state.get("kan_k", args.kan_k)).to(device)
        imp.load_state_dict(state["model_state"])
        imp.eval()
        key = M.imputer_method_key(kind)
        by_method[key] = {"per_wl": {}}
        for wl in range(M.NUM_WAVELENGTHS):
            preds, t_fill = [], 0.0
            for i in range(0, n_test, args.batch_size):
                chunk = all_pc[i:i + args.batch_size]
                b = chunk.size(0)
                wl_t = torch.full((b,), wl, dtype=torch.long, device=device)
                xyz, vis, _, ret = M.split_pc(chunk, wl)
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred4 = imp(xyz, vis, wl_t)
                filled = M.assemble_pc16(xyz, vis, pred4, ret, wl)
                if cuda:
                    torch.cuda.synchronize()
                t_fill += time.perf_counter() - t0
                preds.append(classifier(filled).argmax(-1).cpu().numpy())
            m = M.report_metrics(all_y, np.concatenate(preds))
            m["fill_time_ms_per_tree"] = 1000.0 * t_fill / max(1, n_test)
            by_method[key]["per_wl"][wl] = m
    return by_method


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbones = _split(args.backbones)
    seeds = [int(s) for s in _split(args.seeds)]
    imputers = _split(args.imputers)
    for k in imputers:
        if k not in M.IMPUTERS:
            raise SystemExit(f"unknown imputer '{k}'; choose from {sorted(M.IMPUTERS)}")

    print("=" * 88)
    print("  Imputation experiments -- baselines + both WaveFill-Net variants")
    print("=" * 88)
    print(f"  Backbones : {', '.join(backbones)}")
    print(f"  Seeds     : {seeds}")
    print(f"  Imputers  : {', '.join(imputers)}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Device    : {device}")
    print("=" * 88, flush=True)

    # --- test set + shared baseline cache (built once, backbone-agnostic) ---
    print("\n--- Loading test set ---", flush=True)
    loader = DataLoader(M.build_dataset("test"), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True,
                        collate_fn=M.collate)
    pcs, ys = [], []
    for pc, y in loader:
        pcs.append(pc.to(device))
        ys.append(y)
    all_pc = torch.cat(pcs, 0)
    all_y = torch.cat(ys, 0).numpy()
    test_sig = M._test_set_signature(all_pc.detach().cpu().numpy())
    print(f"  test: {tuple(all_pc.shape)}", flush=True)

    print("\n--- Baseline imputations (cached, shared by all backbones) ---",
          flush=True)
    baseline_imputed: dict = {}
    baseline_timings: dict = {}

    def _zero():
        return lambda wl: (lambda pc: M.zero_fill(pc, wl))

    def _mean():
        tm = M.compute_train_per_channel_mean(train_ds)
        return lambda wl: (lambda pc: M.mean_fill(pc, wl, tm))

    def _linear():
        lin = M.LinearImputer(train_ds)
        return lambda wl: (lambda pc: lin.fill(pc, wl))

    def _knn():
        path = M.knn_cache_path()
        # See note in main.py: the validator needs the train set.
        knn = M.KNNImputer(train_ds, k=M.KNN_K,
                           cache_path=path)
        return lambda wl: (lambda pc: knn.fill(pc, wl))

    def _mf():
        path = M.missforest_cache_path()
        mf = M.MissForestImputer(train_ds, cache_path=path)
        return lambda wl: (lambda pc: mf.fill(pc, wl))

    plan = [("zero", _zero)]
    if not args.no_mean:
        plan.append(("mean", _mean))
    if not args.no_linear:
        plan.append(("linear", _linear))
    if not args.no_knn:
        plan.append(("knn", _knn))
    if not args.no_missforest:
        plan.append(("missforest", _mf))
    # Validate prediction caches against the same data used for fitting.
    train_ds = M.build_dataset("train") if len(plan) > 1 else None
    train_fp = M._train_fingerprint(train_ds)
    for name, factory in plan:
        imp, tim = M.load_or_build_imputed_test(
            name, factory, all_pc, args.batch_size, device, test_sig,
            train_fp=train_fp if name != "zero" else None)
        baseline_imputed[name] = imp
        baseline_timings[name] = tim

    # --- per backbone -------------------------------------------------------
    all_agg: dict[str, dict] = {}
    # CLAIM 5a: collect per-backbone failures so the exit code and the final
    # summary report them, instead of a skipped backbone vanishing silently.
    failures: list[tuple[str, str]] = []
    for bb in backbones:
        cls_dir = ROOT / args.classifier_root / bb / "classifier"
        if not (cls_dir / f"model_seed{seeds[0]}.pt").exists():
            print(f"\n!! skip {bb}: no classifier at {cls_dir}", flush=True)
            failures.append((bb, "no classifier"))
            continue
        out_dir = ROOT / args.output_root / bb
        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 88)
        print(f"#  Backbone: {bb}")
        print("#" * 88, flush=True)

        seed_results = []
        for s in seeds:
            seed_dir = out_dir / f"seed{s}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            # train each variant for this seed (interleaved)
            if not args.skip_train:
                for kind in imputers:
                    # Do NOT skip here: train_one_seed owns the skip logic and
                    # the epoch-budget guard. Short-circuiting on mere file
                    # existence bypassed that guard and could reuse a
                    # different-budget or unfinished checkpoint.
                    targs = SimpleNamespace(
                        imputer=kind, kan_grid=args.kan_grid, kan_k=args.kan_k,
                        hidden=args.hidden,
                        epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, force_train=args.force_train)
                    M.train_one_seed(s, targs, bb, cls_dir, device, out_dir)

            # CLAIM 3a: a retrained checkpoint makes any cached evaluation
            # stale, so --force-train drops the seed's eval_results.json
            # before re-evaluating rather than reporting old numbers.
            if args.force_train:
                stale = seed_dir / "eval_results.json"
                if stale.exists():
                    stale.unlink()
            by_method = eval_seed(bb, s, args, device, seed_dir, imputers,
                                  all_pc, all_y, baseline_imputed,
                                  baseline_timings, cls_dir)
            print_seed_table(by_method, bb, s)

            res = {"seed": s, "backbone": bb, "by_method": by_method}
            seed_results.append(res)
            (seed_dir / "eval_results.json").write_text(
                json.dumps(res, indent=2, default=str))

        if not seed_results:
            continue
        agg = M.aggregate(seed_results)
        all_agg[bb] = agg
        print_backbone_table(agg, bb, len(seed_results))
        (out_dir / "summary.json").write_text(json.dumps(
            {"backbone": bb, "n_seeds": len(seed_results),
             "aggregated": agg}, indent=2, default=str))
        M.write_csv(seed_results, out_dir / "summary.csv")      # per seed
        write_agg_csv(agg, bb, len(seed_results),
                      out_dir / "summary_aggregated.csv")       # mean +- std

    print_final_table(all_agg, len(seeds))

    # One consolidated aggregated CSV across every backbone, so the final
    # table can be re-plotted without walking the per-backbone tree.
    if all_agg:
        rows: list[dict] = []
        for bb, agg in all_agg.items():
            rows += agg_rows(agg, bb,
                             all_agg[bb][next(iter(agg))]["avg"].get("n_seeds",
                                                                    len(seeds)))
        root = ROOT / args.output_root
        root.mkdir(parents=True, exist_ok=True)
        _write_rows(root / "all_backbones_aggregated.csv", rows)
        print(f"  Consolidated CSV     : {root / 'all_backbones_aggregated.csv'}")

    if failures:
        print(f"\n  !! {len(failures)} backbone(s) did not produce results:")
        for bb, why in failures:
            print(f"       {bb:<22} {why}")
    print(f"  Per-backbone results : {args.output_root}/<BB>/summary.json")
    print(f"  Aggregated CSV       : {args.output_root}/<BB>/summary_aggregated.csv")
    print("  LaTeX table          : python build_master_table.py\n")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
