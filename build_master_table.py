#!/usr/bin/env python3
"""
Consolidate every runs/method/<BACKBONE>/summary.json into ONE compilation-ready
LaTeX document, laid out like oc_table.tex:

    rows  = backbones (\\multirow groups), one sub-row per imputation method
    cols  = WL=0 (1550 nm) | WL=1 (905 nm) | WL=2 (532 nm) | Avg over lambda
    cell  = OA +- std / macro-F1 +- std   (%)

Highlighting (per column, decided on OA mean, the 'full' reference excluded):
    \\best{...}   bold   -> best method within a backbone
    \\topval{...} red    -> best value across ALL backbones

Usage
-----
    python build_master_table.py                      # -> master_comparison.tex (+ .pdf)
    python build_master_table.py --out tab.tex        # custom path
    python build_master_table.py --metric f1          # rank/highlight on macro-F1
    python build_master_table.py --no-compile         # skip pdflatex
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Overridable so a smoke run can point at a scratch tree instead of the real
# results (env var, or --method-root).
METHOD_ROOT = Path(os.environ.get("OUTPUT_ROOT", ROOT / "runs" / "method"))
if not METHOD_ROOT.is_absolute():
    METHOD_ROOT = ROOT / METHOD_ROOT

# Display order of backbones (only those with a summary.json are emitted).
BACKBONE_ORDER = [
    "PointTransformerV2",
    "DGCNN",
    "DeepGCN",
    "PCT",
    "PointNet",
]

BACKBONE_LABEL = {
    "PointTransformerV2": "Point Transformer V2",
    "DGCNN":              "DGCNN",
    "DeepGCN":            "DeepGCN",
    "PCT":                "PCT",
    "PointNet":           "PointNet",
}

# Row order + pretty labels. 'wavefill' is renamed here -- change in ONE place.
METHOD_ORDER = ["full", "zero", "mean", "linear", "knn", "missforest",
                "wavefill", "wavefill_relukan", "wavefill_fastkan"]
METHOD_LABEL = {
    "full":             "full",
    "zero":             "zero",
    "mean":             "mean",
    "linear":           "linear",
    "knn":              "kNN",
    "missforest":       "missForest",
    "wavefill":         "WaveFill-Net (ours)",
    "wavefill_relukan": "WaveFill-Net (ReLU-KAN)",
    "wavefill_fastkan": "WaveFill-Net (FastKAN)",
}

# Excluded from the bold/red competition (kept as a displayed reference row).
EXCLUDE_FROM_BEST = {"full"}

WL_KEYS = ["0", "1", "2", "avg"]


def load_summaries() -> dict[str, dict]:
    out = {}
    for bb in BACKBONE_ORDER:
        p = METHOD_ROOT / bb / "summary.json"
        if p.exists():
            out[bb] = json.loads(p.read_text())
    # Any backbone present on disk but not in BACKBONE_ORDER -> append.
    for p in sorted(METHOD_ROOT.glob("*/summary.json")):
        bb = p.parent.name
        if bb not in out:
            out[bb] = json.loads(p.read_text())
    return out


def fmt_cell(c: dict) -> str:
    """OA +- std / F1 +- std, all in %."""
    return (f"{c['oa_mean']:.2f}\\,$\\pm$\\,{c['oa_std']:.2f} / "
            f"{c['f1_mean']:.2f}\\,$\\pm$\\,{c['f1_std']:.2f}")


def metric_value(c: dict, metric: str) -> float:
    return c["oa_mean"] if metric == "oa" else c["f1_mean"]


def build_tex(summaries: dict[str, dict], metric: str) -> str:
    metric_name = "OA" if metric == "oa" else "macro-F1"
    # Seed count read from the data, not hardcoded: a backbone rerun with
    # fewer seeds must not be captioned as 5.
    # Seed count taken from the PER-METHOD counts, not the per-backbone one:
    # a method available on fewer seeds than the run's nominal count would
    # otherwise be captioned with that nominal count.
    per_method = set()
    for summ in summaries.values():
        for m, agg in summ.get("aggregated", {}).items():
            n = agg.get("avg", {}).get("n_seeds")
            if n:
                per_method.add(int(n))
    if not per_method:
        per_method = {int(s.get("n_seeds", 0)) for s in summaries.values()}
    seed_txt = (f"{min(per_method)} seeds" if len(per_method) == 1
                else f"{min(per_method)}--{max(per_method)} seeds "
                     "depending on backbone and method")

    # --- global best per column (across all backbones, non-full methods) ---
    global_best = {wl: float("-inf") for wl in WL_KEYS}
    for bb, summ in summaries.items():
        agg = summ["aggregated"]
        for m in METHOD_ORDER:
            if m in EXCLUDE_FROM_BEST or m not in agg:
                continue
            for wl in WL_KEYS:
                global_best[wl] = max(global_best[wl],
                                      metric_value(agg[m][wl], metric))

    lines: list[str] = []
    A = lines.append

    A(r"\documentclass[11pt]{article}")
    A(r"")
    A(r"% --- Packages ---")
    A(r"\usepackage[T1]{fontenc}")
    A(r"\usepackage[utf8]{inputenc}")
    A(r"\usepackage[a4paper,margin=2cm]{geometry}")
    A(r"\usepackage{booktabs}")
    A(r"\usepackage{multirow}")
    A(r"\usepackage{array}")
    A(r"\usepackage{graphicx}")
    A(r"\usepackage{caption}")
    A(r"\usepackage[table]{xcolor}")
    A(r"")
    A(r"% best method per backbone (bold), best overall value (red)")
    A(r"\newcommand{\best}[1]{\textbf{#1}}")
    A(r"\newcommand{\topval}[1]{\textcolor{red}{\textbf{#1}}}")
    A(r"")
    A(r"\begin{document}")
    A(r"")
    A(r"\begin{table}[htbp]")
    A(r"\centering")
    A(r"\caption{Missing-wavelength imputation performance across LiDAR "
      r"wavelengths and classifier backbones. "
      rf"Each cell reports OA / macro-F1 (\%) as mean $\pm$ std over {seed_txt}. "
      r"\textbf{Bold} marks the best method per backbone (the "
      rf"\textit{{full}} 16D reference excluded), decided on {metric_name}; "
      r"\textcolor{red}{\textbf{red}} marks the best value over all backbones.}")
    A(r"\label{tab:method_wavelength}")
    A(r"\renewcommand{\arraystretch}{1.2}")
    A(r"\setlength{\tabcolsep}{4pt}")
    A(r"\footnotesize")
    A(r"\resizebox{\textwidth}{!}{%")
    A(r"\begin{tabular}{l l c c c c}")
    A(r"\toprule")
    A(r"\textbf{Backbone} & \textbf{Method}")
    A(r"& \textbf{WL=0 (1550\,nm)} & \textbf{WL=1 (905\,nm)}")
    A(r"& \textbf{WL=2 (532\,nm)} & \textbf{Avg over $\lambda$} \\")
    A(r"\midrule")

    backbones = list(summaries.keys())
    for bi, bb in enumerate(backbones):
        agg = summaries[bb]["aggregated"]
        methods = [m for m in METHOD_ORDER if m in agg]

        # per-column best (this backbone, non-full) on the chosen metric
        local_best = {wl: float("-inf") for wl in WL_KEYS}
        for m in methods:
            if m in EXCLUDE_FROM_BEST:
                continue
            for wl in WL_KEYS:
                local_best[wl] = max(local_best[wl],
                                     metric_value(agg[m][wl], metric))

        label = BACKBONE_LABEL.get(bb, bb)
        A(rf"\multirow{{{len(methods)}}}{{*}}{{{label}}}")

        for m in methods:
            cells = []
            for wl in WL_KEYS:
                c = agg[m][wl]
                txt = fmt_cell(c)
                if m not in EXCLUDE_FROM_BEST:
                    v = metric_value(c, metric)
                    if abs(v - global_best[wl]) < 1e-9:
                        txt = rf"\topval{{{txt}}}"
                    elif abs(v - local_best[wl]) < 1e-9:
                        txt = rf"\best{{{txt}}}"
                cells.append(txt)
            A(rf" & {METHOD_LABEL.get(m, m):<20} & "
              + " & ".join(cells) + r" \\")

        A(r"\midrule" if bi < len(backbones) - 1 else r"\bottomrule")

    A(r"\end{tabular}%")
    A(r"}")
    A(r"\end{table}")
    A(r"")
    A(r"\end{document}")
    return "\n".join(lines) + "\n"


def print_console_summary(summaries: dict[str, dict]) -> None:
    """One consolidated console table: every method, every backbone, plus a
    final column averaging each method ACROSS backbones."""
    # Same backbone set as the LaTeX table: BACKBONE_ORDER first, then any
    # extra backbone found on disk (load_summaries appends those, and
    # build_tex keeps them, so dropping them here would make the console
    # table and the LaTeX table disagree).
    backbones = ([b for b in BACKBONE_ORDER if b in summaries]
                 + [b for b in summaries if b not in BACKBONE_ORDER])
    methods = [m for m in METHOD_ORDER
               if any(m in summaries[b]["aggregated"] for b in backbones)]
    if not backbones or not methods:
        return

    def cell(bb, m):
        agg = summaries[bb]["aggregated"]
        return agg[m]["avg"] if m in agg else None

    # Short display names so the header does not get truncated.
    SHORT = {"PointTransformerV2": "PTv2", "PointTransformer": "PT",
             "PointNet2_MSG": "PointNet++", "PointNet": "PointNet",
             "DGCNN": "DGCNN", "DeepGCN": "DeepGCN", "PCT": "PCT",
             "PointMLP": "PointMLP", "GDAN": "GDAN", "KANDGCNN": "KAN-DGCNN"}
    LBLW = 26
    width = LBLW + 17 * len(backbones) + 16
    print("\n" + "=" * width)
    print("  CONSOLIDATED  --  OA mean±std (%), averaged over the 3 wavelengths")
    print("=" * width)
    print(f"  {'method':<{LBLW - 2}}"
          + "".join(f"{SHORT.get(b, b):>17}" for b in backbones)
          + f"{'mean over BB':>16}")
    print("  " + "-" * (width - 4))

    for m in methods:
        cells, means = [], []
        for b in backbones:
            c = cell(b, m)
            if c is None:
                cells.append(f"{'--':>17}")
            else:
                cells.append(f"{c['oa_mean']:10.2f}±{c['oa_std']:4.2f}")
                means.append(c["oa_mean"])
        lbl = METHOD_LABEL.get(m, m)
        if m not in EXCLUDE_FROM_BEST and m.startswith("wavefill"):
            lbl += " *"
        avg = f"{statistics.mean(means):16.2f}" if means else f"{'--':>16}"
        print(f"  {lbl:<{LBLW - 2}}" + "".join(cells) + avg)

    # ours vs strongest baseline, per backbone
    base = [m for m in methods
            if m not in EXCLUDE_FROM_BEST and not m.startswith("wavefill")]
    ours = [m for m in methods if m.startswith("wavefill")]
    if base and ours:
        print("  " + "-" * (width - 4))
        for b in backbones:
            vals = {m: cell(b, m) for m in base + ours}
            avail = [m for m in base if vals[m] is not None]
            if not avail:
                continue
            best = max(avail, key=lambda m: vals[m]["oa_mean"])
            deltas = [
                f"{METHOD_LABEL.get(m, m)} {vals[m]['oa_mean'] - vals[best]['oa_mean']:+.2f}"
                for m in ours if vals[m] is not None
            ]
            if deltas:
                print(f"  {SHORT.get(b, b):<24} vs best baseline "
                      f"({METHOD_LABEL.get(best, best)}): " + ",  ".join(deltas)
                      + " pp OA")
    n = summaries[backbones[0]].get("n_seeds", "?")
    print("=" * width)
    print(f"  * = our method.  {n} seed(s).  "
          f"'mean over BB' = unweighted mean across backbones.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "master_comparison.tex"))
    ap.add_argument("--metric", choices=["oa", "f1"], default="oa",
                    help="metric used to pick the best cell (default: oa)")
    ap.add_argument("--method-root", default=None,
                    help="directory holding <BACKBONE>/summary.json "
                         "(default: $OUTPUT_ROOT or runs/method)")
    ap.add_argument("--no-compile", action="store_true",
                    help="do not run pdflatex")
    args = ap.parse_args()

    if args.method_root:
        global METHOD_ROOT
        METHOD_ROOT = Path(args.method_root)
        if not METHOD_ROOT.is_absolute():
            METHOD_ROOT = ROOT / METHOD_ROOT
    summaries = load_summaries()
    if not summaries:
        raise SystemExit(f"No summary.json found under {METHOD_ROOT}")

    print(f"Backbones included ({len(summaries)}): "
          f"{', '.join(summaries.keys())}")

    print_console_summary(summaries)

    tex = build_tex(summaries, args.metric)
    out = Path(args.out)
    out.write_text(tex)
    print(f"Wrote {out}")

    if args.no_compile:
        return
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        print("pdflatex not found -- skipping compilation.")
        return
    for _ in range(2):  # twice so \multirow / refs settle
        r = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
             out.name],
            cwd=out.parent, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stdout.splitlines()[-25:])
        print(f"pdflatex FAILED:\n{tail}")
        raise SystemExit(r.returncode)
    print(f"Compiled {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
