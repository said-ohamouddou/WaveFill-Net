#!/usr/bin/env python3
"""
Paper-ready exports for the two ablation studies.

Reads the CSVs the ablations already write -- it never retrains, so it is
cheap to re-run while tuning the presentation.

    Experiment 2 (components, cross-classifier)
        -> ablation_components_table.tex
        LaTeX table: evaluator backbone groups with one row per variant;
        columns are WL=0, WL=1, WL=2, and the average over wavelengths.
        The teacher backbone is marked. Include booktabs, graphicx, and
        multirow in the document that imports the table fragment.

    Experiment 3 (hyperparameters)
        -> ablation_hparam_<parameter>.{pdf,png} + ablation_hparam_points.csv
        Separate figures, one per swept hyperparameter. lambda_recon and
        lambda_cls keep the parameter count fixed, so those figures plot OA
        against the swept VALUE. num_grids and hidden change the parameter
        count (27k to 719k across the sweep), so plotting against value alone
        would confound basis resolution with capacity -- those two figures plot
        OA against PARAMETER COUNT instead, annotating each point with its
        value. That way a gain that merely tracks capacity is visible as a
        straight climb, not mistaken for a better basis.

Usage
-----
    python export_ablation_figures.py
    python export_ablation_figures.py --comp-dir runs/ablation_components_kan \
                                      --hp-dir   runs/ablation_hparam_kan
    python export_ablation_figures.py --only table     # or: figures
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Short column headers; falls back to the raw name for anything unlisted.
SHORT = {"PointTransformerV2": "PTv2", "PointNet2_MSG": "PointNet++",
         "DGCNN": "DGCNN", "DeepGCN": "DeepGCN", "PCT": "PCT",
         "PointNet": "PointNet", "PointTransformer": "PT"}

PARAM_LABEL = {"lambda_recon": r"$\lambda_{\mathrm{recon}}$",
               "lambda_cls":   r"$\lambda_{\mathrm{cls}}$",
               "num_grids":    "RBF grid points",
               "hidden":       "hidden width"}
# Plain-text names for the capacity panels' axis label (no math mode there).
PARAM_LABEL_PLAIN = {"num_grids": "RBF grid points", "hidden": "hidden width"}

# Panels whose parameter count is constant across the sweep -> x = value.
VALUE_AXIS = {"lambda_recon", "lambda_cls"}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}\n  Run the ablation first.")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sd(v):
    return statistics.stdev(v) if len(v) > 1 else 0.0


# ---------------------------------------------------------------------------
# Experiment 2: LaTeX table
# ---------------------------------------------------------------------------

def build_components_table(rows: list[dict]) -> str:
    """Backbones as \\multirow groups, wavelengths as columns.

    Same layout as the main comparison table (Table 1) so the two read
    together: one evaluator backbone per group, one variant per sub-row,
    WL=0/1/2 plus the average over lambda as columns. The teacher group is
    marked -- a variant that merely pleases its teacher scores well inside
    that group and drops in the held-out groups below it.
    """
    WLN = {0: r"WL=0 (1550\,nm)", 1: r"WL=1 (905\,nm)", 2: r"WL=2 (532\,nm)"}

    # (variant, evaluator, wl) -> per-seed OA ; and per-seed wavelength means
    # for the avg column, so its std reflects seed variation not wl spread.
    cells: dict = defaultdict(lambda: defaultdict(list))
    teacher = None
    variants, evaluators = [], []
    for r in rows:
        if r.get("is_teacher", "").lower() == "true":
            teacher = r["evaluator"]
        v, e, wl, sd = r["variant"], r["evaluator"], int(r["wavelength"]), int(r["seed"])
        cells[(v, e, wl)][sd].append(_f(r["oa"]))
        if v not in variants:
            variants.append(v)
        if e not in evaluators:
            evaluators.append(e)
    wls = [0, 1, 2]
    evaluators.sort(key=lambda e: (e != teacher, e))   # teacher group first

    def stat(v, e, wl):
        d = cells.get((v, e, wl))
        if not d:
            return None
        per_seed = [statistics.mean(x) for x in d.values()]
        return statistics.mean(per_seed), _sd(per_seed), len(per_seed)

    def stat_avg(v, e):
        """Mean over wavelengths WITHIN each seed, then across seeds."""
        seeds = set()
        for wl in wls:
            seeds |= set(cells.get((v, e, wl), {}))
        per_seed = []
        for sd in sorted(seeds):
            vals = [statistics.mean(cells[(v, e, wl)][sd])
                    for wl in wls if sd in cells.get((v, e, wl), {})]
            if vals:
                per_seed.append(statistics.mean(vals))
        if not per_seed:
            return None
        return statistics.mean(per_seed), _sd(per_seed), len(per_seed)

    counts = {s[2] for v in variants for e in evaluators
              for wl in wls if (s := stat(v, e, wl))}
    counts.update(s[2] for v in variants for e in evaluators
                  if (s := stat_avg(v, e)))
    if len(counts) == 1:
        n = next(iter(counts))
        seed_text = f"{n} seed" + ("s" if n != 1 else "")
    elif counts:
        seed_text = f"{min(counts)}--{max(counts)} seeds (depending on the cell)"
    else:
        seed_text = "0 seeds"

    L = []
    A = L.append
    A(r"\begin{table}[htbp]")
    A(r"\centering")
    A(r"\caption{Component ablation of WaveFill-Net (FastKAN) across LiDAR "
      r"wavelengths and evaluator backbones. Each backbone group contains "
      r"one row per ablation variant. Variants are trained once per seed "
      r"against the teacher and evaluated by seed-matched frozen classifiers. "
      rf"Cells report OA (\%) as mean\,$\pm$\,sample std over {seed_text}. "
      r"For Avg over $\lambda$, wavelengths are averaged within each seed "
      r"before computing the mean and std across seeds.}")
    A(r"\label{tab:abl_components}")
    A(r"\renewcommand{\arraystretch}{1.2}")
    A(r"\setlength{\tabcolsep}{4pt}")
    A(r"\footnotesize")
    A(r"\resizebox{\textwidth}{!}{%")
    A(r"\begin{tabular}{l l c c c c}")
    A(r"\toprule")
    A(r"\textbf{Backbone} & \textbf{Variant} & "
      + " & ".join(rf"\textbf{{{WLN.get(wl, f'WL={wl}')}}}" for wl in wls)
      + r" & \textbf{Avg over $\lambda$} \\")
    A(r"\midrule")

    for gi, e in enumerate(evaluators):
        present = [v for v in variants if stat_avg(v, e)]
        if not present:
            continue
        lbl = SHORT.get(e, e)
        if e == teacher:
            lbl += r"$^{\dagger}$"
        A(rf"\multirow{{{len(present)}}}{{*}}{{{lbl}}}")
        # best variant in this group, on the avg column
        best = max(present, key=lambda v: stat_avg(v, e)[0])
        for v in present:
            row = [v.replace("_", r"\_")]
            for wl in wls:
                st = stat(v, e, wl)
                row.append(f"{st[0]:.2f}\\,$\\pm$\\,{st[1]:.2f}" if st else "--")
            av = stat_avg(v, e)
            txt = f"{av[0]:.2f}\\,$\\pm$\\,{av[1]:.2f}"
            if v == best:
                txt = rf"\textbf{{{txt}}}"
            row.append(txt)
            A(" & " + " & ".join(row) + r" \\")
        A(r"\midrule" if gi < len(evaluators) - 1 else r"\bottomrule")

    A(r"\end{tabular}%")
    A(r"}")
    A(r"\\[2pt]\footnotesize $^{\dagger}$ teacher (in-distribution); "
      r"\textbf{bold} marks the highest average OA within each backbone.")
    A(r"\end{table}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Experiment 3: sweep figures
# ---------------------------------------------------------------------------

def build_hparam_figures(rows: list[dict], out_pdf: Path, out_png: Path,
                         out_csv: Path) -> list[Path]:
    """Export one PDF/PNG pair per parameter and a shared numeric CSV.

    The supplied PDF/PNG names are naming templates: a trailing ``_sweeps``
    is replaced with the parameter name (otherwise the name is appended).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    # param -> value -> {n_params, per-seed wavelength-averaged OA/RMSE}
    tree: dict = defaultdict(lambda: defaultdict(
        lambda: {"oa": defaultdict(list), "rmse": defaultdict(list),
                 "n_params": None, "is_default": False}))
    for r in rows:
        p, v = r["param"], _f(r["value"])
        s = int(r["seed"])
        node = tree[p][v]
        node["oa"][s].append(_f(r["oa"]))
        rm = _f(r.get("rmse"))
        if rm is not None:
            node["rmse"][s].append(rm)
        node["n_params"] = int(float(r["n_params"]))
        node["is_default"] = r.get("is_default", "").lower() == "true"

    params = [p for p in PARAM_LABEL if p in tree] or list(tree)
    pts: list[dict] = []
    written: list[Path] = []

    for p in params:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        vals = sorted(tree[p])
        by_value = {}
        for v in vals:
            node = tree[p][v]
            per_seed = [statistics.mean(o) for o in node["oa"].values()]
            rm = [x for l in node["rmse"].values() for x in l]
            by_value[v] = (statistics.mean(per_seed), _sd(per_seed),
                           node["n_params"], node["is_default"],
                           statistics.mean(rm) if rm else None,
                           len(per_seed))
            pts.append({"param": p, "value": v, "n_params": node["n_params"],
                        "is_default": node["is_default"],
                        "oa_mean": by_value[v][0], "oa_std": by_value[v][1],
                        "rmse_mean": by_value[v][4], "n_seeds": by_value[v][5]})

        value_axis = p in VALUE_AXIS
        # x = swept value when the parameter count is constant; x = parameter
        # count otherwise, so capacity is not mistaken for basis quality.
        xs = [v if value_axis else by_value[v][2] for v in vals]
        ys = [by_value[v][0] for v in vals]
        es = [by_value[v][1] for v in vals]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
        es = [es[i] for i in order]; vs = [vals[i] for i in order]

        ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, lw=1.4, capsize=3,
                    color="tab:blue", zorder=2)
        for v, x, y in zip(vs, xs, ys):
            if by_value[v][3]:                       # the a-priori default
                ax.scatter([x], [y], s=110, facecolors="none",
                           edgecolors="tab:red", lw=1.8, zorder=3,
                           label="default")
            if not value_axis:
                # Annotate each point with the swept value. Bold and dark so
                # it reads against the curve: on these panels the x axis is
                # the parameter count, so the value is the only place the
                # swept quantity appears.
                ax.annotate(f"{v:g}", (x, y), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=9,
                            fontweight="bold", color="0.15")

        # No panel title: the swept hyperparameter is already named on the
        # x axis, and a title repeating it just adds clutter. For the
        # parameter-count panels the x axis names the capacity axis, so the
        # swept quantity is carried in the axis label instead.
        ax.set_xlabel(PARAM_LABEL.get(p, p) if value_axis
                      else f"Number of parameters (log scale) "
                           f"\u2014 {PARAM_LABEL_PLAIN.get(p, p)} annotated",
                      fontsize=11 if value_axis else 10)
        ax.set_ylabel("Test OA (%)")
        ax.grid(alpha=0.3)
        if not value_axis:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        if any(by_value[v][3] for v in vals):
            ax.legend(fontsize=8, loc="best")

        fig.tight_layout()
        for template in (out_pdf, out_png):
            path = template.with_name(
                f"{template.stem.removesuffix('_sweeps')}_{p}{template.suffix}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    if pts:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pts[0]))
            w.writeheader()
            w.writerows(pts)
    return written


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comp-dir", default="runs/ablation_components_kan")
    ap.add_argument("--hp-dir", default="runs/ablation_hparam_kan")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--only", choices=["table", "figures"], default=None)
    args = ap.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.only != "figures":
        rows = read_csv(ROOT / args.comp_dir / "results.csv")
        tex = out / "ablation_components_table.tex"
        tex.write_text(build_components_table(rows))
        print(f"  wrote {tex}")

    if args.only != "table":
        rows = read_csv(ROOT / args.hp_dir / "results.csv")
        written = build_hparam_figures(rows,
                             out / "ablation_hparam_sweeps.pdf",
                             out / "ablation_hparam_sweeps.png",
                             out / "ablation_hparam_points.csv")
        for path in written:
            print(f"  wrote {path}")
        print(f"  wrote {out / 'ablation_hparam_points.csv'}")


if __name__ == "__main__":
    main()
