#!/usr/bin/env python3
"""Export per-species recall, fill time and added neural-imputer parameters.

Run in the project's torch311 environment (no training or CUDA required):
    python build_recall_time_params.py
    python build_recall_time_params.py --compile
    python build_recall_time_params.py --methods all --out recall_all.tex
    python build_recall_time_params.py --methods full,knn,wavefill_fastkan

Default: Full 16D, kNN, random forest, and all three WaveFill-Net variants.
Zero-fill, Mean-fill and Linear are excluded, including with --methods all.
Use --exclude-methods '' to override these exclusions when needed.
Output: an includable .tex table and a .json audit with unrounded seed values.
--compile additionally writes a standalone .pdf using pdflatex.

Recall = diagonal / row support of each saved confusion matrix. First average
the three wavelengths WITHIN each seed, then take mean and sample SD (ddof=1)
across paired seeds. Macro-recall SD is computed from per-seed macro recalls,
not by averaging class SDs. Baseline times are cached/shared across seeds.
Counts come from saved metadata/checkpoints, never the example table.
"""
from __future__ import annotations

import argparse
import codecs
import json
import math
import shutil
import statistics as stats
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES = ['Pine', 'Spruce', 'Birch', 'Maple', 'Aspen', 'Rowan', 'Oak', 'Linden', 'Alder']
WAVELENGTHS = ('0', '1', '2')
LABELS = {
    'full': 'Full 16D (ref.)', 'zero': 'Zero-fill', 'mean': 'Mean-fill',
    'linear': 'Linear', 'knn': 'kNN', 'missforest': 'Random forest',
    'wavefill': 'WaveFill-Net (MLP, ours)',
    'wavefill_relukan': 'WaveFill-Net (ReLU-KAN)',
    'wavefill_fastkan': 'WaveFill-Net (FastKAN)',
}
DEFAULT_METHODS = ('full', 'knn', 'missforest', 'wavefill',
                   'wavefill_relukan', 'wavefill_fastkan')
CHECKPOINTS = {
    'wavefill': 'wavefill_best.pt',
    'wavefill_relukan': 'wavefill_relukan_best.pt',
    'wavefill_fastkan': 'wavefill_fastkan_best.pt',
}


def summary(values):
    return {'mean': stats.mean(values), 'std': stats.stdev(values), 'values': values}


def aggregate(records, methods, backbone):
    """Validate complete paired results before emitting any output."""
    if len(records) < 2:
        raise ValueError('At least two seeds are required for sample SD.')
    seeds = [r['seed'] for r in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate seed IDs in evaluation files.')
    support = None
    per_method = {m: {'recall': [], 'macro': [], 'time': []} for m in methods}
    for record in records:
        if record.get('backbone') != backbone:
            raise ValueError(f"Seed {record['seed']}: backbone does not match {backbone}.")
        for method in methods:
            cells = record.get('by_method', {}).get(method, {}).get('per_wl', {})
            if set(cells) != set(WAVELENGTHS):
                raise ValueError(f"Seed {record['seed']}, {method}: require all three wavelengths 0,1,2.")
            recalls, times = [], []
            for wl in WAVELENGTHS:
                cell = cells[wl]
                cm = cell.get('confusion_matrix', [])
                if len(cm) != len(CLASSES) or any(len(row) != len(CLASSES) for row in cm):
                    raise ValueError(f'{method}/{wl}: expected a 9 x 9 confusion matrix.')
                if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for row in cm for v in row):
                    raise ValueError('Confusion matrices must contain nonnegative integer counts.')
                row_support = [sum(row) for row in cm]
                if any(n == 0 for n in row_support):
                    raise ValueError('Recall is undefined for a class with no test examples.')
                if support is None:
                    support = row_support
                if support != row_support:
                    raise ValueError('Class supports differ across methods, wavelengths or seeds.')
                if record.get('n_test', sum(support)) != sum(support):
                    raise ValueError('Saved n_test disagrees with the confusion matrix.')
                if 'per_class_f1' in cell and list(cell['per_class_f1']) != CLASSES:
                    raise ValueError('Saved class ordering does not match the project class ordering.')
                recall = [100 * cm[i][i] / row_support[i] for i in range(len(CLASSES))]
                if 'macro_recall' in cell and not math.isclose(
                        stats.mean(recall) / 100, cell['macro_recall'], abs_tol=1e-10):
                    raise ValueError('Saved macro recall disagrees with the confusion matrix.')
                timing = cell.get('fill_time_ms_per_tree')
                if not isinstance(timing, (float, int)) or not math.isfinite(timing) or timing < 0:
                    raise ValueError(f'{method}/{wl}: missing or invalid fill timing.')
                recalls.append(recall)
                times.append(timing)
            seed_recalls = [stats.mean(v) for v in zip(*recalls)]
            per_method[method]['recall'].append(seed_recalls)
            per_method[method]['macro'].append(stats.mean(seed_recalls))
            per_method[method]['time'].append(stats.mean(times))
    result = {'backbone': backbone, 'seeds': seeds, 'classes': CLASSES,
              'support': support, 'n_test': sum(support), 'methods': {},
              'aggregation': 'Equal wavelength mean within seed, then mean and sample SD (ddof=1).',
              'timing_note': 'Cached baseline timings are reused across classifier seeds; their SD is not independent timing variability.'}
    for method, data in per_method.items():
        result['methods'][method] = {
            'recall': {name: summary(list(v)) for name, v in zip(CLASSES, zip(*data['recall']))},
            'macro_recall': summary(data['macro']),
            'fill_time_ms_per_tree': summary(data['time']),
        }
    return result


def checkpoint_count(path, classifier=False):
    # No model imports, GPU kernels, forward passes or checkpoint modifications.
    import numpy as np
    import torch
    try:
        from numpy._core.multiarray import scalar  # NumPy 2.x
    except ImportError:
        from numpy.core.multiarray import scalar  # NumPy 1.x

    # Project classifier metadata includes NumPy scalar metrics and byte strings.
    allowed = [scalar, np.dtype, type(np.dtype('float64')), codecs.encode]
    # NumPy 2 renamed core to _core. Legacy checkpoints still reference the old
    # pickle path, so registering the function by its current name is insufficient.
    if scalar.__module__ != 'numpy.core.multiarray':
        allowed.append((scalar, 'numpy.core.multiarray.scalar'))
    previous = list(torch.serialization.get_safe_globals())
    try:
        torch.serialization.add_safe_globals(allowed)
        state = torch.load(path, map_location='cpu', weights_only=True)['model_state']
    finally:
        # Preserve caller state, including on failure; never use unrestricted load.
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(previous)
    # Current classifier buffers: BatchNorm statistics, and KAN-DGCNN spline grid.
    # FastKAN's rbf.grid IS an nn.Parameter (fixed), included in total count.
    def is_buffer(name):
        return name.endswith(('running_mean', 'running_var', 'num_batches_tracked')) or (
            classifier and name.endswith('.grid'))
    return sum(v.numel() for name, v in state.items() if not is_buffer(name))


def add_parameter_counts(report, records, seed_dirs, classifier_root):
    classifier_counts = []
    for record, seed_dir in zip(records, seed_dirs):
        count = record.get('model_params', {}).get('paired_classifier')
        if count is None:
            count = checkpoint_count(classifier_root / report['backbone'] / 'classifier' /
                                     f"model_seed{record['seed']}.pt", classifier=True)
        classifier_counts.append(count)
    if len(set(classifier_counts)) != 1:
        raise ValueError('Classifier parameter count differs across seeds.')
    report['classifier_params'] = classifier_counts[0]
    for method, data in report['methods'].items():
        if method not in CHECKPOINTS:
            data['imputer_params'] = None
            continue
        counts = []
        for record, seed_dir in zip(records, seed_dirs):
            count = record.get('model_params', {}).get(method)
            if count is None:
                count = checkpoint_count(seed_dir / CHECKPOINTS[method])
            counts.append(count)
        if len(set(counts)) != 1:
            raise ValueError(f'{method}: imputer parameter count differs across seeds.')
        data['imputer_params'] = counts[0]


def latex_escape(value):
    escapes = {'\\': r'\textbackslash{}', '_': r'\_', '%': r'\%', '&': r'\&',
               '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}'}
    return ''.join(escapes.get(c, c) for c in str(value))


def build_tex(report, time_decimals=2):
    methods = list(report['methods'])
    bb = latex_escape(report['backbone'])
    caption = (
        r'Per-class recall (\%), per-tree imputation (fill) time (ms), and imputer parameter count '
        rf'for the \textbf{{{bb}}} classifier ({report["n_test"]} test trees). '
        r'Recall and fill time are averaged over the three wavelengths within each seed, '
        rf'then reported as mean $\pm$ sample SD over {len(report["seeds"])} seeds. '
        r'\textbf{Bold} marks the highest mean recall among the displayed imputation methods '
        r'(the \emph{Full 16D} reference excluded; exact ties are all bold). '
        r'Within each seed, all methods share the same frozen classifier '
        rf'(\textasciitilde{report["classifier_params"] / 1e6:.2f}M parameters). '
        r'The parameter row reports added neural-imputer parameters; -- means not applicable '
        r'under this convention, not absence of fitted baseline state. '
        r'Cached baseline fill times are reused across seeds, so their zero SD does not '
        r'measure timing variability. Classifier inference time is excluded.'
    )
    if 'wavefill_fastkan' in methods:
        caption += ' FastKAN counts include the fixed RBF grid parameters.'
    lines = [r'% Generated by build_recall_time_params.py; requires booktabs and graphicx.',
             r'\begin{table}[htbp]\centering', '\\caption{' + caption + '}',
             r'\label{tab:recall_time_params}',
             r'\renewcommand{\arraystretch}{1.2}\setlength{\tabcolsep}{5pt}\footnotesize',
             r'\resizebox{\textwidth}{!}{%', '\\begin{tabular}{l ' + 'c' * len(methods) + '}',
             r'\toprule', r'\textbf{Metric / Species} & ' + ' & '.join(
                 '\\textbf{' + LABELS[m] + '}' for m in methods) + r' \\', r'\midrule']

    def row(label, cells, digits=1, bold=False):
        best = max((cells[m]['mean'] for m in methods if m != 'full'), default=None)
        output = []
        for m in methods:
            c = cells[m]
            text = f"{c['mean']:.{digits}f}\\,$\\pm$\\,{c['std']:.{digits}f}"
            if bold and m != 'full' and math.isclose(c['mean'], best, rel_tol=0, abs_tol=1e-10):
                text = '\\textbf{' + text + '}'
            output.append(text)
        lines.append(label + ' & ' + ' & '.join(output) + r' \\')

    for name in CLASSES:
        row(name, {m: report['methods'][m]['recall'][name] for m in methods}, bold=True)
    lines.append(r'\midrule')
    row(r'\textbf{Macro recall}', {m: report['methods'][m]['macro_recall'] for m in methods}, bold=True)
    row('Fill time (ms/tree)', {m: report['methods'][m]['fill_time_ms_per_tree'] for m in methods}, time_decimals)
    params = [report['methods'][m]['imputer_params'] for m in methods]
    lines.append('Imputer params & ' + ' & '.join('--' if n is None else f'{n / 1e6:.2f}M' for n in params) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}%', '}', r'\end{table}', ''])
    return '\n'.join(lines)


def compile_pdf(tex, output):
    if not shutil.which('pdflatex'):
        raise RuntimeError('pdflatex is required for --compile; the .tex and .json were saved.')
    document = (r'\documentclass[11pt]{article}' '\n'
                r'\usepackage[T1]{fontenc}' '\n'
                r'\usepackage[a4paper,landscape,margin=1.4cm]{geometry}' '\n'
                r'\usepackage{booktabs,graphicx}' '\n'
                r'\begin{document}\thispagestyle{empty}' '\n' + tex + '\n' + r'\end{document}')
    with tempfile.TemporaryDirectory(prefix='recall-table-') as temp:
        work = Path(temp)
        (work / 'table.tex').write_text(document)
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'table.tex'],
                                cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('LaTeX compilation failed:\n' + result.stdout[-5000:])
        shutil.copyfile(work / 'table.pdf', output.with_suffix('.pdf'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method-root', type=Path, default=ROOT / 'runs/method')
    parser.add_argument('--classifier-root', type=Path, default=ROOT / 'runs/all_16d_backbones_1024pts_5seed')
    parser.add_argument('--backbone', default='PointTransformerV2')
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS), help='Comma-separated method keys, or all.')
    parser.add_argument('--exclude-methods', default='zero,mean,linear',
                        help='Comma-separated methods to omit (default: zero,mean,linear); applies to --methods all too.')
    parser.add_argument('--seeds', help='Optional comma-separated seed IDs; default: all saved seeds.')
    parser.add_argument('--out', type=Path, default=ROOT / 'table_recall_time_params.tex')
    parser.add_argument('--time-decimals', type=int, default=2, choices=range(2, 7))
    parser.add_argument('--compile', action='store_true', help='Also compile a standalone PDF preview.')
    args = parser.parse_args()
    try:
        if args.out.suffix != '.tex':
            raise ValueError('--out must end in .tex')
        methods = list(LABELS) if args.methods == 'all' else args.methods.split(',')
        if not methods or len(set(methods)) != len(methods) or any(m not in LABELS for m in methods):
            raise ValueError('Use unique method keys from: ' + ','.join(LABELS))
        excluded = {m.strip() for m in args.exclude_methods.split(',') if m.strip()}
        if excluded - LABELS.keys():
            raise ValueError('Unknown excluded methods: ' + ','.join(sorted(excluded - LABELS.keys())))
        methods = [m for m in methods if m not in excluded]
        if not methods:
            raise ValueError('No methods remain after exclusions.')
        backbone_dir = args.method_root / args.backbone
        if args.seeds:
            files = [backbone_dir / f'seed{int(s)}' / 'eval_results.json' for s in args.seeds.split(',')]
        else:
            files = sorted(backbone_dir.glob('seed*/eval_results.json'))
        pairs = sorted([(json.loads(p.read_text()), p.parent) for p in files], key=lambda pair: pair[0]['seed'])
        records, seed_dirs = [p[0] for p in pairs], [p[1] for p in pairs]
        report = aggregate(records, methods, args.backbone)
        add_parameter_counts(report, records, seed_dirs, args.classifier_root)
        report['source_files'] = [str(p / 'eval_results.json') for p in seed_dirs]
        tex = build_tex(report, args.time_decimals)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(tex)
        args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"Exported {args.backbone}: {report['n_test']} trees; seeds {report['seeds']}; {len(methods)} methods.")
        print(f'Table: {args.out.resolve()}')
        print(f'Audit: {args.out.with_suffix(".json").resolve()}')
        if args.compile:
            compile_pdf(tex, args.out)
            print(f'PDF:   {args.out.with_suffix(".pdf").resolve()}')
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
