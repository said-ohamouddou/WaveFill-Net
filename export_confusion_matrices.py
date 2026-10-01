#!/usr/bin/env python3
"""Export separate kNN and FastKAN confusion matrices from saved evaluations.

Examples (no training, checkpoint loading, or GPU required):
    python export_confusion_matrices.py
    python export_confusion_matrices.py --backbone DGCNN
    python export_confusion_matrices.py --wavelength 1550 --show-std
    python export_confusion_matrices.py --wavelength all

Default output: confusion_matrices/<BACKBONE>/confusion_<METHOD>_avg.{pdf,png,json}
Rows are true species; columns are predicted species. Each row is normalized
to percentages BEFORE averaging. For avg, average all three wavelengths within
each seed, then compute mean and sample SD across seeds. This is an average of
repeated evaluations of the same test trees, not an ensemble confusion matrix.
The diagonal matches the wavelength-averaged per-class recall table.

Plots share a fixed 0--100% color scale and have no title by default. --titles
adds method/classifier labels; --show-std annotates mean +/- sample SD. JSON
sidecars always retain unrounded means, SDs, per-seed matrices and provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from build_recall_time_params import CLASSES

ROOT = Path(__file__).resolve().parent
METHODS = {'knn': 'kNN', 'wavefill_fastkan': 'WaveFill-Net (FastKAN)'}
WAVELENGTHS = {'1550': '0', '905': '1', '532': '2'}


def aggregate_matrices(records, backbone, wavelength='avg'):
    """Validate paired results and return row-percent mean and sample SD."""
    if len(records) < 2:
        raise ValueError('At least two seeds are required for sample SD.')
    seeds = [record['seed'] for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate seed IDs.')
    wl_keys = list(WAVELENGTHS.values()) if wavelength == 'avg' else [WAVELENGTHS[wavelength]]
    support = None
    output = {}
    for method in METHODS:
        seed_matrices = []
        for record in records:
            if record.get('backbone') != backbone:
                raise ValueError(f"Seed {record['seed']}: backbone does not match {backbone}.")
            cells = record.get('by_method', {}).get(method, {}).get('per_wl', {})
            normalized = []
            for wl in wl_keys:
                if wl not in cells:
                    raise ValueError(f"Seed {record['seed']}, {method}: missing wavelength {wl}.")
                cell = cells[wl]
                matrix = np.asarray(cell.get('confusion_matrix', []))
                if matrix.shape != (len(CLASSES), len(CLASSES)):
                    raise ValueError(f'{method}/{wl}: expected a 9 x 9 confusion matrix.')
                if not np.issubdtype(matrix.dtype, np.integer) or (matrix < 0).any():
                    raise ValueError('Confusion matrices must contain nonnegative integer counts.')
                current_support = matrix.sum(axis=1)
                if (current_support == 0).any():
                    raise ValueError('Cannot normalize a class with no test examples.')
                if support is None:
                    support = current_support
                if not np.array_equal(current_support, support):
                    raise ValueError('Class supports differ across seeds, wavelengths or methods.')
                if record.get('n_test', int(support.sum())) != int(support.sum()):
                    raise ValueError('Saved n_test disagrees with the confusion matrix.')
                if 'per_class_f1' in cell and list(cell['per_class_f1']) != CLASSES:
                    raise ValueError('Saved class order does not match the project class order.')
                normalized.append(matrix / current_support[:, None] * 100)
            seed_matrices.append(np.mean(normalized, axis=0))
        stack = np.stack(seed_matrices)
        output[method] = {
            'backbone': backbone, 'method': method, 'wavelength': wavelength,
            'seeds': seeds, 'n_seeds': len(seeds), 'n_test': int(support.sum()),
            'classes': CLASSES, 'class_support': support.tolist(),
            'rows': 'true species', 'columns': 'predicted species', 'unit': 'percent',
            'aggregation': 'Row-normalize each evaluation; average selected wavelengths within each seed; mean and sample SD across seeds (ddof=1).',
            'mean_percent': stack.mean(axis=0).tolist(),
            'std_percent': stack.std(axis=0, ddof=1).tolist(),
            'per_seed_percent': stack.tolist(),
        }
    return output


def plot_matrix(data, output_stem, show_std=False, titles=False):
    # Use a writable cache in restricted environments; figures need no display.
    if 'MPLCONFIGDIR' not in os.environ:
        cache = Path(tempfile.gettempdir()) / 'wavefill-matplotlib'
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['MPLCONFIGDIR'] = str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    mean = np.asarray(data['mean_percent'])
    std = np.asarray(data['std_percent'])
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    fig.subplots_adjust(left=.17, bottom=.18, right=.89, top=.97)
    im = ax.imshow(mean, cmap='Blues', vmin=0, vmax=100, interpolation='nearest')
    positions = np.arange(len(CLASSES))
    ax.set_xticks(positions, CLASSES, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticks(positions, CLASSES)
    ax.set_xlabel('Predicted species', labelpad=9)
    ax.set_ylabel('True species', labelpad=9)
    ax.tick_params(which='both', length=0)
    ax.set_xticks(np.arange(-.5, len(CLASSES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(CLASSES), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.65)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in positions:
        for j in positions:
            annotation = f'{mean[i, j]:.1f}'
            if show_std:
                annotation += f'\n± {std[i, j]:.1f}'
            ax.text(j, i, annotation, ha='center', va='center',
                    fontsize=8 if show_std else 10,
                    color='white' if mean[i, j] >= 55 else '#152B3C')
    bar = fig.colorbar(im, ax=ax, fraction=.047, pad=.035, ticks=np.arange(0, 101, 20))
    bar.set_label('Mean row percentage (%)', labelpad=9)
    bar.outline.set_visible(False)
    if titles:
        wl_label = 'Wavelength average' if data['wavelength'] == 'avg' else data['wavelength'] + ' nm'
        ax.set_title(f"{METHODS[data['method']]} · {data['backbone']}\n{wl_label}", pad=14)
    for suffix in ('pdf', 'png'):
        fig.savefig(output_stem.with_suffix('.' + suffix), dpi=300, facecolor='white',
                    bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backbone', default='PointTransformerV2')
    parser.add_argument('--method-root', type=Path, default=ROOT / 'runs/method')
    parser.add_argument('--out-dir', type=Path, help='Default: confusion_matrices/<BACKBONE>.')
    parser.add_argument('--seeds', help='Comma-separated seed IDs; default: all saved seeds.')
    parser.add_argument('--wavelength', choices=['avg', 'all', *WAVELENGTHS], default='avg',
                        help='avg: average three wavelengths; all: avg plus one figure per wavelength.')
    parser.add_argument('--show-std', action='store_true', help='Annotate cells with mean and sample SD.')
    parser.add_argument('--titles', action='store_true', help='Add method/classifier titles.')
    args = parser.parse_args()
    try:
        folder = args.method_root / args.backbone
        paths = ([folder / f'seed{int(seed)}' / 'eval_results.json' for seed in args.seeds.split(',')]
                 if args.seeds else sorted(folder.glob('seed*/eval_results.json')))
        pairs = sorted([(json.loads(path.read_text()), path) for path in paths],
                       key=lambda pair: pair[0]['seed'])
        records = [record for record, _ in pairs]
        wavelengths = ['avg', *WAVELENGTHS] if args.wavelength == 'all' else [args.wavelength]
        # Validate every requested matrix before writing any figures.
        reports = [aggregate_matrices(records, args.backbone, wl) for wl in wavelengths]
        output_dir = args.out_dir or ROOT / 'confusion_matrices' / args.backbone
        output_dir.mkdir(parents=True, exist_ok=True)
        for report in reports:
            for method, data in report.items():
                data['source_files'] = [str(path.resolve()) for _, path in pairs]
                data['plot_annotation'] = 'mean ± sample SD' if args.show_std else 'mean'
                stem = output_dir / f"confusion_{method}_{data['wavelength']}"
                plot_matrix(data, stem, show_std=args.show_std, titles=args.titles)
                stem.with_suffix('.json').write_text(json.dumps(data, indent=2) + '\n')
                print(f'{stem.resolve()}.{{pdf,png,json}}')
        print(f"Exported {len(reports) * len(METHODS)} matrices; {len(records)} paired seeds; classifier {args.backbone}.")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
