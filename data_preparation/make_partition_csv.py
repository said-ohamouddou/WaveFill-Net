"""Add or re-flag the test_set_flag column on the segments CSV.

Convention: 0 = train, 1 = test, 2 = val.

Modes
-----
1. Re-flag mode (input CSV already has test_set_flag with 0/1):
       Reflags `--val-ratio` of the flag==1 rows to 2, stratified by species.
       (Used to convert the original binary HeliALS partition into the
       three-way one expected by the pipeline.)

2. Bootstrap mode (input CSV has no test_set_flag, e.g. final-segments-with-species.csv):
       Creates a fresh stratified 3-way split with the supplied ratios.

Usage
-----
  # Re-flag the original training-and-test CSV in place
  python "data preparation/make_partition_csv.py"

  # Bootstrap a 3-way partition from final-segments and write to data/
  python "data preparation/make_partition_csv.py" \\
      --csv "data preparation/final-segments-with-species.csv" \\
      --output data/training-and-test-segments-with-species.csv \\
      --train-ratio 0.17 --val-ratio 0.21
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _stratified_three_way(df: pd.DataFrame,
                            train_ratio: float, val_ratio: float,
                            seed: int) -> pd.Series:
    """Stratified train/val/test split. Returns a pd.Series of test_set_flag
    values aligned to df's index (0=train, 1=test, 2=val)."""
    if not (0 < train_ratio < 1):
        raise SystemExit(f"--train-ratio must be in (0, 1); got {train_ratio}")
    if not (0 < val_ratio < 1):
        raise SystemExit(f"--val-ratio must be in (0, 1); got {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise SystemExit("train_ratio + val_ratio must be < 1 "
                          "(remainder goes to test)")

    species = df["species_code"].astype(int).to_numpy()
    idx = df.index.to_numpy()

    # Carve out train; the rest is split into val + test below.
    _, rest_idx = train_test_split(
        idx, train_size=train_ratio, stratify=species, random_state=seed,
    )
    # val_ratio is relative to the full dataset, not the remainder.
    rest_species = df.loc[rest_idx, "species_code"].astype(int).to_numpy()
    val_size_in_remainder = val_ratio / (1.0 - train_ratio)
    val_idx, _ = train_test_split(
        rest_idx, train_size=val_size_in_remainder,
        stratify=rest_species, random_state=seed,
    )

    # Default everyone to train (0), then mark the remainder as test (1),
    # then overwrite the val subset of the remainder to 2.
    flags = pd.Series(0, index=df.index, dtype=np.int64)
    flags.loc[rest_idx] = 1
    flags.loc[val_idx]  = 2
    return flags


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', type=str,
                   default='data preparation/final-segments-with-species.csv',
                   help='input segments CSV')
    p.add_argument('--output', type=str,
                   default='data/final-segments-with-species.csv',
                   help='output path (default: data/final-segments-with-species.csv)')
    p.add_argument('--train-ratio', type=float, default=0.17,
                   help='[bootstrap mode] fraction of segments for train. paper: ~0.17')
    p.add_argument('--val-ratio', type=float, default=0.21,
                   help='[bootstrap mode] fraction of segments for val. '
                        '[re-flag mode] fraction of test pool reflagged to val. paper: ~0.21 / 0.25')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-backup', action='store_true',
                   help='do not write <output>.orig backup on first run')
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if 'species_code' not in df.columns:
        raise SystemExit("CSV is missing species_code; cannot stratify")
    print(f"Loaded {len(df)} rows from {args.csv}")

    mode = 'reflag' if 'test_set_flag' in df.columns else 'bootstrap'
    print(f"Mode: {mode}")

    if mode == 'reflag':
        counts_before = df['test_set_flag'].value_counts().to_dict()
        print(f"  test_set_flag before: {counts_before}")
        test_pool = df[df['test_set_flag'] == 1]
        if test_pool.empty:
            raise SystemExit("No rows with test_set_flag == 1; nothing to re-split.")
        _, val_idx = train_test_split(
            test_pool.index.to_numpy(),
            test_size=args.val_ratio,
            stratify=test_pool['species_code'].astype(int).to_numpy(),
            random_state=args.seed,
        )
        df.loc[val_idx, 'test_set_flag'] = 2
    else:
        df['test_set_flag'] = _stratified_three_way(
            df, args.train_ratio, args.val_ratio, args.seed,
        )

    counts_after = df['test_set_flag'].value_counts().to_dict()
    n_train = int((df['test_set_flag'] == 0).sum())
    n_val   = int((df['test_set_flag'] == 2).sum())
    n_test  = int((df['test_set_flag'] == 1).sum())
    print(f"  test_set_flag after : {counts_after}")
    print(f"  train (flag=0): {n_train} ({n_train/len(df)*100:.1f}%)")
    print(f"  val   (flag=2): {n_val} ({n_val/len(df)*100:.1f}%)")
    print(f"  test  (flag=1): {n_test} ({n_test/len(df)*100:.1f}%)")

    output = args.output or args.csv
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    if mode == 'reflag' and not args.no_backup and os.path.exists(output):
        backup = output + '.orig'
        if not os.path.exists(backup):
            shutil.copy2(output, backup)
            print(f"  Backed up to {backup}")
    df.to_csv(output, index=False)
    print(f"Wrote {output}")

    val_species  = df[df['test_set_flag'] == 2]['species_code'].value_counts().sort_index()
    test_species = df[df['test_set_flag'] == 1]['species_code'].value_counts().sort_index()
    train_species = df[df['test_set_flag'] == 0]['species_code'].value_counts().sort_index()
    print("\nSpecies distribution by partition:")
    print(f"  {'code':>4}  {'train':>6}  {'val':>6}  {'test':>6}")
    for code in sorted(set(train_species.index)
                        | set(val_species.index)
                        | set(test_species.index)):
        tr = int(train_species.get(code, 0))
        v  = int(val_species.get(code, 0))
        ts = int(test_species.get(code, 0))
        print(f"  {code:>4}  {tr:>6}  {v:>6}  {ts:>6}")


if __name__ == '__main__':
    main()
