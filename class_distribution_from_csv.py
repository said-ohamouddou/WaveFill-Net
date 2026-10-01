#!/usr/bin/env python3
"""Report HeliALS species counts from the actual partition CSV.

Uses the same test_set_flag mapping as TreeSpeciesDatasetHELIALS.py:
0 = train, 2 = val, 1 = test. No split is generated and no point clouds or
HDF5 cache are loaded. Counts describe the rows listed in the CSV.

Usage:
    python class_distribution_from_csv.py
    python class_distribution_from_csv.py --csv path/to/labels.csv
    python class_distribution_from_csv.py --out path/to/distribution.csv

Only the Python standard library is required. Percentages are calculated
within each split; total_pct is calculated over all CSV rows.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# HeliALS species codes, matching TreeSpeciesDatasetHELIALS.SPECIES_NAMES.
SPECIES = {
    1: "Pine", 2: "Spruce", 3: "Birch", 4: "Maple", 5: "Aspen",
    6: "Rowan", 7: "Oak", 8: "Linden", 9: "Alder",
}
FLAG_TO_SPLIT = {0: "train", 2: "val", 1: "test"}
SPLITS = ("train", "val", "test")


def read_counts(path: Path) -> dict[str, Counter]:
    """Validate metadata and count each unique segment exactly once."""
    counts = {split: Counter() for split in SPLITS}
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"test_site_name", "segment_id", "species_code", "test_set_flag"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        for row in reader:
            location = f"CSV line {reader.line_num}"
            site = (row["test_site_name"] or "").strip()
            try:
                segment_id = int(row["segment_id"])
                species_code = int(row["species_code"])
                flag = int(row["test_set_flag"])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"{location}: segment_id, species_code and test_set_flag "
                    "must be integers."
                ) from exc
            if not site:
                raise ValueError(f"{location}: empty test_site_name.")
            key = (site, segment_id)
            if key in seen:
                raise ValueError(f"{location}: duplicate segment {site}_{segment_id}.")
            if species_code not in SPECIES:
                raise ValueError(f"{location}: unknown species_code {species_code}.")
            if flag not in FLAG_TO_SPLIT:
                raise ValueError(
                    f"{location}: unknown test_set_flag {flag}; expected 0, 1 or 2."
                )
            seen.add(key)
            counts[FLAG_TO_SPLIT[flag]][species_code] += 1
    if not seen:
        raise ValueError("The CSV contains no segment records.")
    return counts


def percentage(count: int, total: int) -> str:
    return f"{100 * count / total:.2f}" if total else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "data" / "final-segments-with-species.csv",
                        help="Partition CSV (default: data/final-segments-with-species.csv)")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs" / "dataset_stats_csv" / "class_distribution.csv",
                        help="Output CSV report path")
    args = parser.parse_args()
    if args.csv.resolve() == args.out.resolve():
        parser.error("The output must not overwrite the input CSV.")
    try:
        counts = read_counts(args.csv)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))

    totals = {split: sum(counts[split].values()) for split in SPLITS}
    grand_total = sum(totals.values())
    print(f"\nSource: {args.csv.resolve()}")
    print("CSV assignments: train=0, val=2, test=1. No new split generated.")
    print("Cells: number of trees (percentage within that split).\n")
    header = f"{'Species':<12}" + "".join(f"{s.title():>19}" for s in SPLITS)
    header += f"{'Total':>19}"
    print(header)
    print("-" * len(header))
    rows = []
    for code, species in SPECIES.items():
        row = {"species_code": code, "species": species}
        cells = []
        for split in SPLITS:
            count = counts[split][code]
            pct = percentage(count, totals[split])
            row[split] = count
            row[f"{split}_pct"] = pct
            cells.append(f"{count} ({pct}%)" if pct else f"{count} (n/a)")
        row["total"] = sum(counts[split][code] for split in SPLITS)
        row["total_pct"] = percentage(row["total"], grand_total)
        cells.append(f"{row['total']} ({row['total_pct']}%)")
        print(f"{species:<12}" + "".join(f"{cell:>19}" for cell in cells))
        rows.append(row)

    print("-" * len(header))
    print(f"{'TOTAL':<12}" + "".join(f"{totals[s]:>19}" for s in SPLITS)
          + f"{grand_total:>19}")
    total_row = {"species_code": "", "species": "TOTAL",
                 "total": grand_total, "total_pct": "100.00"}
    for split in SPLITS:
        total_row[split] = totals[split]
        total_row[f"{split}_pct"] = "100.00" if totals[split] else ""
        if not totals[split]:
            print(f"Note: the CSV assigns no trees to {split}; percentages are undefined.")
    rows.append(total_row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
