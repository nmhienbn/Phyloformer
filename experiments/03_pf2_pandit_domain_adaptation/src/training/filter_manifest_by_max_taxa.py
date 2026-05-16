#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a synthetic manifest into rows kept under a max num_sequences threshold "
            "and rows excluded above that threshold."
        )
    )
    parser.add_argument("manifest_tsv", help="Input manifest TSV.")
    parser.add_argument("--max-taxa", type=int, default=170, help="Maximum allowed num_sequences.")
    parser.add_argument("--out-tsv", required=True, help="Output TSV for kept rows.")
    parser.add_argument("--excluded-tsv", required=True, help="Output TSV for excluded rows.")
    return parser.parse_args()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f"No header found in {path}")
    return fieldnames, rows


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_tsv(Path(args.manifest_tsv))

    kept = [row for row in rows if int(row["num_sequences"]) <= args.max_taxa]
    excluded = [row for row in rows if int(row["num_sequences"]) > args.max_taxa]

    write_tsv(Path(args.out_tsv), fieldnames, kept)
    write_tsv(Path(args.excluded_tsv), fieldnames, excluded)

    print(f"[OK] kept: {len(kept)}")
    print(f"[OK] excluded: {len(excluded)}")
    print(f"[OK] wrote {args.out_tsv}")
    print(f"[OK] wrote {args.excluded_tsv}")


if __name__ == "__main__":
    main()
