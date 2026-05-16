#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


SEQ_BINS = [
    ("<=50", 0, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
    (">200", 201, 10**12),
]
LEN_BINS = [
    ("<=500", 0, 500),
    ("501-1000", 501, 1000),
    ("1001-5000", 1001, 5000),
    (">5000", 5001, 10**12),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test splits for PANDIT AA MSAs."
    )
    parser.add_argument(
        "--stats-csv",
        default="data/zenodo_raw/msa_stats.csv",
        help="CSV produced by the Zenodo/PANDIT data-preparation workflow.",
    )
    parser.add_argument(
        "--outdir",
        default="runs/pandit_domain_adaptation/stage1_msa_splits",
        help="Output directory for split TSVs and summaries.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    return parser.parse_args()


def assign_bin(value: int, bins: list[tuple[str, int, int]]) -> str:
    for label, lo, hi in bins:
        if lo <= value <= hi:
            return label
    raise ValueError(f"Value {value} did not fit any bin")


def load_pandit_aa(stats_csv: Path) -> list[dict[str, str]]:
    with stats_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") == "pandit" and row.get("seq_type") == "aa"
        ]

    for row in rows:
        num_sequences = int(row["num_sequences"])
        alignment_length = int(row["alignment_length"])
        row["msa_id"] = row["msa_id"]
        row["seq_bin"] = assign_bin(num_sequences, SEQ_BINS)
        row["len_bin"] = assign_bin(alignment_length, LEN_BINS)
        row["stratum"] = f"{row['seq_bin']}|{row['len_bin']}"
    return rows


def split_stratum(
    rows: list[dict[str, str]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> None:
    n = len(rows)
    raw_counts = {
        "train": n * train_frac,
        "val": n * val_frac,
        "test": n * test_frac,
    }
    counts = {split: int(raw_counts[split]) for split in raw_counts}
    remainder = n - sum(counts.values())
    priority = sorted(
        raw_counts,
        key=lambda split: (raw_counts[split] - counts[split], split),
        reverse=True,
    )
    for split in priority[:remainder]:
        counts[split] += 1

    index = 0
    for split in ("train", "val", "test"):
        for row in rows[index : index + counts[split]]:
            row["split"] = split
        index += counts[split]


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "msa_id",
        "relative_path",
        "num_sequences",
        "alignment_length",
        "seq_bin",
        "len_bin",
        "stratum",
        "split",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def write_split_summary(outdir: Path, rows: list[dict[str, str]]) -> None:
    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_stratum: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        by_split[row["split"]].append(row)
        by_stratum[(row["stratum"], row["split"])] += 1

    summary_path = outdir / "split_summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "split",
            "n_msa",
            "mean_num_sequences",
            "median_num_sequences",
            "mean_alignment_length",
            "median_alignment_length",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for split in ("train", "val", "test"):
            items = by_split[split]
            seqs = [int(row["num_sequences"]) for row in items]
            lens = [int(row["alignment_length"]) for row in items]
            writer.writerow(
                {
                    "split": split,
                    "n_msa": len(items),
                    "mean_num_sequences": f"{mean(seqs):.6g}",
                    "median_num_sequences": f"{median(seqs):.6g}",
                    "mean_alignment_length": f"{mean(lens):.6g}",
                    "median_alignment_length": f"{median(lens):.6g}",
                }
            )

    stratum_path = outdir / "split_strata_counts.tsv"
    with stratum_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["stratum", "train", "val", "test", "total"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for stratum in sorted({row["stratum"] for row in rows}):
            counts = {split: by_stratum[(stratum, split)] for split in ("train", "val", "test")}
            writer.writerow(
                {
                    "stratum": stratum,
                    **counts,
                    "total": sum(counts.values()),
                }
            )


def main() -> None:
    args = parse_args()
    total_frac = args.train_frac + args.val_frac + args.test_frac
    if abs(total_frac - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1.0, got {total_frac}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_pandit_aa(Path(args.stats_csv))
    rng = random.Random(args.seed)
    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    for items in strata.values():
        rng.shuffle(items)
        split_stratum(items, args.train_frac, args.val_frac, args.test_frac)

    rows.sort(key=lambda row: int(row["msa_id"]))
    write_tsv(outdir / "pandit_all.tsv", rows)
    for split in ("train", "val", "test"):
        write_tsv(outdir / f"pandit_{split}.tsv", [row for row in rows if row["split"] == split])
    write_split_summary(outdir, rows)

    print(f"[OK] wrote splits to {outdir}")
    print(f"[OK] total PANDIT AA MSAs: {len(rows)}")


if __name__ == "__main__":
    main()
