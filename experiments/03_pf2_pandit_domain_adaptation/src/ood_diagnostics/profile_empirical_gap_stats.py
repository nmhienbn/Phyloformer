#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median

import numpy as np
from Bio import AlignIO


GAP_CHARS = {"-", ".", "?"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile empirical PANDIT train gap statistics for Chapter 3. "
            "The output defines low/mid/high gap buckets and their summary "
            "statistics for comparison with the synthetic Stage 5 v3 data."
        )
    )
    parser.add_argument("split_tsv", help="PANDIT split TSV, normally pandit_train.tsv.")
    parser.add_argument("--input-root", default="data/zenodo_raw")
    parser.add_argument(
        "--outdir",
        default="runs/pandit_domain_adaptation/stage5_empirical_gap_profile",
    )
    parser.add_argument("--prefix", default="pandit_train")
    parser.add_argument(
        "--bucket-quantiles",
        default="0.333333,0.666667",
        help="Two comma-separated quantiles for low/mid/high gap buckets.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_quantiles(raw: str) -> tuple[float, float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != 2:
        raise ValueError("--bucket-quantiles must provide exactly two values.")
    return values[0], values[1]


def infer_alignment_formats(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa", ".fas"}:
        return ["fasta"]
    if suffix in {".phy", ".phylip"}:
        return ["phylip-relaxed", "phylip", "fasta"]
    return ["phylip-relaxed", "phylip", "fasta"]


def load_alignment_matrix(path: Path) -> np.ndarray:
    last_error: Exception | None = None
    for fmt in infer_alignment_formats(path):
        try:
            alignment = AlignIO.read(path, fmt)
            sequences = [str(record.seq).upper() for record in alignment]
            lengths = {len(seq) for seq in sequences}
            if len(lengths) != 1:
                raise ValueError(f"Alignment is not rectangular: {sorted(lengths)}")
            return np.array([list(seq) for seq in sequences], dtype="<U1")
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not parse {path}: {last_error}")


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return values[lo]
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def gap_bucket(gap_ratio: float, low: float, high: float) -> str:
    if gap_ratio <= low:
        return "low-gap"
    if gap_ratio <= high:
        return "mid-gap"
    return "high-gap"


def gap_run_lengths(matrix: np.ndarray) -> list[int]:
    runs: list[int] = []
    for row in matrix:
        current = 0
        for char in row.tolist():
            if char in GAP_CHARS:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
    return runs


def p90(values: list[float]) -> float:
    return quantile(values, 0.9) if values else 0.0


def compute_gap_stats(matrix: np.ndarray) -> dict[str, float]:
    nseq, length = matrix.shape
    gap_mask = np.isin(matrix, list(GAP_CHARS))
    total_cells = nseq * length
    runs = gap_run_lengths(matrix)
    return {
        "gap_ratio": float(gap_mask.sum()) / total_cells if total_cells else 0.0,
        "gappy_column_ratio_any": float(gap_mask.any(axis=0).sum()) / length if length else 0.0,
        "median_gap_run_length": float(median(runs)) if runs else 0.0,
        "p90_gap_run_length": p90([float(run) for run in runs]),
    }


def summarize_bucket(rows: list[dict[str, str]], bucket: str) -> dict[str, str]:
    bucket_rows = [row for row in rows if row["gap_bucket"] == bucket]
    values = {
        "gap_ratio": [float(row["gap_ratio"]) for row in bucket_rows],
        "gappy_column_ratio_any": [float(row["gappy_column_ratio_any"]) for row in bucket_rows],
        "median_gap_run_length": [float(row["median_gap_run_length"]) for row in bucket_rows],
        "p90_gap_run_length": [float(row["p90_gap_run_length"]) for row in bucket_rows],
    }
    record = {"gap_bucket": bucket, "n_msa": str(len(bucket_rows))}
    for name, data in values.items():
        record[f"mean_{name}"] = f"{mean(data):.12g}" if data else ""
        record[f"median_{name}"] = f"{median(data):.12g}" if data else ""
    return record


def main() -> None:
    args = parse_args()
    split_rows = read_tsv(Path(args.split_tsv))
    input_root = Path(args.input_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_stats: list[dict[str, float]] = []
    for row in split_rows:
        raw_stats.append(compute_gap_stats(load_alignment_matrix(input_root / row["relative_path"])))

    q_low, q_high = parse_quantiles(args.bucket_quantiles)
    gap_ratios = [stats["gap_ratio"] for stats in raw_stats]
    threshold_low = quantile(gap_ratios, q_low)
    threshold_high = quantile(gap_ratios, q_high)

    per_msa_rows: list[dict[str, str]] = []
    for split_row, stats in zip(split_rows, raw_stats):
        per_msa_rows.append(
            {
                "msa_id": split_row["msa_id"],
                "split": split_row.get("split", "train"),
                "relative_path": split_row["relative_path"],
                "num_sequences": split_row["num_sequences"],
                "alignment_length": split_row["alignment_length"],
                "gap_bucket": gap_bucket(stats["gap_ratio"], threshold_low, threshold_high),
                "gap_ratio": f"{stats['gap_ratio']:.12g}",
                "gappy_column_ratio_any": f"{stats['gappy_column_ratio_any']:.12g}",
                "median_gap_run_length": f"{stats['median_gap_run_length']:.12g}",
                "p90_gap_run_length": f"{stats['p90_gap_run_length']:.12g}",
                "gap_bucket_q_low": f"{threshold_low:.12g}",
                "gap_bucket_q_high": f"{threshold_high:.12g}",
            }
        )

    per_msa_path = outdir / f"{args.prefix}_gap_stats.tsv"
    summary_path = outdir / f"{args.prefix}_gap_bucket_summary.tsv"
    thresholds_path = outdir / f"{args.prefix}_gap_bucket_thresholds.tsv"
    write_tsv(
        per_msa_path,
        [
            "msa_id",
            "split",
            "relative_path",
            "num_sequences",
            "alignment_length",
            "gap_bucket",
            "gap_ratio",
            "gappy_column_ratio_any",
            "median_gap_run_length",
            "p90_gap_run_length",
            "gap_bucket_q_low",
            "gap_bucket_q_high",
        ],
        per_msa_rows,
    )
    summary_rows = [summarize_bucket(per_msa_rows, bucket) for bucket in ["low-gap", "mid-gap", "high-gap"]]
    write_tsv(summary_path, list(summary_rows[0].keys()), summary_rows)
    write_tsv(
        thresholds_path,
        ["metric", "value"],
        [
            {"metric": "low_mid_threshold", "value": f"{threshold_low:.12g}"},
            {"metric": "mid_high_threshold", "value": f"{threshold_high:.12g}"},
            {"metric": "n_msa", "value": str(len(per_msa_rows))},
        ],
    )
    print(f"[OK] wrote {per_msa_path}")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {thresholds_path}")


if __name__ == "__main__":
    main()
