#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np
from Bio import AlignIO


GAP_CHARS = {"-", ".", "?"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the empirical PANDIT profile needed by the Chapter 3 OOD and synthetic steps."
    )
    parser.add_argument("split_tsv")
    parser.add_argument("--input-root", default="data/zenodo_raw")
    parser.add_argument("--out-tsv", required=True)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def alignment_formats(path: Path) -> list[str]:
    if path.suffix.lower() in {".fa", ".fasta", ".faa", ".fas"}:
        return ["fasta"]
    if path.suffix.lower() in {".phy", ".phylip"}:
        return ["phylip-relaxed", "phylip", "fasta"]
    return ["phylip-relaxed", "phylip", "fasta"]


def load_alignment(path: Path) -> np.ndarray:
    last_error: Exception | None = None
    for fmt in alignment_formats(path):
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


def site_stats(matrix: np.ndarray) -> dict[str, float]:
    nseq, length = matrix.shape
    gap_mask = np.isin(matrix, list(GAP_CHARS))
    invariant_sites = 0
    informative_sites = 0
    all_gap_sites = 0

    for col_idx in range(length):
        chars = [char for char in matrix[:, col_idx].tolist() if char not in GAP_CHARS]
        if not chars:
            all_gap_sites += 1
            continue
        counts = Counter(chars)
        invariant_sites += int(len(counts) == 1)
        informative_sites += int(sum(1 for count in counts.values() if count >= 2) >= 2)

    return {
        "gap_ratio": float(gap_mask.sum()) / (nseq * length) if nseq and length else 0.0,
        "invariant_site_ratio": invariant_sites / length if length else 0.0,
        "informative_site_ratio": informative_sites / length if length else 0.0,
        "all_gap_site_ratio": all_gap_sites / length if length else 0.0,
    }


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    summary_rows = []
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        summary_rows.append(
            {
                "split": split,
                "n_msa": str(len(split_rows)),
                "median_num_sequences": f"{median(int(row['num_sequences']) for row in split_rows):.6g}",
                "median_alignment_length": f"{median(int(row['alignment_length']) for row in split_rows):.6g}",
                "mean_gap_ratio": f"{mean(float(row['gap_ratio']) for row in split_rows):.6g}",
                "median_gap_ratio": f"{median(float(row['gap_ratio']) for row in split_rows):.6g}",
                "median_informative_site_ratio": (
                    f"{median(float(row['informative_site_ratio']) for row in split_rows):.6g}"
                ),
            }
        )
    write_tsv(path, list(summary_rows[0].keys()), summary_rows)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    profile_rows: list[dict[str, str]] = []

    for row in read_tsv(Path(args.split_tsv)):
        stats = site_stats(load_alignment(input_root / row["relative_path"]))
        profile_rows.append(
            {
                "msa_id": row["msa_id"],
                "split": row["split"],
                "relative_path": row["relative_path"],
                "num_sequences": row["num_sequences"],
                "alignment_length": row["alignment_length"],
                "seq_bin": row["seq_bin"],
                "len_bin": row["len_bin"],
                "gap_ratio": f"{stats['gap_ratio']:.12g}",
                "invariant_site_ratio": f"{stats['invariant_site_ratio']:.12g}",
                "informative_site_ratio": f"{stats['informative_site_ratio']:.12g}",
                "all_gap_site_ratio": f"{stats['all_gap_site_ratio']:.12g}",
            }
        )

    out_tsv = Path(args.out_tsv)
    write_tsv(out_tsv, list(profile_rows[0].keys()), profile_rows)
    write_summary(out_tsv.with_suffix(".summary.tsv"), profile_rows)
    print(f"[OK] wrote {out_tsv}")
    print(f"[OK] profiled {len(profile_rows)} MSAs")


if __name__ == "__main__":
    main()
