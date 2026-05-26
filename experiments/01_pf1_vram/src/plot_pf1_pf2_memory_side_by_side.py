#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def load_matrix_csv(path: Path) -> tuple[list[int], list[int], np.ndarray]:
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        lengths = [int(x) for x in header[1:]]
        rows = list(reader)

    seqs = [int(r[0]) for r in rows]
    data = np.full((len(seqs), len(lengths)), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, val in enumerate(row[1:]):
            if val.strip():
                data[i, j] = float(val)
    return seqs, lengths, data


def save_matrix_csv(path: Path, seqs: list[int], lengths: list[int], data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_seq"] + lengths)
        for i, n in enumerate(seqs):
            writer.writerow(
                [n] + [f"{x:.6f}" if not math.isnan(x) else "" for x in data[i]]
            )


def memory_cmap() -> ListedColormap:
    try:
        base = plt.get_cmap("rocket")
    except ValueError:
        base = plt.get_cmap("magma")
    colors = base(np.linspace(0, 1, 256))
    colors[-32:] = colors[-33]
    cmap = ListedColormap(colors, name=f"{base.name}_readable_max")
    cmap.set_bad(color="#d9d9e3")
    return cmap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PF1 and PF2 GPU-memory heatmaps side-by-side from matrix CSV files."
    )
    parser.add_argument("--pf1-csv", required=True, help="Path to PF1 matrix CSV.")
    parser.add_argument("--pf2-csv", required=True, help="Path to PF2 matrix CSV.")
    parser.add_argument("--out", required=True, help="Output figure path, preferably .pdf.")
    parser.add_argument(
        "--pf1-title",
        default="PF1 inference measured",
        help="Title for the PF1 panel.",
    )
    parser.add_argument(
        "--pf2-title",
        default="PF2 inference reference",
        help="Title for the PF2 panel.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional overall figure title.",
    )
    parser.add_argument(
        "--copy-pf1-csv",
        default=None,
        help="Optional output path to copy normalized PF1 matrix CSV.",
    )
    parser.add_argument(
        "--copy-pf2-csv",
        default=None,
        help="Optional output path to copy normalized PF2 matrix CSV.",
    )
    args = parser.parse_args()

    pf1_csv = Path(args.pf1_csv)
    pf2_csv = Path(args.pf2_csv)
    out = Path(args.out)

    seqs1, lengths1, pf1 = load_matrix_csv(pf1_csv)
    seqs2, lengths2, pf2 = load_matrix_csv(pf2_csv)

    if seqs1 != seqs2:
        raise ValueError(f"PF1/PF2 seq axes differ: {seqs1} vs {seqs2}")
    if lengths1 != lengths2:
        raise ValueError(f"PF1/PF2 length axes differ: {lengths1} vs {lengths2}")

    if args.copy_pf1_csv:
        save_matrix_csv(Path(args.copy_pf1_csv), seqs1, lengths1, pf1)
    if args.copy_pf2_csv:
        save_matrix_csv(Path(args.copy_pf2_csv), seqs2, lengths2, pf2)

    vmax = np.nanmax([np.nanmax(pf1), np.nanmax(pf2)])
    cmap = memory_cmap()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    plots = [
        (axes[0], pf1, args.pf1_title),
        (axes[1], pf2, args.pf2_title),
    ]
    for ax, data, title in plots:
        im = ax.imshow(
            data,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
        )
        ax.set_xticks(range(len(lengths1)))
        ax.set_xticklabels([str(x) for x in lengths1])
        ax.set_yticks(range(len(seqs1)))
        ax.set_yticklabels([str(x) for x in seqs1])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Nb. of Sequences")
        ax.set_title(title)
        for i in range(len(seqs1)):
            for j in range(len(lengths1)):
                val = data[i, j]
                if math.isnan(val):
                    continue
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("Max GPU RSS (GB)")
    if args.title:
        fig.suptitle(args.title)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
