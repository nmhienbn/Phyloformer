#!/usr/bin/env python3
"""Plot mean topology metrics and sample count by number of blocks needed.

Two figures:
  1. Bar chart: number of alignments per block count
  2. Line chart: mean nRF / wRF / KF per block count, one line per method

Usage:
    python third_party/tools/plots/plot_metrics_by_nblocks.py \
        --cmp-dir runs/benchmarks/pandit_over2gb/benchmark_results/cmp \
        --aln-dir data/benchmarks/pandit_aa_zenodo/alignments \
        --cap 2.0 \
        --dataset PANDIT \
        --outdir figures/pandit_over2gb_by_nblocks \
        --methods "PF2_direct" "window-site+weighted" "window-site+fastrfs" \
                  "window-rate+weighted" "window-rate+fastrfs" \
                  "softbioblock+weighted" "softbioblock+fastrfs"
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from third_party.tools.vram.pf2_vram import estimate_vram_gb

# ── default method styles ────────────────────────────────────────────────────
DEFAULT_METHODS = [
    ("PF2_direct",             "PF2 direct",            "#2ca02c", "-",  "o"),
    ("window-site+weighted",   "window-site+weighted",  "#ff7f0e", "-",  "o"),
    ("window-site+fastrfs",    "window-site+fastrfs",   "#ff7f0e", "--", "s"),
    ("window-rate+weighted",   "window-rate+weighted",  "#8c564b", "-",  "o"),
    ("window-rate+fastrfs",    "window-rate+fastrfs",   "#8c564b", "--", "s"),
    ("softbioblock+weighted",  "softbioblock+weighted", "#9467bd", "-",  "o"),
    ("softbioblock+fastrfs",   "softbioblock+fastrfs",  "#9467bd", "--", "s"),
]

METRICS = [
    ("norm_rf",     "nRF (normalized RF)"),
    ("weighted_rf", "wRF"),
    ("kf_score",    "KF score"),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def build_block_map(aln_dir: Path, cap: float) -> dict[str, int]:
    """Return {stem: n_blocks} for alignments that exceed `cap` GB VRAM."""
    from Bio import AlignIO
    block_map: dict[str, int] = {}
    for f in sorted(aln_dir.glob("*.fa")):
        try:
            aln = AlignIO.read(f, "fasta")
        except Exception:
            continue
        v = estimate_vram_gb(len(aln), aln.get_alignment_length())
        if v > cap:
            n_seq, n_sites = len(aln), aln.get_alignment_length()
            max_sites = (cap - 0.4573) / (5.12e-7 * n_seq ** 2)
            block_map[f.stem] = math.ceil(n_sites / max_sites)
    return block_map


def load_topo(cmp_dir: Path, key: str, dataset: str, block_map: dict) -> pd.DataFrame | None:
    safe = key.replace("/", "_").replace(" ", "_")
    p = cmp_dir / f"{safe}__{dataset}_topo.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["id"] = df["id"].astype(str)
    df["n_blocks"] = df["id"].map(block_map)
    return df


# ── plots ────────────────────────────────────────────────────────────────────

def plot_sample_count(cnt: Counter, outpath: Path, dataset: str, cap: float) -> None:
    groups = sorted(cnt)
    counts = [cnt[g] for g in groups]
    xlabels = [str(g) for g in groups]

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.7), 4))
    bars = ax.bar(range(len(groups)), counts, color="#4e79a7", edgecolor="white")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlabel("Số block cần chia", fontsize=11)
    ax.set_ylabel("Số alignment", fontsize=11)
    ax.set_title(
        f"{dataset}: Phân phối số block (cap {cap} GB)  —  n={sum(counts)}",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outpath}")


def plot_mean_metrics(
    dfs: dict[str, pd.DataFrame],
    methods_ordered: list,
    cnt: Counter,
    outpath: Path,
    dataset: str,
    cap: float,
) -> None:
    groups = sorted(cnt)
    g_labels = [f"{g}\n(n={cnt[g]})" for g in groups]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS) + 1, 5))

    for ax, (metric, mlabel) in zip(axes, METRICS):
        for key, label, color, ls, mk in methods_ordered:
            if key not in dfs:
                continue
            df = dfs[key]
            means = [df[df["n_blocks"] == g][metric].dropna().mean() for g in groups]
            ax.plot(range(len(groups)), means,
                    marker=mk, color=color, linestyle=ls,
                    linewidth=1.5, markersize=4.5, label=label)

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(g_labels, fontsize=7.5)
        ax.set_xlabel(f"Số block cần chia (cap {cap} GB)", fontsize=10)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.13), frameon=True)
    fig.suptitle(
        f"{dataset}: Mean metrics theo số block (cap {cap} GB)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outpath}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cmp-dir", required=True, help="Directory with *_topo.csv files from run_tree_dir_benchmarks.py")
    p.add_argument("--aln-dir", required=True, help="Alignment directory (.fa files)")
    p.add_argument("--cap", type=float, default=2.0, help="VRAM cap in GB used for block estimation (default: 2.0)")
    p.add_argument("--dataset", default="PANDIT", help="Dataset label used in CSV filenames, e.g. PANDIT or Cherry")
    p.add_argument("--outdir", default="figures/by_nblocks", help="Output directory for figures")
    p.add_argument(
        "--methods", nargs="+",
        default=[m[0] for m in DEFAULT_METHODS],
        help="Method keys to include (must match CSV filename prefix). Order determines plot order.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cmp_dir = Path(args.cmp_dir)
    aln_dir = Path(args.aln_dir)
    outdir  = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build block map
    print(f"Computing block counts (cap={args.cap} GB) from {aln_dir} ...")
    block_map = build_block_map(aln_dir, args.cap)
    cnt = Counter(block_map.values())
    print(f"  {len(block_map)} alignments over cap | block distribution: {dict(sorted(cnt.items()))}")

    # Build method style table (preserve user order, apply default styles where available)
    default_style = {m[0]: m for m in DEFAULT_METHODS}
    methods_ordered = []
    for key in args.methods:
        if key in default_style:
            methods_ordered.append(default_style[key])
        else:
            # fallback style for unknown methods
            methods_ordered.append((key, key, "#888888", "-", "o"))

    # Load topo CSVs
    dfs: dict[str, pd.DataFrame] = {}
    for key, *_ in methods_ordered:
        df = load_topo(cmp_dir, key, args.dataset, block_map)
        if df is not None:
            dfs[key] = df
            print(f"  loaded {key}: {len(df)} rows, {df['n_blocks'].notna().sum()} with block info")
        else:
            print(f"  [missing] {key}")

    # Plot 1: sample count bar chart
    plot_sample_count(
        cnt,
        outdir / f"{args.dataset}_block_distribution.png",
        args.dataset,
        args.cap,
    )

    # Plot 2: mean metrics line chart
    plot_mean_metrics(
        dfs,
        methods_ordered,
        cnt,
        outdir / f"{args.dataset}_mean_metrics_by_nblocks.png",
        args.dataset,
        args.cap,
    )


if __name__ == "__main__":
    main()
