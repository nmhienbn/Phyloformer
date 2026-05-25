#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_labelled_runs(values: List[str]) -> List[Tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            path = value
            label = Path(path).name
        parsed.append((label, Path(path)))
    return parsed


def load_summary(runs: List[Tuple[str, Path]]) -> pd.DataFrame:
    rows = []
    for label, run_dir in runs:
        runtime = pd.read_csv(run_dir / "runtime_summary.csv").iloc[0].to_dict()
        topo = pd.read_csv(run_dir / "cmp_topo.csv")
        rows.append(
            {
                "method": label,
                "n_alignments": int(runtime["n_alignments"]),
                "infer_sec": float(runtime["infer_sec"]),
                "compare_sec": float(runtime["compare_sec"]),
                "total_sec": float(runtime["total_sec"]),
                "mean_sec_per_alignment": float(runtime["mean_sec_per_alignment"]),
                "max_rss_kb": float(runtime["max_rss_kb"]),
                "mean_norm_rf": float(topo["norm_rf"].mean()),
                "median_norm_rf": float(topo["norm_rf"].median()),
                "mean_weighted_rf": float(topo["weighted_rf"].mean()),
                "mean_kf_score": float(topo["kf_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_plots(summary: pd.DataFrame, outdir: Path) -> None:
    sns.set_style("darkgrid")

    plt.figure(figsize=(8, 4))
    sns.barplot(data=summary, x="method", y="total_sec")
    plt.ylabel("Total Runtime (s)")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.savefig(outdir / "runtime_total_sec.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=summary, x="method", y="mean_norm_rf")
    plt.ylabel("Mean Normalized RF")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.savefig(outdir / "accuracy_mean_norm_rf.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=summary, x="total_sec", y="mean_norm_rf", hue="method", s=120)
    for _, row in summary.iterrows():
        plt.text(row["total_sec"], row["mean_norm_rf"], f" {row['method']}", va="center")
    plt.xlabel("Total Runtime (s)")
    plt.ylabel("Mean Normalized RF")
    plt.tight_layout()
    plt.savefig(outdir / "runtime_vs_accuracy.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot runtime/accuracy summary for benchmark run directories."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=run_dir (repeatable). run_dir must contain runtime_summary.csv and cmp_topo.csv.",
    )
    parser.add_argument(
        "--outdir",
        default="figures/runtime_accuracy",
        help="Output directory for summary plots.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(parse_labelled_runs(args.run))
    summary.to_csv(outdir / "summary.csv", index=False)
    make_plots(summary, outdir)
    print(f"[INFO] wrote summary to {outdir / 'summary.csv'}")
    print(f"[INFO] wrote plots to {outdir}")


if __name__ == "__main__":
    main()
