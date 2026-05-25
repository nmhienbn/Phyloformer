#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_labelled_paths(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            path = value
            label = Path(path).stem
        parsed.append((label, Path(path)))
    return parsed


def load_topology_only(paths: list[tuple[str, Path]]) -> pd.DataFrame:
    frames = []
    for label, path in paths:
        df = pd.read_csv(path)
        df["marker"] = label
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["n_tips"] = out["n_tips"].astype(int)
    return out


def plot_box(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="n_tips", y="norm_rf", hue="marker")
    plt.xlabel("Number of tips")
    plt.ylabel("Normalized RF distance")
    plt.tight_layout()
    plt.savefig(outdir / "topology_only_norm_rf_box.png", dpi=200)
    plt.close()


def plot_line(df: pd.DataFrame, outdir: Path) -> None:
    summary = (
        df.groupby(["marker", "n_tips"], as_index=False)["norm_rf"]
        .mean()
        .sort_values(["marker", "n_tips"])
    )
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=summary, x="n_tips", y="norm_rf", hue="marker", marker="o")
    plt.xlabel("Number of tips")
    plt.ylabel("Mean normalized RF distance")
    plt.tight_layout()
    plt.savefig(outdir / "topology_only_norm_rf_line.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot topology-only comparison CSVs."
    )
    parser.add_argument(
        "--topo-only",
        action="append",
        required=True,
        help="LABEL=path_to_topology_only.csv (repeatable). If LABEL omitted, inferred from filename.",
    )
    parser.add_argument(
        "--outdir",
        default="figures/topology_only",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_topology_only(parse_labelled_paths(args.topo_only))
    plot_box(df, outdir)
    plot_line(df, outdir)
    print(f"[INFO] wrote plots to {outdir}")


if __name__ == "__main__":
    main()
