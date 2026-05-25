#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    ("nRF", "norm_rf", "Normalized RF"),
    ("wRF", "weighted_rf", "Weighted RF"),
    ("KF", "kf_score", "KF"),
]

PANDIT_GPARTITION_PALETTE = {
    # Match figures/pandit_domain_adaptation/topology_boxplots_pandit_test_gpartition_kaggle-16gb:
    # PF2 adapt gPart, PF2 full, FastTree, IQTree, MPBoot SPR6.
    "PF2 adapt": "#1f77b4",
    "PF2 adapt gPart": "#1f77b4",
    "PF2 direct": "#ff7f0e",
    "PF2 full": "#ff7f0e",
    "FastTree": "#2ca02c",
    "IQ-TREE": "#d62728",
    "IQTree": "#d62728",
    "MPBoot SPR6": "#9467bd",
    "MPBoot SPR3": "#8c564b",
}


def parse_labelled_paths(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=path, got {value!r}")
        label, path = value.split("=", 1)
        parsed.append((label.strip(), Path(path)))
    return parsed


def load_topology(paths: list[tuple[str, Path]]) -> pd.DataFrame:
    frames = []
    required = {"id", "norm_rf", "weighted_rf", "kf_score"}
    for label, path in paths:
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        df = df.copy()
        df["method"] = label
        frames.append(df[["id", "method", "norm_rf", "weighted_rf", "kf_score"]])
    if not frames:
        raise ValueError("No topology CSVs were provided.")
    return pd.concat(frames, ignore_index=True)


def make_summary(df: pd.DataFrame, outdir: Path) -> None:
    summary = (
        df.groupby("method", as_index=False)
        .agg(
            n_alignments=("id", "nunique"),
            median_nRF=("norm_rf", "median"),
            mean_nRF=("norm_rf", "mean"),
            median_wRF=("weighted_rf", "median"),
            mean_wRF=("weighted_rf", "mean"),
            median_KF=("kf_score", "median"),
            mean_KF=("kf_score", "mean"),
        )
        .sort_values("method")
    )
    summary.to_csv(outdir / "topology_panel_summary.csv", index=False)


def plot_panels(
    df: pd.DataFrame,
    outpath: Path,
    title: str,
    base_methods: list[str],
    nrf_extra_methods: list[str],
    adapt_method: str | None,
    hide_fliers: bool,
    upper_quantile: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("This script requires matplotlib and seaborn.") from exc

    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    for ax, (metric_name, column, ylabel) in zip(axes, METRICS):
        methods = list(base_methods)
        if metric_name == "nRF":
            methods.extend(nrf_extra_methods)
        if adapt_method is not None:
            methods.append(adapt_method)

        methods = [method for method in methods if method in set(df["method"])]
        plot_df = df[df["method"].isin(methods)].copy()
        plot_df["method"] = pd.Categorical(plot_df["method"], categories=methods, ordered=True)

        sns.boxplot(
            data=plot_df,
            x="method",
            hue="method",
            y=column,
            order=methods,
            hue_order=methods,
            ax=ax,
            palette={method: PANDIT_GPARTITION_PALETTE[method] for method in methods},
            legend=False,
            width=0.58,
            showfliers=not hide_fliers,
            fliersize=2.0,
        )
        if 0.0 < upper_quantile < 1.0 and not plot_df.empty:
            ymin = float(plot_df[column].min())
            ymax = float(plot_df[column].quantile(upper_quantile))
            if ymax > ymin:
                pad = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(metric_name)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=25)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    fig.suptitle(title)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create PANDIT topology boxplot panels with MPBoot only on nRF."
    )
    parser.add_argument(
        "--cmp-topo",
        action="append",
        required=True,
        help="METHOD_LABEL=path/to/cmp_topo.csv. Repeat for each method.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dataset-title", default="PANDIT")
    parser.add_argument("--base-method", action="append", required=True)
    parser.add_argument("--nrf-extra-method", action="append", default=[])
    parser.add_argument("--adapt-method", default=None)
    parser.add_argument("--hide-fliers", action="store_true")
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_topology(parse_labelled_paths(args.cmp_topo))
    df.to_csv(outdir / "topology_panel_raw.csv", index=False)
    make_summary(df, outdir)

    plot_panels(
        df,
        outdir / "pandit_topology_base.png",
        args.dataset_title,
        base_methods=args.base_method,
        nrf_extra_methods=args.nrf_extra_method,
        adapt_method=None,
        hide_fliers=args.hide_fliers,
        upper_quantile=args.upper_quantile,
    )
    if args.adapt_method:
        plot_panels(
            df,
            outdir / "pandit_topology_with_pf2_adapt.png",
            f"{args.dataset_title} + PF2 adapt",
            base_methods=args.base_method,
            nrf_extra_methods=args.nrf_extra_method,
            adapt_method=args.adapt_method,
            hide_fliers=args.hide_fliers,
            upper_quantile=args.upper_quantile,
        )

    print(f"[INFO] wrote plots to {outdir}")


if __name__ == "__main__":
    main()
