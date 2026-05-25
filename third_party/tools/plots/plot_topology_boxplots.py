#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def parse_labelled_inputs(values: List[str]) -> List[Tuple[str, str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --cmp-topo value {value!r}. Expected METHOD|DATASET=path/to/cmp_topo.csv"
            )
        label, path = value.split("=", 1)
        if "|" in label:
            method, dataset = label.split("|", 1)
        elif "@" in label:
            method, dataset = label.split("@", 1)
        else:
            method = label
            dataset = Path(path).parent.name
        parsed.append((method.strip(), dataset.strip(), Path(path)))
    return parsed


def load_cmp_topo(inputs: List[Tuple[str, str, Path]]) -> pd.DataFrame:
    frames = []
    for method, dataset, path in inputs:
        df = pd.read_csv(path)
        required = {"id", "norm_rf", "weighted_rf", "kf_score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        df = df.copy()
        df["method"] = method
        df["dataset"] = dataset
        df["nRF"] = pd.to_numeric(df["norm_rf"], errors="coerce")
        df["wRF"] = pd.to_numeric(df["weighted_rf"], errors="coerce")
        df["KF"] = pd.to_numeric(df["kf_score"], errors="coerce")
        frames.append(df[["id", "dataset", "method", "nRF", "wRF", "KF"]])

    if not frames:
        raise ValueError("No cmp_topo inputs were loaded.")
    return pd.concat(frames, ignore_index=True)


def make_long_table(df: pd.DataFrame, nrf_only_methods: set[str] | None = None) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=["id", "dataset", "method"],
        value_vars=["nRF", "wRF", "KF"],
        var_name="metric",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value"]).copy()
    if nrf_only_methods:
        long_df = long_df[
            (long_df["metric"] == "nRF") | (~long_df["method"].isin(nrf_only_methods))
        ].copy()
    return long_df


def write_summary(df: pd.DataFrame, outdir: Path) -> None:
    summary = (
        df.groupby(["dataset", "method"], as_index=False)
        .agg(
            n_alignments=("id", "nunique"),
            mean_nRF=("nRF", "mean"),
            median_nRF=("nRF", "median"),
            mean_wRF=("wRF", "mean"),
            median_wRF=("wRF", "median"),
            mean_KF=("KF", "mean"),
            median_KF=("KF", "median"),
        )
        .sort_values(["dataset", "method"])
    )
    summary.to_csv(outdir / "topology_boxplot_summary.csv", index=False)


def plot_boxplots(
    long_df: pd.DataFrame,
    outdir: Path,
    box_width: float,
    width_per_dataset: float,
    min_panel_width: float,
    combined_metric_width: float,
    single_metric_height: float,
    combined_metric_height: float,
    hide_fliers: bool,
    upper_quantile: float,
    metric_upper_quantiles: dict[str, float],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "plot_topology_boxplots.py requires matplotlib and seaborn in the current environment."
        ) from exc

    sns.set_style("darkgrid")

    metrics = [("nRF", "Normalized RF"), ("wRF", "Weighted RF"), ("KF", "KF Distance")]
    n_datasets = max(1, long_df["dataset"].nunique())
    for metric_key, metric_label in metrics:
        plot_df = long_df[long_df["metric"] == metric_key].copy()
        if plot_df.empty:
            continue

        fig_width = max(min_panel_width, width_per_dataset * n_datasets)
        plt.figure(figsize=(fig_width, single_metric_height))
        ax = sns.boxplot(
            data=plot_df,
            x="dataset",
            y="value",
            hue="method",
            width=box_width,
            whis=1.5,
            fliersize=2.5,
            showfliers=not hide_fliers,
        )
        effective_upper_quantile = metric_upper_quantiles.get(metric_key, upper_quantile)
        if 0.0 < effective_upper_quantile < 1.0:
            ymax = float(plot_df["value"].quantile(effective_upper_quantile))
            ymin = float(plot_df["value"].min())
            if ymax > ymin:
                pad = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by Dataset")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / f"boxplot_{metric_key.lower()}.png", dpi=200)
        plt.close()

    combined_width = max(combined_metric_width, width_per_dataset * n_datasets * 3.0)
    plt.figure(figsize=(combined_width, combined_metric_height))
    g = sns.catplot(
        data=long_df,
        x="dataset",
        y="value",
        hue="method",
        col="metric",
        kind="box",
        width=box_width,
        sharey=False,
        height=combined_metric_height,
        aspect=max(1.1, (combined_width / 3.0) / combined_metric_height),
        fliersize=2.5,
        showfliers=not hide_fliers,
    )
    g.set_axis_labels("Dataset", "Value")
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        title = ax.get_title()
        metric_key = title.strip()
        metric_df = long_df[long_df["metric"] == metric_key]
        effective_upper_quantile = metric_upper_quantiles.get(metric_key, upper_quantile)
        if not metric_df.empty and 0.0 < effective_upper_quantile < 1.0:
            ymax = float(metric_df["value"].quantile(effective_upper_quantile))
            ymin = float(metric_df["value"].min())
            if ymax > ymin:
                pad = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_ha("right")
    g.tight_layout()
    g.savefig(outdir / "boxplot_topology_all_metrics.png", dpi=200)
    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw box plots for nRF, wRF, and KF from one or more cmp_topo.csv files, grouped by dataset and method."
        )
    )
    parser.add_argument(
        "--cmp-topo",
        action="append",
        required=True,
        help=(
            "METHOD|DATASET=path/to/cmp_topo.csv (repeatable). "
            "You can also use METHOD@DATASET=... ."
        ),
    )
    parser.add_argument(
        "--outdir",
        default="figures/topology_boxplots",
        help="Output directory for plots and summary tables.",
    )
    parser.add_argument(
        "--box-width",
        type=float,
        default=0.55,
        help="Width of each box in seaborn boxplots. Lower values make boxes visually narrower.",
    )
    parser.add_argument(
        "--width-per-dataset",
        type=float,
        default=3.8,
        help="Figure width budget allocated to each dataset for single-metric boxplots.",
    )
    parser.add_argument(
        "--min-panel-width",
        type=float,
        default=10.0,
        help="Minimum width of each single-metric figure.",
    )
    parser.add_argument(
        "--combined-metric-width",
        type=float,
        default=22.0,
        help="Minimum width of the combined 3-panel topology figure.",
    )
    parser.add_argument(
        "--single-metric-height",
        type=float,
        default=5.0,
        help="Height of each single-metric figure.",
    )
    parser.add_argument(
        "--combined-metric-height",
        type=float,
        default=4.8,
        help="Height of each panel in the combined 3-panel topology figure.",
    )
    parser.add_argument(
        "--hide-fliers",
        action="store_true",
        help="Hide outlier markers so the box body is easier to read.",
    )
    parser.add_argument(
        "--upper-quantile",
        type=float,
        default=0.99,
        help=(
            "Set the upper y-limit of each metric panel to this quantile. "
            "Use 1.0 to keep the full range."
        ),
    )
    parser.add_argument(
        "--nrf-upper-quantile",
        type=float,
        default=None,
        help="Optional per-metric upper quantile override for nRF.",
    )
    parser.add_argument(
        "--wrf-upper-quantile",
        type=float,
        default=None,
        help="Optional per-metric upper quantile override for wRF.",
    )
    parser.add_argument(
        "--kf-upper-quantile",
        type=float,
        default=None,
        help="Optional per-metric upper quantile override for KF.",
    )
    parser.add_argument(
        "--nrf-only-method",
        action="append",
        default=[],
        help=(
            "METHOD label to keep only in the nRF panel and remove from wRF/KF. "
            "Repeat for multiple methods."
        ),
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inputs = parse_labelled_inputs(args.cmp_topo)
    wide_df = load_cmp_topo(inputs)
    long_df = make_long_table(wide_df, set(args.nrf_only_method))

    wide_df.to_csv(outdir / "topology_boxplot_raw.csv", index=False)
    long_df.to_csv(outdir / "topology_boxplot_long.csv", index=False)
    write_summary(wide_df, outdir)
    metric_upper_quantiles = {
        key: value
        for key, value in (
            ("nRF", args.nrf_upper_quantile),
            ("wRF", args.wrf_upper_quantile),
            ("KF", args.kf_upper_quantile),
        )
        if value is not None
    }
    plot_boxplots(
        long_df,
        outdir,
        box_width=args.box_width,
        width_per_dataset=args.width_per_dataset,
        min_panel_width=args.min_panel_width,
        combined_metric_width=args.combined_metric_width,
        single_metric_height=args.single_metric_height,
        combined_metric_height=args.combined_metric_height,
        hide_fliers=args.hide_fliers,
        upper_quantile=args.upper_quantile,
        metric_upper_quantiles=metric_upper_quantiles,
    )

    print(f"[INFO] wrote raw table to {outdir / 'topology_boxplot_raw.csv'}")
    print(f"[INFO] wrote summary to {outdir / 'topology_boxplot_summary.csv'}")
    print(f"[INFO] wrote plots to {outdir}")


if __name__ == "__main__":
    main()
