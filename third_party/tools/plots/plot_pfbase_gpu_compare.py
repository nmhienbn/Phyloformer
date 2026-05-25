#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


RENAMER = {
    "PF+FastME": "PF_Base+FastME",
    "PF_MRE+FastME": "PF+FastME",
}

METHOD_ORDER = ["PF+FastME", "PF_Base+FastME", "6GPU 20GB", "1 GPU 80GB"]
COLORS = {
    "PF+FastME": "#D62728",
    "PF_Base+FastME": "#BCBD22",
    "6GPU 20GB": "#1F77B4",
    "1 GPU 80GB": "#9467BD",
}
MARKERS = {
    "PF+FastME": "o",
    "PF_Base+FastME": "^",
    "6GPU 20GB": "D",
    "1 GPU 80GB": "s",
}
LINESTYLES = {
    "PF+FastME": "--",
    "PF_Base+FastME": "--",
    "6GPU 20GB": "-.",
    "1 GPU 80GB": ":",
}


@dataclass(frozen=True)
class MethodSpec:
    label: str
    topo_path: Path
    dist_path: Path
    source_run: str
    marker_filter: set[str] | None = None
    marker_renamer: dict[str, str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot PF1 final-test curves comparing paper PF baselines with "
            "pfbase_26GPU and pfbase_1GPU runs."
        )
    )
    parser.add_argument(
        "--tree-metrics-out",
        default="experiments/01_pf1_vram/results/pfbase_gpu_compare_tree_metrics.csv",
        help="Per-tree metric cache.",
    )
    parser.add_argument(
        "--summary-out",
        default="experiments/01_pf1_vram/results/pfbase_gpu_compare_summary.csv",
        help="Mean metric summary CSV.",
    )
    parser.add_argument(
        "--output",
        default="experiments/01_pf1_vram/results/pfbase_gpu_compare_base_vs_mre.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--pdf-output",
        default="experiments/01_pf1_vram/results/pfbase_gpu_compare_base_vs_mre.pdf",
        help="Optional output PDF path. Use empty string to disable.",
    )
    parser.add_argument(
        "--dist-chunksize",
        type=int,
        default=2_000_000,
        help="Rows per chunk when streaming large distance CSV files.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute per-tree metrics even if cache exists.",
    )
    return parser.parse_args()


def specs() -> list[MethodSpec]:
    return [
        MethodSpec(
            label="PF+FastME",
            topo_path=Path("data/topos_lggc.csv"),
            dist_path=Path("data/dists_lggc.csv"),
            source_run="data/topos_lggc.csv + data/dists_lggc.csv",
            marker_filter={"PF_MRE+FastME"},
            marker_renamer=RENAMER,
        ),
        MethodSpec(
            label="PF_Base+FastME",
            topo_path=Path("data/topos_lggc.csv"),
            dist_path=Path("data/dists_lggc.csv"),
            source_run="data/topos_lggc.csv + data/dists_lggc.csv",
            marker_filter={"PF+FastME"},
            marker_renamer=RENAMER,
        ),
        MethodSpec(
            label="6GPU 20GB",
            topo_path=Path("runs/pfbase_26GPU/eval_final_test_set_qsiam/cmp_qsiam_topo.csv"),
            dist_path=Path("runs/pfbase_26GPU/eval_final_test_set_qsiam/cmp_qsiam_dist.csv"),
            source_run="runs/pfbase_26GPU/eval_final_test_set_qsiam",
        ),
        MethodSpec(
            label="1 GPU 80GB",
            topo_path=Path("runs/pfbase_1GPU/eval_final_test_set/cmp_qsiam_topo.csv"),
            dist_path=Path("runs/pfbase_1GPU/eval_final_test_set/cmp_qsiam_dist.csv"),
            source_run="runs/pfbase_1GPU/eval_final_test_set",
        ),
    ]


def extract_n_tips(tree_id: str) -> int:
    match = re.search(r"_(\d+)_tips(?:_|$)", str(tree_id))
    if match:
        return int(match.group(1))
    parts = str(tree_id).split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    raise ValueError(f"Could not infer n_tips from id={tree_id!r}")


def extract_length(tree_id: str) -> int:
    tail = str(tree_id).split("_")[-1]
    if tail.isdigit():
        return int(tail)
    raise ValueError(f"Could not infer alignment length from id={tree_id!r}")


def normalize_marker(series: pd.Series, renamer: dict[str, str] | None) -> pd.Series:
    if renamer is None:
        return series
    return series.map(lambda value: renamer.get(value, value))


def load_topology(spec: MethodSpec) -> pd.DataFrame:
    if not spec.topo_path.exists():
        raise FileNotFoundError(f"Missing topology CSV for {spec.label}: {spec.topo_path}")

    df = pd.read_csv(
        spec.topo_path,
        usecols=lambda col: col
        in {"id", "rf", "norm_rf", "weighted_rf", "kf_score", "n_tips", "length", "marker"},
    )
    required = {"id", "rf", "norm_rf", "weighted_rf", "kf_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{spec.topo_path} missing columns: {sorted(missing)}")

    if spec.marker_filter is not None:
        if "marker" not in df.columns:
            raise ValueError(f"{spec.topo_path} has no marker column")
        df = df[df["marker"].isin(spec.marker_filter)].copy()
    else:
        df = df.copy()

    if df.empty:
        raise ValueError(f"No topology rows selected for {spec.label}")

    if "n_tips" not in df.columns:
        df["n_tips"] = df["id"].map(extract_n_tips)
    if "length" not in df.columns:
        df["length"] = df["id"].map(extract_length)
    df["label"] = spec.label
    df["source_run"] = spec.source_run
    return df.drop_duplicates(subset=["id"])[
        ["label", "source_run", "id", "n_tips", "length", "rf", "norm_rf", "weighted_rf", "kf_score"]
    ]


def distance_tree_metrics(
    spec: MethodSpec,
    id_to_n_tips: dict[str, int],
    id_to_length: dict[str, int],
    chunksize: int,
) -> pd.DataFrame:
    if not spec.dist_path.exists():
        raise FileNotFoundError(f"Missing distance CSV for {spec.label}: {spec.dist_path}")

    grouped_chunks = []
    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            spec.dist_path,
            usecols=["id", "ref_dist", "cmp_dist", "marker"],
            chunksize=max(int(chunksize), 1),
        ),
        start=1,
    ):
        if spec.marker_filter is not None:
            chunk = chunk[chunk["marker"].isin(spec.marker_filter)].copy()
            if chunk.empty:
                continue
        else:
            chunk = chunk.copy()

        ref = pd.to_numeric(chunk["ref_dist"], errors="coerce")
        cmp = pd.to_numeric(chunk["cmp_dist"], errors="coerce")
        abs_err = (ref - cmp).abs()
        chunk["MAE"] = abs_err
        chunk["MRE"] = np.where(ref > 0, abs_err / ref, np.nan)

        grouped = chunk.groupby("id", as_index=False).agg(
            MAE_sum=("MAE", "sum"),
            MAE_count=("MAE", "count"),
            MRE_sum=("MRE", "sum"),
            MRE_count=("MRE", "count"),
        )
        grouped_chunks.append(grouped)
        if chunk_idx % 25 == 0:
            print(f"[dist] {spec.label}: processed {chunk_idx} chunks")

    if not grouped_chunks:
        raise ValueError(f"No distance rows selected for {spec.label}")

    dist = pd.concat(grouped_chunks, ignore_index=True).groupby("id", as_index=False).sum()
    dist["MAE"] = dist["MAE_sum"] / dist["MAE_count"]
    dist["MRE"] = dist["MRE_sum"] / dist["MRE_count"]
    dist["n_tips"] = dist["id"].map(id_to_n_tips)
    dist["length"] = dist["id"].map(id_to_length)
    dist["label"] = spec.label
    return dist[["label", "id", "n_tips", "length", "MAE", "MRE"]]


def build_method_metrics(spec: MethodSpec, chunksize: int) -> pd.DataFrame:
    print(f"[topo] {spec.label}: {spec.topo_path}")
    topo = load_topology(spec)
    id_to_n_tips = topo.set_index("id")["n_tips"].to_dict()
    id_to_length = topo.set_index("id")["length"].to_dict()

    print(f"[dist] {spec.label}: {spec.dist_path}")
    dist = distance_tree_metrics(spec, id_to_n_tips, id_to_length, chunksize)
    metrics = topo.merge(dist[["id", "MAE", "MRE"]], on="id", how="inner")
    if metrics.empty:
        raise ValueError(f"No shared tree ids for {spec.label}")
    return metrics


def load_or_build_metrics(args: argparse.Namespace) -> pd.DataFrame:
    cache_path = Path(args.tree_metrics_out)
    if cache_path.exists() and not args.recompute:
        print(f"[cache] reused {cache_path}")
        return pd.read_csv(cache_path)

    metrics = pd.concat(
        [build_method_metrics(spec, args.dist_chunksize) for spec in specs()],
        ignore_index=True,
    )
    metrics["label"] = pd.Categorical(metrics["label"], METHOD_ORDER, ordered=True)
    metrics = metrics.sort_values(["label", "n_tips", "length", "id"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(cache_path, index=False, float_format="%.6f")
    print(f"[cache] wrote {cache_path}")
    return metrics


def write_summary(metrics: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    summary = (
        metrics.groupby(["label", "source_run"], observed=True)
        .agg(
            n_alignments=("id", "nunique"),
            mean_rf=("rf", "mean"),
            mean_nrf=("norm_rf", "mean"),
            mean_wrf=("weighted_rf", "mean"),
            mean_kf=("kf_score", "mean"),
            mean_mae=("MAE", "mean"),
            mean_mre=("MRE", "mean"),
        )
        .reset_index()
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"[summary] wrote {summary_path}")
    return summary


def plot_line(ax, data: pd.DataFrame, label: str, yvar: str) -> None:
    sub = data[data["label"] == label]
    if sub.empty:
        return
    by_tips = sub.groupby("n_tips", as_index=False)[yvar].mean()
    ax.plot(
        by_tips["n_tips"],
        by_tips[yvar],
        color=COLORS[label],
        marker=MARKERS[label],
        linestyle=LINESTYLES[label],
        linewidth=1.8,
        markersize=5,
        label=label,
    )


def plot_base_vs_mre(metrics: pd.DataFrame, output: Path, pdf_output: Path | None) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(
        {
            "axes.facecolor": "#EAEAF2",
            "grid.color": "white",
            "grid.linewidth": 1.2,
            "font.size": 10,
        }
    )

    fig = plt.figure(layout="constrained", figsize=(9.5, 7.0))
    gs = GridSpec(7, 6, figure=fig, top=0.92, bottom=0.06, right=0.98, left=0.08)

    kf_ax = fig.add_subplot(gs[:3, :3])
    rf_ax = fig.add_subplot(gs[:3, -3:])
    mae_ax = fig.add_subplot(gs[3:-1, :3])
    mre_ax = fig.add_subplot(gs[3:-1, -3:])
    legend_ax = fig.add_subplot(gs[-1, :])

    axes = [kf_ax, rf_ax, mae_ax, mre_ax]
    specs_for_axes = [
        (kf_ax, "kf_score", "Kuhner-Felsenstein distance"),
        (rf_ax, "norm_rf", "Normalized Robinson-Foulds distance"),
        (mae_ax, "MAE", "Mean Absolute Error"),
        (mre_ax, "MRE", "Mean Relative Error"),
    ]
    for ax, yvar, ylabel in specs_for_axes:
        for label in METHOD_ORDER:
            plot_line(ax, metrics, label, yvar)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(10, 110, 10))
        ax.grid(which="major", axis="both", color="white", linewidth=1.2)

    for ax in [kf_ax, rf_ax]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")
    for ax in [mae_ax, mre_ax]:
        ax.set_xlabel("Number of leaves")

    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = kf_ax.get_legend_handles_labels()
    legend_ax.set_axis_off()
    legend_ax.legend(handles, labels, loc="center", ncol=4)
    fig.suptitle("PF1 final test set: GPU-memory runs vs PF baselines", fontsize=13)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(f"[plot] wrote {output}")
    if pdf_output is not None:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output)
        print(f"[plot] wrote {pdf_output}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics = load_or_build_metrics(args)
    write_summary(metrics, Path(args.summary_out))
    pdf_output = Path(args.pdf_output) if args.pdf_output else None
    plot_base_vs_mre(metrics, Path(args.output), pdf_output)


if __name__ == "__main__":
    main()
