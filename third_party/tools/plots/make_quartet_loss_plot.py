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


@dataclass(frozen=True)
class ResultSpec:
    loss: str
    label: str
    source_run: str
    topo_path: Path
    dist_path: Path


METRICS = [
    ("norm_rf", "Normalized Robinson-Foulds distance"),
    ("weighted_rf", "Weighted Robinson-Foulds distance"),
    ("kf_score", "Kuhner-Felsenstein distance"),
    ("MAE", "Mean Absolute Error"),
    ("MRE", "Mean Relative Error"),
]

METHOD_ORDER = ["PFBase", "Close", "Push", "Combined"]
PALETTE = {
    "PFBase": "#4C78A8",
    "Close": "#F58518",
    "Push": "#54A24B",
    "Combined": "#B279A2",
}
MARKERS = {
    "PFBase": "o",
    "Close": "D",
    "Push": "s",
    "Combined": "^",
}
DASHES = {
    "PFBase": "",
    "Close": (4, 2),
    "Push": (2, 2),
    "Combined": (6, 2),
}


def default_specs() -> list[ResultSpec]:
    return [
        ResultSpec(
            loss="pf_base",
            label="PFBase",
            source_run="data/final_test_set/results",
            topo_path=Path("data/final_test_set/results/PF+FastME_topo.csv"),
            dist_path=Path("data/final_test_set/results/PF+FastME_dist.csv"),
        ),
        ResultSpec(
            loss="quartet_close",
            label="Close",
            source_run="experiments/02_pf1_quartet_loss/results/quartet_close/eval_final_test_set_qsiam",
            topo_path=Path(
                "experiments/02_pf1_quartet_loss/results/quartet_close/"
                "eval_final_test_set_qsiam/cmp_qsiam_topo.csv"
            ),
            dist_path=Path(
                "experiments/02_pf1_quartet_loss/results/quartet_close/"
                "eval_final_test_set_qsiam/cmp_qsiam_dist.csv"
            ),
        ),
        ResultSpec(
            loss="quartet_push",
            label="Push",
            source_run="experiments/02_pf1_quartet_loss/results/quartet_push/eval_final_test_set",
            topo_path=Path(
                "experiments/02_pf1_quartet_loss/results/quartet_push/"
                "eval_final_test_set/cmp_pf1_topo.csv"
            ),
            dist_path=Path(
                "experiments/02_pf1_quartet_loss/results/quartet_push/"
                "eval_final_test_set/cmp_pf1_dist.csv"
            ),
        ),
        ResultSpec(
            loss="quartet_combined",
            label="Combined",
            source_run="experiments/02_pf1_quartet_loss/results/quartet_combined/eval_final_test_set_qsiam",
            topo_path=Path(
                "experiments/02_pf1_quartet_loss/results/quartet_combined/"
                "eval_final_test_set_qsiam/cmp_qsiam_topo.csv"
            ),
            dist_path=Path(
                "experiments/02_pf1_quartet_loss/results/quartet_combined/"
                "eval_final_test_set_qsiam/cmp_qsiam_dist.csv"
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PF1 quartet-loss final-test curves against PFBase."
    )
    parser.add_argument(
        "--tree-metrics-out",
        default="experiments/02_pf1_quartet_loss/results/final_test_tree_metrics.csv",
        help="Per-tree metrics cache used for plotting.",
    )
    parser.add_argument(
        "--summary-out",
        default="experiments/02_pf1_quartet_loss/results/final_test_summary.csv",
        help="Mean metric summary CSV.",
    )
    parser.add_argument(
        "--output",
        default="experiments/02_pf1_quartet_loss/results/quartet_loss_metrics.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--pdf-output",
        default="experiments/02_pf1_quartet_loss/results/quartet_loss_metrics.pdf",
        help="Optional PDF output path. Use empty string to disable.",
    )
    parser.add_argument(
        "--dist-chunksize",
        type=int,
        default=1_000_000,
        help="Rows per chunk when reading large cmp_dist CSV files.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute per-tree metrics even if --tree-metrics-out exists.",
    )
    return parser.parse_args()


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


def load_topology(spec: ResultSpec) -> pd.DataFrame:
    if not spec.topo_path.exists():
        raise FileNotFoundError(f"Missing topology CSV for {spec.label}: {spec.topo_path}")
    df = pd.read_csv(
        spec.topo_path,
        usecols=lambda col: col
        in {"id", "rf", "norm_rf", "weighted_rf", "kf_score", "n_tips", "length"},
    )
    required = {"id", "rf", "norm_rf", "weighted_rf", "kf_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{spec.topo_path} missing columns: {sorted(missing)}")

    df = df.copy()
    if "n_tips" not in df.columns:
        df["n_tips"] = df["id"].map(extract_n_tips)
    if "length" not in df.columns:
        df["length"] = df["id"].map(extract_length)
    return df.drop_duplicates(subset=["id"])


def load_distance_tree_metrics(
    spec: ResultSpec,
    id_to_n_tips: dict[str, int],
    id_to_length: dict[str, int],
    chunksize: int,
) -> pd.DataFrame:
    if not spec.dist_path.exists():
        raise FileNotFoundError(f"Missing distance CSV for {spec.label}: {spec.dist_path}")

    grouped_chunks = []
    for chunk in pd.read_csv(
        spec.dist_path,
        usecols=["id", "ref_dist", "cmp_dist"],
        chunksize=max(int(chunksize), 1),
    ):
        ref = pd.to_numeric(chunk["ref_dist"], errors="coerce")
        cmp = pd.to_numeric(chunk["cmp_dist"], errors="coerce")
        chunk = chunk.assign(
            MAE=(ref - cmp).abs(),
            MRE=np.where(ref > 0, (ref - cmp).abs() / ref, np.nan),
        )
        grouped = chunk.groupby("id", as_index=False).agg(
            MAE_sum=("MAE", "sum"),
            MAE_count=("MAE", "count"),
            MRE_sum=("MRE", "sum"),
            MRE_count=("MRE", "count"),
        )
        grouped_chunks.append(grouped)

    if not grouped_chunks:
        raise ValueError(f"No distance rows found in {spec.dist_path}")

    dist = pd.concat(grouped_chunks, ignore_index=True).groupby("id", as_index=False).sum()
    dist["MAE"] = dist["MAE_sum"] / dist["MAE_count"]
    dist["MRE"] = dist["MRE_sum"] / dist["MRE_count"]
    dist["n_tips"] = dist["id"].map(id_to_n_tips)
    dist["length"] = dist["id"].map(id_to_length)
    return dist[["id", "n_tips", "length", "MAE", "MRE"]]


def build_metrics_for_spec(spec: ResultSpec, chunksize: int) -> pd.DataFrame:
    print(f"[metrics] {spec.label}: topology {spec.topo_path}")
    topo = load_topology(spec)
    id_to_n_tips = topo.set_index("id")["n_tips"].to_dict()
    id_to_length = topo.set_index("id")["length"].to_dict()

    print(f"[metrics] {spec.label}: distances {spec.dist_path}")
    dist = load_distance_tree_metrics(spec, id_to_n_tips, id_to_length, chunksize)
    merged = topo.merge(dist[["id", "MAE", "MRE"]], on="id", how="inner")
    if merged.empty:
        raise ValueError(f"No shared tree ids between {spec.topo_path} and {spec.dist_path}")

    merged["loss"] = spec.loss
    merged["label"] = spec.label
    merged["source_run"] = spec.source_run
    return merged[
        [
            "loss",
            "label",
            "source_run",
            "id",
            "n_tips",
            "length",
            "rf",
            "norm_rf",
            "weighted_rf",
            "kf_score",
            "MAE",
            "MRE",
        ]
    ]


def load_or_build_tree_metrics(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.tree_metrics_out)
    if path.exists() and not args.recompute:
        print(f"[metrics] reused {path}")
        return pd.read_csv(path)

    metrics = pd.concat(
        [build_metrics_for_spec(spec, args.dist_chunksize) for spec in default_specs()],
        ignore_index=True,
    )
    metrics["label"] = pd.Categorical(metrics["label"], METHOD_ORDER, ordered=True)
    metrics = metrics.sort_values(["label", "n_tips", "length", "id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False, float_format="%.6f")
    print(f"[metrics] wrote {path}")
    return metrics


def write_summary(tree_metrics: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    summary = (
        tree_metrics.groupby(["loss", "label", "source_run"], observed=True)
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
    summary["status"] = "complete"
    summary["raw_dist_copied"] = summary["source_run"].str.startswith("experiments/")
    summary = summary[
        [
            "loss",
            "label",
            "status",
            "source_run",
            "n_alignments",
            "mean_rf",
            "mean_nrf",
            "mean_wrf",
            "mean_kf",
            "mean_mae",
            "mean_mre",
            "raw_dist_copied",
        ]
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"[summary] wrote {summary_path}")
    return summary


def plot_metric(ax, data: pd.DataFrame, metric: str, ylabel: str) -> None:
    for label in METHOD_ORDER:
        sub = data[data["label"] == label]
        if sub.empty:
            continue
        mean_by_tips = sub.groupby("n_tips", as_index=False)[metric].mean()
        linestyle = "-" if DASHES[label] == "" else (0, DASHES[label])
        ax.plot(
            mean_by_tips["n_tips"],
            mean_by_tips[metric],
            color=PALETTE[label],
            linestyle=linestyle,
            marker=MARKERS[label],
            label=label,
            linewidth=1.8,
            markersize=5,
        )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(10, 110, 10))
    ax.grid(which="major", axis="both", color="white", linewidth=1.2)
    ax.legend().remove()


def plot_final_test(tree_metrics: pd.DataFrame, output: Path, pdf_output: Path | None) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(
        {
            "axes.facecolor": "#EAEAF2",
            "grid.color": "white",
            "grid.linewidth": 1.2,
            "font.size": 10,
        }
    )

    data = tree_metrics.copy()
    data["label"] = pd.Categorical(data["label"], METHOD_ORDER, ordered=True)
    data = data.sort_values(["label", "n_tips", "length", "id"])

    fig = plt.figure(layout="constrained", figsize=(12, 7.2))
    gs = GridSpec(7, 6, figure=fig, top=0.92, bottom=0.06, right=0.98, left=0.07)

    axes = [
        fig.add_subplot(gs[:3, :2]),
        fig.add_subplot(gs[:3, 2:4]),
        fig.add_subplot(gs[:3, 4:6]),
        fig.add_subplot(gs[3:6, :3]),
        fig.add_subplot(gs[3:6, 3:6]),
    ]
    legend_ax = fig.add_subplot(gs[-1, :])

    for ax, (metric, ylabel) in zip(axes, METRICS):
        plot_metric(ax, data, metric, ylabel)

    for ax in axes[:3]:
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in axes[3:]:
        ax.set_xlabel("Number of leaves")

    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.set_axis_off()
    legend_ax.legend(handles, labels, loc="center", ncol=len(METHOD_ORDER))
    fig.suptitle("PF1 final test set: PFBase vs quartet losses", fontsize=14)

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
    tree_metrics = load_or_build_tree_metrics(args)
    write_summary(tree_metrics, Path(args.summary_out))
    pdf_output = Path(args.pdf_output) if args.pdf_output else None
    plot_final_test(tree_metrics, Path(args.output), pdf_output)


if __name__ == "__main__":
    main()
