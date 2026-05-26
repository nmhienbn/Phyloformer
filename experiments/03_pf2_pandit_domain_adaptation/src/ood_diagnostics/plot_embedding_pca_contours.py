#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.stats import gaussian_kde  # noqa: E402


DEFAULT_STAGE3 = Path("runs/pandit_domain_adaptation/stage3_evopf_embeddings")
DEFAULT_STAGE7 = Path("runs/pandit_domain_adaptation/stage7_postft_evopf_embeddings")


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    manifest: Path


@dataclass(frozen=True)
class LoadedDataset:
    label: str
    rows: list[dict[str, str]]
    matrix: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EvoPF embedding PCA as KDE contour subplots for the PANDIT adaptation experiment."
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional extra output directory. By default, write only next to the existing scatter figures.",
    )
    parser.add_argument(
        "--training-fixed-manifest",
        default=str(DEFAULT_STAGE3 / "synthetic_pf2_native_fixed" / "embedding_manifest.tsv"),
        help="Fixed PF2-native training manifest used in figure 1.",
    )
    parser.add_argument(
        "--training-size-matched-manifest",
        default=str(DEFAULT_STAGE3 / "synthetic_pf2_native_size_matched" / "embedding_manifest.tsv"),
        help="Size-matched PF2-native training manifest used in figure 1.",
    )
    parser.add_argument(
        "--simulated-pandit-like-manifest",
        default=str(DEFAULT_STAGE7 / "synthetic_pandit_like_finetune" / "embedding_manifest.tsv"),
        help="Manifest used as the simulated PANDIT-like density in figure 2.",
    )
    parser.add_argument(
        "--pandit-stage3-manifest",
        action="append",
        default=None,
        help="PANDIT manifest for figure 1. Repeat for train/test; all are merged.",
    )
    parser.add_argument(
        "--pandit-stage7-manifest",
        action="append",
        default=None,
        help="PANDIT manifest for figure 2. Repeat for train/test; all are merged.",
    )
    parser.add_argument("--max-per-label", type=int, default=20000, help="Deterministic cap per density label.")
    parser.add_argument("--grid-size", type=int, default=180)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--also-write-scatter-dirs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write contour PNG/PDF files into result directories that contain the older scatter plots.",
    )
    args = parser.parse_args()
    if args.pandit_stage3_manifest is None:
        args.pandit_stage3_manifest = [
            str(DEFAULT_STAGE3 / "pandit_train" / "embedding_manifest.tsv"),
            str(DEFAULT_STAGE3 / "pandit_test" / "embedding_manifest.tsv"),
        ]
    if args.pandit_stage7_manifest is None:
        args.pandit_stage7_manifest = [
            str(DEFAULT_STAGE7 / "pandit_train" / "embedding_manifest.tsv"),
            str(DEFAULT_STAGE7 / "pandit_test" / "embedding_manifest.tsv"),
        ]
    return args


def load_manifest(label: str, manifest: Path) -> LoadedDataset:
    rows: list[dict[str, str]] = []
    vectors: list[np.ndarray] = []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("status") != "done":
                continue
            path = Path(row["pooled_embedding"])
            rows.append({**row, "density_label": label, "source_manifest": str(manifest)})
            vectors.append(np.load(path).reshape(-1))
    if not vectors:
        raise ValueError(f"No successful embeddings in {manifest}")
    return LoadedDataset(label=label, rows=rows, matrix=np.stack(vectors))


def merge_datasets(label: str, manifests: list[Path]) -> LoadedDataset:
    datasets = [load_manifest(label, manifest) for manifest in manifests]
    rows: list[dict[str, str]] = []
    matrices: list[np.ndarray] = []
    for dataset in datasets:
        rows.extend(dataset.rows)
        matrices.append(dataset.matrix)
    return LoadedDataset(label=label, rows=rows, matrix=np.concatenate(matrices, axis=0))


def deterministic_cap(dataset: LoadedDataset, max_rows: int) -> LoadedDataset:
    if max_rows <= 0 or len(dataset.rows) <= max_rows:
        return dataset
    indices = np.linspace(0, len(dataset.rows) - 1, max_rows, dtype=int)
    rows = [dataset.rows[int(index)] for index in indices]
    matrix = dataset.matrix[indices]
    return LoadedDataset(label=dataset.label, rows=rows, matrix=matrix)


def pca_two_components(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    explained = (singular_values[:2] ** 2) / np.sum(singular_values**2)
    return centered @ components.T, components, explained


def write_coords(out_tsv: Path, datasets: list[LoadedDataset], coords_by_label: dict[str, np.ndarray]) -> None:
    fieldnames = [
        "density_label",
        "split",
        "msa_id",
        "pc1",
        "pc2",
        "source_manifest",
        "pooled_embedding",
        "num_sequences",
        "alignment_length",
    ]
    with out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for dataset in datasets:
            coords = coords_by_label[dataset.label]
            for row, coord in zip(dataset.rows, coords):
                writer.writerow({**row, "pc1": f"{coord[0]:.10g}", "pc2": f"{coord[1]:.10g}"})


def density_grid(coords_by_label: dict[str, np.ndarray], grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    all_coords = np.concatenate(list(coords_by_label.values()), axis=0)
    low = all_coords.min(axis=0)
    high = all_coords.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    pad = span * 0.45
    x = np.linspace(low[0] - pad[0], high[0] + pad[0], grid_size)
    y = np.linspace(low[1] - pad[1], high[1] + pad[1], grid_size)
    return np.meshgrid(x, y)


def kde_values(coords: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    kde = gaussian_kde(coords.T)
    grid = np.vstack([xx.ravel(), yy.ravel()])
    return kde(grid).reshape(xx.shape)


def positive_levels(values: np.ndarray, level_count: int) -> np.ndarray:
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        raise ValueError("KDE produced no positive density values")
    max_density = float(positive.max())
    levels = np.linspace(max_density * 0.04, max_density * 0.95, level_count)
    if levels.size < 2:
        levels = np.linspace(float(positive.min()), float(positive.max()), level_count)
    return levels


def save_figure(fig: plt.Figure, out_base: Path, dpi: int) -> list[str]:
    written: list[str] = []
    for suffix in (".png", ".pdf"):
        path = out_base.with_suffix(suffix)
        fig.savefig(path, dpi=dpi)
        written.append(str(path))
    return written


def plot_contour_subplots(
    out_base: Path,
    title: str,
    coords_by_label: dict[str, np.ndarray],
    colors: dict[str, str],
    cmaps: dict[str, str],
    grid_size: int,
    level_count: int,
    dpi: int,
) -> list[str]:
    xx, yy = density_grid(coords_by_label, grid_size)
    ncols = len(coords_by_label)
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 3.9), sharex=True, sharey=True)
    if ncols == 1:
        axes = np.array([axes])

    values_by_label: dict[str, np.ndarray] = {}
    levels_by_label: dict[str, np.ndarray] = {}
    active_mask = np.zeros(xx.shape, dtype=bool)
    for label, coords in coords_by_label.items():
        values = kde_values(coords, xx, yy)
        levels = positive_levels(values, level_count)
        values_by_label[label] = values
        levels_by_label[label] = levels
        active_mask |= values >= levels[0]

    if active_mask.any():
        active_x = xx[active_mask]
        active_y = yy[active_mask]
        x_low, x_high = float(active_x.min()), float(active_x.max())
        y_low, y_high = float(active_y.min()), float(active_y.max())
        x_pad = max((x_high - x_low) * 0.12, 1e-6)
        y_pad = max((y_high - y_low) * 0.12, 1e-6)
        xlim = (x_low - x_pad, x_high + x_pad)
        ylim = (y_low - y_pad, y_high + y_pad)
    else:
        xlim = (float(xx.min()), float(xx.max()))
        ylim = (float(yy.min()), float(yy.max()))

    for ax, (label, coords) in zip(axes, coords_by_label.items()):
        ax.set_facecolor("#eef0f6")
        values = values_by_label[label]
        levels = levels_by_label[label]
        ax.contourf(xx, yy, values, levels=levels, cmap=cmaps[label], alpha=0.62)
        ax.contour(xx, yy, values, levels=levels, colors=colors[label], linewidths=0.75, alpha=0.82)
        ax.set_title(f"{label}\n(n={len(coords):,})", fontsize=10, weight="bold", pad=8)
        ax.set_xlabel("PC1")
        ax.grid(color="white", linewidth=0.9)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    axes[0].set_ylabel("PC2")
    fig.suptitle(title, fontsize=12, weight="bold", y=0.985)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.16, top=0.74, wspace=0.08)
    written = save_figure(fig, out_base, dpi)
    plt.close(fig)
    return written


def make_figure(
    outdir: Path,
    stem: str,
    title: str,
    datasets: list[LoadedDataset],
    max_per_label: int,
    grid_size: int,
    level_count: int,
    dpi: int,
) -> dict[str, object]:
    capped = [deterministic_cap(dataset, max_per_label) for dataset in datasets]
    matrix = np.concatenate([dataset.matrix for dataset in capped], axis=0)
    coords, _, explained = pca_two_components(matrix)

    coords_by_label: dict[str, np.ndarray] = {}
    offset = 0
    for dataset in capped:
        end = offset + len(dataset.rows)
        coords_by_label[dataset.label] = coords[offset:end]
        offset = end

    palette = [
        ("#1f77b4", "Blues"),
        ("#2ca02c", "Greens"),
        ("#d62728", "Reds"),
        ("#9467bd", "Purples"),
    ]
    colors = {dataset.label: palette[index % len(palette)][0] for index, dataset in enumerate(capped)}
    cmaps = {dataset.label: palette[index % len(palette)][1] for index, dataset in enumerate(capped)}
    out_base = outdir / stem
    out_tsv = outdir / f"{stem}_coords.tsv"
    figures = plot_contour_subplots(out_base, title, coords_by_label, colors, cmaps, grid_size, level_count, dpi)
    write_coords(out_tsv, capped, coords_by_label)
    return {
        "figures": figures,
        "coords": str(out_tsv),
        "explained_variance_pc1": float(explained[0]),
        "explained_variance_pc2": float(explained[1]),
        "counts": {dataset.label: len(dataset.rows) for dataset in capped},
    }


def copy_summary_figure(
    outdir: Path,
    stem: str,
    title: str,
    datasets: list[LoadedDataset],
    max_per_label: int,
    grid_size: int,
    level_count: int,
    dpi: int,
) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    result = make_figure(outdir, stem, title, datasets, max_per_label, grid_size, level_count, dpi)
    coords_path = Path(result["coords"])
    if coords_path.exists():
        coords_path.unlink()
    result.pop("coords", None)
    return result


def main() -> None:
    args = parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.cpu_threads)

    training_fixed = load_manifest("Training fixed", Path(args.training_fixed_manifest))
    training_size_matched = load_manifest("Training size-matched", Path(args.training_size_matched_manifest))
    pandit_stage3 = merge_datasets("PANDIT data", [Path(item) for item in args.pandit_stage3_manifest])
    simulated = load_manifest("Simulated PANDIT-like", Path(args.simulated_pandit_like_manifest))
    pandit_stage7 = merge_datasets("PANDIT data", [Path(item) for item in args.pandit_stage7_manifest])

    summary: dict[str, object] = {}
    if args.outdir is not None:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        summary["training_vs_pandit"] = make_figure(
            outdir,
            "pca_contour_training_vs_pandit",
            "Training Data vs PANDIT Data",
            [training_fixed, training_size_matched, pandit_stage3],
            args.max_per_label,
            args.grid_size,
            args.levels,
            args.dpi,
        )
        summary["simulated_pandit_like_vs_pandit"] = make_figure(
            outdir,
            "pca_contour_simulated_pandit_like_vs_pandit",
            "Simulated PANDIT-like vs PANDIT Data",
            [simulated, pandit_stage7],
            args.max_per_label,
            args.grid_size,
            args.levels,
            args.dpi,
        )
        with (outdir / "pca_contour_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if args.also_write_scatter_dirs:
        paper_dir = Path("experiments/03_pf2_pandit_domain_adaptation/results/00_paper_figures")
        fixed_dir = Path("experiments/03_pf2_pandit_domain_adaptation/results/05_ood_pca/fixed")
        size_matched_dir = Path("experiments/03_pf2_pandit_domain_adaptation/results/05_ood_pca/size_matched")
        summary["scatter_dirs"] = {
            "paper_training_vs_pandit": copy_summary_figure(
                paper_dir,
                "pca_contour_training_vs_pandit",
                "Training Data vs PANDIT Data",
                [training_fixed, training_size_matched, pandit_stage3],
                args.max_per_label,
                args.grid_size,
                args.levels,
                args.dpi,
            ),
            "paper_simulated_pandit_like_vs_pandit": copy_summary_figure(
                paper_dir,
                "pca_contour_simulated_pandit_like_vs_pandit",
                "Simulated PANDIT-like vs PANDIT Data",
                [simulated, pandit_stage7],
                args.max_per_label,
                args.grid_size,
                args.levels,
                args.dpi,
            ),
            "ood_fixed": copy_summary_figure(
                fixed_dir,
                "embedding_pca_contour",
                "Fixed Training Data vs PANDIT Data",
                [training_fixed, pandit_stage3],
                args.max_per_label,
                args.grid_size,
                args.levels,
                args.dpi,
            ),
            "ood_size_matched": copy_summary_figure(
                size_matched_dir,
                "embedding_pca_contour",
                "Size-matched Training Data vs PANDIT Data",
                [training_size_matched, pandit_stage3],
                args.max_per_label,
                args.grid_size,
                args.levels,
                args.dpi,
            ),
        }
        with (paper_dir / "pca_contour_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def print_figures(value: object) -> None:
        if not isinstance(value, dict):
            return
        figures = value.get("figures")
        if isinstance(figures, list):
            for figure in figures:
                print(f"[OK] wrote {figure}")

    for value in summary.values():
        if isinstance(value, dict) and "figures" in value:
            print_figures(value)
    scatter_dirs = summary.get("scatter_dirs")
    if isinstance(scatter_dirs, dict):
        for value in scatter_dirs.values():
            print_figures(value)


if __name__ == "__main__":
    main()
