#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GPARTITION_DIR = REPO_ROOT / "tools" / "gpartition"
if str(GPARTITION_DIR) not in sys.path:
    sys.path.insert(0, str(GPARTITION_DIR))

from run_softbioblock_pf2 import (  # type: ignore
    build_feature_weights,
    fast_tiger_rates,
    infer_seqtype,
    load_alignment,
    preprocess_columns,
    seq_alphabet,
    site_feature_matrix,
    to_matrix,
    weighted_standardize,
)

DEFAULT_CASES = [
    (
        "pandit_over1gb__138",
        REPO_ROOT / "runs/benchmarks/pandit_over1gb",
        "138",
        "PANDIT over-1GB",
    ),
    (
        "cherry_over2gb__1029_100_tips_250",
        REPO_ROOT / "runs/benchmarks/cherry_over2gb",
        "1029_100_tips_250",
        "Cherry over-2GB",
    ),
    (
        "superaln_60k__rep_003",
        REPO_ROOT / "runs/benchmarks/superaln_16gb/60k",
        "rep_003",
        "Superalignment 60k",
    ),
]

METHOD_SPECS = [
    ("window-site", "partition_window_pos", "block_id"),
    ("window-rate", "partition_window_rate", "block_id"),
    ("softbioblock", "partition_softbioblock", "parent_regime_id"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot saved partition assignments for a few large benchmark cases."
    )
    parser.add_argument(
        "--outdir",
        default=str(REPO_ROOT / "figures/test_cluster"),
        help="Output directory.",
    )
    parser.add_argument(
        "--dot-size",
        type=float,
        default=8.0,
        help="Marker size for site dots.",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.95,
        help="Gap threshold used before computing feature space.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=6000,
        help="Maximum plotted points per case.",
    )
    parser.add_argument(
        "--max-feature-sites",
        type=int,
        default=8000,
        help="Maximum number of filtered sites used to build PCA for very long alignments.",
    )
    parser.add_argument(
        "--max-tsne-sites",
        type=int,
        default=1000,
        help="Maximum number of sites used for exact t-SNE.",
    )
    return parser.parse_args()


def load_case_meta(case_root: Path) -> dict:
    return json.loads((case_root / "metadata/pipeline.json").read_text(encoding="utf-8"))


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_site_labels(
    rows: list[dict[str, str]],
    label_field: str,
) -> tuple[dict[int, str], int]:
    site_to_labels: dict[int, list[str]] = {}
    for row in rows:
        label = row[label_field]
        cols = ast.literal_eval(row["original_columns_json"])
        for ci in cols:
            site_to_labels.setdefault(int(ci), []).append(label)
    site_to_primary = {ci: labels[0] for ci, labels in site_to_labels.items()}
    overlap_sites = sum(1 for labels in site_to_labels.values() if len(labels) > 1)
    return site_to_primary, overlap_sites


def palette_for_labels(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab20", max(1, len(uniq)))
    return {label: cmap(i) for i, label in enumerate(uniq)}


def pca_2d(X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0, 2), dtype=float)
    X0 = X - X.mean(axis=0, keepdims=True)
    u, s, _vh = np.linalg.svd(X0, full_matrices=False)
    coords = u[:, :2] * s[:2]
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=float)])
    return coords


def sample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points, dtype=int)


def subsample_sites(site_indices: list[int], limit: int) -> list[int]:
    if len(site_indices) <= limit:
        return list(site_indices)
    idx = np.linspace(0, len(site_indices) - 1, limit, dtype=int)
    return [site_indices[i] for i in idx]


def pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    sum_X = np.sum(np.square(X), axis=1, keepdims=True)
    D = sum_X + sum_X.T - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)
    return D


def conditional_probabilities(D: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    n = D.shape[0]
    P = np.zeros((n, n), dtype=float)
    log_perp = math.log(perplexity)
    for i in range(n):
        Di = np.concatenate([D[i, :i], D[i, i + 1 :]])
        beta = 1.0
        betamin = -np.inf
        betamax = np.inf
        for _ in range(50):
            Pi = np.exp(-Di * beta)
            sumPi = Pi.sum()
            if sumPi <= 0:
                Pi = np.full_like(Pi, 1.0 / len(Pi))
                H = 0.0
            else:
                Pi /= sumPi
                H = -np.sum(Pi * np.log(Pi + 1e-12))
            Hdiff = H - log_perp
            if abs(Hdiff) < 1e-5:
                break
            if Hdiff > 0:
                betamin = beta
                beta = beta * 2 if np.isinf(betamax) else 0.5 * (beta + betamax)
            else:
                betamax = beta
                beta = beta / 2 if np.isinf(betamin) else 0.5 * (beta + betamin)
        P[i, np.r_[0:i, i + 1 : n]] = Pi
    return P


def tsne_2d(
    X: np.ndarray,
    perplexity: float = 25.0,
    n_iter: int = 250,
    learning_rate: float = 180.0,
    seed: int = 0,
) -> np.ndarray:
    n = len(X)
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    if n == 1:
        return np.zeros((1, 2), dtype=float)
    base = pca_2d(X)
    if n <= 2:
        return base

    rng = np.random.default_rng(seed)
    Y = 1e-4 * rng.standard_normal((n, 2))
    Y += 1e-3 * base[:, :2]
    dY = np.zeros_like(Y)
    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)

    D = pairwise_sq_dists(X)
    P = conditional_probabilities(D, perplexity=min(perplexity, max(5.0, (n - 1) / 3.0)))
    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, 1e-12)
    P *= 4.0

    for it in range(n_iter):
        sum_Y = np.sum(np.square(Y), axis=1, keepdims=True)
        num = 1.0 / (1.0 + sum_Y + sum_Y.T - 2.0 * (Y @ Y.T))
        np.fill_diagonal(num, 0.0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        PQ = (P - Q) * num
        for i in range(n):
            dY[i] = 4.0 * np.sum((PQ[:, i][:, None]) * (Y[i] - Y), axis=0)

        momentum = 0.5 if it < 100 else 0.8
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains = np.maximum(gains, 0.01)
        iY = momentum * iY - learning_rate * (gains * dY)
        Y += iY
        Y -= Y.mean(axis=0, keepdims=True)

        if it == 100:
            P /= 4.0
    return Y


def plot_embedding(
    coords: np.ndarray,
    method_payloads: list[dict[str, object]],
    draw_idx: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    dot_size: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, payload in zip(axes, method_payloads):
        if len(coords):
            colors = payload["colors"]  # type: ignore[index]
            ax.scatter(
                coords[draw_idx, 0],
                coords[draw_idx, 1],
                c=[colors[i] for i in draw_idx],
                s=dot_size,
                linewidths=0.0,
                alpha=0.8,
                rasterized=True,
            )
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_title(
            f"{payload['label']}  |  groups={payload['n_groups']}  |  overlap-sites={payload['overlap_sites']}",
            fontsize=11,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_case(
    outdir: Path,
    case_key: str,
    benchmark_root: Path,
    case_id: str,
    benchmark_label: str,
    dot_size: float,
    gap_threshold: float,
    max_points: int,
    max_feature_sites: int,
    max_tsne_sites: int,
) -> dict[str, str | int]:
    soft_case_root = benchmark_root / "partition_softbioblock" / "cases" / case_id
    meta = load_case_meta(soft_case_root)
    aln_path = REPO_ROOT / meta["input_alignment"]

    ids, seqs = load_alignment(aln_path, "auto")
    seqtype = infer_seqtype(seqs, "auto")
    alphabet = seq_alphabet(seqtype)
    matrix = to_matrix(seqs)
    filtered, kept, _variant, _invariant, _gap_ratios = preprocess_columns(matrix, gap_threshold)
    pca_sites = subsample_sites(kept, max_feature_sites)
    rates = fast_tiger_rates(matrix, pca_sites, count_gap_gap=False)
    global_to_local = {ci: i for i, ci in enumerate(kept)}
    features, _stats = site_feature_matrix(filtered, rates, global_to_local, pca_sites, alphabet)
    X = weighted_standardize(features, build_feature_weights(alphabet))
    pca_coords = pca_2d(X)
    pca_draw_idx = sample_indices(len(pca_coords), max_points)

    tsne_sites = subsample_sites(pca_sites, max_tsne_sites)
    tsne_lookup = {ci: i for i, ci in enumerate(pca_sites)}
    tsne_rows = np.array([tsne_lookup[ci] for ci in tsne_sites], dtype=int)
    X_tsne = X[tsne_rows]
    tsne_coords = tsne_2d(X_tsne)
    tsne_draw_idx = sample_indices(len(tsne_coords), max_points)

    method_payloads = []
    for method_label, partition_dir, label_field in METHOD_SPECS:
        manifest_path = benchmark_root / partition_dir / "cases" / case_id / "block_manifest.csv"
        rows = read_manifest_rows(manifest_path)
        site_to_label, overlap_sites = build_site_labels(rows, label_field)
        labels = [site_to_label.get(ci, "__missing__") for ci in pca_sites]
        tsne_labels = [site_to_label.get(ci, "__missing__") for ci in tsne_sites]
        palette = palette_for_labels(labels)
        colors = [palette[label] for label in labels]
        tsne_colors = [palette[label] for label in tsne_labels]
        n_groups = len(set(labels)) - (1 if "__missing__" in labels else 0)
        method_payloads.append(
            {
                "label": method_label,
                "labels": labels,
                "colors": colors,
                "tsne_labels": tsne_labels,
                "tsne_colors": tsne_colors,
                "n_groups": n_groups,
                "overlap_sites": overlap_sites,
            }
        )

    summary: dict[str, str | int] = {
        "case_key": case_key,
        "benchmark": benchmark_label,
        "case_id": case_id,
        "input_alignment": str(aln_path.relative_to(REPO_ROOT)),
        "n_taxa": len(ids),
        "alignment_length": matrix.shape[1],
        "filtered_length": len(kept),
        "pca_sites": len(pca_sites),
        "tsne_sites": len(tsne_sites),
    }

    for payload in method_payloads:
        summary[f"{payload['label']}_groups"] = int(payload["n_groups"])
        summary[f"{payload['label']}_overlap_sites"] = int(payload["overlap_sites"])

    pca_out = outdir / f"{case_key}_pca.png"
    plot_embedding(
        pca_coords,
        method_payloads,
        pca_draw_idx,
        f"{benchmark_label}  |  case {case_id}  |  PCA  |  taxa={len(ids)}  sites={matrix.shape[1]}  filtered={len(kept)}",
        "PC1",
        "PC2",
        pca_out,
        dot_size,
    )
    tsne_payloads = []
    for payload in method_payloads:
        tsne_payloads.append(
            {
                "label": payload["label"],
                "colors": payload["tsne_colors"],
                "n_groups": payload["n_groups"],
                "overlap_sites": payload["overlap_sites"],
            }
        )
    tsne_out = outdir / f"{case_key}_tsne.png"
    plot_embedding(
        tsne_coords,
        tsne_payloads,
        tsne_draw_idx,
        f"{benchmark_label}  |  case {case_id}  |  t-SNE  |  taxa={len(ids)}  sites={matrix.shape[1]}  filtered={len(kept)}",
        "tSNE-1",
        "tSNE-2",
        tsne_out,
        dot_size,
    )
    summary["pca_figure"] = str(pca_out.relative_to(REPO_ROOT))
    summary["tsne_figure"] = str(tsne_out.relative_to(REPO_ROOT))
    return summary


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_key, benchmark_root, case_id, benchmark_label in DEFAULT_CASES:
        rows.append(
            plot_case(
                outdir=outdir,
                case_key=case_key,
                benchmark_root=benchmark_root,
                case_id=case_id,
                benchmark_label=benchmark_label,
                dot_size=args.dot_size,
                gap_threshold=args.gap_threshold,
                max_points=args.max_points,
                max_feature_sites=args.max_feature_sites,
                max_tsne_sites=args.max_tsne_sites,
            )
        )

    summary_path = outdir / "case_summary.csv"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} figures to {outdir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
