#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA/OOD diagnostics on pooled EvoPF embeddings.")
    parser.add_argument("--embedding-manifest", action="append", required=True, help="LABEL=embedding_manifest.tsv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--variance", type=float, default=0.95)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def labelled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=path, got {value!r}")
    label, path = value.split("=", 1)
    return label, Path(path)


def load_embeddings(label: str, manifest: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    rows: list[dict[str, str]] = []
    vectors: list[np.ndarray] = []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["status"] != "done":
                continue
            rows.append({**row, "label": label})
            vectors.append(np.load(row["pooled_embedding"]))
    if not vectors:
        raise ValueError(f"No successful embeddings in {manifest}")
    return rows, np.stack(vectors)


def write_reconstruction_errors(outdir: Path, rows: list[dict[str, str]], errors: np.ndarray) -> None:
    with (outdir / "embedding_pca_recon_errors.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "split", "msa_id", "recon_mse"], delimiter="\t")
        writer.writeheader()
        for row, error in zip(rows, errors):
            writer.writerow(
                {
                    "label": row["label"],
                    "split": row.get("split", ""),
                    "msa_id": row["msa_id"],
                    "recon_mse": f"{float(error):.12g}",
                }
            )


def plot_outputs(outdir: Path, rows: list[dict[str, str]], matrix: np.ndarray, errors: np.ndarray) -> None:
    labels = sorted({row["label"] for row in rows})

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [[float(error) for row, error in zip(rows, errors) if row["label"] == label] for label in labels],
        labels=labels,
        showfliers=False,
    )
    plt.yscale("log")
    plt.ylabel("PCA reconstruction MSE")
    plt.tight_layout()
    plt.savefig(outdir / "embedding_pca_recon_error.png", dpi=200)
    plt.close()

    coords = PCA(n_components=2).fit_transform(matrix)
    plt.figure(figsize=(7, 6))
    for label in labels:
        indices = [idx for idx, row in enumerate(rows) if row["label"] == label]
        plt.scatter(coords[indices, 0], coords[indices, 1], s=8, alpha=0.55, label=label)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(outdir / "embedding_pca_scatter.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.cpu_threads)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    matrices: list[np.ndarray] = []
    train_matrix: np.ndarray | None = None
    for item in args.embedding_manifest:
        label, path = labelled_path(item)
        item_rows, matrix = load_embeddings(label, path)
        rows.extend(item_rows)
        matrices.append(matrix)
        if label == args.train_label:
            train_matrix = matrix

    if train_matrix is None:
        raise ValueError(f"No manifest had train label {args.train_label!r}")

    matrix = np.concatenate(matrices, axis=0)
    pca = PCA(n_components=args.variance, svd_solver="full").fit(train_matrix)
    errors = ((matrix - pca.inverse_transform(pca.transform(matrix))) ** 2).mean(axis=1)

    write_reconstruction_errors(outdir, rows, errors)
    plot_outputs(outdir, rows, matrix, errors)
    print(f"[OK] wrote {outdir / 'embedding_pca_recon_errors.tsv'}")
    print(f"[OK] wrote {outdir / 'embedding_pca_scatter.png'}")


if __name__ == "__main__":
    main()
