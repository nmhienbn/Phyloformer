#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median

import numpy as np
from Bio import AlignIO


GAP_CHARS = {"-", ".", "?"}
BRANCH_LENGTH_RE = re.compile(r":([0-9eE.+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the final Chapter 3 synthetic PANDIT-like dataset.")
    parser.add_argument("manifest_tsv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--empirical-gap-summary",
        default="runs/pandit_domain_adaptation/stage5_empirical_gap_profile/pandit_train_gap_bucket_summary.tsv",
    )
    parser.add_argument(
        "--empirical-model-counts",
        default="runs/pandit_domain_adaptation/pandit_iqtree_model_analysis/pandit_train_model_counts.tsv",
    )
    parser.add_argument(
        "--empirical-gamma-quantiles",
        default="runs/pandit_domain_adaptation/pandit_iqtree_model_analysis/pandit_train_gamma_alpha_quantiles.tsv",
    )
    parser.add_argument(
        "--empirical-tree-quantiles",
        default="runs/pandit_domain_adaptation/pandit_iqtree_model_analysis/pandit_train_tree_quantiles.tsv",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | None) -> float | None:
    if value in {None, "", "N/A"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    return values[lo] if lo == hi else float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def read_empirical_quantiles(path: Path) -> dict[str, float | None]:
    rows = read_tsv(path)
    if not rows:
        return {}

    if "value" in rows[0]:
        return {row["metric"]: as_float(row["value"]) for row in rows}

    if {"q25", "median", "q75"}.issubset(rows[0]):
        values: dict[str, float | None] = {}
        for row in rows:
            if row.get("metric") in {"gamma_alpha", "total_tree_length"}:
                values["q25"] = as_float(row["q25"])
                values["median"] = as_float(row["median"])
                values["q75"] = as_float(row["q75"])
                return values
        row = rows[0]
        return {
            "q25": as_float(row["q25"]),
            "median": as_float(row["median"]),
            "q75": as_float(row["q75"]),
        }

    raise ValueError(
        f"{path} must contain either metric/value rows or metric/q25/median/q75 rows"
    )


def matrix_from_alignment(path: Path) -> np.ndarray:
    last_error: Exception | None = None
    for fmt_name in ["fasta", "phylip-relaxed", "phylip"]:
        try:
            alignment = AlignIO.read(path, fmt_name)
            sequences = [str(record.seq).upper() for record in alignment]
            return np.array([list(seq) for seq in sequences], dtype="<U1")
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not parse alignment {path}: {last_error}")


def gap_runs(matrix: np.ndarray) -> list[int]:
    runs = []
    for row in matrix:
        current = 0
        for char in row.tolist():
            if char in GAP_CHARS:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
    return runs


def gap_stats(path: Path) -> dict[str, float]:
    matrix = matrix_from_alignment(path)
    nseq, length = matrix.shape
    mask = np.isin(matrix, list(GAP_CHARS))
    runs = gap_runs(matrix)
    return {
        "gap_ratio": float(mask.sum()) / (nseq * length) if nseq and length else 0.0,
        "gappy_column_ratio_any": float(mask.any(axis=0).sum()) / length if length else 0.0,
        "median_gap_run_length": float(median(runs)) if runs else 0.0,
        "p90_gap_run_length": quantile([float(run) for run in runs], 0.9) or 0.0,
    }


def tree_length(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    return sum(float(match.group(1)) for match in BRANCH_LENGTH_RE.finditer(text))


def model_counts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    counts: dict[str, int] = {}
    for row in rows:
        model = row.get("model_without_alpha") or "MISSING"
        counts[model] = counts.get(model, 0) + 1
    total = sum(counts.values())
    return [
        {"model": model, "count": str(count), "fraction": f"{count / total:.12g}"}
        for model, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def gap_bucket_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary = []
    for bucket in ["low-gap", "mid-gap", "high-gap"]:
        bucket_rows = [row for row in rows if row["gap_bucket"] == bucket]
        if not bucket_rows:
            continue
        record = {"gap_bucket": bucket, "n_msa": str(len(bucket_rows))}
        for metric in ["gap_ratio", "gappy_column_ratio_any", "median_gap_run_length", "p90_gap_run_length"]:
            values = [float(row[metric]) for row in bucket_rows]
            record[f"mean_{metric}"] = f"{mean(values):.12g}"
            record[f"median_{metric}"] = f"{median(values):.12g}"
        summary.append(record)
    return summary


def quantile_rows(values: list[float], empirical: dict[str, float | None], value_name: str) -> list[dict[str, str]]:
    synthetic = {
        "q25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "q75": quantile(values, 0.75),
        "n": float(len(values)),
    }
    return [
        {"metric": metric, value_name: fmt(synthetic[metric]), "empirical_value": fmt(empirical.get(metric))}
        for metric in ["q25", "median", "q75", "n"]
    ]


def write_markdown_report(outdir: Path, n_rows: int, synthetic_models: list[dict[str, str]], gap_summary: list[dict[str, str]]) -> None:
    top_models = ", ".join(f"{row['model']} ({row['count']})" for row in synthetic_models[:4])
    lines = [
        "# Stage 5 v3 Synthetic Dataset Summary",
        "",
        f"- synthetic MSAs: `{n_rows}`",
        f"- top synthetic models: `{top_models}`",
        "",
        "## Gap Buckets",
        "",
        "| bucket | n | median gap ratio | median gap-run length |",
        "|---|---:|---:|---:|",
    ]
    for row in gap_summary:
        lines.append(f"| {row['gap_bucket']} | {row['n_msa']} | {row['median_gap_ratio']} | {row['median_median_gap_run_length']} |")
    (outdir / "synthetic_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_tsv(Path(args.manifest_tsv))

    profile_rows = []
    for row in manifest_rows:
        gaps = gap_stats(Path(row["relative_path"]))
        profile_rows.append(
            {
                "msa_id": row["msa_id"],
                "gap_bucket": row["gap_bucket"],
                "length_bin": row.get("length_bin", ""),
                "model_without_alpha": row.get("model_without_alpha", ""),
                "model": row.get("model", ""),
                "source_gamma_alpha": row.get("source_gamma_alpha", ""),
                "target_total_tree_length": row.get("target_total_tree_length", ""),
                "gap_ratio": f"{gaps['gap_ratio']:.12g}",
                "gappy_column_ratio_any": f"{gaps['gappy_column_ratio_any']:.12g}",
                "median_gap_run_length": f"{gaps['median_gap_run_length']:.12g}",
                "p90_gap_run_length": f"{gaps['p90_gap_run_length']:.12g}",
                "total_tree_length": f"{tree_length(Path(row['tree_path'])):.12g}",
            }
        )

    synthetic_models = model_counts(manifest_rows)
    gap_summary = gap_bucket_summary(profile_rows)
    gamma_rows = quantile_rows(
        [value for value in (as_float(row.get("source_gamma_alpha")) for row in manifest_rows) if value is not None],
        read_empirical_quantiles(Path(args.empirical_gamma_quantiles)),
        "synthetic_value",
    )
    tree_rows = quantile_rows(
        [value for value in (as_float(row.get("target_total_tree_length")) for row in manifest_rows) if value is not None],
        read_empirical_quantiles(Path(args.empirical_tree_quantiles)),
        "synthetic_target_total_tree_length",
    )

    write_tsv(outdir / "synthetic_profile.tsv", profile_rows)
    write_tsv(outdir / "synthetic_gap_bucket_summary.tsv", gap_summary)
    write_tsv(outdir / "synthetic_model_counts.tsv", synthetic_models)
    write_tsv(outdir / "synthetic_gamma_alpha_quantiles.tsv", gamma_rows)
    write_tsv(outdir / "synthetic_tree_quantiles.tsv", tree_rows)
    (outdir / "synthetic_eval_report.json").write_text(
        json.dumps(
            {
                "n_synthetic": len(manifest_rows),
                "empirical_top_models": read_tsv(Path(args.empirical_model_counts))[:4],
                "synthetic_top_models": synthetic_models[:4],
                "empirical_gap_summary": read_tsv(Path(args.empirical_gap_summary)),
                "synthetic_gap_summary": gap_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown_report(outdir, len(manifest_rows), synthetic_models, gap_summary)
    print(f"[OK] wrote {outdir / 'synthetic_eval_report.md'}")


if __name__ == "__main__":
    main()
