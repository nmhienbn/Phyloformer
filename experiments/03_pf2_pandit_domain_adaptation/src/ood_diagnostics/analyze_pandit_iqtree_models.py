#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import io
import math
import re
from pathlib import Path
from statistics import median

from Bio import Phylo


MODEL_RE = re.compile(r"Model of substitution:\s+(.+)")
ALPHA_RE = re.compile(r"Gamma shape alpha:\s+([0-9eE.+-]+)")
PINV_RE = re.compile(r"Proportion of invariable sites:\s+([0-9eE.+-]+)")
INPUT_RE = re.compile(r"Input data:\s+(\d+)\s+sequences\s+with\s+(\d+)\s+amino-acid sites")
LOGL_RE = re.compile(r"Log-likelihood of the tree:\s+([0-9eE.+-]+)")
AIC_RE = re.compile(r"Akaike information criterion \(AIC\) score:\s+([0-9eE.+-]+)")
BIC_RE = re.compile(r"Bayesian information criterion \(BIC\) score:\s+([0-9eE.+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join PANDIT train metadata with IQ-TREE ModelFinder outputs. "
            "This is intentionally narrow: it emits only the tables needed by "
            "the Chapter 3 PANDIT-like synthetic generator and dataset summary."
        )
    )
    parser.add_argument(
        "split_tsv",
        nargs="?",
        default=None,
        help="PANDIT split TSV, normally pandit_train.tsv.",
    )
    parser.add_argument(
        "--all-split",
        default=None,
        help="Backward-compatible alias for split_tsv.",
    )
    parser.add_argument(
        "--profile-tsv",
        default="runs/pandit_domain_adaptation/stage2_profiles/pandit_train_profile.tsv",
        help="Profile TSV with gap_ratio and informative_site_ratio.",
    )
    parser.add_argument(
        "--iqtree-work-dir",
        default="runs/pandit_domain_adaptation/iqtree_mfp_pandit_train/work",
        help="Directory containing <msa_id>.iqtree and <msa_id>.treefile.",
    )
    parser.add_argument(
        "--outdir",
        default="runs/pandit_domain_adaptation/pandit_iqtree_model_analysis",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(text: str | None) -> float:
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def fmt_float(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.12g}"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def model_flags(model: str) -> dict[str, str]:
    tokens = [token.strip() for token in model.split("+") if token.strip()]
    suffixes = tokens[1:]
    rate_tokens = [token for token in suffixes if token.startswith(("G", "R"))]
    rate_token = rate_tokens[0] if rate_tokens else ""
    return {
        "best_model_base": tokens[0] if tokens else "",
        "has_G": str(rate_token.startswith("G")),
        "has_I": str("I" in suffixes),
        "has_F": str("F" in suffixes),
    }


def parse_iqtree_report(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {
            "fitted_model": "",
            "best_model_base": "",
            "has_G": "",
            "has_I": "",
            "has_F": "",
            "gamma_alpha": "",
            "p_invariant": "",
            "log_likelihood": "",
            "aic": "",
            "bic": "",
            "iqtree_num_sequences": "",
            "iqtree_alignment_length": "",
        }

    text = read_text(path)
    model_match = MODEL_RE.search(text)
    input_match = INPUT_RE.search(text)
    fitted_model = model_match.group(1).strip() if model_match else ""
    flags = model_flags(fitted_model) if fitted_model else {
        "best_model_base": "",
        "has_G": "",
        "has_I": "",
        "has_F": "",
    }
    alpha_match = ALPHA_RE.search(text)
    pinv_match = PINV_RE.search(text)
    logl_match = LOGL_RE.search(text)
    aic_match = AIC_RE.search(text)
    bic_match = BIC_RE.search(text)
    return {
        "fitted_model": fitted_model,
        **flags,
        "gamma_alpha": alpha_match.group(1) if alpha_match else "",
        "p_invariant": pinv_match.group(1) if pinv_match else "",
        "log_likelihood": logl_match.group(1) if logl_match else "",
        "aic": aic_match.group(1) if aic_match else "",
        "bic": bic_match.group(1) if bic_match else "",
        "iqtree_num_sequences": input_match.group(1) if input_match else "",
        "iqtree_alignment_length": input_match.group(2) if input_match else "",
    }


def parse_treefile(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {
            "total_tree_length": "",
            "mean_branch_length": "",
            "median_branch_length": "",
            "max_root_to_leaf_depth": "",
            "median_root_to_leaf_depth": "",
            "n_leaves": "",
            "n_branches": "",
        }

    tree = Phylo.read(io.StringIO(read_text(path).strip()), "newick")
    branch_lengths = [
        float(clade.branch_length)
        for clade in tree.find_clades()
        if clade.branch_length is not None
    ]
    terminal_depths = [
        float(depth)
        for clade, depth in tree.depths().items()
        if clade.is_terminal()
    ]
    total = sum(branch_lengths)
    return {
        "total_tree_length": fmt_float(total),
        "mean_branch_length": fmt_float(total / len(branch_lengths)) if branch_lengths else "",
        "median_branch_length": fmt_float(float(median(branch_lengths))) if branch_lengths else "",
        "max_root_to_leaf_depth": fmt_float(max(terminal_depths)) if terminal_depths else "",
        "median_root_to_leaf_depth": fmt_float(float(median(terminal_depths))) if terminal_depths else "",
        "n_leaves": str(len(tree.get_terminals())),
        "n_branches": str(len(branch_lengths)),
    }


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def write_quantiles(path: Path, values: list[float]) -> None:
    rows = [
        {"metric": "q25", "value": fmt_float(quantile(values, 0.25))},
        {"metric": "median", "value": fmt_float(quantile(values, 0.50))},
        {"metric": "q75", "value": fmt_float(quantile(values, 0.75))},
        {"metric": "n", "value": str(len(values))},
    ]
    write_tsv(path, ["metric", "value"], rows)


def count_models(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    counts: dict[str, int] = {}
    for row in rows:
        model = row.get("fitted_model") or "MISSING"
        counts[model] = counts.get(model, 0) + 1
    total = sum(counts.values())
    return [
        {
            "fitted_model": model,
            "count": str(count),
            "fraction": f"{count / total:.12g}" if total else "",
        }
        for model, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def main() -> None:
    args = parse_args()
    split_path = Path(args.split_tsv or args.all_split or "runs/pandit_domain_adaptation/stage1_msa_splits/pandit_train.tsv")
    outdir = Path(args.outdir)
    iqtree_work_dir = Path(args.iqtree_work_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    split_rows = read_tsv(split_path)
    profile_by_id = {row["msa_id"]: row for row in read_tsv(Path(args.profile_tsv))}

    joined_rows: list[dict[str, str]] = []
    for split_row in split_rows:
        msa_id = split_row["msa_id"]
        profile = profile_by_id.get(msa_id, {})
        iqtree = parse_iqtree_report(iqtree_work_dir / f"{msa_id}.iqtree")
        tree = parse_treefile(iqtree_work_dir / f"{msa_id}.treefile")
        joined_rows.append(
            {
                "msa_id": msa_id,
                "split": split_row.get("split", "train"),
                "fitted_model": iqtree["fitted_model"],
                "best_model_base": iqtree["best_model_base"],
                "has_G": iqtree["has_G"],
                "has_I": iqtree["has_I"],
                "has_F": iqtree["has_F"],
                "gamma_alpha": iqtree["gamma_alpha"],
                "p_invariant": iqtree["p_invariant"],
                "log_likelihood": iqtree["log_likelihood"],
                "aic": iqtree["aic"],
                "bic": iqtree["bic"],
                "total_tree_length": tree["total_tree_length"],
                "mean_branch_length": tree["mean_branch_length"],
                "median_branch_length": tree["median_branch_length"],
                "max_root_to_leaf_depth": tree["max_root_to_leaf_depth"],
                "median_root_to_leaf_depth": tree["median_root_to_leaf_depth"],
                "n_leaves": tree["n_leaves"],
                "n_branches": tree["n_branches"],
                "gap_ratio": profile.get("gap_ratio", ""),
                "informative_site_ratio": profile.get("informative_site_ratio", ""),
                "num_sequences": split_row.get("num_sequences") or profile.get("num_sequences", ""),
                "alignment_length": split_row.get("alignment_length") or profile.get("alignment_length", ""),
            }
        )

    fieldnames = list(joined_rows[0].keys()) if joined_rows else [
        "msa_id",
        "split",
        "fitted_model",
        "best_model_base",
        "has_G",
        "has_I",
        "has_F",
        "gamma_alpha",
        "p_invariant",
        "log_likelihood",
        "aic",
        "bic",
        "total_tree_length",
        "mean_branch_length",
        "median_branch_length",
        "max_root_to_leaf_depth",
        "median_root_to_leaf_depth",
        "n_leaves",
        "n_branches",
        "gap_ratio",
        "informative_site_ratio",
        "num_sequences",
        "alignment_length",
    ]
    write_tsv(outdir / "pandit_train_iqtree_per_msa_joined.tsv", fieldnames, joined_rows)
    write_tsv(outdir / "pandit_iqtree_per_msa_joined.tsv", fieldnames, joined_rows)

    tree_fields = [
        "msa_id",
        "total_tree_length",
        "mean_branch_length",
        "median_branch_length",
        "max_root_to_leaf_depth",
        "median_root_to_leaf_depth",
        "n_leaves",
        "n_branches",
    ]
    write_tsv(
        outdir / "pandit_train_iqtree_tree_stats.tsv",
        tree_fields,
        [{field: row[field] for field in tree_fields} for row in joined_rows],
    )
    write_tsv(
        outdir / "pandit_train_model_counts.tsv",
        ["fitted_model", "count", "fraction"],
        count_models(joined_rows),
    )

    gamma_values = [safe_float(row["gamma_alpha"]) for row in joined_rows]
    gamma_values = [value for value in gamma_values if not math.isnan(value)]
    tree_values = [safe_float(row["total_tree_length"]) for row in joined_rows]
    tree_values = [value for value in tree_values if not math.isnan(value)]
    write_quantiles(outdir / "pandit_train_gamma_alpha_quantiles.tsv", gamma_values)
    write_quantiles(outdir / "pandit_train_tree_quantiles.tsv", tree_values)

    coverage = [
        {
            "n_rows": str(len(joined_rows)),
            "n_iqtree_report_found": str(sum(1 for row in joined_rows if row["fitted_model"])),
            "n_treefile_found": str(sum(1 for row in joined_rows if row["total_tree_length"])),
            "n_gamma_alpha_found": str(len(gamma_values)),
        }
    ]
    write_tsv(outdir / "coverage_summary.tsv", list(coverage[0].keys()), coverage)

    print(f"[OK] rows={len(joined_rows)}")
    print(f"[OK] wrote {outdir / 'pandit_train_iqtree_per_msa_joined.tsv'}")
    print(f"[OK] wrote {outdir / 'pandit_train_iqtree_tree_stats.tsv'}")


if __name__ == "__main__":
    main()
