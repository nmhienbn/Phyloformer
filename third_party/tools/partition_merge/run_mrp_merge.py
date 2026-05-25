#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_MERGE_DIR = REPO_ROOT / "experiments" / "04_pf2_partition_merge_refit" / "src" / "merge"
sys.path.insert(0, str(EXPERIMENT_MERGE_DIR))

from helper import read_manifest_rows, write_tree
from iqtree_refit import add_refit_args, refit_if_requested, topology_path_for_refit
from run_consensus_merge import weighted_consensus_tree


def duplication_count(support: float, mode: str, scale: float, max_duplications: int) -> int:
    if mode == "none":
        return 1
    value = support
    if mode == "sqrt":
        value = math.sqrt(value)
    elif mode == "log":
        value = math.log1p(value) / math.log(2.0)
    return min(max_duplications, max(1, int(round(scale * value))))


def write_mrp_matrix(
    taxa: list[str],
    split_rows: list[dict[str, str]],
    output_path: Path,
    weight_mode: str,
    weight_scale: float,
    max_duplications: int,
) -> tuple[int, list[dict[str, str]]]:
    taxon_to_chars = {taxon: [] for taxon in taxa}
    weight_rows: list[dict[str, str]] = []
    char_index = 0

    for row in split_rows:
        members = set(row["split"].split("|"))
        n_dup = duplication_count(
            support=float(row["support"]),
            mode=weight_mode,
            scale=weight_scale,
            max_duplications=max_duplications,
        )
        for dup in range(n_dup):
            char_index += 1
            for taxon in taxa:
                taxon_to_chars[taxon].append("1" if taxon in members else "0")
            weight_rows.append(
                {
                    "char_id": f"mrp_{char_index:05d}",
                    "source_split": row["split"],
                    "support": row["support"],
                    "duplication_index": str(dup + 1),
                    "duplication_total": str(n_dup),
                }
            )

    n_chars = len(next(iter(taxon_to_chars.values()), []))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(taxa)} {n_chars}\n")
        for taxon in taxa:
            handle.write(f"{taxon} {''.join(taxon_to_chars[taxon])}\n")
    return n_chars, weight_rows


def write_weight_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["char_id", "source_split", "support", "duplication_index", "duplication_total"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_iqtree_morph_backend(
    *,
    backend_bin: str,
    matrix_path: Path,
    topology_path: Path,
    threads: int,
    extra_args: str,
) -> None:
    prefix = topology_path.with_suffix(topology_path.suffix + ".mrp_backend")
    cmd = [
        backend_bin,
        "-s",
        str(matrix_path),
        "-st",
        "MORPH",
        "-m",
        "MK+ASC",
        "-nt",
        str(threads),
        "-pre",
        str(prefix),
        "-redo",
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    subprocess.run(cmd, check=True)

    treefile = Path(f"{prefix}.treefile")
    if not treefile.exists():
        raise FileNotFoundError(f"MRP backend completed but did not write {treefile}")
    topology_path.write_text(treefile.read_text(encoding="utf-8"), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge PF2 block trees with MRP, then optional IQ-TREE branch-length refit."
    )
    parser.add_argument("manifest")
    parser.add_argument("trees_dir")
    parser.add_argument("output_tree")
    parser.add_argument("--prefer-samples", action="store_true")
    parser.add_argument("--matrix", default=None, help="Optional output PHYLIP MRP matrix path.")
    parser.add_argument("--weights", default=None, help="Optional MRP character weight table path.")
    parser.add_argument("--mrp-weight-mode", choices=["none", "linear", "sqrt", "log"], default="linear")
    parser.add_argument("--mrp-weight-scale", type=float, default=8.0)
    parser.add_argument("--mrp-max-duplications", type=int, default=20)
    parser.add_argument("--backend-bin", default="bin/bin_linux/iqtree_2.2.0")
    parser.add_argument("--backend-extra-args", default="")
    add_refit_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    trees_dir = Path(args.trees_dir)
    output_tree = Path(args.output_tree)
    topology_tree = topology_path_for_refit(output_tree, args.fit_branch_lengths)
    matrix_path = Path(args.matrix) if args.matrix else output_tree.with_suffix(output_tree.suffix + ".mrp.phy")
    weights_path = Path(args.weights) if args.weights else output_tree.with_suffix(output_tree.suffix + ".mrp_weights.csv")

    rows = read_manifest_rows(manifest_path)
    consensus_tree, split_rows, accepted_count = weighted_consensus_tree(
        rows=rows,
        trees_dir=trees_dir,
        prefer_samples=args.prefer_samples,
    )
    if not split_rows:
        write_tree(consensus_tree, topology_tree)
        n_chars = 0
        fallback = " fallback=weighted:no_splits"
    else:
        taxa = sorted({taxon for row in rows for taxon in row.get("taxa", "").split("|") if taxon})
        if not taxa:
            from run_consensus_merge import collect_source_trees

            taxa, _, _, _ = collect_source_trees(rows, trees_dir, args.prefer_samples)
        n_chars, weight_rows = write_mrp_matrix(
            taxa=taxa,
            split_rows=split_rows,
            output_path=matrix_path,
            weight_mode=args.mrp_weight_mode,
            weight_scale=args.mrp_weight_scale,
            max_duplications=args.mrp_max_duplications,
        )
        write_weight_table(weights_path, weight_rows)
        run_iqtree_morph_backend(
            backend_bin=args.backend_bin,
            matrix_path=matrix_path,
            topology_path=topology_tree,
            threads=args.iqtree_threads,
            extra_args=args.backend_extra_args,
        )
        fallback = ""

    refit_if_requested(
        fit_branch_lengths=args.fit_branch_lengths,
        manifest_path=manifest_path,
        topology_tree_path=topology_tree,
        output_tree_path=output_tree,
        alignment=args.alignment,
        iqtree_bin=args.iqtree_bin,
        iqtree_model=args.iqtree_model,
        iqtree_threads=args.iqtree_threads,
    )
    print(
        f"[OK] method=mrp chars={n_chars} source_splits={len(split_rows)} "
        f"consensus_accepted_splits={accepted_count}{fallback} tree={output_tree}"
    )


if __name__ == "__main__":
    main()
