#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_MERGE_DIR = REPO_ROOT / "experiments" / "04_pf2_partition_merge_refit" / "src" / "merge"
sys.path.insert(0, str(EXPERIMENT_MERGE_DIR))

from helper import read_manifest_rows, write_source_tree_list, write_tree
from iqtree_refit import add_refit_args, refit_if_requested, topology_path_for_refit
from run_consensus_merge import weighted_consensus_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge PF2 block trees with SuperFine, then optional IQ-TREE branch-length refit."
    )
    parser.add_argument("manifest")
    parser.add_argument("trees_dir")
    parser.add_argument("output_tree")
    parser.add_argument("--prefer-samples", action="store_true")
    parser.add_argument("--source-list", default=None)
    parser.add_argument(
        "--superfine-cmd",
        required=True,
        help=(
            "Shell command template for SuperFine. Use {source_list}; {output_tree} is optional. "
            "If the command does not create {output_tree}, stdout is written to {output_tree}. "
            "Example: 'PYTHONPATH=... python2 .../runReup.py -r qmc {source_list}'."
        ),
    )
    add_refit_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    trees_dir = Path(args.trees_dir)
    output_tree = Path(args.output_tree)
    source_list = (
        Path(args.source_list)
        if args.source_list
        else output_tree.with_suffix(output_tree.suffix + ".sources.nwk")
    )
    topology_tree = topology_path_for_refit(output_tree, args.fit_branch_lengths)

    rows = read_manifest_rows(manifest_path)
    n_sources = write_source_tree_list(rows, trees_dir, source_list, prefer_samples=args.prefer_samples)

    fallback_reason = ""
    if n_sources >= 3:
        cmd_text = args.superfine_cmd.format(source_list=source_list, output_tree=topology_tree)
        result = subprocess.run(cmd_text, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and (not topology_tree.exists() or topology_tree.stat().st_size == 0):
            stdout = result.stdout.strip()
            if stdout:
                topology_tree.parent.mkdir(parents=True, exist_ok=True)
                topology_tree.write_text(stdout + "\n", encoding="utf-8")
        if result.returncode != 0 or not topology_tree.exists() or topology_tree.stat().st_size == 0:
            fallback_reason = "superfine_crash"
    else:
        fallback_reason = "too_few_sources"

    if fallback_reason:
        tree, _, _ = weighted_consensus_tree(
            rows=rows,
            trees_dir=trees_dir,
            prefer_samples=args.prefer_samples,
        )
        write_tree(tree, topology_tree)

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
    fallback = f" fallback=weighted:{fallback_reason}" if fallback_reason else ""
    print(f"[OK] method=superfine sources={n_sources}{fallback} tree={output_tree}")


if __name__ == "__main__":
    main()
