#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from Bio.Phylo.Newick import Clade, Tree

REPO_ROOT = Path(__file__).resolve().parents[4]
THIRD_PARTY_MERGE_DIR = REPO_ROOT / "third_party" / "tools" / "partition_merge"
sys.path.insert(0, str(THIRD_PARTY_MERGE_DIR))

from helper import (
    block_weight,
    load_taxa_from_first_block,
    read_block_trees,
    read_manifest_rows,
    write_split_table,
    write_tree,
)
from iqtree_refit import add_refit_args, refit_if_requested, topology_path_for_refit


@dataclass
class SplitStats:
    score: float = 0.0
    presence_weight: float = 0.0


def canonical_split(taxa_subset: frozenset[str], all_taxa: frozenset[str]) -> frozenset[str] | None:
    if not taxa_subset or len(taxa_subset) == len(all_taxa):
        return None
    complement = all_taxa - taxa_subset
    if len(taxa_subset) == 1 or len(complement) == 1:
        return None
    if len(taxa_subset) < len(complement):
        return taxa_subset
    if len(complement) < len(taxa_subset):
        return frozenset(complement)
    return min(taxa_subset, frozenset(complement), key=lambda group: tuple(sorted(group)))


def extract_splits(tree: Tree, all_taxa: frozenset[str]) -> set[frozenset[str]]:
    splits = set()
    for clade in tree.get_nonterminals(order="postorder"):
        split = canonical_split(frozenset(t.name for t in clade.get_terminals()), all_taxa)
        if split is not None:
            splits.add(split)
    return splits


def split_compatible(a: frozenset[str], b: frozenset[str], taxa: frozenset[str]) -> bool:
    ac = taxa - a
    bc = taxa - b
    return not (a & b and a & bc and ac & b and ac & bc)


def collect_source_trees(
    rows: list[dict[str, str]],
    trees_dir: Path,
    prefer_samples: bool,
) -> tuple[list[str], frozenset[str], float, dict[frozenset[str], SplitStats]]:
    taxa = load_taxa_from_first_block(rows)
    taxa_set = frozenset(taxa)
    total_weight = 0.0
    split_stats: dict[frozenset[str], SplitStats] = defaultdict(SplitStats)

    for row in rows:
        trees = read_block_trees(row, trees_dir, prefer_samples)
        if not trees:
            continue
        weight = block_weight(row)
        per_tree_weight = weight / len(trees)
        total_weight += weight
        for tree in trees:
            for split in extract_splits(tree, taxa_set):
                stats = split_stats[split]
                stats.score += per_tree_weight
                stats.presence_weight += per_tree_weight

    if total_weight <= 0:
        raise ValueError(f"No usable block trees found in {trees_dir}")
    return taxa, taxa_set, total_weight, split_stats


def split_rows_from_stats(
    split_stats: dict[frozenset[str], SplitStats],
    total_weight: float,
) -> list[dict[str, str]]:
    rows = [
        {
            "split": "|".join(sorted(split)),
            "size": str(len(split)),
            "weight_score": f"{stats.score:.6f}",
            "weighted_presence": f"{stats.presence_weight:.6f}",
            "support": f"{stats.score / total_weight:.6f}",
        }
        for split, stats in split_stats.items()
    ]
    return sorted(rows, key=lambda row: (-float(row["weight_score"]), -int(row["size"]), row["split"]))


def accept_splits_greedily(
    split_rows: list[dict[str, str]],
    taxa_set: frozenset[str],
) -> tuple[list[frozenset[str]], dict[frozenset[str], float]]:
    accepted: list[frozenset[str]] = []
    support_map: dict[frozenset[str], float] = {}
    split_limit = max(0, len(taxa_set) - 3)
    for row in split_rows:
        split = frozenset(row["split"].split("|"))
        if all(split_compatible(split, other, taxa_set) for other in accepted):
            accepted.append(split)
            support_map[split] = float(row["support"])
            if len(accepted) >= split_limit:
                break
    return accepted, support_map


def orient_cluster(split: frozenset[str], root_taxon: str, taxa: frozenset[str]) -> frozenset[str]:
    return frozenset(taxa - split) if root_taxon in split else split


def build_rooted_clade(
    cluster: frozenset[str],
    all_clusters: set[frozenset[str]],
    support_map: dict[frozenset[str], float],
) -> Clade:
    children = [
        candidate
        for candidate in all_clusters
        if candidate != cluster
        and candidate.issubset(cluster)
        and not any(candidate < other < cluster for other in all_clusters)
    ]
    clade = Clade()
    if cluster in support_map:
        clade.confidence = support_map[cluster]

    covered: set[str] = set()
    for child in sorted(children, key=lambda group: (len(group), tuple(sorted(group)))):
        covered.update(child)
        clade.clades.append(build_rooted_clade(child, all_clusters, support_map))
    for taxon in sorted(cluster - covered):
        clade.clades.append(Clade(name=taxon))
    return clade


def build_consensus_tree(
    taxa: list[str],
    accepted_splits: list[frozenset[str]],
    support_map: dict[frozenset[str], float],
) -> Tree:
    taxa_set = frozenset(taxa)
    root_taxon = sorted(taxa)[0]
    oriented = {
        orient_cluster(split, root_taxon, taxa_set)
        for split in accepted_splits
        if 0 < len(orient_cluster(split, root_taxon, taxa_set)) < len(taxa_set) - 1
    }
    oriented_support = {
        orient_cluster(split, root_taxon, taxa_set): support
        for split, support in support_map.items()
    }
    root = Clade()
    root.clades.append(Clade(name=root_taxon))
    others = frozenset(taxa_set - {root_taxon})
    if others:
        root.clades.append(build_rooted_clade(others, oriented, oriented_support))
    return Tree(root=root, rooted=False)


def weighted_consensus_tree(
    rows: list[dict[str, str]],
    trees_dir: Path,
    prefer_samples: bool,
) -> tuple[Tree, list[dict[str, str]], int]:
    taxa, taxa_set, total_weight, split_stats = collect_source_trees(
        rows=rows,
        trees_dir=trees_dir,
        prefer_samples=prefer_samples,
    )
    split_rows = split_rows_from_stats(split_stats, total_weight)
    accepted, support_map = accept_splits_greedily(split_rows, taxa_set)
    tree = build_consensus_tree(taxa=taxa, accepted_splits=accepted, support_map=support_map)
    return tree, split_rows, len(accepted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PF2 block trees with weighted split consensus.")
    parser.add_argument("manifest", help="Block manifest CSV.")
    parser.add_argument("trees_dir", help="Directory containing block .nwk / .samples.nwk trees.")
    parser.add_argument("output_tree", help="Final merged Newick path.")
    parser.add_argument("--split-table", default=None, help="Optional split summary CSV.")
    parser.add_argument("--prefer-samples", action="store_true", help="Use .samples.nwk when present.")
    add_refit_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_tree = Path(args.output_tree)
    split_table = Path(args.split_table) if args.split_table else output_tree.with_suffix(output_tree.suffix + ".splits.csv")
    topology_tree = topology_path_for_refit(output_tree, args.fit_branch_lengths)

    rows = read_manifest_rows(manifest_path)
    tree, split_rows, accepted_count = weighted_consensus_tree(
        rows=rows,
        trees_dir=Path(args.trees_dir),
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

    write_split_table(split_table, split_rows)
    print(f"[OK] weighted accepted_splits={accepted_count} tree={output_tree} split_table={split_table}")


if __name__ == "__main__":
    main()
