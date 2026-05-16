#!/usr/bin/env python3

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from Bio import AlignIO, Phylo
from Bio.Phylo.Newick import Tree


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def load_taxa_from_first_block(rows: list[dict[str, str]]) -> list[str]:
    alignment = AlignIO.read(Path(rows[0]["path"]), "fasta")
    return [record.id for record in alignment]


def block_weight(row: dict[str, str]) -> float:
    if row.get("soft_block_weight"):
        return float(row["soft_block_weight"])
    if row.get("effective_length") and row.get("informative_ratio"):
        return float(row["effective_length"]) * float(row["informative_ratio"])
    if row.get("effective_length"):
        return float(row["effective_length"])
    return float(row.get("length", 1.0))


def parse_newick_texts(path: Path) -> list[Tree]:
    trees = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            trees.append(Phylo.read(StringIO(line), "newick"))
    return trees


def read_block_trees(row: dict[str, str], trees_dir: Path, prefer_samples: bool) -> list[Tree]:
    block_id = row["block_id"]
    sample_path = trees_dir / f"{block_id}.samples.nwk"
    tree_path = trees_dir / f"{block_id}.nwk"
    if prefer_samples and sample_path.exists():
        return parse_newick_texts(sample_path)
    if tree_path.exists():
        return [Phylo.read(tree_path, "newick")]
    return []


def write_source_tree_list(
    rows: list[dict[str, str]],
    trees_dir: Path,
    output_path: Path,
    prefer_samples: bool,
) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            block_id = row["block_id"]
            sample_path = trees_dir / f"{block_id}.samples.nwk"
            tree_path = trees_dir / f"{block_id}.nwk"
            if prefer_samples and sample_path.exists():
                source_lines = sample_path.read_text(encoding="utf-8").splitlines()
            elif tree_path.exists():
                source_lines = [tree_path.read_text(encoding="utf-8")]
            else:
                source_lines = []
            for line in source_lines:
                line = line.strip()
                if line:
                    handle.write(line + "\n")
                    count += 1
    if count == 0:
        raise ValueError(f"No usable block trees found in {trees_dir}")
    return count


def write_tree(tree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        Phylo.write(tree, handle, "newick")


def write_split_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "size", "weight_score", "weighted_presence", "support"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
