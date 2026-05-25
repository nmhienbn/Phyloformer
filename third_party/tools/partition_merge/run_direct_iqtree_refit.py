#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from iqtree_refit import fit_branch_lengths_with_iqtree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit branch lengths for an existing directory of PF2 direct topologies with IQ-TREE -te."
    )
    parser.add_argument("alignments", help="Directory containing full input alignments.")
    parser.add_argument("topologies", help="Directory containing PF2 direct topology .nwk files.")
    parser.add_argument("output_dir", help="Directory where refitted .nwk files are written.")
    parser.add_argument("--iqtree-bin", default="bin/bin_linux/iqtree_2.2.0")
    parser.add_argument("--iqtree-model", default="LG+G4")
    parser.add_argument("--iqtree-threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def find_alignment(alignments_dir: Path, stem: str) -> Path:
    for ext in (".fa", ".fasta", ".faa"):
        path = alignments_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No alignment found for {stem} in {alignments_dir}")


def refit_one(
    *,
    topology_path: Path,
    alignments_dir: Path,
    output_dir: Path,
    iqtree_bin: str,
    iqtree_model: str,
    iqtree_threads: int,
    skip_existing: bool,
) -> Path:
    output_path = output_dir / topology_path.name
    if skip_existing and output_path.exists():
        return output_path
    fit_branch_lengths_with_iqtree(
        alignment_path=find_alignment(alignments_dir, topology_path.stem),
        topology_tree_path=topology_path,
        output_tree_path=output_path,
        iqtree_bin=iqtree_bin,
        iqtree_model=iqtree_model,
        iqtree_threads=iqtree_threads,
    )
    return output_path


def main() -> None:
    args = parse_args()
    alignments_dir = Path(args.alignments)
    topologies_dir = Path(args.topologies)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topologies = sorted(topologies_dir.glob("*.nwk"))
    if not topologies:
        raise FileNotFoundError(f"No .nwk topologies found in {topologies_dir}")

    print(f"[INFO] topologies: {len(topologies)}")
    print(f"[INFO] alignments: {alignments_dir}")
    print(f"[INFO] output: {output_dir}")
    print(f"[INFO] IQ-TREE model: {args.iqtree_model}")

    if args.workers <= 1:
        for topology_path in topologies:
            out = refit_one(
                topology_path=topology_path,
                alignments_dir=alignments_dir,
                output_dir=output_dir,
                iqtree_bin=args.iqtree_bin,
                iqtree_model=args.iqtree_model,
                iqtree_threads=args.iqtree_threads,
                skip_existing=args.skip_existing,
            )
            print(f"[DONE] {out}")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                refit_one,
                topology_path=topology_path,
                alignments_dir=alignments_dir,
                output_dir=output_dir,
                iqtree_bin=args.iqtree_bin,
                iqtree_model=args.iqtree_model,
                iqtree_threads=args.iqtree_threads,
                skip_existing=args.skip_existing,
            )
            for topology_path in topologies
        ]
        for future in as_completed(futures):
            print(f"[DONE] {future.result()}")


if __name__ == "__main__":
    main()
