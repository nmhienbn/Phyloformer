#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

from tqdm import tqdm


def get_alignment(case_dir: Path) -> Path | None:
    pipeline = case_dir / "metadata" / "pipeline.json"
    if not pipeline.exists():
        return None
    payload = json.loads(pipeline.read_text(encoding="utf-8"))
    raw = payload.get("input_alignment")
    return Path(raw) if raw else None


def build_merge_cmd(
    merge: str,
    manifest: str,
    trees_dir: str,
    output_tree: str,
    fit_branch_lengths: bool,
    alignment: str | None,
    iqtree_bin: str,
    iqtree_model: str,
    iqtree_threads: int,
) -> list[str]:
    bl_args = []
    if fit_branch_lengths and alignment:
        bl_args = [
            "--fit-branch-lengths",
            "--alignment", alignment,
            "--iqtree-bin", iqtree_bin,
            "--iqtree-model", iqtree_model,
            "--iqtree-threads", str(iqtree_threads),
        ]

    if merge == "weighted":
        return ["python", "experiments/04_pf2_partition_merge_refit/src/merge/run_consensus_merge.py",
                manifest, trees_dir, output_tree] + bl_args
    if merge == "fastrfs":
        return ["python", "experiments/04_pf2_partition_merge_refit/src/merge/run_fastrfs_merge.py",
                manifest, trees_dir, output_tree, "--fastrfs-bin", "bin/bin_linux/FastRFS"] + bl_args
    raise ValueError(f"Unknown merge method: {merge}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all partition × merge combinations with per-combination tqdm."
    )
    parser.add_argument("--base", default="runs/benchmarks/pandit_over2gb")
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=["window_pos", "window_rate", "softbioblock"],
    )
    parser.add_argument(
        "--merges",
        nargs="+",
        default=[
            "weighted",
            "fastrfs",
        ],
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases whose output tree already exists (resume mode).",
    )
    parser.add_argument(
        "--fit-branch-lengths",
        action="store_true",
        help="After merging topology, refit branch lengths with IQ-TREE -te.",
    )
    parser.add_argument("--iqtree-bin", default="bin/bin_linux/iqtree_2.2.0")
    parser.add_argument("--iqtree-model", default="LG+G4")
    parser.add_argument("--iqtree-threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N cases per combo (for testing).")
    parser.add_argument("--workers", type=int, default=1, help="Number of cases to merge in parallel per combo.")
    return parser.parse_args()


def run_combo(partition_dir: Path, merge: str, output_dir: Path, args: argparse.Namespace) -> dict:
    output_dir.joinpath("final_trees").mkdir(parents=True, exist_ok=True)
    cases = sorted(partition_dir.joinpath("cases").glob("*/"))
    cases = [c for c in cases if c.joinpath("block_manifest.csv").exists()
             and c.joinpath("pf2_blocks").is_dir()]

    if args.limit is not None:
        cases = cases[:args.limit]

    failed = skipped = 0
    t0 = time.perf_counter()
    bar = tqdm(total=len(cases), desc=f"{partition_dir.name} × {merge}", unit="case", leave=True)

    def _run_case(case_dir: Path) -> str | None:
        stem = case_dir.name
        output_tree = output_dir / "final_trees" / f"{stem}.nwk"
        if args.skip_existing and output_tree.exists():
            return "skipped"

        alignment = None
        if args.fit_branch_lengths:
            alignment = get_alignment(case_dir)
            if alignment is None or not alignment.exists():
                tqdm.write(f"  [warn] {stem}: alignment not found, skipping refit")

        cmd = build_merge_cmd(
            merge=merge,
            manifest=str(case_dir / "block_manifest.csv"),
            trees_dir=str(case_dir / "pf2_blocks"),
            output_tree=str(output_tree),
            fit_branch_lengths=args.fit_branch_lengths and alignment is not None,
            alignment=str(alignment) if alignment else None,
            iqtree_bin=args.iqtree_bin,
            iqtree_model=args.iqtree_model,
            iqtree_threads=args.iqtree_threads,
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr_lines = result.stderr.strip().splitlines()
            error_msg = "\n    ".join(stderr_lines[-3:]) if stderr_lines else (result.stdout.strip() or "?")
            tqdm.write(f"  [failed] {stem}: {error_msg}")
            return "failed"
        return None

    workers = getattr(args, "workers", 1)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_case, c): c for c in cases}
            for fut in as_completed(futures):
                bar.update(1)
                status = fut.result()
                if status == "skipped":
                    skipped += 1
                elif status == "failed":
                    failed += 1
    else:
        for case_dir in cases:
            status = _run_case(case_dir)
            bar.update(1)
            if status == "skipped":
                skipped += 1
            elif status == "failed":
                failed += 1

    bar.close()
    elapsed = time.perf_counter() - t0
    return {"elapsed_sec": elapsed, "n_cases": len(cases), "n_failed": failed, "n_skipped": skipped}


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    timing_path = base / "merge_timing.csv"

    combos = list(product(args.partitions, args.merges))
    rows: list[dict] = []

    for part, merge in combos:
        stats = run_combo(
            base / f"partition_{part}",
            merge,
            base / f"merge_{part}_{merge}",
            args,
        )
        rows.append({
            "partition": part,
            "merge": merge,
            "n_cases": stats["n_cases"],
            "n_failed": stats["n_failed"],
            "elapsed_sec": f"{stats['elapsed_sec']:.3f}",
        })

    with timing_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["partition", "merge", "n_cases", "n_failed", "elapsed_sec"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] {len(combos)} combinations done  timing={timing_path}")


if __name__ == "__main__":
    main()
