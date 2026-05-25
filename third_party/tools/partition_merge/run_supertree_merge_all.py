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
    *,
    merge: str,
    manifest: str,
    trees_dir: str,
    output_tree: str,
    fit_branch_lengths: bool,
    alignment: str | None,
    args: argparse.Namespace,
) -> list[str]:
    bl_args = []
    if fit_branch_lengths and alignment:
        bl_args = [
            "--fit-branch-lengths",
            "--alignment",
            alignment,
            "--iqtree-bin",
            args.iqtree_bin,
            "--iqtree-model",
            args.iqtree_model,
            "--iqtree-threads",
            str(args.iqtree_threads),
        ]

    if merge == "mrp":
        return [
            "python",
            "third_party/tools/partition_merge/run_mrp_merge.py",
            manifest,
            trees_dir,
            output_tree,
            "--backend-bin",
            args.mrp_backend_bin,
            "--mrp-weight-mode",
            args.mrp_weight_mode,
            "--mrp-weight-scale",
            str(args.mrp_weight_scale),
            "--mrp-max-duplications",
            str(args.mrp_max_duplications),
        ] + bl_args
    if merge == "superfine":
        if not args.superfine_cmd:
            raise ValueError("--superfine-cmd is required when --merges includes superfine")
        return [
            "python",
            "third_party/tools/partition_merge/run_superfine_merge.py",
            manifest,
            trees_dir,
            output_tree,
            "--superfine-cmd",
            args.superfine_cmd,
        ] + bl_args
    raise ValueError(f"Unknown merge method: {merge}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PF2 partition results through SuperFine/MRP merge methods.")
    parser.add_argument("--base", default="runs/benchmarks/pandit_over1gb")
    parser.add_argument("--partitions", nargs="+", default=["window_pos", "window_rate", "softbioblock"])
    parser.add_argument("--merges", nargs="+", default=["mrp", "superfine"])
    parser.add_argument("--timing-file", default="merge_timing_supertree.csv")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fit-branch-lengths", action="store_true")
    parser.add_argument("--iqtree-bin", default="bin/bin_linux/iqtree_2.2.0")
    parser.add_argument("--iqtree-model", default="LG+G4")
    parser.add_argument("--iqtree-threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mrp-backend-bin", default="bin/bin_linux/iqtree_2.2.0")
    parser.add_argument("--mrp-weight-mode", choices=["none", "linear", "sqrt", "log"], default="linear")
    parser.add_argument("--mrp-weight-scale", type=float, default=8.0)
    parser.add_argument("--mrp-max-duplications", type=int, default=20)
    parser.add_argument("--superfine-cmd", default=None)
    return parser.parse_args()


def run_combo(partition_dir: Path, merge: str, output_dir: Path, args: argparse.Namespace) -> dict:
    output_dir.joinpath("final_trees").mkdir(parents=True, exist_ok=True)
    cases = sorted(partition_dir.joinpath("cases").glob("*/"))
    cases = [c for c in cases if c.joinpath("block_manifest.csv").exists() and c.joinpath("pf2_blocks").is_dir()]
    if args.limit is not None:
        cases = cases[: args.limit]

    failed = skipped = 0
    t0 = time.perf_counter()
    bar = tqdm(total=len(cases), desc=f"{partition_dir.name} x {merge}", unit="case", leave=True)

    def _run_case(case_dir: Path) -> str | None:
        output_tree = output_dir / "final_trees" / f"{case_dir.name}.nwk"
        if args.skip_existing and output_tree.exists():
            return "skipped"

        alignment = None
        if args.fit_branch_lengths:
            alignment = get_alignment(case_dir)
            if alignment is None or not alignment.exists():
                tqdm.write(f"  [warn] {case_dir.name}: alignment not found, skipping refit")

        cmd = build_merge_cmd(
            merge=merge,
            manifest=str(case_dir / "block_manifest.csv"),
            trees_dir=str(case_dir / "pf2_blocks"),
            output_tree=str(output_tree),
            fit_branch_lengths=args.fit_branch_lengths and alignment is not None,
            alignment=str(alignment) if alignment else None,
            args=args,
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr_lines = result.stderr.strip().splitlines()
            error_msg = "\n    ".join(stderr_lines[-3:]) if stderr_lines else (result.stdout.strip() or "?")
            tqdm.write(f"  [failed] {case_dir.name}: {error_msg}")
            return "failed"
        return None

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_case, c): c for c in cases}
            for fut in as_completed(futures):
                bar.update(1)
                status = fut.result()
                skipped += int(status == "skipped")
                failed += int(status == "failed")
    else:
        for case_dir in cases:
            status = _run_case(case_dir)
            bar.update(1)
            skipped += int(status == "skipped")
            failed += int(status == "failed")

    bar.close()
    return {
        "elapsed_sec": time.perf_counter() - t0,
        "n_cases": len(cases),
        "n_failed": failed,
        "n_skipped": skipped,
    }


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    timing_path = base / args.timing_file
    rows: list[dict[str, str]] = []

    for part, merge in product(args.partitions, args.merges):
        stats = run_combo(
            partition_dir=base / f"partition_{part}",
            merge=merge,
            output_dir=base / f"merge_{part}_{merge}",
            args=args,
        )
        rows.append(
            {
                "partition": part,
                "merge": merge,
                "n_cases": str(stats["n_cases"]),
                "n_failed": str(stats["n_failed"]),
                "n_skipped": str(stats["n_skipped"]),
                "elapsed_sec": f"{stats['elapsed_sec']:.3f}",
            }
        )

    with timing_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["partition", "merge", "n_cases", "n_failed", "n_skipped", "elapsed_sec"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] {len(rows)} combinations done timing={timing_path}")


if __name__ == "__main__":
    main()
