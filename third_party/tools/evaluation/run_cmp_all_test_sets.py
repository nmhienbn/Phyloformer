#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = [
    "final_test_set",
    "LGGC+gaps",
    "cherry_test_data",
    "pastek_test_data",
]

# Keep the order stable so the resulting legends are predictable.
BASELINE_CANDIDATES = [
    "IQTree_LG+GC",
    "IQTree_MF",
    "FastTree",
    "FastME",
    "Hamming+FastME",
    "PF+FastME",
    "PF_MRE+FastME",
    "PF_cherry+FastME",
    "PF_pastek+FastME",
    "PFnoft+FastME",
    "BioNJ",
    "bionj",
]

TRADITIONAL_ONLY = {
    "IQTree_LG+GC",
    "IQTree_MF",
    "FastTree",
    "FastME",
    "BioNJ",
    "bionj",
}


def build_baselines(dataset_dir: Path, traditional_only: bool) -> tuple[list[str], list[str]]:
    results_dir = dataset_dir / "results"
    topo_args: list[str] = []
    dist_args: list[str] = []
    if not results_dir.exists():
        return topo_args, dist_args

    candidates = (
        [c for c in BASELINE_CANDIDATES if c in TRADITIONAL_ONLY]
        if traditional_only
        else BASELINE_CANDIDATES
    )
    for label in candidates:
        topo_path = results_dir / f"{label}_topo.csv"
        dist_path = results_dir / f"{label}_dist.csv"
        if topo_path.exists() and dist_path.exists():
            topo_args.extend(["--cmp-topo", f"{label}={topo_path}"])
            dist_args.extend(["--cmp-dist", f"{label}={dist_path}"])
    return topo_args, dist_args


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run shared make_plots2.py cmp mode for multiple test sets with auto-loaded baselines."
    )
    parser.add_argument("--model-label", required=True, help="Display label, e.g. evoPF")
    parser.add_argument(
        "--run-root",
        required=True,
        help="Directory containing eval_<dataset>/cmp files, e.g. runs/completed/eval/evopf",
    )
    parser.add_argument(
        "--cmp-prefix",
        required=True,
        help="Prefix of cmp outputs inside each eval dir, e.g. cmp_evopf",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Directory where per-dataset comparison figures will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset directory names under data/.",
    )
    parser.add_argument(
        "--traditional-only",
        action="store_true",
        help="Only compare against traditional baselines (IQTree/FastTree/FastME/BioNJ).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    plot_script = repo_root / "third_party" / "benchmark" / "make_plots2.py"
    run_root = (repo_root / args.run_root).resolve()
    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []

    for dataset in args.datasets:
        dataset_dir = repo_root / "data" / dataset
        eval_dir = run_root / f"eval_{dataset}"
        pred_topo = eval_dir / f"{args.cmp_prefix}_topo.csv"
        pred_dist = eval_dir / f"{args.cmp_prefix}_dist.csv"
        if not pred_topo.exists() or not pred_dist.exists():
            failures.append(
                f"{dataset}: missing predicted cmp files ({pred_topo.name}, {pred_dist.name})"
            )
            continue

        topo_args = ["--cmp-topo", f"{args.model_label}={pred_topo}"]
        dist_args = ["--cmp-dist", f"{args.model_label}={pred_dist}"]
        base_topo_args, base_dist_args = build_baselines(
            dataset_dir, traditional_only=args.traditional_only
        )
        topo_args.extend(base_topo_args)
        dist_args.extend(base_dist_args)

        outdir = out_root / dataset
        cmd = [
            sys.executable,
            str(plot_script),
            "--full",
            *topo_args,
            *dist_args,
            "--outdir",
            str(outdir),
        ]
        print(f"[cmp-all] {dataset}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    if failures:
        print("[cmp-all] Skipped datasets:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
