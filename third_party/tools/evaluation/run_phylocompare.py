#!/usr/bin/env python3

import argparse
import subprocess
import time
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare predicted tree directory with true tree directory using phylocompare."
    )
    parser.add_argument(
        "--bin-dir",
        required=True,
        help="Directory containing the phylocompare binary.",
    )
    parser.add_argument(
        "--pred-trees",
        required=True,
        help="Directory containing predicted .nwk trees.",
    )
    parser.add_argument(
        "--true-trees",
        required=True,
        help="Directory containing ground-truth trees.",
    )
    parser.add_argument(
        "--cmp-out",
        required=True,
        help="Output prefix for phylocompare (-o).",
    )
    parser.add_argument(
        "--method-name",
        default="HybridA",
        help="Method label passed to phylocompare via -m.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label shown in progress/time logs.",
    )
    parser.add_argument(
        "--phylocompare-log",
        default=None,
        help="Path to phylocompare log file. Default: <pred-trees>/phylocompare.log.",
    )
    return parser.parse_args()


def run_compare(
    phylocompare_bin: Path,
    true_trees: Path,
    pred_trees: Path,
    cmp_out: Path,
    method_name: str,
    phylocompare_log: Path,
    label: str,
) -> float:
    cmd = [
        str(phylocompare_bin),
        "-t",
        "-d",
        "-n",
        "-m",
        method_name,
        "-o",
        str(cmp_out),
        str(true_trees),
        str(pred_trees),
    ]
    start = time.perf_counter()
    phylocompare_log.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=1, desc=f"phylocompare [{label}]", unit="job") as pbar:
        with open(phylocompare_log, "w", encoding="utf-8") as f_log:
            subprocess.run(
                cmd,
                check=True,
                stdout=f_log,
                stderr=subprocess.STDOUT,
            )
        pbar.update(1)
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()

    bin_dir = Path(args.bin_dir)
    phylocompare_bin = bin_dir / "phylocompare"
    if not phylocompare_bin.exists():
        raise FileNotFoundError(f"phylocompare binary not found: {phylocompare_bin}")

    pred_trees = Path(args.pred_trees)
    true_trees = Path(args.true_trees)
    label = args.label or pred_trees.parent.name or pred_trees.name
    phylocompare_log = (
        Path(args.phylocompare_log)
        if args.phylocompare_log is not None
        else pred_trees / "phylocompare.log"
    )

    print(f"[INFO] [{label}] Predicted trees: {pred_trees}")
    print(f"[INFO] [{label}] True trees: {true_trees}")
    print(f"[INFO] [{label}] phylocompare log: {phylocompare_log}")

    compare_sec = run_compare(
        phylocompare_bin=phylocompare_bin,
        true_trees=true_trees,
        pred_trees=pred_trees,
        cmp_out=Path(args.cmp_out),
        method_name=args.method_name,
        phylocompare_log=phylocompare_log,
        label=label,
    )
    print(f"[TIME] [{label}] phylocompare total: {compare_sec:.2f}s")


if __name__ == "__main__":
    main()
