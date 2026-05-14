#!/usr/bin/env python3

import argparse
import subprocess
import time
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FASTME on .phy matrices and optionally compare trees with phylocompare."
    )
    parser.add_argument(
        "--bin-dir",
        required=True,
        help="Directory containing fastme and phylocompare binaries.",
    )
    parser.add_argument(
        "--mats-dir",
        required=True,
        help="Directory containing input .phy distance matrices.",
    )
    parser.add_argument(
        "--trees-dir",
        required=True,
        help="Directory where FASTME output trees (.nwk) will be written.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads passed to FASTME via -T.",
    )
    parser.add_argument(
        "--skip-nni",
        action="store_true",
        help="Disable FASTME --nni optimization.",
    )
    parser.add_argument(
        "--skip-spr",
        action="store_true",
        help="Disable FASTME --spr optimization.",
    )
    parser.add_argument(
        "--true-trees",
        default=None,
        help="Directory containing true trees for phylocompare.",
    )
    parser.add_argument(
        "--cmp-out",
        default=None,
        help="Output prefix for phylocompare (-o). Required with --true-trees.",
    )
    parser.add_argument(
        "--method-name",
        default="PF_QCLOSE+FastME",
        help="Method label passed to phylocompare via -m.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label shown in progress/time logs (e.g. dataset name).",
    )
    parser.add_argument(
        "--fastme-log-dir",
        default=None,
        help=(
            "Directory to store FASTME logs. "
            "Default: <trees-dir>/fastme_logs."
        ),
    )
    parser.add_argument(
        "--phylocompare-log",
        default=None,
        help=(
            "Path to phylocompare log file. "
            "Default: <trees-dir>/phylocompare.log."
        ),
    )
    return parser.parse_args()


def run_fastme(
    fastme_bin: Path,
    mats_dir: Path,
    trees_dir: Path,
    fastme_log_dir: Path,
    threads: int,
    use_nni: bool,
    use_spr: bool,
    label: str,
) -> tuple[int, float]:
    mats = sorted(mats_dir.glob("*.phy"))
    if not mats:
        raise FileNotFoundError(f"No .phy files found in {mats_dir}")

    trees_dir.mkdir(parents=True, exist_ok=True)
    fastme_log_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    for mat in tqdm(mats, desc=f"FASTME [{label}]", unit="tree"):
        out_tree = trees_dir / f"{mat.stem}.nwk"
        log_file = fastme_log_dir / f"{mat.stem}.log"
        cmd = [
            str(fastme_bin),
            "-T",
            str(threads),
            f"--input_data={mat}",
            f"--output_tree={out_tree}",
        ]
        if use_nni:
            cmd.append("--nni")
        if use_spr:
            cmd.append("--spr")
        with open(log_file, "w", encoding="utf-8") as f_log:
            subprocess.run(
                cmd,
                check=True,
                stdout=f_log,
                stderr=subprocess.STDOUT,
            )
    elapsed = time.perf_counter() - start
    return len(mats), elapsed


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

    if args.threads < 1:
        raise ValueError("--threads must be >= 1")

    bin_dir = Path(args.bin_dir)
    fastme_bin = bin_dir / "fastme"
    phylocompare_bin = bin_dir / "phylocompare"

    if not fastme_bin.exists():
        raise FileNotFoundError(f"fastme binary not found: {fastme_bin}")

    mats_dir = Path(args.mats_dir)
    trees_dir = Path(args.trees_dir)
    label = args.label or mats_dir.parent.name or mats_dir.name
    fastme_log_dir = (
        Path(args.fastme_log_dir)
        if args.fastme_log_dir is not None
        else trees_dir / "fastme_logs"
    )

    print(f"[INFO] [{label}] FASTME input: {mats_dir}")
    print(f"[INFO] [{label}] FASTME output: {trees_dir}")
    print(f"[INFO] [{label}] FASTME logs: {fastme_log_dir}")
    n_mats, fastme_sec = run_fastme(
        fastme_bin=fastme_bin,
        mats_dir=mats_dir,
        trees_dir=trees_dir,
        fastme_log_dir=fastme_log_dir,
        threads=args.threads,
        use_nni=not args.skip_nni,
        use_spr=not args.skip_spr,
        label=label,
    )
    print(
        f"[TIME] [{label}] FASTME total: {fastme_sec:.2f}s "
        f"({n_mats} matrices, {fastme_sec / n_mats:.2f}s/matrix)"
    )

    if args.true_trees is None and args.cmp_out is None:
        return
    if args.true_trees is None or args.cmp_out is None:
        raise ValueError("--true-trees and --cmp-out must be provided together")
    if not phylocompare_bin.exists():
        raise FileNotFoundError(f"phylocompare binary not found: {phylocompare_bin}")
    phylocompare_log = (
        Path(args.phylocompare_log)
        if args.phylocompare_log is not None
        else trees_dir / "phylocompare.log"
    )
    print(f"[INFO] [{label}] phylocompare log: {phylocompare_log}")

    compare_sec = run_compare(
        phylocompare_bin=phylocompare_bin,
        true_trees=Path(args.true_trees),
        pred_trees=trees_dir,
        cmp_out=Path(args.cmp_out),
        method_name=args.method_name,
        phylocompare_log=phylocompare_log,
        label=label,
    )
    print(f"[TIME] [{label}] phylocompare total: {compare_sec:.2f}s")
    print(f"[TIME] [{label}] FASTME + phylocompare total: {fastme_sec + compare_sec:.2f}s")


if __name__ == "__main__":
    main()
