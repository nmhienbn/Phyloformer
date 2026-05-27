#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(path for path in SCRIPT_DIR.parents if (path / "third_party").is_dir())
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))


def _preseed_numba_threads(argv: list[str]) -> None:
    if "--cpu-threads" not in argv:
        return
    idx = argv.index("--cpu-threads")
    if idx + 1 >= len(argv):
        return
    os.environ["NUMBA_NUM_THREADS"] = argv[idx + 1]


_preseed_numba_threads(sys.argv)

from run_softbioblock_pf2 import prepare_alignment  # noqa: E402
from testset_runner import run_testset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SoftBioBlock prepare + PF2 block inference over a FASTA directory.")
    parser.add_argument("alignments")
    parser.add_argument("output_root")
    parser.add_argument("--checkpoint", default="models/phyloformer2/pf2.tch")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--include-glob", default="*.fa")
    parser.add_argument("--cap-length", type=int, default=None)
    parser.add_argument(
        "--cap-policy",
        choices=["profile-max", "practical", "formula"],
        default="practical",
        help="How to derive the default PF2 block-length cap.",
    )
    parser.add_argument(
        "--vram-gb",
        type=float,
        default=None,
        help="VRAM budget in GB. When set, forces formula cap policy.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_case(alignment: Path, case_root: Path, args: argparse.Namespace) -> None:
    prepare_alignment(
        argparse.Namespace(
            alignment=str(alignment),
            output_root=str(case_root),
            seqtype="AA",
            input_format="auto",
            gap_threshold=0.95,
            cap_length=args.cap_length,
            cap_policy=args.cap_policy,
            vram_gb=args.vram_gb,
            max_gmm_k=6,
            gmm_subsample=None,
            tau=0.20,
            max_membership=3,
            duplicate_budget=0.15,
            overlap_frac=0.05,
            cpu_threads=args.cpu_threads,
            checkpoint=args.checkpoint,
        )
    )


def main() -> None:
    run_testset(parse_args(), "PF2_SOFTBIOBLOCK", prepare_case)


if __name__ == "__main__":
    main()
