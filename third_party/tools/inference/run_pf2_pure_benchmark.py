#!/usr/bin/env python3

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PF2 in pure tree-inference mode (no FastME), compare predicted "
            "trees with reference trees, and optionally generate summary plots."
        )
    )
    parser.add_argument("checkpoint", help="Path to the PF2 checkpoint.")
    parser.add_argument("alignments", help="Directory containing FASTA alignments.")
    parser.add_argument("output_root", help="Directory where benchmark outputs are written.")
    parser.add_argument(
        "--true-trees",
        required=True,
        help="Directory of reference trees with matching stems.",
    )
    parser.add_argument(
        "--bin-dir",
        default="bin/bin_linux",
        help="Directory containing phylocompare.",
    )
    parser.add_argument(
        "--method-name",
        default="PF2",
        help="Method label passed to phylocompare and plots.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the inference scripts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output root before running.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate summary plots from cmp_topo.csv and cmp_dist.csv after benchmarking.",
    )
    parser.add_argument(
        "--plot-outdir",
        default=None,
        help="Directory for plot outputs. Default: <output_root>/figures.",
    )
    parser.add_argument(
        "--plot-python",
        default=None,
        help="Python executable used for plotting. Default: same as --python.",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="Only keep alignments with number of sequences <= this limit.",
    )
    parser.add_argument(
        "--include-glob",
        default="*.fa",
        help="Glob pattern used to select alignments before benchmarking, e.g. 'prot_*.fa'.",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Pass --measure to third_party/phyloformer2/infer.py to collect per-alignment timing and memory stats.",
    )
    parser.add_argument(
        "--cpu-set",
        default=None,
        help="Pin process to these CPUs via os.sched_setaffinity. Example: 0-3 or 100-107.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose to third_party/phyloformer2/infer.py.",
    )
    return parser.parse_args()


def _apply_cpu_set(spec: str) -> None:
    cpus: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(chunk))
    os.sched_setaffinity(0, cpus)


def run_cmd(cmd: List[str], label: str) -> float:
    print(f"[CMD] [{label}] {' '.join(shlex.quote(x) for x in cmd)}")
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - start


def fasta_nseqs(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.startswith(b">"):
                count += 1
    return count


def build_filtered_subset(
    alignments_dir: Path,
    output_root: Path,
    true_trees_dir: Path,
    include_glob: str,
    max_seqs: Optional[int],
) -> Tuple[Path, Path]:
    subset_root = output_root / "_filtered_inputs"
    subset_alns = subset_root / "alignments"
    subset_trees = subset_root / "trees"
    subset_alns.mkdir(parents=True, exist_ok=True)
    subset_trees.mkdir(parents=True, exist_ok=True)

    selected = []
    for aln in sorted(alignments_dir.glob(include_glob)):
        if max_seqs is not None and fasta_nseqs(aln) > max_seqs:
            continue
        selected.append(aln)

    if not selected:
        raise FileNotFoundError(
            f"No alignments matched include_glob={include_glob!r} with max_seqs={max_seqs} in {alignments_dir}"
        )

    for aln in selected:
        target = subset_alns / aln.name
        if not target.exists():
            os.symlink(aln.resolve(), target)
        for ext in (".nwk", ".newick", ".tree"):
            src_tree = true_trees_dir / f"{aln.stem}{ext}"
            if src_tree.exists():
                target_tree = subset_trees / src_tree.name
                if not target_tree.exists():
                    os.symlink(src_tree.resolve(), target_tree)
                break

    print(
        f"[INFO] [filter] kept {len(selected)} alignments from {alignments_dir} "
        f"(glob={include_glob!r}, max_seqs={max_seqs})"
    )
    return subset_alns, subset_trees


def enrich_cmp_dist_with_n_tips(cmp_prefix: Path) -> None:
    topo_path = Path(f"{cmp_prefix}_topo.csv")
    dist_path = Path(f"{cmp_prefix}_dist.csv")
    if not topo_path.exists() or not dist_path.exists():
        return

    with topo_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        id_to_n_tips = {}
        for row in reader:
            tree_id = row.get("id")
            n_tips = row.get("n_tips")
            if tree_id and n_tips:
                id_to_n_tips[tree_id] = n_tips

    if not id_to_n_tips:
        return

    with dist_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if "n_tips" in fieldnames:
            return
        rows = list(reader)

    if not rows:
        return

    fieldnames.append("n_tips")
    for row in rows:
        row["n_tips"] = id_to_n_tips.get(row.get("id", ""), "")

    with dist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_runtime_summary(
    output_root: Path,
    method_name: str,
    infer_sec: float,
    compare_sec: float,
    n_alignments: int,
) -> None:
    total_sec = infer_sec + compare_sec
    runtime_path = output_root / "runtime_summary.csv"
    with runtime_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "n_alignments",
                "infer_sec",
                "compare_sec",
                "total_sec",
                "mean_sec_per_alignment",
                "max_rss_kb",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": method_name,
                "n_alignments": n_alignments,
                "infer_sec": f"{infer_sec:.6f}",
                "compare_sec": f"{compare_sec:.6f}",
                "total_sec": f"{total_sec:.6f}",
                "mean_sec_per_alignment": f"{(infer_sec / n_alignments) if n_alignments else 0.0:.6f}",
                "max_rss_kb": -1,
            }
        )


def main() -> None:
    args = parse_args()
    if args.cpu_set:
        _apply_cpu_set(args.cpu_set)

    output_root = Path(args.output_root)
    pred_trees = output_root / "trees"
    cmp_prefix = output_root / "cmp"

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use --overwrite to replace it."
            )
        subprocess.run(["rm", "-rf", str(output_root)], check=True)

    alignments_dir = Path(args.alignments)
    true_trees_dir = Path(args.true_trees)
    if args.max_seqs is not None or args.include_glob != "*.fa":
        alignments_dir, true_trees_dir = build_filtered_subset(
            alignments_dir=alignments_dir,
            output_root=output_root,
            true_trees_dir=true_trees_dir,
            include_glob=args.include_glob,
            max_seqs=args.max_seqs,
        )
    n_alignments = len(list(alignments_dir.glob("*.fa"))) + len(list(alignments_dir.glob("*.fasta")))

    infer_cmd = [
        args.python,
        "third_party/phyloformer2/infer.py",
        "--mode",
        "max-sample",
        str(alignments_dir),
        args.checkpoint,
        str(pred_trees),
    ]
    if args.measure:
        infer_cmd.append("--measure")
    if args.verbose:
        infer_cmd.append("--verbose")

    infer_sec = run_cmd(infer_cmd, f"{output_root.name}/infer")
    print(f"[TIME] [{output_root.name}] PF2 pure inference total: {infer_sec:.2f}s")

    cmp_cmd = [
        args.python,
        "third_party/tools/evaluation/run_phylocompare.py",
        "--bin-dir",
        args.bin_dir,
        "--pred-trees",
        str(pred_trees),
        "--true-trees",
        str(true_trees_dir),
        "--cmp-out",
        str(cmp_prefix),
        "--method-name",
        args.method_name,
        "--label",
        output_root.name,
    ]
    cmp_sec = run_cmd(cmp_cmd, f"{output_root.name}/compare")
    print(f"[TIME] [{output_root.name}] phylocompare total: {cmp_sec:.2f}s")

    enrich_cmp_dist_with_n_tips(cmp_prefix)
    write_runtime_summary(output_root, args.method_name, infer_sec, cmp_sec, n_alignments)

    if args.plot:
        plot_outdir = Path(args.plot_outdir) if args.plot_outdir is not None else output_root / "figures"
        plot_python = args.plot_python or args.python
        plot_cmd = [
            plot_python,
            "third_party/tools/plots/make_plots2.py",
            "--full",
            "--cmp-topo",
            f"{args.method_name}={cmp_prefix}_topo.csv",
            "--cmp-dist",
            f"{args.method_name}={cmp_prefix}_dist.csv",
            "--outdir",
            str(plot_outdir),
        ]
        plot_sec = run_cmd(plot_cmd, f"{output_root.name}/plot")
        print(f"[TIME] [{output_root.name}] plot total: {plot_sec:.2f}s")
        print(f"[INFO] [{output_root.name}] plots written to: {plot_outdir}")


if __name__ == "__main__":
    main()
