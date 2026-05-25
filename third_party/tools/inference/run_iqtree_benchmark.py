#!/usr/bin/env python3

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run IQ-TREE on a directory of FASTA alignments, compare predicted "
            "trees with reference trees, and optionally generate summary plots."
        )
    )
    parser.add_argument("alignments", help="Directory containing FASTA alignments.")
    parser.add_argument("output_root", help="Directory where benchmark outputs are written.")
    parser.add_argument(
        "--true-trees",
        required=True,
        help="Directory of reference trees with matching stems.",
    )
    parser.add_argument(
        "--iqtree-bin",
        default="bin/bin_linux/iqtree_2.2.0",
        help="Path to the IQ-TREE executable.",
    )
    parser.add_argument(
        "--bin-dir",
        default="bin/bin_linux",
        help="Directory containing phylocompare.",
    )
    parser.add_argument(
        "--model",
        default="LG+G4",
        help="IQ-TREE model string, e.g. LG+G4, LG+F+G4, MFP.",
    )
    parser.add_argument(
        "--threads",
        default="1",
        help="Value passed to IQ-TREE via -nt.",
    )
    parser.add_argument(
        "--method-name",
        default="IQ-TREE",
        help="Method label passed to phylocompare and plots.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch helper scripts.",
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
    return parser.parse_args()


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


def parse_time_file(path: Path) -> Tuple[float, int]:
    elapsed = 0.0
    max_rss_kb = -1
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("elapsed_sec="):
                elapsed = float(line.split("=", 1)[1])
            elif line.startswith("max_rss_kb="):
                max_rss_kb = int(float(line.split("=", 1)[1]))
    return elapsed, max_rss_kb


def write_runtime_summary(
    output_root: Path,
    method_name: str,
    per_tree_rows: List[dict],
    compare_sec: float,
) -> None:
    runtime_path = output_root / "runtime_summary.csv"
    n = len(per_tree_rows)
    infer_sec = sum(float(row["elapsed_sec"]) for row in per_tree_rows)
    total_sec = infer_sec + compare_sec
    max_rss_kb = max((int(row["max_rss_kb"]) for row in per_tree_rows), default=-1)
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
                "n_alignments": n,
                "infer_sec": f"{infer_sec:.6f}",
                "compare_sec": f"{compare_sec:.6f}",
                "total_sec": f"{total_sec:.6f}",
                "mean_sec_per_alignment": f"{(infer_sec / n) if n else 0.0:.6f}",
                "max_rss_kb": max_rss_kb,
            }
        )

    per_tree_path = output_root / "runtime_per_tree.csv"
    with per_tree_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "elapsed_sec", "max_rss_kb", "n_tips", "model"],
        )
        writer.writeheader()
        writer.writerows(per_tree_rows)


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    work_dir = output_root / "iqtree_work"
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

    iqtree_bin = Path(args.iqtree_bin)
    if not iqtree_bin.exists():
        raise FileNotFoundError(f"IQ-TREE binary not found: {iqtree_bin}")
    time_bin = shutil.which("time") or "/usr/bin/time"

    work_dir.mkdir(parents=True, exist_ok=True)
    pred_trees.mkdir(parents=True, exist_ok=True)

    per_tree_rows = []
    selected = sorted(list(alignments_dir.glob("*.fa")) + list(alignments_dir.glob("*.fasta")))
    if not selected:
        raise FileNotFoundError(f"No FASTA alignments found in {alignments_dir}")

    for aln in selected:
        stem = aln.stem
        prefix = work_dir / stem
        time_log = work_dir / f"{stem}.time.txt"
        iqtree_log = work_dir / f"{stem}.iqtree.log"
        cmd = [
            time_bin,
            "-f",
            "elapsed_sec=%e\nmax_rss_kb=%M",
            "-o",
            str(time_log),
            str(iqtree_bin),
            "-s",
            str(aln),
            "-pre",
            str(prefix),
            "-nt",
            str(args.threads),
            "-m",
            args.model,
            "-quiet",
            "-redo",
        ]
        print(f"[CMD] [{output_root.name}/iqtree] {' '.join(shlex.quote(x) for x in cmd)}")
        with iqtree_log.open("w", encoding="utf-8") as f_log:
            subprocess.run(cmd, check=True, stdout=f_log, stderr=subprocess.STDOUT)

        treefile = prefix.with_suffix(".treefile")
        if not treefile.exists():
            raise FileNotFoundError(f"Missing IQ-TREE output tree: {treefile}")
        shutil.copy2(treefile, pred_trees / f"{stem}.nwk")

        elapsed_sec, max_rss_kb = parse_time_file(time_log)
        per_tree_rows.append(
            {
                "id": stem,
                "elapsed_sec": f"{elapsed_sec:.6f}",
                "max_rss_kb": max_rss_kb,
                "n_tips": fasta_nseqs(aln),
                "model": args.model,
            }
        )

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
    compare_sec = run_cmd(cmp_cmd, f"{output_root.name}/compare")
    enrich_cmp_dist_with_n_tips(cmp_prefix)
    write_runtime_summary(output_root, args.method_name, per_tree_rows, compare_sec)

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
        run_cmd(plot_cmd, f"{output_root.name}/plot")


if __name__ == "__main__":
    main()
