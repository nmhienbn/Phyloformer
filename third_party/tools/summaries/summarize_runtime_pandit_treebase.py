#!/usr/bin/env python3

import argparse
import csv
import math
import re
from pathlib import Path


METHOD_ORDER = ["IQTREE", "FASTTREE", "PF2", "MPBOOT_SPR3", "MPBOOT_SPR6"]
DATASET_ORDER = ["PANDIT", "TREEBASE"]

DEFAULT_CLASSICAL_SOURCES = {
    ("PANDIT", "IQTREE"): Path("runs/completed/topology/iqtree_modeltxt_pandit_aa/runtime_per_alignment.csv"),
    ("PANDIT", "FASTTREE"): Path("runs/completed/topology/fasttree_pandit_aa/runtime_per_alignment.csv"),
    ("PANDIT", "MPBOOT_SPR3"): Path("runs/completed/topology/mpboot_spr3_uniform_pandit_aa/runtime_per_alignment.csv"),
    ("PANDIT", "MPBOOT_SPR6"): Path("runs/completed/topology/mpboot_spr6_uniform_pandit_aa/runtime_per_alignment.csv"),
    ("TREEBASE", "IQTREE"): Path("runs/completed/topology/iqtree_treebase_aa/runtime_per_alignment.csv"),
    ("TREEBASE", "FASTTREE"): Path("runs/completed/topology/fasttree_treebase_aa/runtime_per_alignment.csv"),
    ("TREEBASE", "MPBOOT_SPR3"): Path("runs/completed/topology/mpboot_spr3_uniform_treebase_aa/runtime_per_alignment.csv"),
    ("TREEBASE", "MPBOOT_SPR6"): Path("runs/completed/topology/mpboot_spr6_uniform_treebase_aa/runtime_per_alignment.csv"),
}

DEFAULT_PF2_RUNTIME_SUMMARY = {
    "PANDIT": Path("runs/completed/topology/runtime_pf2_pandit_aa/runtime_summary.csv"),
    "TREEBASE": Path("runs/completed/topology/runtime_pf2_treebase_aa/runtime_summary.csv"),
}

DEFAULT_PF2_EXEC_DIR = {
    "PANDIT": Path("runs/completed/topology/runtime_pf2_pandit_aa/exec"),
    "TREEBASE": Path("runs/completed/topology/runtime_pf2_treebase_aa/exec"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write one total-runtime table for IQ-TREE, FastTree, PF2, MPBoot SPR3, and MPBoot SPR6 on PANDIT and TREEBASE."
    )
    parser.add_argument(
        "--out",
        default="runs/completed/topology/runtime_benchmark_pandit_treebase/runtime_total_by_method_dataset.tsv",
        help="Output TSV path.",
    )
    parser.add_argument("--pandit-iqtree-csv", default=str(DEFAULT_CLASSICAL_SOURCES[("PANDIT", "IQTREE")]))
    parser.add_argument("--pandit-fasttree-csv", default=str(DEFAULT_CLASSICAL_SOURCES[("PANDIT", "FASTTREE")]))
    parser.add_argument("--pandit-mpboot-spr3-csv", default=str(DEFAULT_CLASSICAL_SOURCES[("PANDIT", "MPBOOT_SPR3")]))
    parser.add_argument("--pandit-mpboot-spr6-csv", default=str(DEFAULT_CLASSICAL_SOURCES[("PANDIT", "MPBOOT_SPR6")]))
    parser.add_argument("--treebase-iqtree-csv", default=str(DEFAULT_CLASSICAL_SOURCES[("TREEBASE", "IQTREE")]))
    parser.add_argument("--treebase-fasttree-csv", default=str(DEFAULT_CLASSICAL_SOURCES[("TREEBASE", "FASTTREE")]))
    parser.add_argument("--treebase-mpboot-spr3-csv", default=str(DEFAULT_CLASSICAL_SOURCES[("TREEBASE", "MPBOOT_SPR3")]))
    parser.add_argument("--treebase-mpboot-spr6-csv", default=str(DEFAULT_CLASSICAL_SOURCES[("TREEBASE", "MPBOOT_SPR6")]))
    parser.add_argument("--pandit-pf2-runtime-summary", default=str(DEFAULT_PF2_RUNTIME_SUMMARY["PANDIT"]))
    parser.add_argument("--treebase-pf2-runtime-summary", default=str(DEFAULT_PF2_RUNTIME_SUMMARY["TREEBASE"]))
    parser.add_argument("--pandit-pf2-exec-dir", default=str(DEFAULT_PF2_EXEC_DIR["PANDIT"]))
    parser.add_argument("--treebase-pf2-exec-dir", default=str(DEFAULT_PF2_EXEC_DIR["TREEBASE"]))
    return parser.parse_args()


def parse_pf2_time_file(path: Path) -> float:
    elapsed_sec = math.nan
    elapsed_re = re.compile(r"Elapsed .*:\s*(\d+):(\d+(?:\.\d+)?)")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = elapsed_re.search(line.strip())
            if match:
                elapsed_sec = int(match.group(1)) * 60.0 + float(match.group(2))
    return elapsed_sec


def load_classical_total(path: Path) -> tuple[float | None, int, int]:
    if not path.exists():
        return None, 0, 0
    total = 0.0
    n_total = 0
    n_done = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n_total += 1
            if row.get("status") == "done" and row.get("elapsed_sec"):
                total += float(row["elapsed_sec"])
                n_done += 1
    return total, n_total, n_done


def load_pf2_total(runtime_summary_path: Path, exec_dir: Path) -> tuple[float | None, int, int]:
    if runtime_summary_path.exists():
        with runtime_summary_path.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle), None)
        if row is not None:
            return float(row["total_sec"]), int(row["n_alignments"]), int(row["n_alignments"])

    if exec_dir.exists():
        total = 0.0
        n_done = 0
        for path in sorted(exec_dir.glob("*.time")):
            elapsed_sec = parse_pf2_time_file(path)
            if math.isnan(elapsed_sec):
                continue
            total += elapsed_sec
            n_done += 1
        n_total = n_done + len(list(exec_dir.glob("*.time.oom")))
        if n_done > 0 or n_total > 0:
            return total, n_total, n_done

    return None, 0, 0


def main() -> None:
    args = parse_args()
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    sources = {
        ("PANDIT", "IQTREE"): Path(args.pandit_iqtree_csv),
        ("PANDIT", "FASTTREE"): Path(args.pandit_fasttree_csv),
        ("PANDIT", "MPBOOT_SPR3"): Path(args.pandit_mpboot_spr3_csv),
        ("PANDIT", "MPBOOT_SPR6"): Path(args.pandit_mpboot_spr6_csv),
        ("TREEBASE", "IQTREE"): Path(args.treebase_iqtree_csv),
        ("TREEBASE", "FASTTREE"): Path(args.treebase_fasttree_csv),
        ("TREEBASE", "MPBOOT_SPR3"): Path(args.treebase_mpboot_spr3_csv),
        ("TREEBASE", "MPBOOT_SPR6"): Path(args.treebase_mpboot_spr6_csv),
    }

    rows = []
    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            if method == "PF2":
                total_sec, n_total, n_done = load_pf2_total(
                    Path(args.pandit_pf2_runtime_summary if dataset == "PANDIT" else args.treebase_pf2_runtime_summary),
                    Path(args.pandit_pf2_exec_dir if dataset == "PANDIT" else args.treebase_pf2_exec_dir),
                )
            else:
                total_sec, n_total, n_done = load_classical_total(sources[(dataset, method)])
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "total_sec": "" if total_sec is None else f"{total_sec:.6f}",
                    "total_hms": "" if total_sec is None else seconds_to_hms(total_sec),
                    "n_total": n_total,
                    "n_done": n_done,
                }
            )

    rows.sort(key=lambda row: (DATASET_ORDER.index(row["dataset"]), METHOD_ORDER.index(row["method"])))
    with outpath.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "method", "total_sec", "total_hms", "n_total", "n_done"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote {outpath}")


def seconds_to_hms(total_sec: float) -> str:
    total_sec = int(round(total_sec))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    main()
