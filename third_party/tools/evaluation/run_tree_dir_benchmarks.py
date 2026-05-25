#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def parse_labelled_path(value: str) -> tuple[str, str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid value {value!r}. Expected METHOD|DATASET=path."
        )
    label, path = value.split("=", 1)
    if "|" not in label:
        raise argparse.ArgumentTypeError(
            f"Invalid label {label!r}. Expected METHOD|DATASET=path."
        )
    method, dataset = label.split("|", 1)
    return method.strip(), dataset.strip(), Path(path)


def parse_reference(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Invalid value {value!r}. Expected DATASET=path.")
    dataset, path = value.split("=", 1)
    return dataset.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one or more predicted tree directories against dataset references with phylocompare. "
            "Tree names are normalized by removing common PF2 database prefixes."
        )
    )
    parser.add_argument(
        "--reference",
        action="append",
        type=parse_reference,
        required=True,
        help="DATASET=reference_tree_dir. Repeat for each dataset.",
    )
    parser.add_argument(
        "--pred",
        action="append",
        type=parse_labelled_path,
        required=True,
        help="METHOD|DATASET=predicted_tree_dir. Repeat for each method/dataset.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for cmp_topo.csv/cmp_dist.csv files.",
    )
    parser.add_argument(
        "--bin-dir",
        default="bin/bin_linux",
        help="Directory containing phylocompare.",
    )
    parser.add_argument(
        "--keep-staged",
        action="store_true",
        help="Keep normalized staged tree dirs under outdir/staged.",
    )
    parser.add_argument(
        "--impute-missing-branch-lengths",
        type=float,
        default=None,
        help=(
            "If set, staged trees without branch lengths are copied with missing branch "
            "lengths filled by this value. This keeps topology metrics valid, but "
            "weighted RF/KF for those trees are based on imputed lengths."
        ),
    )
    parser.add_argument(
        "--cpu-set",
        default=None,
        help="Pin process to these CPUs via os.sched_setaffinity. Example: 0-3 or 100-107.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel phylocompare processes. Default: 1.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Threads passed to each phylocompare process (--threads). Default: 4.",
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


def normalize_stem(stem: str) -> str:
    for prefix in ("pandit__aa__", "treebase__aa__", "pandit__dna__", "treebase__dna__"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def collect_tree_map(tree_dir: Path) -> dict[str, Path]:
    paths = sorted(list(tree_dir.glob("*.nwk")) + list(tree_dir.glob("*.tre")) + list(tree_dir.glob("*.tree")))
    return {normalize_stem(path.stem): path for path in paths}


def add_missing_branch_lengths(newick: str, length: float) -> str:
    """Fill branch lengths absent from a simple Newick tree.

    The root branch is left unmodified because many Newick writers omit it.
    This is intended for parsimony trees such as MPBoot output that contain
    topology but no branch lengths.
    """
    branch_length = f":{length:g}"
    output: list[str] = []
    endpoint_open = False
    endpoint_has_length = False
    in_quote = False

    def finalize_endpoint() -> None:
        nonlocal endpoint_open, endpoint_has_length
        if endpoint_open and not endpoint_has_length:
            output.append(branch_length)
        endpoint_open = False
        endpoint_has_length = False

    for char in newick:
        if char == "'":
            in_quote = not in_quote
            output.append(char)
            endpoint_open = True
            continue
        if in_quote:
            output.append(char)
            continue

        if char == "(":
            output.append(char)
            endpoint_open = False
            endpoint_has_length = False
        elif char == ",":
            finalize_endpoint()
            output.append(char)
        elif char == ")":
            finalize_endpoint()
            output.append(char)
            endpoint_open = True
            endpoint_has_length = False
        elif char == ";":
            output.append(char)
            endpoint_open = False
            endpoint_has_length = False
        elif char == ":":
            output.append(char)
            endpoint_open = True
            endpoint_has_length = True
        elif char.isspace():
            output.append(char)
        else:
            output.append(char)
            endpoint_open = True

    return "".join(output)


def stage_tree(source: Path, target: Path, missing_branch_length: float | None) -> bool:
    if missing_branch_length is None:
        target.symlink_to(source.resolve())
        return False

    text = source.read_text(encoding="utf-8").strip()
    if ":" in text:
        target.symlink_to(source.resolve())
        return False

    target.write_text(
        add_missing_branch_lengths(text, missing_branch_length) + "\n",
        encoding="utf-8",
    )
    return True


def stage_pair(
    ref_dir: Path,
    pred_dir: Path,
    stage_root: Path,
    method: str,
    dataset: str,
    missing_branch_length: float | None,
) -> tuple[Path, Path, list[str], int]:
    ref_map = collect_tree_map(ref_dir)
    pred_map = collect_tree_map(pred_dir)
    common = sorted(set(ref_map).intersection(pred_map))
    if not common:
        raise RuntimeError(f"No common tree ids for {method}|{dataset}: ref={ref_dir}, pred={pred_dir}")

    safe_method = method.replace("/", "_").replace(" ", "_")
    safe_dataset = dataset.replace("/", "_").replace(" ", "_")
    ref_stage = stage_root / f"{safe_method}__{safe_dataset}" / "ref"
    pred_stage = stage_root / f"{safe_method}__{safe_dataset}" / "pred"
    if ref_stage.parent.exists():
        shutil.rmtree(ref_stage.parent)
    ref_stage.mkdir(parents=True, exist_ok=True)
    pred_stage.mkdir(parents=True, exist_ok=True)

    n_imputed = 0
    for tree_id in common:
        ref_target = ref_stage / f"{tree_id}.nwk"
        pred_target = pred_stage / f"{tree_id}.nwk"
        stage_tree(ref_map[tree_id], ref_target, missing_branch_length)
        if stage_tree(pred_map[tree_id], pred_target, missing_branch_length):
            n_imputed += 1
    return ref_stage, pred_stage, common, n_imputed


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def run_phylocompare(
    phylocompare_bin: Path,
    ref_stage: Path,
    pred_stage: Path,
    cmp_prefix: Path,
    method: str,
    log_path: Path,
    threads: int = 4,
) -> float:
    cmd = [
        str(phylocompare_bin),
        "-t",
        "-d",
        "-n",
        "--threads", str(threads),
        "-m",
        method,
        "-o",
        str(cmp_prefix),
        str(ref_stage),
        str(pred_stage),
    ]
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    if args.cpu_set:
        _apply_cpu_set(args.cpu_set)
    outdir = Path(args.outdir)
    cmp_dir = outdir / "cmp"
    log_dir = outdir / "logs"
    stage_root = outdir / "staged"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    references = dict(args.reference)
    phylocompare_bin = Path(args.bin_dir) / "phylocompare"
    if not phylocompare_bin.exists():
        raise FileNotFoundError(f"phylocompare binary not found: {phylocompare_bin}")

    # Stage all pairs first (fast, sequential)
    staged = []
    for method, dataset, pred_dir in args.pred:
        if dataset not in references:
            raise KeyError(f"No --reference was provided for dataset {dataset!r}")
        ref_dir = references[dataset]
        ref_stage, pred_stage, common, n_imputed = stage_pair(
            ref_dir=ref_dir,
            pred_dir=pred_dir,
            stage_root=stage_root,
            method=method,
            dataset=dataset,
            missing_branch_length=args.impute_missing_branch_lengths,
        )
        safe_method = method.replace("/", "_").replace(" ", "_")
        safe_dataset = dataset.replace("/", "_").replace(" ", "_")
        cmp_prefix = cmp_dir / f"{safe_method}__{safe_dataset}"
        log_path = log_dir / f"{safe_method}__{safe_dataset}.phylocompare.log"
        staged.append((method, dataset, pred_dir, ref_stage, pred_stage, common, n_imputed, cmp_prefix, log_path))

    # Run phylocompare in parallel
    results: dict[str, tuple[float, int]] = {}

    def _run(item):
        method, dataset, pred_dir, ref_stage, pred_stage, common, n_imputed, cmp_prefix, log_path = item
        elapsed = run_phylocompare(
            phylocompare_bin=phylocompare_bin,
            ref_stage=ref_stage,
            pred_stage=pred_stage,
            cmp_prefix=cmp_prefix,
            method=method,
            log_path=log_path,
            threads=args.threads,
        )
        return method, dataset, elapsed

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, item): item for item in staged}
        for fut in tqdm(as_completed(futures), total=len(staged), desc="phylocompare", unit="run"):
            method, dataset, elapsed = fut.result()
            results[(method, dataset)] = elapsed

    manifest_rows = []
    for method, dataset, pred_dir, ref_stage, pred_stage, common, n_imputed, cmp_prefix, log_path in staged:
        elapsed = results[(method, dataset)]
        manifest_rows.append(
            {
                "method": method,
                "dataset": dataset,
                "reference_dir": str(references[dataset]),
                "pred_dir": str(pred_dir),
                "n_common": len(common),
                "n_imputed_missing_branch_lengths": n_imputed,
                "imputed_branch_length": (
                    "" if args.impute_missing_branch_lengths is None else args.impute_missing_branch_lengths
                ),
                "cmp_topo": f"{cmp_prefix}_topo.csv",
                "cmp_dist": f"{cmp_prefix}_dist.csv",
                "log_path": str(log_path),
                "elapsed_sec": f"{elapsed:.6f}",
            }
        )

    manifest_path = outdir / "benchmark_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    if not args.keep_staged:
        shutil.rmtree(stage_root)
    print(f"[OK] wrote {manifest_path}")


if __name__ == "__main__":
    main()
