#!/usr/bin/env python3

import argparse
import csv
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run IQ-TREE sequentially on a directory of FASTA alignments with simple batch progress, "
            "per-alignment timing, ETA, CPU pinning, and CSV logging."
        )
    )
    parser.add_argument("alignments", help="Directory containing FASTA alignments.")
    parser.add_argument("output_root", help="Directory where IQ-TREE outputs are written.")
    parser.add_argument(
        "--iqtree-bin",
        default="bin/bin_linux/iqtree_2.2.0",
        help="Path to the IQ-TREE executable.",
    )
    parser.add_argument(
        "--model",
        default="LG+G4",
        help="Fixed IQ-TREE model string, e.g. LG+G4. Ignored when --model-dir is set.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Directory containing per-alignment model files named <stem>.model.txt. "
            "When provided, the script reads one raw model string per alignment and passes it "
            "directly to IQ-TREE."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Value passed to IQ-TREE via -T.",
    )
    parser.add_argument(
        "--cpu-set",
        default="0-3",
        help=(
            "CPU set passed to taskset -c. Example: 0-3 or 100-107. "
            "For IQ-TREE with -T > 1, prefer a CPU set wider than the thread count."
        ),
    )
    parser.add_argument(
        "--nice",
        type=int,
        default=10,
        help="nice priority value. Higher means lower priority.",
    )
    parser.add_argument(
        "--seqtype",
        default="AA",
        help="Value passed to IQ-TREE via --seqtype.",
    )
    parser.add_argument(
        "--include-glob",
        default="*.fa",
        help="Glob pattern used to select alignments.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output_root before running.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output_root and skip alignments already marked done with an existing tree.",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Do not delete per-prefix work files before rerun.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Pass --quiet to IQ-TREE.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass -v to IQ-TREE.",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Pass --redo to IQ-TREE.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries after the first failed attempt for one alignment.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Optional base seed. If set, attempt k for alignment i uses seed_base + i*1000 + k.",
    )
    parser.add_argument(
        "--show-iqtree-log",
        action="store_true",
        help="Stream IQ-TREE stdout/stderr to the terminal instead of writing them only to per-alignment log files.",
    )
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def cpu_set_size(spec: str) -> int:
    total = 0
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid cpu-set range: {chunk}")
            total += end - start + 1
        else:
            int(chunk)
            total += 1
    return total


def resolve_model(stem: str, args: argparse.Namespace) -> str:
    if args.model_dir is None:
        return args.model
    model_path = Path(args.model_dir) / f"{stem}.model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file for {stem}: {model_path}")
    return model_path.read_text(encoding="utf-8").strip()


def run_cmd(cmd: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    pinned_cpus = cpu_set_size(args.cpu_set)
    if args.threads > 1 and pinned_cpus <= args.threads:
        tqdm.write(
            "[WARN] cpu-set is not wider than --threads. "
            "On some PANDIT runs, IQ-TREE 2.2.0 failed with 'Tree taxa and alignment sequence do not match' "
            "under tight pinning. Prefer a wider cpu-set, e.g. --threads 2 --cpu-set 0-3 or "
            "--threads 4 --cpu-set 0-7."
        )

    alignments_dir = Path(args.alignments)
    model_dir = Path(args.model_dir) if args.model_dir is not None else None
    output_root = Path(args.output_root)
    work_dir = output_root / "work"
    trees_dir = output_root / "trees"
    logs_dir = output_root / "logs"
    runtime_csv = output_root / "runtime_per_alignment.csv"

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together.")

    if output_root.exists():
        if args.resume:
            pass
        elif not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use --overwrite to replace it."
            )
        else:
            shutil.rmtree(output_root)

    work_dir.mkdir(parents=True, exist_ok=True)
    trees_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    alignments = sorted(alignments_dir.glob(args.include_glob))
    if not alignments:
        raise FileNotFoundError(
            f"No alignments matched include_glob={args.include_glob!r} in {alignments_dir}"
        )
    if model_dir is not None and not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    existing_rows_by_id = {}
    if args.resume and runtime_csv.exists():
        with runtime_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                existing_rows_by_id[row["id"]] = row

    done_ids = set()
    if args.resume:
        for stem, row in existing_rows_by_id.items():
            tree_path = Path(row.get("tree_path", "")) if row.get("tree_path") else trees_dir / f"{stem}.nwk"
            if row.get("status") == "done" and tree_path.exists():
                done_ids.add(stem)

    rows_by_id = dict(existing_rows_by_id)
    total = len(alignments)
    pending_alignments = [aln for aln in alignments if aln.stem not in done_ids]
    skipped_done = total - len(pending_alignments)
    batch_start = time.perf_counter()
    elapsed_samples = []

    pbar = tqdm(
        pending_alignments,
        total=len(pending_alignments),
        desc="IQ-TREE",
        unit="msa",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    if args.resume:
        tqdm.write(
            f"[OK] resume mode: total={total} already_done={skipped_done} remaining={len(pending_alignments)}"
        )

    for run_index, aln in enumerate(pbar, start=1):
        index = skipped_done + run_index
        stem = aln.stem
        model = resolve_model(stem, args)
        prefix = work_dir / stem
        treefile = work_dir / f"{stem}.treefile"
        out_tree = trees_dir / f"{stem}.nwk"
        log_path = logs_dir / f"{stem}.iqtree.log"

        if not args.keep_work:
            for old in work_dir.glob(f"{stem}.*"):
                old.unlink()

        cmd = [
            "taskset",
            "-c",
            args.cpu_set,
            "nice",
            "-n",
            str(args.nice),
            str(args.iqtree_bin),
            "-s",
            str(aln),
            "--seqtype",
            args.seqtype,
            "-m",
            model,
            "-T",
            str(args.threads),
            "--prefix",
            str(prefix),
        ]
        if args.redo:
            cmd.append("--redo")
        if args.quiet:
            cmd.append("--quiet")
        if args.verbose:
            cmd.append("-v")

        pbar.set_description(f"IQ-TREE {index}/{total} {stem}")

        start = time.perf_counter()
        status = "done"
        note = ""
        attempts = 0
        max_attempts = 1 + max(0, args.retries)
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            attempt_cmd = list(cmd)
            if args.seed_base is not None:
                seed = args.seed_base + index * 1000 + attempt
                attempt_cmd.extend(["--seed", str(seed)])
            mode = "ab" if attempt > 1 and not args.show_iqtree_log else "wb"
            try:
                with log_path.open(mode) as log_handle:
                    if attempt > 1 and not args.show_iqtree_log:
                        header = (
                            f"\n\n=== RETRY {attempt}/{max_attempts} "
                            f"cmd={' '.join(shlex.quote(x) for x in attempt_cmd)} ===\n"
                        )
                        log_handle.write(header.encode("utf-8"))
                    subprocess.run(
                        attempt_cmd,
                        stdout=None if args.show_iqtree_log else log_handle,
                        stderr=None if args.show_iqtree_log else subprocess.STDOUT,
                        check=True,
                    )
                if treefile.exists():
                    shutil.copy2(treefile, out_tree)
                    status = "done"
                    note = "" if attempt == 1 else f"recovered_after_retry_{attempt}"
                    break
                status = "failed"
                note = "missing_treefile"
            except subprocess.CalledProcessError as exc:
                status = "failed"
                note = f"returncode={exc.returncode}"
            if attempt < max_attempts and not args.keep_work:
                for old in work_dir.glob(f"{stem}.*"):
                    old.unlink()
        elapsed = time.perf_counter() - start
        elapsed_samples.append(elapsed)

        done_count = len(elapsed_samples)
        mean_elapsed = sum(elapsed_samples) / done_count
        remaining = total - index
        eta_sec = mean_elapsed * remaining
        batch_elapsed = time.perf_counter() - batch_start

        pbar.set_postfix_str(
            f"status={status} last={format_seconds(elapsed)} avg={format_seconds(mean_elapsed)} "
            f"eta={format_seconds(eta_sec)}"
        )
        if status != "done":
            tqdm.write(
                f"[WARN] {stem} failed: {note} | cmd={' '.join(shlex.quote(x) for x in cmd)}"
            )

        rows_by_id[stem] = (
            {
                "id": stem,
                "alignment": str(aln),
                "status": status,
                "note": note,
                "attempts": attempts,
                "elapsed_sec": f"{elapsed:.6f}",
                "threads": args.threads,
                "cpu_set": args.cpu_set,
                "model": model,
                "log_path": str(log_path),
                "tree_path": str(out_tree) if out_tree.exists() else "",
            }
        )

        with runtime_csv.open("w", encoding="utf-8", newline="") as handle:
            ordered_rows = [rows_by_id[aln.stem] for aln in alignments if aln.stem in rows_by_id]
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "alignment",
                    "status",
                    "note",
                    "attempts",
                    "elapsed_sec",
                    "threads",
                    "cpu_set",
                    "model",
                    "log_path",
                    "tree_path",
                ],
            )
            writer.writeheader()
            writer.writerows(ordered_rows)

    pbar.close()
    total_elapsed = time.perf_counter() - batch_start
    final_rows = [rows_by_id[aln.stem] for aln in alignments if aln.stem in rows_by_id]
    done = sum(1 for row in final_rows if row["status"] == "done")
    failed = sum(1 for row in final_rows if row["status"] == "failed")
    print(
        f"[OK] total={total} done={done} failed={failed} total_elapsed={format_seconds(total_elapsed)} "
        f"log={runtime_csv}"
    )


if __name__ == "__main__":
    main()
