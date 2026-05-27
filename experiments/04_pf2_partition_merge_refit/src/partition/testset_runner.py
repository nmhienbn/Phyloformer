from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(path for path in SCRIPT_DIR.parents if (path / "third_party" / "phyloformer2").is_dir())
sys.path.insert(0, str(REPO_ROOT / "third_party" / "phyloformer2"))


PrepareCase = Callable[[Path, Path, argparse.Namespace], None]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def infer_blocks(case_root: Path, model, device) -> None:
    pf2_blocks = case_root / "pf2_blocks"
    if pf2_blocks.exists():
        shutil.rmtree(pf2_blocks)
    pf2_blocks.mkdir(parents=True, exist_ok=True)

    from infer import process_alns

    process_alns(
        msadir=str(case_root / "blocks"),
        model=model,
        outdir=str(pf2_blocks),
        device=device,
        mode="max-sample",
        nsamples=30,
        save_splits=False,
        measure_execution=False,
        verbose=False,
    )


def run_testset(args: argparse.Namespace, method_name: str, prepare_case: PrepareCase) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = str(args.cpu_threads)

    output_root = Path(args.output_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    (output_root / "cases").mkdir(parents=True, exist_ok=True)

    alignments = sorted(Path(args.alignments).glob(args.include_glob))
    if not alignments:
        raise ValueError(
            f"No alignments matched include_glob={args.include_glob!r} under {args.alignments!r}. "
            "For BigAln replicate directories, use ALIGN_DIR=data/bigaln_benchmark/<size> "
            'and --include-glob "*/big.fa".'
        )

    import torch
    from infer import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(torch.load(args.checkpoint, map_location=device), device)
    model.eval()

    rows = []
    stem_counts = Counter(path.stem for path in alignments)
    for alignment in tqdm(alignments, desc="cases", unit="aln"):
        case_id = alignment.parent.name if stem_counts[alignment.stem] > 1 else alignment.stem
        case_root = output_root / "cases" / case_id
        start = time.perf_counter()
        status = "done"
        note = ""
        try:
            prepare_case(alignment, case_root, args)
            infer_blocks(case_root, model, device)
        except Exception as exc:
            status = "failed"
            note = f"{type(exc).__name__}: {exc}"
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rows.append(
            {
                "id": case_id,
                "case_id": case_id,
                "alignment": str(alignment),
                "status": status,
                "note": note,
                "elapsed_sec": f"{time.perf_counter() - start:.6f}",
                "case_root": str(case_root),
            }
        )

    write_csv(output_root / "pipeline_summary.csv", rows)
    total = sum(float(row["elapsed_sec"]) for row in rows)
    write_csv(
        output_root / "runtime_summary.csv",
        [
            {
                "method": method_name,
                "n_alignments": str(len(rows)),
                "n_done": str(sum(row["status"] == "done" for row in rows)),
                "n_failed": str(sum(row["status"] == "failed" for row in rows)),
                "total_sec": f"{total:.6f}",
                "mean_sec_per_alignment": f"{total / len(rows):.6f}" if rows else "0.000000",
            }
        ],
    )
    print(f"[OK] wrote {output_root / 'pipeline_summary.csv'}")
