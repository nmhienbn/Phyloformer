#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


CHECKPOINT = "third_party/phyloformer2/pretrained/pf2.tch"
EPOCHS = 5
LEARNING_RATE = "1e-6"
WARMUP = "0.005"
BATCH_SIZE = 1
VALIDATE_EVERY = 500
CPU_THREADS = 4
CPU_SET = "102-105"
CUDA_VISIBLE_DEVICES = "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed PF2 fine-tune layout used in Chapter 3."
    )
    parser.add_argument("manifest_tsv")
    parser.add_argument("outdir")
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("relative_path") and row.get("tree_path")
        ]
    return sorted(rows, key=lambda row: row["msa_id"])


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def split_by_taxa(
    rows: list[dict[str, str]],
    val_frac: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["num_sequences"], []).append(row)

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for group in groups.values():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_frac)) if len(group) > 1 else 0
        val_rows.extend(group[:n_val])
        train_rows.extend(group[n_val:])
    return sorted(train_rows, key=lambda r: r["msa_id"]), sorted(val_rows, key=lambda r: r["msa_id"])


def link_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def materialize(rows: list[dict[str, str]], split: str, outdir: Path) -> list[dict[str, str]]:
    records = []
    for row in rows:
        taxa = row["num_sequences"]
        stem = row["msa_id"]
        root = outdir / f"{split}_{taxa}"
        tree_path = root / "trees" / f"{stem}.nwk"
        msa_path = root / "msas" / f"{stem}.fa"
        link_file(Path(row["tree_path"]), tree_path)
        link_file(Path(row["relative_path"]), msa_path)
        records.append(
            {
                "split": split,
                "taxa": taxa,
                "msa_id": stem,
                "tree_path": str(tree_path),
                "msa_path": str(msa_path),
            }
        )
    return records


def write_arg_files(outdir: Path, train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> None:
    args_dir = outdir / "args"
    args_dir.mkdir(parents=True, exist_ok=True)
    for name, split, rows, leaf in [
        ("train_trees", "train", train_rows, "trees"),
        ("train_alns", "train", train_rows, "msas"),
        ("val_trees", "val", val_rows, "trees"),
        ("val_alns", "val", val_rows, "msas"),
    ]:
        paths = sorted({str(outdir / f"{split}_{row['num_sequences']}" / leaf) for row in rows})
        (args_dir / f"{name}.args").write_text(" ".join(paths) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["split"], row["taxa"])
        counts[key] = counts.get(key, 0) + 1
    summary = [
        {"split": split, "taxa": taxa, "n_msa": str(count)}
        for (split, taxa), count in sorted(counts.items(), key=lambda item: (item[0][0], int(item[0][1])))
    ]
    write_tsv(path, ["split", "taxa", "n_msa"], summary)


def write_finetune_script(outdir: Path) -> None:
    script = outdir / "run_pf2_adapt_finetune.sh"
    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-{CPU_THREADS}}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-{CPU_THREADS}}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-{CPU_THREADS}}}"
export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-{CPU_THREADS}}}"
export TORCH_NUM_THREADS="${{TORCH_NUM_THREADS:-{CPU_THREADS}}}"
export PYTORCH_ALLOC_CONF="${{PYTORCH_ALLOC_CONF:-expandable_segments:True}}"
export PF_DATALOADER_WORKERS="${{PF_DATALOADER_WORKERS:-1}}"
export PF_DISABLE_WANDB_LOGGER=true
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-{CUDA_VISIBLE_DEVICES}}}"

TRAIN_TREES=($(cat {outdir}/args/train_trees.args))
TRAIN_ALNS=($(cat {outdir}/args/train_alns.args))
VAL_TREES=($(cat {outdir}/args/val_trees.args))
VAL_ALNS=($(cat {outdir}/args/val_alns.args))

taskset -c {CPU_SET} python third_party/phyloformer2/train.py finetune {CHECKPOINT} \\
  --batch-size {BATCH_SIZE} \\
  --base-batch-size {BATCH_SIZE} \\
  --epochs {EPOCHS} \\
  --warmup {WARMUP} \\
  --learning-rate {LEARNING_RATE} \\
  --unambiguous-order \\
  --project PF2_PANDIT_ADAPT \\
  --project-root runs \\
  --log-every 50 \\
  --validate-every {VALIDATE_EVERY} \\
  --early-stop-patience -1 \\
  --cache-root {outdir}/cache \\
  --train-trees "${{TRAIN_TREES[@]}}" \\
  --train-alns "${{TRAIN_ALNS[@]}}" \\
  --val-trees "${{VAL_TREES[@]}}" \\
  --val-alns "${{VAL_ALNS[@]}}"
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | 0o111)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = split_by_taxa(read_manifest(Path(args.manifest_tsv)), args.val_frac, args.seed)
    layout_rows = materialize(train_rows, "train", outdir) + materialize(val_rows, "val", outdir)

    write_tsv(outdir / "layout_manifest.tsv", list(layout_rows[0].keys()), layout_rows)
    write_arg_files(outdir, train_rows, val_rows)
    write_summary(outdir / "layout_summary.tsv", layout_rows)
    write_finetune_script(outdir)

    print(f"[OK] train MSAs: {len(train_rows)}")
    print(f"[OK] val MSAs: {len(val_rows)}")
    print(f"[OK] wrote {outdir / 'run_pf2_adapt_finetune.sh'}")


if __name__ == "__main__":
    main()
