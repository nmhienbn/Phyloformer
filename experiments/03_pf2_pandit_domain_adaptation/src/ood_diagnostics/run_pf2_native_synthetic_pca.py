#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PF2-native synthetic MSAs for the Chapter 3 OOD PCA."
    )
    parser.add_argument("--outdir", default="runs/pandit_domain_adaptation/stage3_pf2_native_pca_fixed")
    parser.add_argument("--sampling", choices=["pf2-native-fixed", "pf2-native-size-matched"], default="pf2-native-fixed")
    parser.add_argument("--profile-tsv", default="runs/pandit_domain_adaptation/stage2_profiles/pandit_train_profile.tsv")
    parser.add_argument("--n-msas", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-taxa", type=int, default=50)
    parser.add_argument("--alignment-length", type=int, default=500)
    parser.add_argument("--iqtree-bin", default="bin/bin_linux/iqtree_2.2.0")
    parser.add_argument("--tree-simulator", default="third_party/phyloformer1/simulate_trees.py")
    parser.add_argument("--alignment-simulator", default="third_party/phyloformer1/alisim.py")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=200)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sample_records(args: argparse.Namespace) -> list[dict[str, str]]:
    rng = random.Random(args.seed)
    records = []
    profile = read_tsv(Path(args.profile_tsv)) if args.sampling == "pf2-native-size-matched" else []
    for idx in range(args.n_msas):
        if profile:
            source = rng.choice(profile)
            taxa = source["num_sequences"]
            length = source["alignment_length"]
            source_msa_id = source["msa_id"]
        else:
            taxa = str(args.n_taxa)
            length = str(args.alignment_length)
            source_msa_id = ""
        records.append(
            {
                "msa_id": f"syn_{idx:06d}",
                "num_sequences": taxa,
                "alignment_length": length,
                "source_train_msa_id": source_msa_id,
                "sampling": args.sampling,
            }
        )
    return records


def run(cmd: list[str], env: dict[str, str]) -> None:
    process = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode == 0:
        return
    print("[FAILED CMD]", " ".join(cmd), file=sys.stderr)
    print(process.stdout, file=sys.stderr)
    print(process.stderr, file=sys.stderr)
    raise subprocess.CalledProcessError(process.returncode, cmd)


def fasta_shape(path: Path) -> tuple[int, int]:
    n_records = 0
    first_length = 0
    seq_chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if seq_chunks:
                n_records += 1
                first_length = first_length or len("".join(seq_chunks))
                seq_chunks = []
        else:
            seq_chunks.append(line.strip())
    if seq_chunks:
        n_records += 1
        first_length = first_length or len("".join(seq_chunks))
    return n_records, first_length


def existing_msa_ok(path: Path, taxa: int, length: int) -> bool:
    return path.exists() and fasta_shape(path) == (taxa, length)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "msa_id",
        "split",
        "relative_path",
        "tree_path",
        "num_sequences",
        "alignment_length",
        "source_train_msa_id",
        "sampling",
        "model",
        "indels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    trees_root = outdir / "trees_by_taxa"
    msa_root = outdir / "msas"
    metadata_root = outdir / "metadata"
    for path in (trees_root, msa_root, metadata_root):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.cpu_threads)

    by_taxa: dict[int, list[dict[str, str]]] = {}
    for row in sample_records(args):
        by_taxa.setdefault(int(row["num_sequences"]), []).append(row)

    manifest_rows: list[dict[str, str]] = []
    progress = tqdm(total=sum(len(rows) for rows in by_taxa.values()), desc="PF2-native MSAs", unit="msa")
    for taxa, rows in sorted(by_taxa.items()):
        tree_dir = trees_root / f"taxa_{taxa}"
        tree_dir.mkdir(parents=True, exist_ok=True)
        if len(list(tree_dir.glob("*.nwk"))) < len(rows):
            run(
                [
                    sys.executable,
                    args.tree_simulator,
                    "--ntrees",
                    str(len(rows)),
                    "--ntips",
                    str(taxa),
                    "--type",
                    "birth-death",
                    "--output",
                    str(tree_dir),
                ],
                env,
            )

        for row, source_tree in zip(rows, sorted(tree_dir.glob("*.nwk"))[: len(rows)]):
            msa_id = row["msa_id"]
            length = int(row["alignment_length"])
            tree_dir_for_sample = metadata_root / msa_id / "tree"
            alisim_outdir = metadata_root / msa_id / "alisim_out"
            tree_dir_for_sample.mkdir(parents=True, exist_ok=True)
            tree_path = tree_dir_for_sample / f"{msa_id}.nwk"
            tree_path.write_text(source_tree.read_text(encoding="utf-8"), encoding="utf-8")

            msa_path = msa_root / f"{msa_id}.fa"
            if not existing_msa_ok(msa_path, taxa, length):
                if alisim_outdir.exists():
                    shutil.rmtree(alisim_outdir)
                run(
                    [
                        sys.executable,
                        args.alignment_simulator,
                        str(tree_dir_for_sample),
                        "--outdir",
                        str(alisim_outdir),
                        "--substitution",
                        "LG",
                        "--gamma",
                        "G8",
                        "--length",
                        str(length),
                        "--processes",
                        str(args.cpu_threads),
                        "--max-attempts",
                        str(args.max_attempts),
                        "--allow-duplicate-sequences",
                        "--keep-logfiles",
                        "--iqtree",
                        args.iqtree_bin,
                        "--indels",
                    ],
                    env,
                )
                shutil.move(str(alisim_outdir / f"{msa_id}.fa"), str(msa_path))

            manifest_rows.append(
                {
                    **row,
                    "split": "synthetic",
                    "relative_path": str(msa_path),
                    "tree_path": str(tree_path),
                    "model": "LG+G8",
                    "indels": "yes",
                }
            )
            progress.update(1)
    progress.close()

    write_manifest(outdir / "manifest.tsv", manifest_rows)
    (outdir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    print(f"[OK] wrote {outdir / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
