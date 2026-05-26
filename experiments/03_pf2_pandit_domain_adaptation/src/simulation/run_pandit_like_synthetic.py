#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


GAP_THRESHOLDS = (0.0592369477912, 0.143665158371)
GAP_BASE_POLICY = {
    "low-gap": ("0.0035", "GEO{2},GEO{2}"),
    "mid-gap": ("0.006", "GEO{2},GEO{2}"),
    "high-gap": ("0.007", "GEO{3},GEO{3}"),
}
LENGTH_AWARE_POLICY = {
    "lt50": {
        "low-gap": ("0.002", "GEO{1},GEO{1}"),
        "mid-gap": ("0.003", "GEO{1},GEO{1}"),
        "high-gap": ("0.004", "GEO{2},GEO{2}"),
    },
    "50_99": {
        "low-gap": ("0.0025", "GEO{2},GEO{2}"),
        "mid-gap": ("0.0045", "GEO{2},GEO{2}"),
        "high-gap": ("0.0055", "GEO{2},GEO{2}"),
    },
    "100_299": {
        "low-gap": ("0.0035", "GEO{2},GEO{2}"),
        "mid-gap": ("0.006", "GEO{2},GEO{2}"),
        "high-gap": ("0.0065", "GEO{3},GEO{3}"),
    },
    "ge300": GAP_BASE_POLICY,
}
MANIFEST_FIELDS = [
    "msa_id",
    "split",
    "relative_path",
    "tree_path",
    "num_sequences",
    "alignment_length",
    "source_train_msa_id",
    "source_gap_ratio",
    "source_informative_site_ratio",
    "source_gamma_alpha",
    "model",
    "model_without_alpha",
    "best_model",
    "best_model_base",
    "has_G",
    "has_I",
    "has_F",
    "gap_bucket",
    "base_indel_rate",
    "base_indel_size",
    "indel_rate",
    "indel_size",
    "length_bin",
    "override_applied",
    "override_reason",
    "indel_fallback_note",
    "tree_bucket",
    "source_total_tree_length",
    "target_total_tree_length",
    "tree_scale",
    "indels",
]
BRANCH_LENGTH_RE = re.compile(r":([0-9eE.+-]+)")
GEO_RE = re.compile(r"GEO\{(\d+)\},GEO\{(\d+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the final Chapter 3 PANDIT-like synthetic dataset.")
    parser.add_argument(
        "--profile-tsv",
        default="runs/pandit_domain_adaptation/pandit_iqtree_model_analysis/pandit_train_iqtree_per_msa_joined.tsv",
    )
    parser.add_argument("--outdir", default="runs/pandit_domain_adaptation/stage5_synthetic_pandit_like_realism_v3_raw_20k")
    parser.add_argument("--n-msas", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iqtree-bin", default="bin/bin_linux/iqtree_2.2.0")
    parser.add_argument("--tree-simulator", default="third_party/phyloformer1/simulate_trees.py")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--memory-threshold-mib", type=float, default=70000)
    parser.add_argument("--memory-checkpoint", default="models/phyloformer2/pf2.tch")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def length_bin(length: int) -> str:
    if length < 50:
        return "lt50"
    if length < 100:
        return "50_99"
    if length < 300:
        return "100_299"
    return "ge300"


def gap_bucket(gap_ratio: str) -> str:
    value = float(gap_ratio or 0.0)
    if value <= GAP_THRESHOLDS[0]:
        return "low-gap"
    if value <= GAP_THRESHOLDS[1]:
        return "mid-gap"
    return "high-gap"


def indel_policy(length: int, bucket: str) -> dict[str, str]:
    base_rate, base_size = GAP_BASE_POLICY[bucket]
    final_rate, final_size = LENGTH_AWARE_POLICY[length_bin(length)][bucket]
    changed = base_rate != final_rate or base_size != final_size
    return {
        "base_indel_rate": base_rate,
        "base_indel_size": base_size,
        "indel_rate": final_rate,
        "indel_size": final_size,
        "length_bin": length_bin(length),
        "override_applied": "yes" if changed else "no",
        "override_reason": f"length_bin:{length_bin(length)}" if changed else "none",
    }


def inject_gamma_alpha(model: str, gamma_alpha: str) -> str:
    if not model or not gamma_alpha:
        return model
    parts = [part.strip() for part in model.split("+") if part.strip()]
    for idx, part in enumerate(parts):
        if idx and part.startswith("G") and "{" not in part:
            parts[idx] = f"{part}{{{gamma_alpha}}}"
            break
    return "+".join(parts)


def sample_records(profile: list[dict[str, str]], n_msas: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    records = []
    for idx in range(n_msas):
        row = rng.choice(profile)
        bucket = gap_bucket(row["gap_ratio"])
        policy = indel_policy(int(row["alignment_length"]), bucket)
        model = row.get("fitted_model") or "LG+G4"
        gamma_alpha = row.get("gamma_alpha", "")
        records.append(
            {
                "synthetic_id": f"pandit_like_{idx:06d}",
                "num_sequences": row["num_sequences"],
                "alignment_length": row["alignment_length"],
                "source_train_msa_id": row["msa_id"],
                "source_gap_ratio": row.get("gap_ratio", ""),
                "source_informative_site_ratio": row.get("informative_site_ratio", ""),
                "source_gamma_alpha": gamma_alpha,
                "model": inject_gamma_alpha(model, gamma_alpha),
                "model_without_alpha": model,
                "best_model": row.get("best_model", ""),
                "best_model_base": row.get("best_model_base", ""),
                "has_G": row.get("has_G", ""),
                "has_I": row.get("has_I", ""),
                "has_F": row.get("has_F", ""),
                "gap_bucket": bucket,
                **policy,
                "tree_bucket": "source-row-tree",
                "source_total_tree_length": row["total_tree_length"],
                "target_total_tree_length": row["total_tree_length"],
            }
        )
    return records


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def run_checked(cmd: list[str], env: dict[str, str]) -> None:
    process = run(cmd, env)
    if process.returncode == 0:
        return
    print("[FAILED CMD]", " ".join(cmd), file=sys.stderr)
    print(process.stdout, file=sys.stderr)
    print(process.stderr, file=sys.stderr)
    raise subprocess.CalledProcessError(process.returncode, cmd)


def parse_geo(raw: str) -> tuple[int, int]:
    match = GEO_RE.fullmatch(raw)
    if not match:
        raise ValueError(f"Unsupported AliSim indel-size: {raw}")
    return int(match.group(1)), int(match.group(2))


def format_geo(ins_mean: int, del_mean: int) -> str:
    return f"GEO{{{ins_mean}}},GEO{{{del_mean}}}"


def fallback_indels(rate: str, size: str, max_attempts: int) -> list[tuple[str, str, str]]:
    configs = [(rate, size, "planned")]
    rate_value = float(rate)
    ins_mean, del_mean = parse_geo(size)
    for retry in range(1, max_attempts):
        rate_value = max(0.001, rate_value * 0.7)
        ins_mean = max(2, int(round(ins_mean * 0.7)))
        del_mean = max(2, int(round(del_mean * 0.7)))
        configs.append((f"{rate_value:.12g}", format_geo(ins_mean, del_mean), f"fallback_retry_{retry}"))
    configs.append(("", "", "fallback_no_indels"))
    return configs


def simulate_alignment(
    args: argparse.Namespace,
    env: dict[str, str],
    tree_path: Path,
    out_prefix: Path,
    model: str,
    length: int,
    indel_rate: str,
    indel_size: str,
) -> dict[str, str]:
    for rate, size, note in fallback_indels(indel_rate, indel_size, args.max_attempts):
        indel_args = ["--indel", f"{rate},{rate}", "--indel-size", size] if rate and size else []
        cmd = [
            args.iqtree_bin,
            "--alisim",
            str(out_prefix),
            "-t",
            str(tree_path),
            "-m",
            model,
            "-mwopt",
            "-af",
            "fasta",
            "--seqtype",
            "AA",
            "--length",
            str(length),
            "--threads",
            "1" if indel_args else str(args.cpu_threads),
            *indel_args,
        ]
        process = run(cmd, env)
        if process.returncode == 0:
            return {"indel_rate": rate, "indel_size": size, "note": note}
        stderr = f"{process.stdout}\n{process.stderr}"
        if "Could not select a valid position" not in stderr:
            print("[FAILED CMD]", " ".join(cmd), file=sys.stderr)
            print(stderr, file=sys.stderr)
            raise subprocess.CalledProcessError(process.returncode, cmd)
    raise RuntimeError(f"AliSim indel fallback exhausted for {out_prefix}")


def trim_fasta(source: Path, target: Path, length: int) -> None:
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        header = None
        seq = ""
        for line in src:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    dst.write(f"{header}\n{seq[:length]}\n")
                header, seq = line, ""
            else:
                seq += line
        if header is not None:
            dst.write(f"{header}\n{seq[:length]}\n")


def fasta_shape(path: Path) -> tuple[int, int]:
    n_records = 0
    first_length = 0
    seq = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if seq:
                n_records += 1
                first_length = first_length or len(seq)
                seq = ""
        else:
            seq += line.strip()
    if seq:
        n_records += 1
        first_length = first_length or len(seq)
    return n_records, first_length


def scale_newick(newick: str, scale: float) -> str:
    return BRANCH_LENGTH_RE.sub(lambda match: f":{float(match.group(1)) * scale:.10g}", newick)


def total_tree_length(newick: str) -> float:
    return sum(float(match.group(1)) for match in BRANCH_LENGTH_RE.finditer(newick))


def estimate_memory(rows: list[dict[str, str]], checkpoint: Path) -> list[dict[str, str]]:
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu")
    hparams = dict(ckpt.get("hparams", {}))
    n_params = sum(value.numel() for value in ckpt.get("model", {}).values())
    h_dim = int(hparams.get("embed_dim", 128))
    pair_dim = int(hparams.get("pair_dim", 256))
    n_blocks = int(hparams.get("n_blocks", 12))
    n_heads = int(hparams.get("n_heads", 4))
    symmetric = bool(hparams.get("symmetric", True))

    def mib(elements: float) -> float:
        return elements * 4 / 1024**2

    risk_rows = []
    for row in rows:
        n = int(row["num_sequences"])
        length = int(row["alignment_length"])
        pair_count = n * (n - 1) // 2 if symmetric else n * n
        node_count = 2 * n - 1
        node_pair_count = node_count * (node_count - 1) // 2
        per_block = (
            length * n_heads * n * n
            + 10 * length * n * h_dim
            + pair_count * 32 * 32
            + 8 * pair_count * pair_dim
        )
        bayesnj_q = (n - 1) * (2 * node_pair_count * h_dim + node_pair_count + 3 * node_count * node_count)
        total_mib = mib(n_params * 4) + mib(per_block * n_blocks) * 2.0 + mib(bayesnj_q)
        risk_rows.append(
            {
                "msa_id": row["msa_id"],
                "num_sequences": row["num_sequences"],
                "alignment_length": row["alignment_length"],
                "pair_count": str(pair_count),
                "node_count": str(node_count),
                "total_flash_mib": f"{total_mib:.1f}",
            }
        )
    return sorted(risk_rows, key=lambda row: float(row["total_flash_mib"]), reverse=True)


def write_filtered_manifests(outdir: Path, rows: list[dict[str, str]], risk_rows: list[dict[str, str]], threshold: float) -> None:
    write_tsv(outdir / "manifest_full.tsv", MANIFEST_FIELDS, rows)
    write_tsv(outdir / "pf2_memory_risk.tsv", list(risk_rows[0].keys()), risk_rows)
    risk_by_id = {row["msa_id"]: float(row["total_flash_mib"]) for row in risk_rows}
    kept = [row for row in rows if risk_by_id[row["msa_id"]] <= threshold]
    excluded = [
        {**row, "estimated_total_flash_mib": f"{risk_by_id[row['msa_id']]:.1f}"}
        for row in rows
        if risk_by_id[row["msa_id"]] > threshold
    ]
    write_tsv(outdir / "manifest.tsv", MANIFEST_FIELDS, kept)
    write_tsv(outdir / "manifest_excluded_over_threshold.tsv", [*MANIFEST_FIELDS, "estimated_total_flash_mib"], excluded)
    print(f"[OK] kept by memory threshold: {len(kept)}")
    print(f"[OK] excluded by memory threshold: {len(excluded)}")


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
    for row in sample_records(read_tsv(Path(args.profile_tsv)), args.n_msas, args.seed):
        by_taxa.setdefault(int(row["num_sequences"]), []).append(row)

    manifest_rows: list[dict[str, str]] = []
    progress = tqdm(total=args.n_msas, desc="PANDIT-like synthetic MSAs", unit="msa")
    for taxa, rows in sorted(by_taxa.items()):
        tree_dir = trees_root / f"taxa_{taxa}"
        tree_dir.mkdir(parents=True, exist_ok=True)
        if len(list(tree_dir.glob("*.nwk"))) < len(rows):
            run_checked(
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
            msa_id = row["synthetic_id"]
            length = int(row["alignment_length"])
            tmp_root = metadata_root / msa_id
            tree_path = tmp_root / "tree" / f"{msa_id}.nwk"
            raw_tree_path = tmp_root / "tree" / f"{msa_id}.source.nwk"
            tree_path.parent.mkdir(parents=True, exist_ok=True)

            raw_newick = source_tree.read_text(encoding="utf-8")
            raw_tree_path.write_text(raw_newick, encoding="utf-8")
            source_tree_length = total_tree_length(raw_newick)
            scale = float(row["target_total_tree_length"]) / source_tree_length if source_tree_length else 1.0
            tree_path.write_text(scale_newick(raw_newick, scale), encoding="utf-8")

            msa_path = msa_root / f"{msa_id}.fa"
            applied = {"indel_rate": row["indel_rate"], "indel_size": row["indel_size"], "note": "existing_skip"}
            if not (msa_path.exists() and fasta_shape(msa_path) == (taxa, length)):
                alisim_prefix = tmp_root / "alisim" / msa_id
                alisim_prefix.parent.mkdir(parents=True, exist_ok=True)
                applied = simulate_alignment(
                    args,
                    env,
                    tree_path,
                    alisim_prefix,
                    row["model"],
                    length,
                    row["indel_rate"],
                    row["indel_size"],
                )
                generated = alisim_prefix.with_suffix(".fa")
                untrimmed = generated.with_suffix(".fa.untrimmed")
                shutil.move(str(generated), str(untrimmed))
                trim_fasta(untrimmed, msa_path, length)

            manifest_row = {key: row[key] for key in MANIFEST_FIELDS if key in row}
            manifest_row.update(
                {
                    "msa_id": msa_id,
                    "split": "synthetic_pandit_like",
                    "relative_path": str(msa_path),
                    "tree_path": str(tree_path),
                    "tree_scale": f"{scale:.12g}",
                    "indel_rate": applied["indel_rate"],
                    "indel_size": applied["indel_size"],
                    "indel_fallback_note": applied["note"],
                    "indels": "no" if not applied["indel_rate"] else f"{applied['indel_rate']},{applied['indel_rate']}",
                }
            )
            manifest_rows.append(manifest_row)
            progress.update(1)
    progress.close()

    risk_rows = estimate_memory(manifest_rows, Path(args.memory_checkpoint))
    write_filtered_manifests(outdir, manifest_rows, risk_rows, args.memory_threshold_mib)
    (outdir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    print(f"[OK] wrote {outdir / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
