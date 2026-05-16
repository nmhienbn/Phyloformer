#!/usr/bin/env python3
"""
Sinh protein superalignment từ tập cây PF1-style.

Mỗi replicate (= 1 cây):
  - Sinh gene liên tiếp với gene_len ~ lognormal (clip [100, 4500]) cho tới khi đủ target_sites
  - Model: LG+G8, alpha sample từ HOGENOM empirical
  - Concatenate thành 1 superalignment, kiểm tra duplicate ở cấp superalignment
  - Ghi partition.tsv và meta.json
"""

import argparse
import json
import os
import pickle
import random
import subprocess
from pathlib import Path

import numpy as np
from Bio import SeqIO
from tqdm import tqdm

# Lognormal params cho gene_len ~ Fungi protein distribution
# lognormal(mu, sigma) → mean ≈ exp(mu + sigma²/2)
# mu=6.215 (≈ln(500)), sigma=0.8 → mean≈557, mode≈368, clip [100, 4500]
GENE_LEN_MU = 6.215
GENE_LEN_SIGMA = 0.8
GENE_LEN_MIN = 100
GENE_LEN_MAX = 4500


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "experiments").is_dir() and (path / "third_party").is_dir():
            return path
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def sample_alpha(alphas):
    mean = random.choice(alphas)
    alpha = np.random.normal(loc=mean, scale=mean / 10)
    return max(alpha, 0.05)


def sample_gene_len():
    length = int(np.random.lognormal(mean=GENE_LEN_MU, sigma=GENE_LEN_SIGMA))
    return max(GENE_LEN_MIN, min(GENE_LEN_MAX, length))


def run_alisim(treefile, outfa, alpha, gene_len, iqtree, threads):
    model = f"LG+G8{{{alpha:.4f}}}"
    stem = outfa.replace(".fa", "")
    cmd = [
        iqtree, "--alisim", stem,
        "-t", treefile,
        "-m", model,
        "-mwopt", "-af", "fasta",
        "--seqtype", "AA",
        "--length", str(gene_len),
        "--threads", str(threads),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    return proc.returncode == 0


def has_duplicates(fa_path):
    seqs, n = set(), 0
    for rec in SeqIO.parse(fa_path, "fasta"):
        n += 1
        seqs.add(str(rec.seq))
    return n != len(seqs)


def concatenate_genes(gene_fas, gene_lens, out_fa):
    """Concatenate gene alignments theo thứ tự taxa nhất quán."""
    taxa_seqs = {}
    taxa_order = None
    for gfa, glen in zip(gene_fas, gene_lens):
        records = {rec.id: str(rec.seq) for rec in SeqIO.parse(gfa, "fasta")}
        if taxa_order is None:
            taxa_order = list(records.keys())
        for tid in taxa_order:
            seq = records[tid]
            # Trim về đúng gene_len nếu AliSim sinh dài hơn
            taxa_seqs.setdefault(tid, "")
            taxa_seqs[tid] += seq[:glen]
    with open(out_fa, "w") as fh:
        for tid in taxa_order:
            fh.write(f">{tid}\n{taxa_seqs[tid]}\n")


def main():
    parser = argparse.ArgumentParser(description="Simulate protein superalignments with true trees")
    parser.add_argument("--trees", required=True, help="Directory of .nwk tree files")
    parser.add_argument("--outdir", required=True, help="Output directory for replicates")
    parser.add_argument("--target-sites", type=int, required=True,
                        help="Target total sites per superalignment (e.g. 30000 for 30k)")
    parser.add_argument("--iqtree", default="iqtree2", help="Path to iqtree2 binary")
    parser.add_argument("--processes", type=int, default=1, help="Threads for AliSim per gene")
    parser.add_argument("--max-attempts", type=int, default=20,
                        help="Max retries per gene if AliSim fails")
    args = parser.parse_args()

    # Resolve empirical alpha distribution (PF1-style)
    data_dir = REPO_ROOT / "data"
    alphas_path = data_dir / "hogenom_alphas.txt"
    if not alphas_path.exists():
        raise FileNotFoundError(f"Missing {alphas_path}. Expected data/hogenom_alphas.txt")
    alphas = load_pickle(alphas_path)

    trees_dir = Path(args.trees)
    out_root = Path(args.outdir)
    treefiles = sorted(trees_dir.glob("*.nwk"))
    if not treefiles:
        raise FileNotFoundError(f"No .nwk files found in {trees_dir}")

    print(f"Found {len(treefiles)} trees → {len(treefiles)} replicates")
    print(f"Target sites: {args.target_sites:,}  |  gene_len ~ lognormal(mu={GENE_LEN_MU}, sigma={GENE_LEN_SIGMA}) clip [{GENE_LEN_MIN}, {GENE_LEN_MAX}]")

    for idx, treefile in enumerate(tqdm(treefiles, desc="Replicates")):
        rep_dir = out_root / f"rep_{idx:03d}"
        genes_dir = rep_dir / "genes"
        genes_dir.mkdir(parents=True, exist_ok=True)

        # Symlink true tree
        true_link = rep_dir / "true.nwk"
        if not true_link.exists():
            true_link.symlink_to(treefile.resolve())

        gene_fas = []
        gene_lens_actual = []
        partition = []
        pos = 1
        gene_idx = 0

        while pos - 1 < args.target_sites:
            gene_len = sample_gene_len()
            # Không lấy gene dài hơn phần còn thiếu (optional: tắt nếu muốn giữ đủ gene_len)
            alpha = sample_alpha(alphas)
            out_fa = str(genes_dir / f"gene_{gene_idx + 1:03d}.fa")

            ok = False
            for _ in range(args.max_attempts):
                run_alisim(str(treefile), out_fa, alpha, gene_len, args.iqtree, args.processes)
                if Path(out_fa).exists():
                    ok = True
                    break
                # AliSim thất bại, thử lại với alpha mới
                alpha = sample_alpha(alphas)

            if not ok:
                tqdm.write(f"[WARN] rep_{idx:03d} gene_{gene_idx+1:03d}: AliSim failed after {args.max_attempts} attempts, skipping")
                gene_idx += 1
                continue

            # Dọn file log của iqtree
            for ext in [".log", ".ckp.gz"]:
                p = Path(str(treefile) + ext)
                if p.exists():
                    p.unlink()

            gene_fas.append(out_fa)
            gene_lens_actual.append(gene_len)
            end = pos + gene_len - 1
            partition.append({
                "gene_id": f"gene_{gene_idx + 1:03d}",
                "start": pos,
                "end": end,
                "model": "LG",
                "gamma": "G8",
                "alpha": round(alpha, 4),
                "gene_len": gene_len,
            })
            pos = end + 1
            gene_idx += 1

        if not gene_fas:
            tqdm.write(f"[ERROR] rep_{idx:03d}: no genes generated, skipping")
            continue

        # Concatenate → super.fa
        super_fa = rep_dir / "super.fa"
        concatenate_genes(gene_fas, gene_lens_actual, str(super_fa))

        # Kiểm tra duplicate ở cấp superalignment (warn only — rất khó xảy ra với alignment dài)
        if has_duplicates(str(super_fa)):
            tqdm.write(f"[WARN] rep_{idx:03d}: superalignment has duplicate sequences")

        # Partition table
        part_tsv = rep_dir / "partition.tsv"
        with open(part_tsv, "w") as fh:
            fh.write("gene_id\tstart\tend\tmodel\tgamma\talpha\tgene_len\n")
            for row in partition:
                fh.write("\t".join(str(row[k]) for k in
                                   ["gene_id", "start", "end", "model", "gamma", "alpha", "gene_len"]) + "\n")

        # Meta
        n_taxa = sum(1 for _ in SeqIO.parse(str(super_fa), "fasta"))
        gene_lens_list = [r["gene_len"] for r in partition]
        meta = {
            "tree": str(treefile.resolve()),
            "n_taxa": n_taxa,
            "n_genes": len(gene_fas),
            "target_sites": args.target_sites,
            "total_sites": pos - 1,
            "gene_len_mean": round(float(np.mean(gene_lens_list)), 1),
            "gene_len_min": min(gene_lens_list),
            "gene_len_max": max(gene_lens_list),
        }
        with open(rep_dir / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main()
