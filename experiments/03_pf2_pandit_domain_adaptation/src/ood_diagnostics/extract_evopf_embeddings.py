#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from Bio import AlignIO
from tqdm import tqdm


def repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "third_party" / "phyloformer2").is_dir():
            return path
    raise RuntimeError("Could not locate repo root with third_party/phyloformer2")


sys.path.insert(0, str(repo_root() / "third_party" / "phyloformer2"))

from BayesNJ.pf_sdk.pf.data import load_alignment as load_pf2_alignment  # noqa: E402
from infer import load_model  # noqa: E402


FASTA_EXTS = {".fa", ".fasta", ".faa", ".fas"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pooled EvoPF embeddings for the Chapter 3 PCA/OOD diagnostic."
    )
    parser.add_argument("manifest_tsv", help="TSV with msa_id and relative_path columns.")
    parser.add_argument("checkpoint", help="PF2 checkpoint, e.g. models/phyloformer2/pf2.tch.")
    parser.add_argument("outdir")
    parser.add_argument("--input-root", default="data/zenodo_raw")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows.sort(key=lambda row: (row.get("split", ""), row["msa_id"]))
    return rows


def alignment_formats(path: Path) -> list[str]:
    if path.suffix.lower() in FASTA_EXTS:
        return ["fasta"]
    if path.suffix.lower() in {".phy", ".phylip"}:
        return ["phylip-relaxed", "phylip", "fasta"]
    return ["phylip-relaxed", "phylip", "fasta"]


def ensure_fasta(source: Path, cache_dir: Path, msa_id: str) -> Path:
    if source.suffix.lower() in FASTA_EXTS and source.read_bytes()[:1] == b">":
        return source

    target = cache_dir / f"{msa_id}.fa"
    if target.exists() and target.stat().st_size > 0:
        return target

    last_error: Exception | None = None
    for fmt in alignment_formats(source):
        try:
            alignment = AlignIO.read(source, fmt)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                for record in alignment:
                    handle.write(f">{record.id}\n{str(record.seq).upper()}\n")
            return target
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not parse alignment {source}: {last_error}")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "msa_id",
        "split",
        "relative_path",
        "status",
        "note",
        "pooled_embedding",
        "num_sequences",
        "alignment_length",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)

    outdir = Path(args.outdir)
    pooled_dir = outdir / "pooled"
    cache_dir = outdir / "_fasta_inputs"
    pooled_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = load_model(checkpoint, device)
    model.eval()

    output_rows: list[dict[str, str]] = []
    input_root = Path(args.input_root)
    with torch.no_grad():
        for row in tqdm(read_tsv(Path(args.manifest_tsv)), desc="EvoPF embeddings", unit="msa"):
            msa_id = row["msa_id"]
            pooled_path = pooled_dir / f"{msa_id}.npy"
            output_row = {
                "msa_id": msa_id,
                "split": row.get("split", ""),
                "relative_path": row["relative_path"],
                "status": "done",
                "note": "",
                "pooled_embedding": str(pooled_path),
                "num_sequences": row.get("num_sequences", ""),
                "alignment_length": row.get("alignment_length", ""),
            }
            try:
                fasta = ensure_fasta(input_root / row["relative_path"], cache_dir, msa_id)
                msa, _ = load_pf2_alignment(str(fasta))
                features = model.extract_features(msa.to(device).float().unsqueeze(0))
                seq_repr = features["seq_repr"].squeeze(0).detach().cpu().numpy()
                np.save(pooled_path, seq_repr.mean(axis=0))
            except torch.cuda.OutOfMemoryError:
                output_row["status"] = "failed"
                output_row["note"] = "cuda_oom"
                torch.cuda.empty_cache()
            except Exception as exc:
                output_row["status"] = "failed"
                output_row["note"] = repr(exc)
            output_rows.append(output_row)

    write_manifest(outdir / "embedding_manifest.tsv", output_rows)
    print(f"[OK] wrote {outdir / 'embedding_manifest.tsv'}")


if __name__ == "__main__":
    main()
