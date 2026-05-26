#!/usr/bin/env python3
"""Prepare PF2 blocks with positional or rate-sorted sliding windows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio import AlignIO

from third_party.tools.vram.pf2_vram import derive_pf2_cap_length

GAP_CHARS = {"-", ".", "?"}
DNA_CHARS = set("ACGTUN")


@dataclass
class WindowBlock:
    block_id: str
    site_indices: list[int]
    mean_rate: float
    gap_ratio: float
    informative_ratio: float
    effective_length: float
    soft_block_weight: float = 0.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WindowPF2 blocks.")
    parser.add_argument("alignment")
    parser.add_argument("output_root")
    parser.add_argument("--seqtype", choices=["auto", "DNA", "AA"], default="auto")
    parser.add_argument(
        "--input-format",
        choices=["auto", "fasta", "phylip", "phylip-relaxed"],
        default="auto",
    )
    parser.add_argument("--gap-threshold", type=float, default=0.95)
    parser.add_argument("--cap-length", type=int, default=None)
    parser.add_argument(
        "--cap-policy", choices=["profile-max", "practical", "formula"], default="practical",
        help="How to derive the default PF2 block-length cap. Use 'formula' with --vram-gb.",
    )
    parser.add_argument(
        "--vram-gb", type=float, default=None,
        help="VRAM budget in GB. When set, forces --cap-policy=formula.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["position", "rate"],
        default="position",
        help="Sort sites by alignment column position or by fastTIGER-like rate before windowing. Default: position.",
    )
    parser.add_argument(
        "--overlap-frac",
        type=float,
        default=0.20,
        help="Fractional overlap between adjacent windows. Default: 0.20.",
    )
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()

def infer_alignment_format(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa", ".fas", ".aln"}:
        return ["fasta"]
    if suffix in {".phy", ".phylip"}:
        return ["phylip-relaxed", "phylip"]
    return ["phylip", "phylip-relaxed", "fasta"]


def load_alignment(path: Path, input_format: str) -> tuple[list[str], list[str]]:
    formats = [input_format] if input_format != "auto" else infer_alignment_format(path)
    last_error = None
    for fmt in formats:
        try:
            aln = AlignIO.read(path, fmt)
            return [r.id for r in aln], [str(r.seq).upper() for r in aln]
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Failed to parse {path}: {last_error}")


def infer_seqtype(seqs: list[str], seqtype: str) -> str:
    if seqtype != "auto":
        return seqtype
    observed = {c for seq in seqs for c in seq.upper() if c not in GAP_CHARS}
    return "DNA" if observed and observed.issubset(DNA_CHARS) else "AA"


def to_matrix(seqs: list[str]) -> np.ndarray:
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"Non-rectangular alignment: lengths={sorted(lengths)}")
    return np.array([list(s) for s in seqs], dtype="<U1")

def is_gap(c: str) -> bool:
    return c in GAP_CHARS


def non_gap_chars(col: np.ndarray) -> list[str]:
    return [c for c in col.tolist() if not is_gap(c)]


def is_variant_site(col: np.ndarray) -> bool:
    return len(set(non_gap_chars(col))) > 1


def is_parsimony_informative(col: np.ndarray) -> bool:
    counts = Counter(non_gap_chars(col))
    return sum(1 for v in counts.values() if v >= 2) >= 2

def preprocess_columns(
    matrix: np.ndarray, gap_threshold: float
) -> tuple[np.ndarray, list[int], list[int], list[int], list[float]]:
    kept, variant, invariant, gap_ratios = [], [], [], []
    for ci in range(matrix.shape[1]):
        col = matrix[:, ci]
        gr = float(sum(is_gap(c) for c in col.tolist())) / float(len(col))
        if gr == 1.0 or gr > gap_threshold:
            continue
        kept.append(ci)
        gap_ratios.append(gr)
        (variant if is_variant_site(col) else invariant).append(ci)
    filtered = matrix[:, kept] if kept else matrix[:, :0]
    return filtered, kept, variant, invariant, gap_ratios

def compute_pair_similarity(
    matrix: np.ndarray, count_gap_gap: bool = False
) -> tuple[np.ndarray, float]:
    nseq = matrix.shape[0]
    sim = np.zeros((nseq, nseq), dtype=float)
    total = 0.0
    for i in range(nseq):
        for j in range(i + 1, nseq):
            same = 0
            for a, b in zip(matrix[i], matrix[j]):
                if a != b:
                    continue
                if is_gap(a) or is_gap(b):
                    if count_gap_gap and is_gap(a) and is_gap(b):
                        same += 1
                    continue
                same += 1
            sim[i, j] = sim[j, i] = same
            total += same
    return sim, total


def fast_tiger_rates(
    matrix: np.ndarray,
    kept_columns: list[int],
    count_gap_gap: bool = False,
) -> dict[int, float]:
    if not kept_columns:
        return {}
    filtered = matrix[:, kept_columns]
    sim, total = compute_pair_similarity(filtered, count_gap_gap=count_gap_gap)
    if total <= 0:
        return {ci: 0.0 for ci in kept_columns}
    rates: dict[int, float] = {}
    for local_idx, ci in enumerate(kept_columns):
        col = filtered[:, local_idx]
        same_w = sum(
            sim[i, j]
            for i in range(filtered.shape[0])
            for j in range(i + 1, filtered.shape[0])
            if (
                col[i] == col[j]
                and (
                    (not is_gap(col[i]) and not is_gap(col[j]))
                    or (count_gap_gap and is_gap(col[i]) and is_gap(col[j]))
                )
            )
        )
        rates[ci] = 1.0 - (same_w / total)
    return rates

def sliding_windows(
    site_indices: list[int],
    cap_length: int,
    overlap_frac: float,
) -> list[list[int]]:
    """Slide cap_length windows over site_indices (already in desired order).

    Tiny trailing windows (< cap_length // 4) are merged into the last window.
    """
    if not site_indices:
        return []
    if len(site_indices) <= cap_length:
        return [site_indices]

    step = max(1, int(cap_length * (1.0 - overlap_frac)))
    windows: list[list[int]] = []
    i = 0
    while i < len(site_indices):
        chunk = site_indices[i : i + cap_length]
        if len(chunk) < cap_length // 4 and windows:
            # Merge tiny last chunk; preserve unique sites
            merged = sorted(set(windows[-1]) | set(chunk))
            windows[-1] = merged
            break
        windows.append(chunk)
        i += step
    return windows

def write_fasta(path: Path, ids: list[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for seq_id, row in zip(ids, matrix):
            fh.write(f">{seq_id}\n{''.join(row.tolist())}\n")

def prepare_alignment(args: argparse.Namespace) -> None:
    input_path = Path(args.alignment)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ids, seqs = load_alignment(input_path, args.input_format)
    seqtype = infer_seqtype(seqs, args.seqtype)
    matrix = to_matrix(seqs)

    filtered, kept_columns, variant_columns, invariant_columns, _ = preprocess_columns(
        matrix, gap_threshold=args.gap_threshold
    )
    global_to_local = {ci: li for li, ci in enumerate(kept_columns)}
    rates = fast_tiger_rates(
        matrix,
        kept_columns,
        count_gap_gap=getattr(args, "count_gap_gap", False),
    )

    _vram_gb = getattr(args, "vram_gb", None) or 16.0
    _cap_policy = "formula" if getattr(args, "vram_gb", None) is not None else args.cap_policy
    cap_length = args.cap_length or derive_pf2_cap_length(len(ids), _cap_policy, _vram_gb)
    filtered_len = filtered.shape[1]

    blocks_dir = output_root / "blocks"
    metadata_dir = output_root / "metadata"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    pipeline_meta: dict = {
        "pipeline_version": "window_pf2",
        "input_alignment": str(input_path),
        "seqtype": seqtype,
        "n_sequences": len(ids),
        "original_length": matrix.shape[1],
        "filtered_length": filtered_len,
        "variant_sites": len(variant_columns),
        "invariant_sites": len(invariant_columns),
        "pf2_cap_length": cap_length,
        "sort_by": args.sort_by,
        "overlap_frac": args.overlap_frac,
        "count_gap_gap": bool(getattr(args, "count_gap_gap", False)),
    }

    # Per-site informativeness weight: (1 - gap_ratio) * is_parsimony_informative
    site_info_map: dict[int, float] = {}
    for ci in kept_columns:
        li = global_to_local[ci]
        col = filtered[:, li]
        gr = float(sum(is_gap(c) for c in col.tolist())) / float(len(col))
        site_info_map[ci] = (1.0 - gr) * (1.0 if is_parsimony_informative(col) else 0.0)

    all_blocks: list[WindowBlock] = []

    if filtered_len <= cap_length:
        block = _make_block("win_b0000", kept_columns, filtered, global_to_local, rates)
        all_blocks = [block]
        pipeline_meta["routing"] = "direct"
        pipeline_meta["n_windows"] = 1
    else:
        if args.sort_by == "rate":
            ordered = sorted(kept_columns, key=lambda ci: rates.get(ci, 0.0))
        else:
            ordered = sorted(kept_columns)  # positional

        windows = sliding_windows(ordered, cap_length, args.overlap_frac)
        for idx, win_sites in enumerate(windows):
            block = _make_block(f"win_b{idx:04d}", win_sites, filtered, global_to_local, rates)
            all_blocks.append(block)

        pipeline_meta["routing"] = "windows"
        pipeline_meta["n_windows"] = len(windows)

    # Compute repeat_count per site, then soft_block_weight per block
    site_count: Counter[int] = Counter()
    for block in all_blocks:
        site_count.update(block.site_indices)

    for block in all_blocks:
        block.soft_block_weight = sum(
            site_info_map.get(ci, 0.0) / max(1, site_count[ci])
            for ci in block.site_indices
        )

    # Export block FASTAs and manifest
    block_rows = []
    for block in sorted(all_blocks, key=lambda b: (-len(b.site_indices), b.block_id)):
        local_idx = [global_to_local[ci] for ci in block.site_indices]
        block_path = blocks_dir / f"{block.block_id}.fa"
        write_fasta(block_path, ids, filtered[:, local_idx])

        block_rows.append({
            "block_id": block.block_id,
            "path": str(block_path),
            "n_sequences": len(ids),
            "length": len(block.site_indices),
            "effective_length": f"{block.effective_length:.6f}",
            "mean_rate": f"{block.mean_rate:.6f}",
            "gap_ratio": f"{block.gap_ratio:.6f}",
            "informative_ratio": f"{block.informative_ratio:.6f}",
            "soft_block_weight": f"{block.soft_block_weight:.8f}",
            "original_columns_json": json.dumps(block.site_indices),
        })

    manifest_path = output_root / "block_manifest.csv"
    fieldnames = [
        "block_id", "path", "n_sequences", "length",
        "effective_length", "mean_rate", "gap_ratio", "informative_ratio",
        "soft_block_weight", "original_columns_json",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(block_rows)

    pipeline_meta["n_blocks"] = len(block_rows)
    pipeline_meta["n_sites_with_overlap"] = int(sum(len(b.site_indices) for b in all_blocks))
    pipeline_meta["n_unique_sites"] = int(len(site_count))
    with (metadata_dir / "pipeline.json").open("w", encoding="utf-8") as fh:
        json.dump(pipeline_meta, fh, indent=2)

    ckpt = args.checkpoint or "<PF2_CHECKPOINT>"
    helper = output_root / "run_pf2_blocks.sh"
    with helper.open("w", encoding="utf-8") as fh:
        fh.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        fh.write(
            "python third_party/phyloformer2/infer.py --mode max-sample "
            f"{shlex.quote(str(blocks_dir))} "
            f"{shlex.quote(ckpt)} "
            f"{shlex.quote(str(output_root / 'pf2_blocks'))}\n"
        )
    os.chmod(helper, 0o755)

    print(
        f"[OK] seqtype={seqtype} filtered_len={filtered_len} cap={cap_length} "
        f"sort={args.sort_by} routing={pipeline_meta['routing']} "
        f"blocks={len(block_rows)} manifest={manifest_path}"
    )


def _make_block(
    block_id: str,
    site_indices: list[int],
    filtered: np.ndarray,
    global_to_local: dict[int, int],
    rates: dict[int, float],
) -> WindowBlock:
    local_idx = [global_to_local[ci] for ci in site_indices]
    bm = filtered[:, local_idx]
    n = bm.shape[1]
    gap_ratios = [float(sum(is_gap(c) for c in bm[:, j].tolist())) / float(bm.shape[0]) for j in range(n)]
    inf_flags = [1.0 if is_parsimony_informative(bm[:, j]) else 0.0 for j in range(n)]
    return WindowBlock(
        block_id=block_id,
        site_indices=site_indices,
        mean_rate=float(np.mean([rates.get(ci, 0.0) for ci in site_indices])) if site_indices else 0.0,
        gap_ratio=float(np.mean(gap_ratios)) if gap_ratios else 0.0,
        informative_ratio=float(np.mean(inf_flags)) if inf_flags else 0.0,
        effective_length=float(sum(1.0 - gr for gr in gap_ratios)),
    )


def main() -> None:
    prepare_alignment(parse_args())


if __name__ == "__main__":
    main()
