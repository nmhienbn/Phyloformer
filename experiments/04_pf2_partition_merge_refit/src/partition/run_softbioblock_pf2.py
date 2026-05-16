#!/usr/bin/env python3
"""Prepare PF2 blocks with SoftBioBlock site-heterogeneity clustering."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio import AlignIO

from pf2_vram import derive_pf2_cap_length

GAP_CHARS = {"-", ".", "?"}
DNA_CHARS = set("ACGTUN")

# Feature weights applied after per-feature standardisation.
_WEIGHT_RATE = 3.0
_WEIGHT_ENTROPY = 2.0
_WEIGHT_GAP_PATTERN_ENTROPY = 1.0
_WEIGHT_INFORMATIVE = 1.5
_WEIGHT_GAP_RATIO = 1.0
_WEIGHT_COMPOSITION = 0.5


@dataclass
class BlockPartition:
    block_id: str
    parent_regime_id: str
    site_indices: list[int]
    site_probs: list[float]        # P[i,k] per site (GMM membership probability)
    mean_rate: float
    mean_entropy: float
    gap_ratio: float
    informative_ratio: float
    effective_length: float
    soft_block_weight: float = 0.0  # filled after repeat_count is known

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SoftBioBlock PF2 blocks.")
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
        "--max-gmm-k",
        type=int,
        default=6,
        help="Maximum number of GMM components to try when selecting K by BIC. Default: 6.",
    )
    parser.add_argument(
        "--gmm-subsample",
        type=int,
        default=None,
        metavar="N",
        help="Subsample N sites for GMM fitting when alignment is very long. Default: no subsampling.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.20,
        help="Membership threshold: site assigned to regime k if P[i,k] >= tau. Default: 0.20.",
    )
    parser.add_argument(
        "--max-membership",
        type=int,
        default=3,
        help="Maximum regimes a single site can be assigned to. Default: 3.",
    )
    parser.add_argument(
        "--duplicate-budget",
        type=float,
        default=0.15,
        help="Max extra (secondary) assignments as fraction of n_sites. Default: 0.15.",
    )
    parser.add_argument(
        "--overlap-frac",
        type=float,
        default=0.05,
        help="Fractional overlap between adjacent windows within a regime. Default: 0.05.",
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


def seq_alphabet(seqtype: str) -> list[str]:
    return sorted("ACGT") if seqtype == "DNA" else sorted("ACDEFGHIKLMNPQRSTVWY")


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


def site_entropy(col: np.ndarray, alphabet_size: int) -> float:
    chars = non_gap_chars(col)
    if not chars:
        return 0.0
    counts = Counter(chars)
    total = float(len(chars))
    h = -sum((v / total) * math.log2(v / total) for v in counts.values() if v > 0)
    return h / (math.log2(alphabet_size) if alphabet_size > 1 else 1.0)


def gap_pattern_entropy(col: np.ndarray) -> float:
    n = len(col)
    if n == 0:
        return 0.0
    n_gap = sum(1 for c in col if is_gap(c))
    if n_gap in (0, n):
        return 0.0
    p = n_gap / n
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def composition_vector(col: np.ndarray, alphabet: list[str]) -> list[float]:
    chars = non_gap_chars(col)
    if not chars:
        return [0.0] * len(alphabet)
    counts = Counter(chars)
    total = float(sum(counts.values()))
    return [counts.get(c, 0) / total for c in alphabet]


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


def build_feature_weights(alphabet: list[str]) -> np.ndarray:
    return np.array(
        [_WEIGHT_RATE, _WEIGHT_ENTROPY, _WEIGHT_GAP_PATTERN_ENTROPY,
         _WEIGHT_INFORMATIVE, _WEIGHT_GAP_RATIO]
        + [_WEIGHT_COMPOSITION] * len(alphabet),
        dtype=float,
    )


def site_feature_matrix(
    matrix: np.ndarray,
    global_rates: dict[int, float],
    global_to_local: dict[int, int],
    site_indices: list[int],
    alphabet: list[str],
) -> tuple[np.ndarray, list[dict]]:
    alpha_size = max(len(alphabet), 1)
    rows, stats = [], []
    for ci in site_indices:
        col = matrix[:, global_to_local[ci]]
        gr = float(sum(is_gap(c) for c in col.tolist())) / float(len(col))
        inf = 1.0 if is_parsimony_informative(col) else 0.0
        ent = site_entropy(col, alpha_size)
        gpe = gap_pattern_entropy(col)
        comp = composition_vector(col, alphabet)
        rows.append([global_rates.get(ci, 0.0), ent, gpe, inf, gr, *comp])
        stats.append({"rate": global_rates.get(ci, 0.0), "entropy": ent,
                      "gap_pattern_entropy": gpe, "gap_ratio": gr, "informative": inf})
    n_alpha = len(alphabet)
    if not rows:
        return np.zeros((0, 5 + n_alpha)), stats
    return np.array(rows, dtype=float), stats


def weighted_standardize(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return features
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (features - mean) / std * weights[np.newaxis, :]


def select_gmm_by_bic(
    X: np.ndarray,
    k_range: range | list[int] = range(2, 7),
    seed: int = 0,
) -> object:
    """Fit GaussianMixture for each K, return model with lowest BIC."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError("sklearn is required for SoftBioBlock-PF2: pip install scikit-learn") from exc

    best_bic, best_gmm = np.inf, None
    for k in k_range:
        k = min(int(k), len(X) - 1)
        if k < 2:
            continue
        try:
            gmm = GaussianMixture(
                n_components=k, covariance_type="full",
                random_state=seed, n_init=3, max_iter=200,
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm
        except Exception:
            continue

    if best_gmm is None:
        from sklearn.mixture import GaussianMixture
        best_gmm = GaussianMixture(n_components=2, random_state=seed, n_init=3)
        best_gmm.fit(X)
    return best_gmm


def overlap_windows(
    site_indices: list[int],
    confidences: list[float],
    cap_length: int,
    overlap_frac: float = 0.12,
) -> list[tuple[list[int], list[float]]]:
    """Slide cap_length windows with overlap_frac over positionally-sorted sites.

    Returns list of (site_indices_chunk, confidence_chunk) pairs.
    Tiny trailing windows are merged into the last window.
    """
    if not site_indices:
        return []

    # Sort by original alignment column position
    order = sorted(range(len(site_indices)), key=lambda i: site_indices[i])
    pos_sites = [site_indices[i] for i in order]
    pos_conf = [confidences[i] for i in order]

    if len(pos_sites) <= cap_length:
        return [(pos_sites, pos_conf)]

    step = max(1, int(cap_length * (1.0 - overlap_frac)))
    windows: list[tuple[list[int], list[float]]] = []
    i = 0
    while i < len(pos_sites):
        s = pos_sites[i : i + cap_length]
        c = pos_conf[i : i + cap_length]
        if len(s) < cap_length // 4 and windows:
            # Merge tiny last window; deduplicate keeping higher confidence
            prev_s, prev_c = windows[-1]
            site_to_conf: dict[int, float] = dict(zip(prev_s, prev_c))
            for si, ci in zip(s, c):
                site_to_conf[si] = max(site_to_conf.get(si, 0.0), ci)
            merged = sorted(site_to_conf)
            windows[-1] = (merged, [site_to_conf[x] for x in merged])
            break
        windows.append((s, c))
        i += step
    return windows


def build_block_partition(
    matrix: np.ndarray,
    block_id: str,
    parent_regime_id: str,
    site_indices: list[int],
    site_probs: list[float],
    global_rates: dict[int, float],
    global_to_local: dict[int, int],
    alphabet: list[str],
) -> BlockPartition:
    local_idx = [global_to_local[ci] for ci in site_indices]
    bm = matrix[:, local_idx]
    n = bm.shape[1]
    alpha_size = max(len(alphabet), 1)
    gap_ratios = [float(sum(is_gap(c) for c in bm[:, j].tolist())) / float(bm.shape[0]) for j in range(n)]
    inf_flags = [1.0 if is_parsimony_informative(bm[:, j]) else 0.0 for j in range(n)]
    entropies = [site_entropy(bm[:, j], alpha_size) for j in range(n)]
    return BlockPartition(
        block_id=block_id,
        parent_regime_id=parent_regime_id,
        site_indices=site_indices,
        site_probs=site_probs,
        mean_rate=float(np.mean([global_rates.get(ci, 0.0) for ci in site_indices])) if site_indices else 0.0,
        mean_entropy=float(np.mean(entropies)) if entropies else 0.0,
        gap_ratio=float(np.mean(gap_ratios)) if gap_ratios else 0.0,
        informative_ratio=float(np.mean(inf_flags)) if inf_flags else 0.0,
        effective_length=float(sum(1.0 - gr for gr in gap_ratios)),
    )


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
    alphabet = seq_alphabet(seqtype)
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
        "pipeline_version": "softbioblock_pf2",
        "input_alignment": str(input_path),
        "seqtype": seqtype,
        "n_sequences": len(ids),
        "original_length": matrix.shape[1],
        "filtered_length": filtered_len,
        "variant_sites": len(variant_columns),
        "invariant_sites": len(invariant_columns),
        "pf2_cap_length": cap_length,
        "overlap_frac": args.overlap_frac,
    }

    all_blocks: list[BlockPartition] = []

    # ------------------------------------------------------------------
    # Per-site informativeness weight: (1 - gap_ratio) * is_parsimony_informative
    # Used later to compute soft_block_weight.
    # ------------------------------------------------------------------
    site_info_map: dict[int, float] = {}
    for ci in kept_columns:
        li = global_to_local[ci]
        col = filtered[:, li]
        gr = float(sum(is_gap(c) for c in col.tolist())) / float(len(col))
        site_info_map[ci] = (1.0 - gr) * (1.0 if is_parsimony_informative(col) else 0.0)

    # ------------------------------------------------------------------
    # Fast path: entire filtered alignment fits in cap → single block
    # ------------------------------------------------------------------
    if filtered_len <= cap_length:
        block = build_block_partition(
            filtered, "direct_b000", "direct",
            kept_columns, [1.0] * len(kept_columns),
            rates, global_to_local, alphabet,
        )
        all_blocks = [block]
        pipeline_meta["routing"] = "direct"
        pipeline_meta["gmm_k"] = 1
        pipeline_meta["n_secondary_assignments"] = 0

    # ------------------------------------------------------------------
    # Partition path: GMM soft membership + overlap windows
    # ------------------------------------------------------------------
    else:
        features, _stats = site_feature_matrix(
            filtered, rates, global_to_local, kept_columns, alphabet
        )
        weights = build_feature_weights(alphabet)
        X = weighted_standardize(features, weights)

        if args.gmm_subsample and len(X) > args.gmm_subsample:
            rng = np.random.default_rng(0)
            sub_idx = rng.choice(len(X), size=args.gmm_subsample, replace=False)
            gmm = select_gmm_by_bic(X[sub_idx], k_range=range(2, args.max_gmm_k + 1))
        else:
            gmm = select_gmm_by_bic(X, k_range=range(2, args.max_gmm_k + 1))

        probs: np.ndarray = gmm.predict_proba(X)   # (n_sites, K)
        K_actual = gmm.n_components

        pipeline_meta["routing"] = "gmm"
        pipeline_meta["gmm_k"] = int(K_actual)
        pipeline_meta["gmm_bic"] = float(gmm.bic(X))

        # ----------------------------------------------------------
        # Soft membership assignment
        # Primary: argmax always kept.
        # Secondary: any k where P[i,k] >= tau, capped at max_membership total.
        # Global budget: at most duplicate_budget * n_sites extra assignments.
        # ----------------------------------------------------------
        tau = args.tau
        max_mem = args.max_membership

        regime_to_sites: dict[int, list[int]] = defaultdict(list)
        regime_to_probs: dict[int, list[float]] = defaultdict(list)

        secondary_candidates: list[tuple[float, int, int]] = []  # (prob, site_col, regime_k)

        for local_idx, ci in enumerate(kept_columns):
            p_row = probs[local_idx]
            order = np.argsort(p_row)[::-1]

            # Primary assignment (always)
            k0 = int(order[0])
            regime_to_sites[k0].append(ci)
            regime_to_probs[k0].append(float(p_row[k0]))

            # Secondary / tertiary candidates
            assigned = 1
            for rank in range(1, K_actual):
                if assigned >= max_mem:
                    break
                k = int(order[rank])
                p = float(p_row[k])
                if p >= tau:
                    secondary_candidates.append((p, ci, k))
                    assigned += 1

        # Apply global duplicate budget
        max_extra = int(args.duplicate_budget * len(kept_columns))
        secondary_candidates.sort(key=lambda t: t[0], reverse=True)
        for p, ci, k in secondary_candidates[:max_extra]:
            regime_to_sites[k].append(ci)
            regime_to_probs[k].append(p)

        pipeline_meta["n_secondary_assignments"] = min(max_extra, len(secondary_candidates))

        # Sort regimes by mean rate (slow → fast)
        regime_mean_rates = {
            k: float(np.mean([rates.get(ci, 0.0) for ci in sites]))
            if sites else 0.0
            for k, sites in regime_to_sites.items()
        }
        regime_order = sorted(regime_to_sites.keys(), key=lambda k: regime_mean_rates[k])

        block_counter = 0
        for k in regime_order:
            r_sites = regime_to_sites[k]
            r_probs = regime_to_probs[k]
            if not r_sites:
                continue
            regime_id = f"regime_{k:02d}"
            for w_sites, w_probs in overlap_windows(r_sites, r_probs, cap_length, args.overlap_frac):
                block = build_block_partition(
                    filtered,
                    f"{regime_id}_b{block_counter:04d}",
                    regime_id,
                    w_sites, w_probs,
                    rates, global_to_local, alphabet,
                )
                all_blocks.append(block)
                block_counter += 1

    # ------------------------------------------------------------------
    # Compute repeat_count per site, then soft_block_weight per block.
    # soft_block_weight = sum( site_info[ci] * P[ci,k] / repeat_count[ci] )
    # ------------------------------------------------------------------
    site_count: Counter[int] = Counter()
    for block in all_blocks:
        site_count.update(block.site_indices)

    for block in all_blocks:
        w = sum(
            site_info_map.get(ci, 0.0) * p / max(1, site_count[ci])
            for ci, p in zip(block.site_indices, block.site_probs)
        )
        block.soft_block_weight = w

    # ------------------------------------------------------------------
    # Export block FASTAs and manifest
    # ------------------------------------------------------------------
    block_rows = []
    for block in sorted(all_blocks, key=lambda b: (-len(b.site_indices), b.block_id)):
        local_idx = [global_to_local[ci] for ci in block.site_indices]
        block_path = blocks_dir / f"{block.block_id}.fa"
        write_fasta(block_path, ids, filtered[:, local_idx])

        block_rows.append({
            "block_id": block.block_id,
            "parent_regime_id": block.parent_regime_id,
            "path": str(block_path),
            "n_sequences": len(ids),
            "length": len(block.site_indices),
            "effective_length": f"{block.effective_length:.6f}",
            "mean_rate": f"{block.mean_rate:.6f}",
            "mean_entropy": f"{block.mean_entropy:.6f}",
            "gap_ratio": f"{block.gap_ratio:.6f}",
            "informative_ratio": f"{block.informative_ratio:.6f}",
            "soft_block_weight": f"{block.soft_block_weight:.8f}",
            "original_columns_json": json.dumps(block.site_indices),
        })

    manifest_path = output_root / "block_manifest.csv"
    fieldnames = [
        "block_id", "parent_regime_id", "path", "n_sequences", "length",
        "effective_length", "mean_rate", "mean_entropy", "gap_ratio",
        "informative_ratio", "soft_block_weight",
        "original_columns_json",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(block_rows)

    pipeline_meta["n_blocks"] = len(block_rows)
    pipeline_meta["n_sites_with_overlap"] = int(sum(len(b.site_indices) for b in all_blocks))
    pipeline_meta["n_unique_sites"] = int(len(site_count))
    pipeline_meta["tau"] = getattr(args, "tau", None)
    pipeline_meta["max_membership"] = getattr(args, "max_membership", None)
    pipeline_meta["duplicate_budget"] = getattr(args, "duplicate_budget", None)
    with (metadata_dir / "pipeline.json").open("w", encoding="utf-8") as fh:
        json.dump(pipeline_meta, fh, indent=2)

    # Helper shell script
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

    routing = pipeline_meta["routing"]
    k_info = f"gmm_k={pipeline_meta['gmm_k']}" if routing == "gmm" else "direct"
    print(
        f"[OK] seqtype={seqtype} filtered_len={filtered_len} cap={cap_length} "
        f"routing={routing} {k_info} blocks={len(block_rows)} "
        f"manifest={manifest_path}"
    )


def main() -> None:
    prepare_alignment(parse_args())


if __name__ == "__main__":
    main()
