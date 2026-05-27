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
from numba import njit, prange, set_num_threads

from third_party.tools.vram.pf2_vram import derive_pf2_cap_length
from fast_tiger import encode_alignment, fast_tiger_rates_encoded

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
    parser.add_argument("--cpu-threads", type=int, default=None)
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


@njit(cache=True, parallel=True)
def _preprocess_columns_numba(
    encoded: np.ndarray,
    gap_threshold: float,
    max_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nseq, nsites = encoded.shape
    keep_mask = np.zeros(nsites, dtype=np.bool_)
    variant_mask = np.zeros(nsites, dtype=np.bool_)
    gap_ratios = np.zeros(nsites, dtype=np.float64)

    for ci in prange(nsites):
        counts = np.zeros(max_code + 1, dtype=np.int64)
        n_gap = 0
        n_states = 0
        for ri in range(nseq):
            code = encoded[ri, ci]
            if code == 0:
                n_gap += 1
            else:
                if counts[code] == 0:
                    n_states += 1
                counts[code] += 1

        gr = n_gap / nseq
        gap_ratios[ci] = gr
        if gr != 1.0 and gr <= gap_threshold:
            keep_mask[ci] = True
            variant_mask[ci] = n_states > 1

    return keep_mask, variant_mask, gap_ratios

@njit(cache=True, parallel=True)
def _site_features_numba(
    encoded: np.ndarray,
    rates: np.ndarray,
    alphabet_codes: np.ndarray,
    alphabet_size: int,
    max_code: int,
) -> tuple[np.ndarray, np.ndarray]:
    nseq, nsites = encoded.shape
    features = np.zeros((nsites, 5 + alphabet_size), dtype=np.float64)
    site_info = np.zeros(nsites, dtype=np.float64)
    entropy_norm = math.log2(alphabet_size) if alphabet_size > 1 else 1.0

    for ci in prange(nsites):
        counts = np.zeros(max_code + 1, dtype=np.int64)
        n_gap = 0
        n_non_gap = 0
        informative_states = 0

        for ri in range(nseq):
            code = encoded[ri, ci]
            if code == 0:
                n_gap += 1
            else:
                counts[code] += 1
                n_non_gap += 1

        entropy = 0.0
        if n_non_gap > 0:
            for code in range(1, max_code + 1):
                count = counts[code]
                if count > 0:
                    p = count / n_non_gap
                    entropy -= p * math.log2(p)
                if count >= 2:
                    informative_states += 1
            entropy /= entropy_norm

        gap_ratio = n_gap / nseq
        gap_pattern_entropy = 0.0
        if n_gap != 0 and n_gap != nseq:
            p_gap = n_gap / nseq
            gap_pattern_entropy = -(p_gap * math.log2(p_gap) + (1.0 - p_gap) * math.log2(1.0 - p_gap))

        informative = 1.0 if informative_states >= 2 else 0.0
        features[ci, 0] = rates[ci]
        features[ci, 1] = entropy
        features[ci, 2] = gap_pattern_entropy
        features[ci, 3] = informative
        features[ci, 4] = gap_ratio
        for ai in range(alphabet_size):
            code = alphabet_codes[ai]
            features[ci, 5 + ai] = counts[code] / n_non_gap if n_non_gap > 0 else 0.0
        site_info[ci] = (1.0 - gap_ratio) * informative

    return features, site_info


@njit(cache=True)
def _block_stats_numba(
    local_idx: np.ndarray,
    rates: np.ndarray,
    entropies: np.ndarray,
    gap_ratios: np.ndarray,
    informative_flags: np.ndarray,
) -> tuple[float, float, float, float, float]:
    n = len(local_idx)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    rate_sum = 0.0
    entropy_sum = 0.0
    gap_sum = 0.0
    informative_sum = 0.0
    effective_length = 0.0
    for i in range(n):
        li = local_idx[i]
        rate_sum += rates[li]
        entropy_sum += entropies[li]
        gap_sum += gap_ratios[li]
        informative_sum += informative_flags[li]
        effective_length += 1.0 - gap_ratios[li]

    return (
        rate_sum / n,
        entropy_sum / n,
        gap_sum / n,
        informative_sum / n,
        effective_length,
    )

def preprocess_columns_encoded(
    matrix: np.ndarray,
    encoded: np.ndarray,
    gap_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[int], list[float]]:
    max_code = int(encoded.max()) if encoded.size else 0
    keep_mask, variant_mask, all_gap_ratios = _preprocess_columns_numba(
        encoded, gap_threshold, max_code
    )
    kept_array = np.flatnonzero(keep_mask).astype(np.int64)
    kept = kept_array.tolist()
    variant = [ci for ci in kept if bool(variant_mask[ci])]
    invariant = [ci for ci in kept if not bool(variant_mask[ci])]
    gap_ratios = [float(all_gap_ratios[ci]) for ci in kept]
    filtered = matrix[:, kept] if kept else matrix[:, :0]
    filtered_encoded = encoded[:, kept_array] if len(kept_array) else encoded[:, :0]
    return filtered, filtered_encoded, kept, variant, invariant, gap_ratios


def build_feature_weights(alphabet: list[str]) -> np.ndarray:
    return np.array(
        [_WEIGHT_RATE, _WEIGHT_ENTROPY, _WEIGHT_GAP_PATTERN_ENTROPY,
         _WEIGHT_INFORMATIVE, _WEIGHT_GAP_RATIO]
        + [_WEIGHT_COMPOSITION] * len(alphabet),
        dtype=float,
    )


def site_feature_matrix_encoded(
    filtered_encoded: np.ndarray,
    rate_values: np.ndarray,
    alphabet: list[str],
    code_by_char: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    alphabet_codes = np.array([code_by_char[c] for c in alphabet], dtype=np.int64)
    max_code = int(filtered_encoded.max()) if filtered_encoded.size else 0
    return _site_features_numba(
        filtered_encoded,
        rate_values,
        alphabet_codes,
        len(alphabet),
        max_code,
    )


def weighted_standardize(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return features
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (features - mean) / std * weights[np.newaxis, :]

# ===================================================
# ====================== GMM ========================
# ===================================================
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
    The final window is anchored at the end, so a short tail is padded by
    overlapping previous sites instead of creating a block longer than cap_length.
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
    while i + cap_length < len(pos_sites):
        windows.append((pos_sites[i : i + cap_length], pos_conf[i : i + cap_length]))
        i += step

    last_start = max(0, len(pos_sites) - cap_length)
    last = (pos_sites[last_start:], pos_conf[last_start:])
    if not windows or windows[-1][0] != last[0]:
        windows.append(last)
    return windows


def build_block_partition_from_metrics(
    block_id: str,
    parent_regime_id: str,
    site_indices: list[int],
    site_probs: list[float],
    global_to_local: dict[int, int],
    rate_values: np.ndarray,
    features: np.ndarray,
) -> BlockPartition:
    local_idx = np.array([global_to_local[ci] for ci in site_indices], dtype=np.int64)
    mean_rate, mean_entropy, gap_ratio, informative_ratio, effective_length = _block_stats_numba(
        local_idx,
        rate_values,
        features[:, 1],
        features[:, 4],
        features[:, 3],
    )
    return BlockPartition(
        block_id=block_id,
        parent_regime_id=parent_regime_id,
        site_indices=site_indices,
        site_probs=site_probs,
        mean_rate=float(mean_rate),
        mean_entropy=float(mean_entropy),
        gap_ratio=float(gap_ratio),
        informative_ratio=float(informative_ratio),
        effective_length=float(effective_length),
    )


def write_fasta(path: Path, ids: list[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for seq_id, row in zip(ids, matrix):
            fh.write(f">{seq_id}\n{''.join(row.tolist())}\n")


def prepare_alignment(args: argparse.Namespace) -> None:
    cpu_threads = getattr(args, "cpu_threads", None)
    if cpu_threads is not None:
        set_num_threads(max(1, int(cpu_threads)))

    input_path = Path(args.alignment)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ids, seqs = load_alignment(input_path, args.input_format)
    seqtype = infer_seqtype(seqs, args.seqtype)
    alphabet = seq_alphabet(seqtype)
    matrix = to_matrix(seqs)
    encoded_matrix, code_by_char = encode_alignment(seqs, alphabet)

    filtered, filtered_encoded, kept_columns, variant_columns, invariant_columns, _ = preprocess_columns_encoded(
        matrix, encoded_matrix, gap_threshold=args.gap_threshold
    )
    global_to_local = {ci: li for li, ci in enumerate(kept_columns)}
    rates, rate_values = fast_tiger_rates_encoded(
        filtered_encoded,
        kept_columns,
        count_gap_gap=getattr(args, "count_gap_gap", False),
    )
    features, site_info_values = site_feature_matrix_encoded(
        filtered_encoded,
        rate_values,
        alphabet,
        code_by_char,
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
        "cpu_threads": int(cpu_threads) if cpu_threads is not None else None,
    }

    all_blocks: list[BlockPartition] = []

    # Per-site informativeness weight: (1 - gap_ratio) * is_parsimony_informative.
    site_info_map = {ci: float(site_info_values[li]) for li, ci in enumerate(kept_columns)}

    # ------------------------------------------------------------------
    # Fast path: entire filtered alignment fits in cap → single block
    # ------------------------------------------------------------------
    if filtered_len <= cap_length:
        block = build_block_partition_from_metrics(
            "direct_b000", "direct",
            kept_columns, [1.0] * len(kept_columns),
            global_to_local, rate_values, features,
        )
        all_blocks = [block]
        pipeline_meta["routing"] = "direct"
        pipeline_meta["gmm_k"] = 1
        pipeline_meta["n_secondary_assignments"] = 0

    # ------------------------------------------------------------------
    # Partition path: GMM soft membership + overlap windows
    # ------------------------------------------------------------------
    else:
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
                block = build_block_partition_from_metrics(
                    f"{regime_id}_b{block_counter:04d}",
                    regime_id,
                    w_sites, w_probs,
                    global_to_local, rate_values, features,
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
