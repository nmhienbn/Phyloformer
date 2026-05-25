"""PF2 VRAM cap table and cap-length derivation — shared across all gpartition scripts.

Measured RSS heatmap (GB) on 16 GB GPU, fitted to:

    vram_gb(n_seq, n_sites) ≈ 0.5 + 5.12e-7 × n_seq² × n_sites

Mean fit error: 3.5%.  Worst case (small alignments): 27% (constant term dominates).

Inverting gives the maximum safe n_sites for a given VRAM budget:

    max_sites(n_seq, vram_gb) = (vram_gb - 0.5) / (5.12e-7 × n_seq²)

A safety factor (default 0.85) is applied to leave ~15% headroom.

The table-based ``derive_pf2_cap_length`` is kept for backward compatibility; new
code should prefer ``max_sites_for_vram`` which is portable across VRAM sizes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Formula constants  (fitted from measured RSS heatmap)
# ---------------------------------------------------------------------------

_VRAM_INTERCEPT = 0.53   # GB — fixed overhead (model weights, activations)
_VRAM_SLOPE     = 5.12e-7  # GB / (n_seq² × n_sites)
_VRAM_SAFETY    = 0.85      # leave 15% headroom


def estimate_vram_gb(n_seq: int, n_sites: int) -> float:
    """Estimate peak VRAM (GB) for a PF2 block of shape (n_seq, n_sites)."""
    return _VRAM_INTERCEPT + _VRAM_SLOPE * n_seq * n_seq * n_sites


def max_sites_for_vram(n_seq: int, vram_gb: float, safety: float = _VRAM_SAFETY) -> int:
    """Return the maximum n_sites that fits in *vram_gb* for *n_seq* sequences.

    Args:
        n_seq:    Number of sequences in the block.
        vram_gb:  VRAM budget in GB (e.g. 8.0, 16.0, 24.0).
        safety:   Fraction of budget to actually use (default 0.85 = 15% headroom).

    Returns:
        Maximum safe block length (n_sites), at least 50.
    """
    headroom = vram_gb * safety - _VRAM_INTERCEPT
    if headroom <= 0:
        return 50
    raw = headroom / (_VRAM_SLOPE * n_seq * n_seq)
    return max(50, int(raw))


# ---------------------------------------------------------------------------
# Table-based derivation (backward compat — used when cap_policy is given)
# ---------------------------------------------------------------------------

PF2_SEQ_BINS = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]

PF2_MAX_LEN_BY_SEQ_BIN = {
    10: 10000,
    50: 10000,
    100: 10000,
    150: 5000,
    200: 3000,
    250: 2000,
    300: 1000,
    350: 1000,
    400: 500,
    450: 500,
    500: 250,
    550: 250,
    600: 80,    # measured: 600 × 100 = 19.2 GB OOM → 80 ≈ 13 GB safe
}


def derive_pf2_cap_length(
    n_sequences: int,
    cap_policy: str,
    vram_gb: float = 16.0,
) -> int:
    """Return the maximum block length (n_sites) that fits in *vram_gb* VRAM.

    Args:
        n_sequences: Number of sequences in the alignment.
        cap_policy:  ``"formula"``   — use the fitted formula (recommended for
                                       non-16 GB targets, e.g. 8 GB or 24 GB).
                     ``"practical"`` — table-based conservative thresholds
                                       calibrated for 16 GB; ignores *vram_gb*.
                     ``"profile-max"`` — raw table upper bound; ignores *vram_gb*.
        vram_gb:     VRAM budget in GB.  Used only when cap_policy == "formula".
    """
    if cap_policy == "formula":
        return max_sites_for_vram(n_sequences, vram_gb)

    for seq_bin in PF2_SEQ_BINS:
        if n_sequences <= seq_bin:
            max_len = PF2_MAX_LEN_BY_SEQ_BIN[seq_bin]
            if cap_policy == "profile-max":
                return max_len
            # "practical": conservative thresholds calibrated for 16 GB GPU RSS
            if n_sequences <= 80:
                return min(max_len, 5000)
            if n_sequences <= 100:
                return min(max_len, 2000)   # 100 × 2000 = 10.6 GB ✓
            if n_sequences <= 170:
                return min(max_len, 1000)   # 150 × 1000 = 11.8 GB ✓
            if n_sequences <= 200:
                return min(max_len, 500)    # 200 × 500  = 10.5 GB ✓
            if n_sequences <= 350:
                return min(max_len, 250)    # 350 × 250  = 16.0 GB borderline
            return min(max_len, 100)        # 400 × 100  =  8.7 GB ✓
    return 80
