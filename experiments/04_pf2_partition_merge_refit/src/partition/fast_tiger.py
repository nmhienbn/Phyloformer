"""Shared fastTIGER site-rate helpers for partition-based PF2 pipelines."""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange

GAP_CHARS = {"-", ".", "?"}


def encode_alignment(
    seqs: list[str],
    alphabet: list[str],
    gap_chars: set[str] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode alignment characters as small ints; 0 is reserved for gap/missing."""
    gaps = gap_chars or GAP_CHARS
    observed = sorted({c for seq in seqs for c in seq.upper() if c not in gaps})
    ordered = []
    for c in alphabet + observed:
        if c not in ordered:
            ordered.append(c)
    code_by_char = {c: i + 1 for i, c in enumerate(ordered)}
    for c in gaps:
        code_by_char[c] = 0

    encoded = np.empty((len(seqs), len(seqs[0]) if seqs else 0), dtype=np.int16)
    for i, seq in enumerate(seqs):
        encoded[i] = [code_by_char.get(c, 0) for c in seq.upper()]
    return encoded, code_by_char


@njit(parallel=True)
def _fast_tiger_rates_numba(
    encoded: np.ndarray,
    count_gap_gap: bool,
    max_code: int,
) -> np.ndarray:
    nseq, nsites = encoded.shape
    nthreads = get_num_threads()
    sim_parts = np.zeros((nthreads, nseq, nseq), dtype=np.float64)
    sim = np.zeros((nseq, nseq), dtype=np.float64)
    total = 0.0

    for ci in prange(nsites):
        tid = get_thread_id()
        for i in range(nseq):
            a = encoded[i, ci]
            for j in range(i + 1, nseq):
                b = encoded[j, ci]
                if a == b and (a != 0 or count_gap_gap):
                    sim_parts[tid, i, j] += 1.0

    for tid in range(nthreads):
        for i in range(nseq):
            for j in range(i + 1, nseq):
                same = sim_parts[tid, i, j]
                sim[i, j] += same
                total += same

    for i in range(nseq):
        for j in range(i + 1, nseq):
            sim[j, i] = sim[i, j]

    rates = np.zeros(nsites, dtype=np.float64)
    if total <= 0.0:
        return rates

    for ci in prange(nsites):
        same_w = 0.0
        for code in range(0 if count_gap_gap else 1, max_code + 1):
            members = np.zeros(nseq, dtype=np.int64)
            n_members = 0
            for ri in range(nseq):
                if encoded[ri, ci] == code:
                    members[n_members] = ri
                    n_members += 1
            for ai in range(n_members):
                i = members[ai]
                for aj in range(ai + 1, n_members):
                    same_w += sim[i, members[aj]]
        rates[ci] = 1.0 - (same_w / total)

    return rates


def fast_tiger_rates_encoded(
    filtered_encoded: np.ndarray,
    kept_columns: list[int],
    count_gap_gap: bool = False,
) -> tuple[dict[int, float], np.ndarray]:
    """Compute fastTIGER rates for already-filtered encoded columns."""
    if not kept_columns:
        return {}, np.zeros(0, dtype=np.float64)
    max_code = int(filtered_encoded.max()) if filtered_encoded.size else 0
    rate_values = _fast_tiger_rates_numba(filtered_encoded, count_gap_gap, max_code)
    return {ci: float(rate_values[li]) for li, ci in enumerate(kept_columns)}, rate_values
