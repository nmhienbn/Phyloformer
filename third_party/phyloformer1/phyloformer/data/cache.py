import os
from pathlib import Path

import torch

from .io import load_alignment, load_distance_matrix


def distance_cache_file(treefile, cache_dir):
    return Path(cache_dir) / f"{Path(treefile).stem}.pt"


def alignment_cache_file(alnfile, cache_dir):
    return Path(cache_dir) / f"{Path(alnfile).stem}.pt"


def _atomic_torch_save(obj, cache_file):
    tmp_file = cache_file.with_suffix(f".{os.getpid()}.tmp")
    torch.save(obj, tmp_file)
    os.replace(tmp_file, cache_file)


def precompute_distance_cache(pairs, cache_dir, overwrite=False):
    """
    Precompute and cache all target distance vectors as <stem>.pt files.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for treefile, alnfile in pairs:
        cache_file = distance_cache_file(treefile, cache_dir)
        if cache_file.exists() and not overwrite:
            continue

        _, ids = load_alignment(alnfile)
        y = load_distance_matrix(treefile, ids)
        _atomic_torch_save(y, cache_file)


def precompute_alignment_cache(pairs, cache_dir, overwrite=False):
    """
    Precompute and cache all one-hot alignments as <stem>.pt files.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for _, alnfile in pairs:
        cache_file = alignment_cache_file(alnfile, cache_dir)
        if cache_file.exists() and not overwrite:
            continue

        x, ids = load_alignment(alnfile)
        _atomic_torch_save({"x": x, "ids": ids}, cache_file)
