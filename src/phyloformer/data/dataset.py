from pathlib import Path

import torch
from torch.utils.data import Dataset

from .cache import (
    _atomic_torch_save,
    alignment_cache_file,
    distance_cache_file,
)
from .io import load_alignment, load_distance_matrix


class PhyloDataset(Dataset):
    """
    Simple pytorch dataset that reads tree/alignment pairs
    and returns the corresponding tensor objects
    """

    def __init__(self, pairs, distance_cache_dir=None, alignment_cache_dir=None):
        """
        pairs: List[(str,str)] = a list of (treefile, alnfile) paths
        """
        self.pairs = pairs
        self.distance_cache_dir = distance_cache_dir
        self.alignment_cache_dir = alignment_cache_dir
        if distance_cache_dir is not None:
            Path(distance_cache_dir).mkdir(parents=True, exist_ok=True)
        if alignment_cache_dir is not None:
            Path(alignment_cache_dir).mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        treefile, alnfile = self.pairs[index]
        x = None
        ids = None
        y = None

        if self.alignment_cache_dir is not None:
            cache_file = alignment_cache_file(alnfile, self.alignment_cache_dir)
            if cache_file.exists():
                payload = torch.load(cache_file, map_location="cpu")
                if isinstance(payload, dict) and "x" in payload and "ids" in payload:
                    x = payload["x"]
                    ids = payload["ids"]
            if x is None or ids is None:
                x, ids = load_alignment(alnfile)
                _atomic_torch_save({"x": x, "ids": ids}, cache_file)
        else:
            x, ids = load_alignment(alnfile)

        if self.distance_cache_dir is not None:
            cache_file = distance_cache_file(treefile, self.distance_cache_dir)
            if cache_file.exists():
                y = torch.load(cache_file, map_location="cpu")
            else:
                if ids is None:
                    _, ids = load_alignment(alnfile)
                y = load_distance_matrix(treefile, ids)
                _atomic_torch_save(y, cache_file)
        else:
            if ids is None:
                _, ids = load_alignment(alnfile)
            y = load_distance_matrix(treefile, ids)

        return x, y
