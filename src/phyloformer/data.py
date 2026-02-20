import os
from pathlib import Path
from itertools import combinations

import dendropy
import torch
from torch.utils.data import Dataset

ALPHABET = b"ARNDCQEGHILKMFPSTWYVX-"
LOOKUP = {char: index for index, char in enumerate(ALPHABET)}


def load_alignment(filepath):
    """
    Reads a fasta formater alignment and returns a one-hot encoded
    tensor of the MSA and the corresponding taxa label order
    """
    sequences, ids = [], []

    with open(filepath, "rb") as aln:
        for line in aln:
            line = line.strip()
            if line.startswith(b">"):
                ids.append(line[1:].decode("utf8"))
                sequences.append([])
            else:
                for char in line:
                    sequences[-1].append(LOOKUP[char])

    seqs = torch.tensor(sequences)
    seqs = torch.nn.functional.one_hot(seqs, num_classes=len(ALPHABET)).permute(2, 1, 0)

    return seqs, ids


def load_distance_matrix(filepath, ids):
    """
    Reads a newick formatted tree and returns a vector of the
    upper triangle of the corresponding pairwise distance matrix.
    The order of taxa in the rows and columns of the corresponding
    distance matrix is given by the `ids` input list.
    """

    distances = []

    with open(filepath, "r") as treefile:
        tree = dendropy.Tree.get(file=treefile, schema="newick")
    taxa = tree.taxon_namespace
    dm = tree.phylogenetic_distance_matrix()
    for tip1, tip2 in combinations(ids, 2):
        l1, l2 = taxa.get_taxon(tip1), taxa.get_taxon(tip2)
        distances.append(dm.distance(l1, l2))

    return torch.tensor(distances, dtype=torch.float32)


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
