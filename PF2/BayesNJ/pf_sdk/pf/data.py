import os
from math import floor, sqrt
import pathlib
from typing import Iterator, Optional

import dendropy
import numpy as np
import pandas as pd
from scipy.special import binom
from torch import (
    eye,
    from_numpy,
    randperm,
    stack,
    tril_indices,
    triu_indices,
    zeros,
    Generator,
)
from torch.nn.functional import pad
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

PADDING_TOKEN = -1

ALPHABET = b"ARNDCQEGHILKMFPSTWYVX-~"  # '~' is the CLS token
LOOKUP = {char: index for index, char in enumerate(ALPHABET)}
IDENT = eye(len(ALPHABET))
# One hot vectors for each character
VECS = {char: IDENT[index] for index, char in enumerate(ALPHABET)}
# X is any of the amino acids (except -)
VECS[ord("X")] = sum(VECS[c] for c in ALPHABET[:-2]) / (len(ALPHABET) - 2)  # type: ignore
# IUPAC: B is D or N
VECS[ord("B")] = (VECS[ord("D")] + VECS[ord("N")]) / 2
# IUPAC: Z is E or Q
VECS[ord("Z")] = (VECS[ord("E")] + VECS[ord("Q")]) / 2


def make_pairs(treedir, msadir):
    """
    Given a tree directory and an alignemnt directory,
    generate data (tree, msa) pairs with matching
    file names.
    """

    def _stemify(root):
        return {
            pathlib.PurePath(f).stem: os.path.join(root, f) for f in os.listdir(root)
        }

    trees = _stemify(treedir)
    msas = _stemify(msadir)

    return [(t, m) for k, m in msas.items() if (t := trees.get(k)) is not None]


def parse_alignment(stream):
    """
    Parses a fasta formatted alignment from a binary file strean
    """

    sequence_tensors, ids = [], []

    first = True
    currseq = [VECS[ord("~")]]  # Starting the sequence with a [CLS] token
    for line in stream:
        line = line.strip()
        if line.startswith(b">"):
            ids.append(line[1:].decode("utf8"))
            if not first:
                sequence_tensors.append(stack(currseq))
                currseq = [VECS[ord("~")]]
            first = False
        else:
            for char in line:
                # Any unknown characters such as '?' are treated as 'X'
                currseq.append(VECS.get(char, VECS[ord("X")]))

    # Append last sequence to alignment
    sequence_tensors.append(stack(currseq))

    tensor_aln = stack(sequence_tensors).permute(2, 1, 0)

    return tensor_aln, ids


def load_alignment(filepath):
    """
    Reads a fasta formater alignment and returns a one-hot encoded
    tensor of the MSA and the corresponding taxa label order
    """

    with open(filepath, "rb") as stream:
        seqs, ids = parse_alignment(stream)

    return seqs, ids


def vec_to_mat(vec, lower: bool = True):
    """
    Takes a vector (1 * nChoose2) representing the lower triangle of a pairwise distance matrix,
    and returns the corresponding distance matrix (n*n)
    """
    n_pairs = vec.shape[-1]
    n_seqs = (1 + sqrt(1 + 8 * n_pairs)) / 2
    assert n_seqs == int(
        n_seqs
    ), f"Invalid number of distances: {n_pairs} is not a valid number of sequence pairs"
    n = int(n_seqs)

    mat = zeros((n, n)).type_as(vec)
    if lower:
        idx = tril_indices(n, n, offset=-1)
    else:
        idx = triu_indices(n, n, offset=1)

    mat[idx[0], idx[1]] = vec

    return mat + mat.T


def load_distance_matrix(filepath, ids, get_vec: bool = True, lower: bool = True):
    """
    Reads a newick formatted tree and returns a vector of the
    lower triangle of the corresponding pairwise distance matrix.
    The order of taxa in the rows and columns of the corresponding
    distance matrix is given by the `ids` input list.
    """

    tree = dendropy.Tree.get(path=filepath, schema="newick", preserve_underscores=True)
    pdm = tree.phylogenetic_distance_matrix()

    dm = from_numpy(pd.DataFrame(pdm.as_data_table()._data).loc[ids, ids].values)
    if get_vec:
        # Return triangular vector
        if lower:
            row, col = tril_indices(*dm.shape, offset=-1)
        else:
            row, col = triu_indices(*dm.shape, offset=1)

        return dm[row, col]

    return dm


class PhyloDataset(Dataset):
    """
    Simple pytorch dataset that reads tree/alignment pairs
    and returns the corresponding tensor objects
    """

    @staticmethod
    def pad_batches(batches):
        shapes = [x[0].shape[1:] for x in batches]
        max_seqlen, max_nseqs = max(x[0] for x in shapes), max(x[1] for x in shapes)
        max_npairs = int(binom(max_nseqs, 2))

        batched_x, batched_y = [], []
        for x, y in batches:
            _, seqlen, nseqs = x.shape
            npairs = y.shape[0]
            batched_x.append(
                pad(
                    x,
                    (0, max_nseqs - nseqs, 0, max_seqlen - seqlen),
                    mode="constant",
                    value=PADDING_TOKEN,
                )
            )
            # This padding of y only works if it's the *lower* trianglular vector
            # of the distance matrix
            batched_y.append(
                pad(y, (0, max_npairs - npairs), mode="constant", value=PADDING_TOKEN)
            )

        return stack(batched_x), stack(batched_y)

    def __init__(self, pairs):
        """
        pairs: List[(str,str)] = a list of (treefile, alnfile) paths
        """
        self.pairs = np.array(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        treefile, alnfile = self.pairs[index]
        x, ids = load_alignment(alnfile)
        y = load_distance_matrix(treefile, ids)

        return x, y

    def state_dict(self):
        return dict(pairs=self.pairs)

    @staticmethod
    def from_state_dict(state_dict):
        return PhyloDataset(**state_dict)


class CheckpointableSequentialSampler(SequentialSampler):
    def __init__(self, dataset, batch_size: int):
        super().__init__(dataset)
        self.dataset = dataset

        self.batch_size = batch_size
        self.start_iter = 0

    @property
    def num_samples(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        ordering = np.fromiter(super().__iter__(), int)
        start_index = self.start_iter * self.batch_size

        return iter(ordering[start_index:])

    def set_starting_step(self, step: int):
        self.start_iter = step

    def state_dict(self):
        return dict(batch_size=self.batch_size)

    def load_state_dict(self, state):
        self.batch_size = state["batch_size"]


class CheckpointableRandomSampler(RandomSampler):
    def __init__(self, dataset, batch_size: int, seed: int):
        # Init RNG
        super().__init__(dataset)

        self.seed = seed
        self.batch_size = batch_size
        self.start_iter = 0
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        generator = Generator()
        generator.manual_seed(self.seed + self.epoch)

        ordering = randperm(self.num_samples, generator=generator).numpy()
        start_index = self.start_iter * self.batch_size

        return iter(ordering[start_index:])

    def set_starting_step(self, step: int):
        self.start_iter = step

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def state_dict(self):
        return dict(batch_size=self.batch_size, seed=self.seed)

    def load_state_dict(self, state):
        self.batch_size = state["batch_size"]
        self.seed = state["seed"]


# Basically stolen from VISSL
# https://github.com/facebookresearch/vissl/blob/main/vissl/data/data_helper.py#L93
class CheckpointableDsitributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        frac_subsample: Optional[float] = None,
    ):
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive, non-zero integer")

        # Sets: self.seed, self.shuffle, self.drop_last, self.rank, seld.num_replicas
        # self.total_size, sel.num_samples
        super().__init__(dataset, shuffle=shuffle, drop_last=drop_last, seed=seed)

        self.batch_size = batch_size
        self.start_iter = 0  # Used for resuming checkpoint
        if frac_subsample is not None:
            assert frac_subsample > 0. and frac_subsample <= 1.0, "Fraction must be in ]0,1]"
        self.frac_subsample = frac_subsample

    def __iter__(self):
        # Generate ordering with PyTorch's `DistributedSampler`
        ordering = np.array(list(super().__iter__()))

        if self.frac_subsample is not None:
            lim = floor(self.frac_subsample * len(ordering))
            ordering = ordering[:lim]

        # Compute starting index
        start_index = self.start_iter * self.batch_size

        return iter(ordering[start_index:])

    def set_starting_step(self, step: int):
        """Call this to set the starting step if resuming from a checkpoint"""
        self.start_iter = step

    def state_dict(self):
        return dict(
            batch_size=self.batch_size,
            seed=self.seed,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

    def load_state_dict(self, state):
        self.batch_size = state["batch_size"]
        self.seed = state["seed"]
        self.shuffle = state["shuffle"]
        self.drop_last = state["drop_last"]

    @staticmethod
    def from_state_dict(dataset, state, start_step: int = 0):
        sampler = CheckpointableDsitributedSampler(dataset, **state)
        sampler.set_starting_step(start_step)

        return sampler
