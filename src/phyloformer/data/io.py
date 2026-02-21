from itertools import combinations

import dendropy
import torch

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
