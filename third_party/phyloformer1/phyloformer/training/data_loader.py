import os
import pathlib
import random
import re

def listdir_paths(root):
    return [os.path.join(root, file) for file in os.listdir(root)]


# Removes all extensions (useful for still matching on predicted trees)
def stem(path):
    filename = pathlib.PurePath(path)
    return str(filename.stem).removesuffix("".join(filename.suffixes))


def make_pairs(treefiles, alnfiles, regex):
    """Find pairs of corresponding trees and MSAs"""
    alndict = {
        stem(alnfile): alnfile
        for alnfile in alnfiles
        if alnfile.endswith(".fa") or alnfile.endswith(".fasta")
    }
    pairs = []
    for treefile in treefiles:
        if not (treefile.endswith(".nwk") or treefile.endswith(".newick")):
            continue
        if regex is not None and not regex.search(treefile):
            continue
        alnfile = alndict.get(stem(treefile))
        if alnfile is None:
            print(f"Tree: {treefile}")
            print(f"Tree stem: {stem(treefile)}")
            raise IndexError(f"Tree: {treefile} has no corresponding alignment.")
        pairs.append((treefile, alnfile))
    return pairs


def choose_data(
    train_alignments, train_trees, train_regex, val_alignments, val_trees, val_regex
):
    """Find and select training and validation tree/MSA pairs"""
    # Choose training and validation examples
    if val_alignments is None and val_trees is None:
        regex = re.compile(train_regex) if train_regex is not None else None
        pairs = make_pairs(
            listdir_paths(train_trees), listdir_paths(train_alignments), regex
        )
        # Split data
        val_index = int(len(pairs) * 0.1)
        random.shuffle(pairs)
        val_pairs = pairs[:val_index]
        train_pairs = pairs[val_index:]
    elif val_alignments is not None and val_trees is not None:
        train_regex = re.compile(train_regex) if train_regex is not None else None
        val_regex = re.compile(val_regex) if val_regex is not None else None
        train_pairs = make_pairs(
            listdir_paths(train_trees),
            listdir_paths(train_alignments),
            train_regex,
        )
        val_pairs = make_pairs(
            listdir_paths(val_trees), listdir_paths(val_alignments), val_regex
        )
    else:
        raise ValueError(
            "You must either specify both validation trees and alignments "
            "or none of them."
        )

    return train_pairs, val_pairs