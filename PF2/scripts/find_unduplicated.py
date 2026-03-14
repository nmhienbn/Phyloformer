import os
from glob import glob
from pathlib import Path

from Bio import SeqIO
from tqdm import tqdm


def has_duplicates(msa):
    return len(set(x.seq for x in msa)) != len(msa)


def find_unduplicated(msa, l):
    tot_len = len(msa[0])
    for offset in range(tot_len - l):
        sub = [x[offset : offset + l] for x in msa]
        if not has_duplicates(sub):
            return offset
    return None


def seq_is_tip(rec):
    return rec.id.startswith("T")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find unduplicated seqgments of length L in alignments"
    )
    parser.add_argument("root")
    parser.add_argument("outdir")
    parser.add_argument("-l", "--length", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    all_msas = glob(f"{args.root}/*untrimmed")

    for path in tqdm(all_msas):
        filename = Path(path).name.strip(".untrimmed")

        msa = list(SeqIO.parse(path, format="fasta"))
        tips = list(filter(seq_is_tip, msa))

        if has_duplicates(tips):
            continue

        if (offset := find_unduplicated(tips, l=args.length)) is not None:
            sub = (x[offset : offset + args.length] for x in msa)
            SeqIO.write(sub, os.path.join(args.outdir, filename), format="fasta")
