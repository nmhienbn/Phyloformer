import os
import re
import shutil
from glob import glob
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Optional

from tqdm import tqdm

LIK_REGEX = re.compile(rb"Final LogLikelihood: (-?\d+.?\d+)")


def get_file_pairs(treedir: str, msadir: str) -> list[tuple[str, str, str]]:
    """Matches treefiles and msafiles by file stem"""

    trees = {Path(f).stem: f for f in glob(f"{treedir}/*.nwk")}
    msas = {Path(f).stem: f for f in glob(f"{msadir}/*.fa")}

    return [
        (k, tree, msa) for k, tree in trees.items() if (msa := msas.get(k)) is not None
    ]


def infer_likelihood(
    treepath: str,
    msapath: str,
    prefix: str,
    model: str,
    opt_brlen: bool,
    rename_output: bool,
    threads: Optional[int],
) -> Optional[float]:
    """Runs likelihood inference with RaxML-NG"""
    cmd = [
        shutil.which("raxml-ng"),
        "--evaluate",
        "--msa",
        msapath,
        "--tree",
        treepath,
        "--model",
        model,
        "--opt-model",
        "on",
        "--prefix",
        prefix,
    ]

    if opt_brlen:
        cmd = cmd + ["--opt-branches", "on"]
    else:
        cmd = cmd + ["--opt-branches", "off"]

    if threads is not None:
        cmd = cmd + ["--threads", f"auto{{{threads}}}"]

    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, _ = process.communicate()
    lik = LIK_REGEX.findall(stdout)

    if rename_output:
        os.rename(f"{prefix}.raxml.bestTree", f"{prefix}.nwk")

    return float(lik[0]) if len(lik) > 0 else None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Use RAxML-NG to compute phylogenetic likelihood"
    )
    parser.add_argument("treedir")
    parser.add_argument("msadir")
    parser.add_argument("-o", "--outdir", required=False)
    parser.add_argument("-b", "--opt-brlens", action="store_true")
    parser.add_argument("-r", "--rename-treefiles", action="store_true")
    parser.add_argument("-m", "--model", default="LG+G8")
    parser.add_argument("-t", "--threads", required=False)

    args = parser.parse_args()

    outdir = args.outdir or os.path.join(
        args.treedir, f"likelihoods.RAxML.{args.model}.Brlens_{args.opt_brlens}"
    )
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "likelihoods.csv"), "w") as csvout:
        csvout.write("tree,msa,model,opt_branches,log_likelihood\n")
        for id, tree, msa in tqdm(get_file_pairs(args.treedir, args.msadir)):
            lik = infer_likelihood(
                tree,
                msa,
                f"{outdir}/{id}",
                args.model,
                args.opt_brlens,
                args.rename_treefiles,
                args.threads,
            )
            csvout.write(f"{tree},{msa},{args.model},{args.opt_brlens},{lik}\n")


if __name__ == "__main__":
    main()
