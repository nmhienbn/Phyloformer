# pyright: basic

import os
import re
import tarfile
from glob import glob
from pathlib import Path
from time import time

import torch
from tqdm import tqdm

from BayesNJ.core import (
    merges_to_tree,
    sample_tree_topology,
)
from BayesNJ.evopf import EvoPF
from BayesNJ.pf_sdk.pf.data import load_alignment
from BayesNJ.pf_sdk.pf.modules import PhyloformerSeq
from BayesNJ.treefuncs import batch_sample_trees

KEY_RENAMER = {
    "dm_MLP": "distance_mlp",
    "embed_dim": "h_dim",
    "use_unambiguous_order": "use_bilinear_embedder",
}


def vec_to_mat(preds, n):
    batch_size, _ = preds.shape
    dm = torch.zeros((batch_size, n, n)).type_as(preds)
    i = torch.tril_indices(row=n, col=n, offset=-1)
    dm[:, i[0], i[1]] = preds

    return dm + dm.transpose(-1, -2)


def mat_to_phylip(dm, ids):
    n = len(ids)

    s = f"{n}\n"
    for id, row in zip(ids, dm):
        row_s = " ".join([f"{x:.10f}" for x in row])
        s += f"{id} {row_s}\n"

    return s


def load_model(ckpt, device):
    hparams = ckpt["hparams"]

    common_keys = {"n_blocks", "n_heads", "embed_dim"}
    if hparams.get("evopf", False):
        allowed_keys = {
            "pair_dim",
            "use_opm",
            "dm_MLP",
            "symmetric",
            "use_unambiguous_order",
            *common_keys,
        }
        model_params = {
            KEY_RENAMER.get(k, k): v for k, v in hparams.items() if k in allowed_keys
        }

        if "pair_dim" not in model_params:
            model_params["pair_dim"] = 4 * model_params["h_dim"]

        model = EvoPF(**model_params)
    else:
        model_params = {k: v for k, v in hparams.items() if k in common_keys}
        model = PhyloformerSeq(**model_params)

    model = model.to(device)
    weights = ckpt["model"]
    model.load_state_dict(weights, strict=False)

    return model


def write_dm(pred_dm, ids, outdir, stem):
    phylip = mat_to_phylip(pred_dm, ids)
    with open(os.path.join(outdir, f"{stem}.phy"), "w") as matfile:
        matfile.write(phylip)


def sample_trees(pred_dm, ids, outdir, stem, nsamples):
    with open(os.path.join(outdir, f"{stem}.samples.nwk"), "w") as treesfile:
        for _ in range(nsamples):
            treesfile.write(
                sample_tree_topology(pred_dm, ids, use_max_proba=False) + "\n"
            )


def sample_max(pred_dm, ids, outdir, stem):
    with open(os.path.join(outdir, f"{stem}.nwk"), "w") as treesfile:
        treesfile.write(sample_tree_topology(pred_dm, ids, use_max_proba=True) + "\n")


def sample_trees_with_gamma(dm, msa, model, ids, outdir, stem, nsamples, verbose):
    with open(os.path.join(outdir, f"{stem}.samples.nwk"), "w") as treesfile:
        for _ in range(nsamples):
            merges, brlens, *_ = batch_sample_trees(
                model, msa, dm, use_max_proba=False, verbose=verbose
            )
            tree_newick = merges_to_tree(merges.squeeze(), brlens.squeeze(), ids)
            treesfile.write(tree_newick + "\n")


def sample_max_with_gamma(dm, msa, model, ids, outdir, stem, save_splits, verbose):
    merges, brlens, _, _ = batch_sample_trees(
        model,
        msa,
        dm,
        use_max_proba=True,
        verbose=verbose,
    )
    tree_newick = merges_to_tree(merges.squeeze(), brlens.squeeze(), ids)
    with open(os.path.join(outdir, f"{stem}.nwk"), "w") as treefile:
        treefile.write(tree_newick)

    if save_splits:
        torch.save(
            (merges, brlens, ids), os.path.join(outdir, "splits", f"{stem}.ckpt")
        )


def process_alns(
    msadir,
    model,
    outdir,
    device,
    mode,
    nsamples,
    save_splits,
    measure_execution,
    verbose,
):
    use_unambiguous = isinstance(model, EvoPF) and model.output_msa_emb

    measure_dir = os.path.join(outdir, "exec")
    if measure_execution:
        os.makedirs(measure_dir)

    # Needed to measure execution
    start_time, mem_peak = -1, -1

    model.eval()
    with torch.no_grad():
        for msapath in tqdm(glob(f"{msadir}/*fa") + glob(f"{msadir}/*fasta")):
            stem = Path(msapath).stem

            # Initialize timing if needed
            if measure_execution:
                start_time = time()
                torch.cuda.reset_peak_memory_stats()

            msa, ids = load_alignment(msapath)

            msa_emb = None
            try:
                if use_unambiguous:
                    pred_dists, msa_emb = model(msa.to(device).float().unsqueeze(0))
                else:
                    pred_dists = model(msa.to(device).float().unsqueeze(0))
            except torch.cuda.OutOfMemoryError:
                with open(os.path.join(measure_dir, f"{stem}.time.oom"), "w") as out:
                    out.write(f"{stem} ran out of CUDA memory\n")
                continue  # skip MSA

            # Predict square DM
            n = len(ids)
            if pred_dists.size(-1) == n:  # Already square
                dm = pred_dists
            else:  # Triangular vec
                dm = vec_to_mat(pred_dists, n)

            # Write output
            if mode == "dm":
                write_dm(dm, ids, outdir, stem)
            elif mode == "samples":
                if use_unambiguous:
                    sample_trees_with_gamma(
                        dm, msa_emb, model, ids, outdir, stem, nsamples, verbose
                    )
                else:
                    sample_trees(dm.squeeze(0), ids, outdir, stem, nsamples)
            elif mode == "max-sample":
                if use_unambiguous:
                    sample_max_with_gamma(
                        dm, msa_emb, model, ids, outdir, stem, save_splits, verbose
                    )
                else:
                    sample_max(dm, ids, outdir, stem)
            else:
                raise ValueError(f"Unknown prediction mode: {mode}")

            # Write executions stats to file
            if measure_execution:
                elapsed = time() - start_time
                mins = int(elapsed // 60)
                secs = elapsed % 60

                mem_peak = torch.cuda.max_memory_allocated()

                with open(os.path.join(measure_dir, f"{stem}.time"), "w") as out:
                    out.write(
                        f"Inference being timed: {msapath}\n"
                        f"\tElapsed (wall clock) time (h:mm:ss or m:ss): {mins}:{secs:.5f}\n"
                        f"\tMaximum resident set size (kbytes): {mem_peak / 1024}"
                    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Infer distance matrices or sample trees from MSAs using checkpoint weights"
    )
    parser.add_argument("msas", help="Path to directory containing fasta files")
    parser.add_argument("ckptfile", help="Path to tarfile or single checkpoint")
    parser.add_argument("output", help="Path to output directory")
    parser.add_argument(
        "--save-splits",
        action="store_true",
        help="Store predicted splits and brlen vectors",
    )
    parser.add_argument(
        "--mode", "-m", choices=["dm", "samples", "max-sample"], help="prediction mode"
    )
    parser.add_argument(
        "--regex",
        "-r",
        default=".*",
        help="Only infer for checkpoints that match the regex",
    )
    parser.add_argument(
        "--nsamples",
        "-n",
        type=int,
        default=30,
        help="Number of trees to sample (only valid if mode is 'samples')",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Measure memory and inference time for each MSA",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print out sampled objects (can slow down execution)",
    )
    args = parser.parse_args()

    msadir = args.msas
    outroot = args.output
    ckptpath = args.ckptfile

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    # Check that execution measurement is only done with CUDA devices
    if not torch.cuda.is_available() and args.measure:
        raise ValueError("The --measure argument is only available on CUDA devices")

    # Chek if checkpoint is not tarfile
    if not tarfile.is_tarfile(ckptpath):
        ckpt = torch.load(ckptpath, map_location=device)
        model = load_model(ckpt, device)
        model.eval()

        print(model)

        os.makedirs(outroot)
        if args.save_splits:
            os.makedirs(os.path.join(outroot, "splits"))
        process_alns(
            msadir,
            model,
            outroot,
            device,
            args.mode,
            args.nsamples,
            args.save_splits,
            args.measure,
            args.verbose,
        )
        return

    filter = re.compile(args.regex)
    with tarfile.open(ckptpath, "r") as ckpts:
        while (member := ckpts.next()) is not None:
            if not filter.match(member.path):
                continue

            outdir = os.path.join(outroot, member.path.replace("/", ""))
            os.makedirs(outdir)
            if args.save_splits:
                os.makedirs(os.path.join(outdir, "splits"))

            ckpt = torch.load(ckpts.extractfile(member), map_location=device)
            model = load_model(ckpt, device)
            model.eval()

            process_alns(
                msadir,
                model,
                outdir,
                device,
                args.mode,
                args.nsamples,
                args.save_splits,
                args.measure,
                args.verbose,
            )


if __name__ == "__main__":
    main()
