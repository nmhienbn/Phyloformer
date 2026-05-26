#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PF2_ROOT = REPO_ROOT / "third_party" / "phyloformer2"
sys.path.insert(0, str(PF2_ROOT))


SAMPLER_BASE_SIZE = 50
torch = None
F = None
get_merge_order = None
EvoPF = None
PADDING_TOKEN = None
load_alignment = None


def load_dependencies() -> None:
    global torch, F, get_merge_order, EvoPF, PADDING_TOKEN, load_alignment

    import torch as torch_module
    import torch.nn.functional as functional_module
    from BayesNJ.core import get_merge_order as get_merge_order_function
    from BayesNJ.evopf import EvoPF as EvoPFClass
    from BayesNJ.pf_sdk.pf.data import PADDING_TOKEN as padding_token
    from BayesNJ.pf_sdk.pf.data import load_alignment as load_alignment_function

    torch = torch_module
    F = functional_module
    get_merge_order = get_merge_order_function
    EvoPF = EvoPFClass
    PADDING_TOKEN = padding_token
    load_alignment = load_alignment_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile PF2 fine-tune peak VRAM on real manifest rows. "
            "Rows are grouped by n_seqs, and each tested batch uses the longest "
            "MSAs in that group, so the result reflects both n_seqs and length."
        )
    )
    parser.add_argument("manifest_tsv")
    parser.add_argument("--checkpoint", default="models/phyloformer2/pf2.tch")
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--base-batch-sizes", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument(
        "--max-taxa",
        type=int,
        default=None,
        help="Optional taxa cap for profiling. Default: use all manifest rows.",
    )
    parser.add_argument(
        "--taxa",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit taxa groups to test. Default: choose the heaviest groups.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=24,
        help="Number of taxa groups to test when --taxa is not provided.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--unambiguous-order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-oom", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path, max_taxa: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("relative_path")
            and row.get("tree_path")
            and (max_taxa is None or int(row["num_sequences"]) <= max_taxa)
        ]
    if not rows:
        raise ValueError(f"No usable rows found in {path}")
    return rows


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "base_batch_size",
        "taxa",
        "requested_batch_size",
        "actual_batch_size",
        "max_alignment_length",
        "mean_alignment_length",
        "status",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "elapsed_sec",
        "loss",
        "msa_ids",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def scheduled_batch_size(taxa: int, base_batch_size: int) -> int:
    batch_size = int((SAMPLER_BASE_SIZE / taxa) ** 2 * base_batch_size)
    return min(max(batch_size, 1), base_batch_size)


def select_groups(
    rows: list[dict[str, str]],
    taxa: list[int] | None,
    max_groups: int,
) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(int(row["num_sequences"]), []).append(row)

    for group in groups.values():
        group.sort(key=lambda row: int(row["alignment_length"]), reverse=True)

    if taxa is not None:
        return {n: groups[n] for n in taxa if n in groups}

    ranked = sorted(
        groups.items(),
        key=lambda item: int(item[0]) * int(item[0]) * int(item[1][0]["alignment_length"]),
        reverse=True,
    )
    selected = dict(ranked[:max_groups])

    for n in sorted(groups):
        if len(selected) >= max_groups:
            break
        selected.setdefault(n, groups[n])
    return dict(sorted(selected.items()))


def collate_unambiguous(batch):
    msas, merge_orders, brlens, ids = zip(*batch)
    max_len = max(msa.shape[1] for msa in msas)
    max_nseqs = max(msa.shape[2] for msa in msas)
    padded_msas = [
        F.pad(
            msa,
            (0, max_nseqs - msa.shape[2], 0, max_len - msa.shape[1]),
            mode="constant",
            value=PADDING_TOKEN,
        )
        for msa in msas
    ]
    return (
        torch.stack(padded_msas),
        torch.stack(merge_orders),
        torch.stack(brlens),
        list(ids),
    )


def load_batch(rows: list[dict[str, str]]):
    batch = []
    for row in rows:
        msa, ids = load_alignment(row["relative_path"])
        merge_order, brlens = get_merge_order(row["tree_path"], ids)
        batch.append((msa, merge_order, brlens, ids))
    return collate_unambiguous(batch)


def build_model(checkpoint: Path, use_unambiguous_order: bool, device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    hparams = ckpt["hparams"]
    get_hparam = hparams.get
    model = EvoPF(
        n_blocks=hparams["n_blocks"],
        n_heads=hparams["n_heads"],
        h_dim=hparams["embed_dim"],
        pair_dim=hparams["pair_dim"],
        use_opm=True,
        distance_mlp=get_hparam("dm_MLP", False),
        symmetric=get_hparam("symmetric", False),
        use_deepspeed=get_hparam("use_deepspeed", False),
        use_flexattention=get_hparam("use_flexattention", False),
        use_bilinear_embedder=use_unambiguous_order or get_hparam("use_unambiguous_order", False),
        use_brlens=get_hparam("optimize_brlens", False),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def run_profile(model, batch, device) -> tuple[float, float, float, float]:
    msa, merge_order, brlens, _ = batch
    msa = msa.to(device)
    merge_order = merge_order.to(device)
    brlens = brlens.to(device)
    n_seqs = msa.shape[-1]

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    model.zero_grad(set_to_none=True)
    (
        _dm,
        _msa_emb,
        logprob_topo,
        _logprob_gamma,
        _logprob_beta,
        logprob_brlens,
        _mse_brlens,
    ) = model(
        msa.float(),
        brlens=brlens,
        merge_order=merge_order,
        temperature=None,
        topo_only=True,
        ignore_topo=False,
        return_tree_logprob=True,
        verbose=False,
    )
    loss = ((-logprob_topo - logprob_brlens) / n_seqs).mean()
    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    else:
        peak_allocated = 0.0
        peak_reserved = 0.0
    elapsed = time.perf_counter() - start

    del msa, merge_order, brlens, loss
    return peak_allocated, peak_reserved, elapsed, float(logprob_topo.detach().mean().cpu())


def main() -> None:
    args = parse_args()
    load_dependencies()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    groups = select_groups(
        read_manifest(Path(args.manifest_tsv), args.max_taxa),
        args.taxa,
        args.max_groups,
    )
    model = build_model(Path(args.checkpoint), args.unambiguous_order, device)
    model.train()

    results: list[dict[str, str]] = []
    for base_batch_size in args.base_batch_sizes:
        for taxa, group in groups.items():
            requested = scheduled_batch_size(taxa, base_batch_size)
            batch_rows = group[:requested]
            lengths = [int(row["alignment_length"]) for row in batch_rows]
            record = {
                "base_batch_size": str(base_batch_size),
                "taxa": str(taxa),
                "requested_batch_size": str(requested),
                "actual_batch_size": str(len(batch_rows)),
                "max_alignment_length": str(max(lengths)),
                "mean_alignment_length": f"{sum(lengths) / len(lengths):.1f}",
                "status": "ok",
                "peak_allocated_mib": "",
                "peak_reserved_mib": "",
                "elapsed_sec": "",
                "loss": "",
                "msa_ids": ",".join(row["msa_id"] for row in batch_rows),
                "error": "",
            }
            try:
                batch = load_batch(batch_rows)
                peak_allocated, peak_reserved, elapsed, loss_proxy = run_profile(model, batch, device)
                record.update(
                    {
                        "peak_allocated_mib": f"{peak_allocated:.1f}",
                        "peak_reserved_mib": f"{peak_reserved:.1f}",
                        "elapsed_sec": f"{elapsed:.3f}",
                        "loss": f"{loss_proxy:.6g}",
                    }
                )
                print(
                    f"[OK] base={base_batch_size} taxa={taxa} "
                    f"batch={len(batch_rows)} max_len={max(lengths)} "
                    f"peak_reserved_mib={peak_reserved:.1f}",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError as exc:
                record["status"] = "oom"
                record["error"] = str(exc).splitlines()[0]
                print(
                    f"[OOM] base={base_batch_size} taxa={taxa} "
                    f"batch={len(batch_rows)} max_len={max(lengths)}",
                    flush=True,
                )
                if args.stop_on_oom:
                    results.append(record)
                    write_tsv(Path(args.out_tsv), results)
                    raise
            finally:
                results.append(record)
                write_tsv(Path(args.out_tsv), results)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print(f"[OK] wrote {args.out_tsv}")


if __name__ == "__main__":
    main()
