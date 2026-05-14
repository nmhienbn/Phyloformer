#!/usr/bin/env python3
"""Measure practical GPU memory for PF1/PF2 over a (n_seq, n_sites) grid.

This script uses a parent/worker design:

- parent process launches one worker per grid cell
- parent polls `nvidia-smi` by PID to estimate actual GPU RSS
- worker runs one synthetic forward or train step and reports torch peak stats

Supported profiles:
    pf1-infer
    pf1-train
    pf2-mae-infer
    pf2-mae-train
    pf2-bayesnj-infer
    pf2-bayesnj-train
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "experiments").is_dir() and (path / "third_party").is_dir():
            return path
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
PF1_ROOT = REPO_ROOT / "third_party" / "phyloformer1"
PF2_ROOT = REPO_ROOT / "third_party" / "phyloformer2"


def _ensure_import_paths() -> None:
    for path in [REPO_ROOT, PF1_ROOT, PF2_ROOT]:
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def format_float(val: float | None) -> str:
    if val is None or np.isnan(val):
        return ""
    return f"{val:.6f}"


def poll_nvidia_smi(pid: int, gpu: int | None) -> float:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    if gpu is not None:
        cmd.extend(["-i", str(gpu)])
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return 0.0

    peak_mib = 0.0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            used_mib = float(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            peak_mib = max(peak_mib, used_mib)
    return peak_mib


def make_balanced_merge_order(n_leaves: int, device, torch_mod):
    """Synthetic rooted merge order compatible with PF2 BayesNJ training."""
    n_nodes = 2 * n_leaves - 1
    num_classes = n_nodes - 1
    current = list(range(n_leaves))
    next_label = n_leaves
    merges: list[tuple[int, int]] = []

    while len(current) > 2:
        nxt = []
        i = 0
        while i + 1 < len(current):
            a = current[i]
            b = current[i + 1]
            merges.append((min(a, b), max(a, b)))
            nxt.append(next_label)
            next_label += 1
            i += 2
        if i < len(current):
            nxt.append(current[i])
        current = nxt

    merges.append((min(current[0], current[1]), max(current[0], current[1])))
    merge_tensor = torch_mod.tensor(merges, device=device, dtype=torch_mod.long)
    merge_order = torch_mod.nn.functional.one_hot(
        merge_tensor, num_classes=num_classes
    ).float()
    merge_order = merge_order.unsqueeze(0)
    brlens = torch_mod.ones((1, num_classes), device=device)
    return merge_order, brlens


def lower_vec_to_mat(vec, n, torch_mod):
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    dm = torch_mod.zeros((vec.shape[0], n, n), device=vec.device, dtype=vec.dtype)
    i, j = torch_mod.tril_indices(n, n, -1, device=vec.device)
    dm[:, i, j] = vec
    return dm + dm.transpose(-1, -2)


def build_worker_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--profile", required=True)
    p.add_argument("--n-seq", type=int, required=True)
    p.add_argument("--n-sites", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--hold-sec", type=float, default=0.5)
    p.add_argument("--json-out", required=True)
    p.add_argument("--pf1-ckpt", default=str(REPO_ROOT / "pretrained_models" / "pf_base.ckpt"))
    p.add_argument("--pf2-ckpt", default=str(PF2_ROOT / "pretrained" / "pf2.tch"))
    p.add_argument("--pf1-blocks", type=int, default=6)
    p.add_argument("--pf1-heads", type=int, default=4)
    p.add_argument("--pf1-embed", type=int, default=64)
    p.add_argument("--pf2-blocks", type=int, default=12)
    p.add_argument("--pf2-heads", type=int, default=4)
    p.add_argument("--pf2-embed", type=int, default=128)
    p.add_argument("--pf2-pair-dim", type=int, default=256)
    p.add_argument("--symmetric", action="store_true", default=True)
    p.add_argument("--no-symmetric", dest="symmetric", action="store_false")
    return p


def worker_main(args: argparse.Namespace) -> int:
    _ensure_import_paths()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch

    import phyloformer.models.phyloformer as pf1_impl
    from phyloformer.models import Phyloformer
    from infer import load_model as load_pf2_model
    from BayesNJ.evopf import EvoPF
    from BayesNJ.treefuncs import batch_sample_trees

    if not torch.cuda.is_available():
        payload = {"status": "error", "error": "CUDA is not available"}
        Path(args.json_out).write_text(json.dumps(payload), encoding="utf-8")
        return 1

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device(args.device)
    autocast_dtype = None
    if args.dtype == "fp16":
        autocast_dtype = torch.float16
    elif args.dtype == "bf16":
        autocast_dtype = torch.bfloat16

    def random_one_hot(batch: int, channels: int, length: int, n_seq: int):
        idx = torch.randint(
            low=0,
            high=channels,
            size=(batch, length, n_seq),
            device=device,
        )
        x = torch.nn.functional.one_hot(idx, num_classes=channels).permute(0, 3, 1, 2)
        return x.float()

    def load_pf1_model():
        if args.n_seq > pf1_impl.SEQ2PAIR.shape[1]:
            pf1_impl.SEQ2PAIR = pf1_impl.seq2pair(args.n_seq)
        ckpt_path = Path(args.pf1_ckpt)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            params = dict(ckpt["hyper_parameters"])
            params["n_seqs"] = max(params.get("n_seqs", args.n_seq), args.n_seq)
            model = Phyloformer(**params)
            model.load_state_dict(
                {
                    k.replace("model.", ""): v
                    for k, v in ckpt["state_dict"].items()
                    if k != "model.seq2pair"
                },
                strict=False,
            )
            return model.to(device)
        return Phyloformer(
            n_blocks=args.pf1_blocks,
            n_heads=args.pf1_heads,
            h_dim=args.pf1_embed,
            n_seqs=max(args.n_seq, 50),
        ).to(device)

    def build_pf2_variant(profile: str):
        if Path(args.pf2_ckpt).exists() and profile.startswith("pf2-bayesnj"):
            ckpt = torch.load(args.pf2_ckpt, map_location="cpu")
            model = load_pf2_model(ckpt, device)
            return model.to(device)
        use_bilinear = profile.startswith("pf2-bayesnj")
        use_brlens = profile.startswith("pf2-bayesnj")
        return EvoPF(
            n_blocks=args.pf2_blocks,
            n_heads=args.pf2_heads,
            h_dim=args.pf2_embed,
            pair_dim=args.pf2_pair_dim,
            use_opm=True,
            distance_mlp=False,
            symmetric=args.symmetric,
            use_bilinear_embedder=use_bilinear,
            use_brlens=use_brlens,
        ).to(device)

    model = None
    x = None
    loss = None
    result: dict[str, Any] = {
        "profile": args.profile,
        "n_seq": args.n_seq,
        "n_sites": args.n_sites,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
    }

    try:
        if args.profile.startswith("pf1"):
            model = load_pf1_model()
            channels = 22
        else:
            model = build_pf2_variant(args.profile)
            channels = 23

        x = random_one_hot(args.batch_size, channels, args.n_sites, args.n_seq)
        is_train = args.profile.endswith("train")
        model.train(is_train)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()

        if args.profile == "pf1-infer":
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=autocast_dtype is not None, dtype=autocast_dtype):
                    out = model(x)
        elif args.profile == "pf1-train":
            with torch.autocast(device_type="cuda", enabled=autocast_dtype is not None, dtype=autocast_dtype):
                out = model(x)
                out2 = out if out.ndim == 2 else out.unsqueeze(0)
                target = torch.rand_like(out2)
                loss = torch.nn.functional.l1_loss(out2, target)
            loss.backward()
        elif args.profile == "pf2-mae-infer":
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=autocast_dtype is not None, dtype=autocast_dtype):
                    out = model(x)
        elif args.profile == "pf2-mae-train":
            with torch.autocast(device_type="cuda", enabled=autocast_dtype is not None, dtype=autocast_dtype):
                out = model(x)
                out2 = out if out.ndim == 2 else out.unsqueeze(0)
                target = torch.rand_like(out2)
                loss = torch.nn.functional.l1_loss(out2, target)
            loss.backward()
        elif args.profile == "pf2-bayesnj-infer":
            if not getattr(model, "output_msa_emb", False):
                raise ValueError(
                    "pf2-bayesnj-infer requires a PF2 checkpoint/model with output_msa_emb enabled."
                )
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=autocast_dtype is not None, dtype=autocast_dtype):
                    dm, msa_emb = model(x)
                    if dm.size(-1) == args.n_seq:
                        dm_sq = dm
                    else:
                        dm_sq = lower_vec_to_mat(dm, args.n_seq, torch)
                    merges, brlens, topo_logsum, brlen_logsum = batch_sample_trees(
                        model,
                        msa_emb,
                        dm_sq,
                        use_max_proba=True,
                        verbose=False,
                    )
        elif args.profile == "pf2-bayesnj-train":
            if not getattr(model, "output_msa_emb", False):
                raise ValueError(
                    "pf2-bayesnj-train requires a PF2 checkpoint/model with output_msa_emb enabled."
                )
            merge_order, brlens = make_balanced_merge_order(args.n_seq, device, torch)
            with torch.autocast(device_type="cuda", enabled=autocast_dtype is not None, dtype=autocast_dtype):
                dm, msa_emb, logprob_topo, logprob_gamma, logprob_beta, logprob_brlens, mse_brlens = model(
                    x,
                    brlens=brlens,
                    merge_order=merge_order,
                    topo_only=False,
                    ignore_topo=False,
                    return_tree_logprob=True,
                    verbose=False,
                )
                norm = float(args.n_seq)
                loss = (-(logprob_topo / norm) - (logprob_brlens / norm)).mean()
            loss.backward()
        else:
            raise ValueError(f"Unknown profile: {args.profile}")

        torch.cuda.synchronize(device)
        elapsed_sec = time.perf_counter() - start
        time.sleep(args.hold_sec)
        torch.cuda.synchronize(device)

        result.update(
            {
                "status": "ok",
                "elapsed_sec": elapsed_sec,
                "torch_max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                "torch_max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"status": "oom", "error": str(exc)})
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            result.update({"status": "oom", "error": str(exc)})
            torch.cuda.empty_cache()
        else:
            result.update({"status": "error", "error": str(exc)})
    except Exception as exc:
        result.update({"status": "error", "error": repr(exc)})

    Path(args.json_out).write_text(json.dumps(result), encoding="utf-8")
    return 0 if result["status"] in {"ok", "oom"} else 1


def plot_heatmap(matrix: np.ndarray, seqs: list[int], lengths: list[int], out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(lengths)), max(6, 0.6 * len(seqs))))
    cmap = plt.get_cmap("rocket").copy() if "rocket" in plt.colormaps() else plt.get_cmap("magma").copy()
    cmap.set_bad(color="#d9d9e3")
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([str(x) for x in lengths])
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels([str(x) for x in seqs])
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Nb. of Sequences")
    ax.set_title(title)

    for i in range(len(seqs)):
        for j in range(len(lengths)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Max GPU RSS (GB)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parent_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        choices=[
            "pf1-infer",
            "pf1-train",
            "pf2-mae-infer",
            "pf2-mae-train",
            "pf2-bayesnj-infer",
            "pf2-bayesnj-train",
        ],
    )
    parser.add_argument("--seqs", required=True, help="Comma-separated sequence counts.")
    parser.add_argument("--lengths", required=True, help="Comma-separated alignment lengths.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index for both worker and nvidia-smi polling.")
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--hold-sec", type=float, default=0.5)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pf1-ckpt", default=str(REPO_ROOT / "pretrained_models" / "pf_base.ckpt"))
    parser.add_argument("--pf2-ckpt", default=str(PF2_ROOT / "pretrained" / "pf2.tch"))
    parser.add_argument("--pf1-blocks", type=int, default=6)
    parser.add_argument("--pf1-heads", type=int, default=4)
    parser.add_argument("--pf1-embed", type=int, default=64)
    parser.add_argument("--pf2-blocks", type=int, default=12)
    parser.add_argument("--pf2-heads", type=int, default=4)
    parser.add_argument("--pf2-embed", type=int, default=128)
    parser.add_argument("--pf2-pair-dim", type=int, default=256)
    parser.add_argument("--symmetric", action="store_true", default=True)
    parser.add_argument("--no-symmetric", dest="symmetric", action="store_false")
    args = parser.parse_args(argv)

    seqs = parse_int_list(args.seqs)
    lengths = parse_int_list(args.lengths)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    matrix = np.full((len(seqs), len(lengths)), np.nan, dtype=float)

    for i, n_seq in enumerate(seqs):
        for j, n_sites in enumerate(lengths):
            with tempfile.NamedTemporaryFile(prefix="pfmem_", suffix=".json", delete=False) as tmp:
                json_path = Path(tmp.name)

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--profile", args.profile,
                "--n-seq", str(n_seq),
                "--n-sites", str(n_sites),
                "--batch-size", str(args.batch_size),
                "--dtype", args.dtype,
                "--hold-sec", str(args.hold_sec),
                "--json-out", str(json_path),
                "--pf1-ckpt", args.pf1_ckpt,
                "--pf2-ckpt", args.pf2_ckpt,
                "--pf1-blocks", str(args.pf1_blocks),
                "--pf1-heads", str(args.pf1_heads),
                "--pf1-embed", str(args.pf1_embed),
                "--pf2-blocks", str(args.pf2_blocks),
                "--pf2-heads", str(args.pf2_heads),
                "--pf2-embed", str(args.pf2_embed),
                "--pf2-pair-dim", str(args.pf2_pair_dim),
            ]
            if args.gpu is not None:
                cmd.extend(["--gpu", str(args.gpu)])
            if args.symmetric:
                cmd.append("--symmetric")
            else:
                cmd.append("--no-symmetric")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                cwd=REPO_ROOT,
            )
            peak_mib = 0.0
            stderr_text = ""
            while proc.poll() is None:
                peak_mib = max(peak_mib, poll_nvidia_smi(proc.pid, args.gpu))
                time.sleep(args.poll_interval)

            if proc.stderr is not None:
                stderr_text = proc.stderr.read().strip()

            peak_mib = max(peak_mib, poll_nvidia_smi(proc.pid, args.gpu))

            payload: dict[str, Any]
            if json_path.exists():
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                json_path.unlink(missing_ok=True)
            else:
                payload = {"status": "error", "error": f"Worker exited with code {proc.returncode}"}

            row = {
                "profile": args.profile,
                "n_seq": n_seq,
                "n_sites": n_sites,
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                "status": payload.get("status", "error"),
                "rss_peak_gb": peak_mib / 1024.0 if peak_mib > 0 else np.nan,
                "torch_max_allocated_gb": payload.get("torch_max_allocated_gb", np.nan),
                "torch_max_reserved_gb": payload.get("torch_max_reserved_gb", np.nan),
                "elapsed_sec": payload.get("elapsed_sec", np.nan),
                "error": payload.get("error", ""),
                "stderr": stderr_text[-500:],
            }
            rows.append(row)
            if row["status"] == "ok" and not np.isnan(row["rss_peak_gb"]):
                matrix[i, j] = float(row["rss_peak_gb"])

            status = row["status"]
            peak_text = "OOM" if status == "oom" else format_float(row["rss_peak_gb"])
            print(f"[{args.profile}] n={n_seq:>4} L={n_sites:>6} -> {status} {peak_text}")

    long_csv = outdir / "memory_grid_long.csv"
    with long_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "profile",
                "n_seq",
                "n_sites",
                "batch_size",
                "dtype",
                "status",
                "rss_peak_gb",
                "torch_max_allocated_gb",
                "torch_max_reserved_gb",
                "elapsed_sec",
                "error",
                "stderr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    matrix_csv = outdir / "memory_grid_matrix.csv"
    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n_seq"] + lengths)
        for i, n_seq in enumerate(seqs):
            writer.writerow([n_seq] + [format_float(v) for v in matrix[i]])

    plot_heatmap(
        matrix=matrix,
        seqs=seqs,
        lengths=lengths,
        out_png=outdir / "memory_grid_heatmap.png",
        title=f"{args.profile} | batch={args.batch_size} | {args.dtype}",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        parser = build_worker_parser()
        return worker_main(parser.parse_args(argv))
    return parent_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
