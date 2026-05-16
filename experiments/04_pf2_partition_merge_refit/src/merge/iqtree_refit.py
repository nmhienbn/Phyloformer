#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path


def infer_alignment_from_manifest(manifest_path: Path) -> Path | None:
    metadata_dir = manifest_path.parent / "metadata"
    pipeline_path = metadata_dir / "pipeline.json"
    if not pipeline_path.exists():
        return None
    payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
    raw = payload.get("input_alignment")
    if not raw:
        return None
    return Path(raw)


def add_refit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fit-branch-lengths", action="store_true", help="Run IQ-TREE -te on the merged topology.")
    parser.add_argument("--alignment", default=None, help="Full alignment for IQ-TREE -te.")
    parser.add_argument("--iqtree-bin", default="bin/bin_linux/iqtree_2.2.0")
    parser.add_argument("--iqtree-model", default="LG+G4")
    parser.add_argument("--iqtree-threads", type=int, default=4)


def topology_path_for_refit(output_tree_path: Path, fit_branch_lengths: bool) -> Path:
    if not fit_branch_lengths:
        return output_tree_path
    return output_tree_path.with_suffix(output_tree_path.suffix + ".topology.nwk")


def refit_if_requested(
    *,
    fit_branch_lengths: bool,
    manifest_path: Path,
    topology_tree_path: Path,
    output_tree_path: Path,
    alignment: str | None,
    iqtree_bin: str,
    iqtree_model: str,
    iqtree_threads: int,
) -> None:
    if not fit_branch_lengths:
        return
    alignment_path = Path(alignment) if alignment else infer_alignment_from_manifest(manifest_path)
    if alignment_path is None:
        raise ValueError("Branch-length refit requested but no alignment was provided or found in metadata.")
    fit_branch_lengths_with_iqtree(
        alignment_path=alignment_path,
        topology_tree_path=topology_tree_path,
        output_tree_path=output_tree_path,
        iqtree_bin=iqtree_bin,
        iqtree_model=iqtree_model,
        iqtree_threads=iqtree_threads,
    )


def _strip_branch_lengths(newick: str) -> str:
    """Remove branch lengths (:value) from a Newick string so IQ-TREE uses its own init."""
    return re.sub(r":[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?", "", newick)


def fit_branch_lengths_with_iqtree(
    alignment_path: Path,
    topology_tree_path: Path,
    output_tree_path: Path,
    iqtree_bin: str,
    iqtree_model: str,
    iqtree_threads: int,
) -> None:
    if shutil.which(iqtree_bin) is None and not Path(iqtree_bin).exists():
        raise FileNotFoundError(f"IQ-TREE binary not found: {iqtree_bin}")

    prefix = output_tree_path.with_suffix(output_tree_path.suffix + ".iqtree_refit")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # Strip branch lengths from topology so IQ-TREE doesn't start from all-zero BL
    topo_for_iqtree = Path(f"{prefix}.topology.nwk")
    raw_topo = topology_tree_path.read_text(encoding="utf-8")
    topo_for_iqtree.write_text(_strip_branch_lengths(raw_topo), encoding="utf-8")

    base_cmd = [
        iqtree_bin,
        "-s", str(alignment_path),
        "-te", str(topo_for_iqtree),
        "-m", iqtree_model,
        "-nt", str(iqtree_threads),
        "-pre", str(prefix),
        "-redo",
    ]

    print(f"[IQTREE CLI] {' '.join(shlex.quote(x) for x in base_cmd)}")
    result = subprocess.run(base_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log = Path(f"{prefix}.log")
        log_text = log.read_text(encoding="utf-8") if log.exists() else (result.stderr or "")
        _numerical = any(kw in log_text.lower() for kw in ("underflow", "abrt", "abort", "assertion"))
        # SIGABRT (returncode < 0) always indicates a crash worth retrying with -safe
        if _numerical or result.returncode < 0:
            retry_cmd = base_cmd + ["-safe"]
            print(f"[IQTREE CLI retry] {' '.join(shlex.quote(x) for x in retry_cmd)}")
            result = subprocess.run(retry_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, base_cmd, output=result.stdout, stderr=result.stderr)

    treefile = Path(f"{prefix}.treefile")
    if not treefile.exists():
        raise FileNotFoundError(f"IQ-TREE completed but did not write {treefile}")
    output_tree_path.write_text(treefile.read_text(encoding="utf-8"), encoding="utf-8")
