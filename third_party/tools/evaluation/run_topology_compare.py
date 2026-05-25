#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import pandas as pd
from ete3 import Tree
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare predicted and reference tree topologies using Robinson-Foulds distance."
    )
    parser.add_argument(
        "--pred-trees",
        required=True,
        help="Directory containing predicted .nwk trees.",
    )
    parser.add_argument(
        "--true-trees",
        required=True,
        help="Directory containing reference .nwk trees.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--method-name",
        default="HybridA",
        help="Marker/method label written into the CSV.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label for tqdm/timing logs.",
    )
    return parser.parse_args()


def load_tree(path: Path) -> Tree:
    tree = Tree(str(path), format=1)
    tree.unroot()
    return tree


def compare_tree_pair(pred_path: Path, true_path: Path, method_name: str) -> dict:
    pred_tree = load_tree(pred_path)
    true_tree = load_tree(true_path)
    metrics = true_tree.compare(pred_tree, unrooted=True)
    return {
        "id": pred_path.stem,
        "rf": float(metrics["rf"]),
        "norm_rf": float(metrics["norm_rf"]),
        "weighted_rf": float("nan"),
        "kf_score": float("nan"),
        "n_tips": int(metrics["effective_tree_size"]),
        "marker": method_name,
    }


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_trees)
    true_dir = Path(args.true_trees)
    out_path = Path(args.out)
    label = args.label or pred_dir.parent.name or pred_dir.name

    pred_paths = sorted(list(pred_dir.glob("*.nwk")) + list(pred_dir.glob("*.tre")))
    if not pred_paths:
        raise FileNotFoundError(f"No predicted trees found in {pred_dir}")

    rows = []
    missing = []
    start = time.perf_counter()
    for pred_path in tqdm(pred_paths, desc=f"topology compare [{label}]", unit="tree"):
        true_path = true_dir / pred_path.name
        if not true_path.exists():
            missing.append(pred_path.name)
            continue
        rows.append(compare_tree_pair(pred_path, true_path, args.method_name))

    if not rows:
        raise RuntimeError("No comparable tree pairs found.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    elapsed = time.perf_counter() - start
    print(f"[INFO] [{label}] compared: {len(rows)} trees")
    if missing:
        print(f"[WARN] [{label}] missing references for {len(missing)} predicted trees")
    print(f"[INFO] [{label}] output: {out_path}")
    print(f"[TIME] [{label}] topology compare total: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
