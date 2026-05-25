#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

from Bio import AlignIO, SeqIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the Zenodo PANDIT/TreeBASE benchmark bundle for PF1/PF2 by "
            "converting PHYLIP alignments to FASTA and optionally collecting trees/models."
        )
    )
    parser.add_argument(
        "dataset_type",
        choices=["pandit", "treebase"],
        help="Which benchmark subset to prepare.",
    )
    parser.add_argument(
        "input_root",
        help="Path to the extracted Zenodo dataset root.",
    )
    parser.add_argument(
        "output_root",
        help="Directory where the prepared benchmark layout will be written.",
    )
    parser.add_argument(
        "--phylip-format",
        default="phylip-relaxed",
        choices=["phylip", "phylip-relaxed"],
        help="Biopython PHYLIP parser to use.",
    )
    parser.add_argument(
        "--treebase-prefix",
        action="append",
        default=None,
        help=(
            "Optional filename prefix filter for TreeBASE .phy files. "
            "Repeatable, e.g. --treebase-prefix prot_. If omitted, all TreeBASE .phy files are included."
        ),
    )
    return parser.parse_args()


def convert_phylip_to_fasta(src: Path, dst: Path, fmt: str) -> None:
    alignment = AlignIO.read(src, fmt)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as handle:
        SeqIO.write(alignment, handle, "fasta")


def prepare_treebase(input_root: Path, output_root: Path, fmt: str, prefixes: list[str] | None) -> None:
    alignments_dir = output_root / "alignments"
    alignments_dir.mkdir(parents=True, exist_ok=True)

    phylips = sorted(input_root.glob("*.phy"))
    if prefixes:
        phylips = [path for path in phylips if any(path.name.startswith(prefix) for prefix in prefixes)]
    if not phylips:
        raise FileNotFoundError(f"No .phy files found in {input_root}")

    for src in phylips:
        convert_phylip_to_fasta(src, alignments_dir / f"{src.stem}.fa", fmt)

    extra = f" filtered by prefixes={prefixes}" if prefixes else ""
    print(f"[OK] Prepared TreeBASE subset with {len(phylips)} alignments at {alignments_dir}{extra}")


def prepare_pandit(input_root: Path, output_root: Path, fmt: str) -> None:
    alignments_dir = output_root / "alignments"
    trees_dir = output_root / "trees"
    models_dir = output_root / "models"
    alignments_dir.mkdir(parents=True, exist_ok=True)
    trees_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    numbered_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not numbered_dirs:
        raise FileNotFoundError(f"No numbered benchmark directories found in {input_root}")

    n_done = 0
    for subdir in numbered_dirs:
        suffix = subdir.name
        data_path = subdir / f"data.{suffix}"
        tree_path = subdir / f"tree.{suffix}"
        model_path = subdir / f"model.{suffix}"
        if not data_path.exists():
            continue

        convert_phylip_to_fasta(data_path, alignments_dir / f"{suffix}.fa", fmt)

        if tree_path.exists():
            shutil.copy2(tree_path, trees_dir / f"{suffix}.nwk")
        if model_path.exists():
            shutil.copy2(model_path, models_dir / f"{suffix}.model.txt")
        n_done += 1

    if n_done == 0:
        raise FileNotFoundError(f"No PANDIT benchmark entries found in {input_root}")

    print(
        "[OK] Prepared PANDIT subset with "
        f"{n_done} alignments at {alignments_dir} and matching trees/models when available"
    )


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    if args.dataset_type == "treebase":
        prepare_treebase(input_root, output_root, args.phylip_format, args.treebase_prefix)
    else:
        prepare_pandit(input_root, output_root, args.phylip_format)


if __name__ == "__main__":
    main()
