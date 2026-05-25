#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

PF2_SEQ_BINS = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
PF2_LEN_BINS = [100, 250, 500, 1000, 2000, 3000, 4000, 5000, 10000]
PF2_MAX_LEN_BY_SEQ_BIN = {
    10: 10000,
    50: 10000,
    100: 10000,
    150: 5000,
    200: 3000,
    250: 2000,
    300: 1000,
    350: 1000,
    400: 500,
    450: 500,
    500: 250,
    550: 250,
    600: 250,
}
PF2_RSS_GB = {
    (10, 100): 0.4,
    (10, 250): 0.4,
    (10, 500): 0.4,
    (10, 1000): 0.4,
    (10, 2000): 0.5,
    (10, 3000): 0.5,
    (10, 4000): 0.6,
    (10, 5000): 0.7,
    (10, 10000): 1.0,
    (50, 100): 0.5,
    (50, 250): 0.7,
    (50, 500): 1.0,
    (50, 1000): 1.7,
    (50, 2000): 3.0,
    (50, 3000): 4.3,
    (50, 4000): 5.6,
    (50, 5000): 6.9,
    (50, 10000): 13.4,
    (100, 100): 0.9,
    (100, 250): 1.6,
    (100, 500): 2.9,
    (100, 1000): 5.5,
    (100, 2000): 10.6,
    (100, 3000): 15.7,
    (100, 4000): 20.8,
    (100, 5000): 25.9,
    (100, 10000): 51.5,
    (150, 100): 1.5,
    (150, 250): 3.2,
    (150, 500): 6.1,
    (150, 1000): 11.8,
    (150, 2000): 23.2,
    (150, 3000): 34.6,
    (150, 4000): 46.0,
    (150, 5000): 57.4,
    (200, 100): 2.4,
    (200, 250): 5.5,
    (200, 500): 10.5,
    (200, 1000): 20.6,
    (200, 2000): 41.0,
    (200, 3000): 61.3,
    (250, 100): 3.6,
    (250, 250): 8.3,
    (250, 500): 16.2,
    (250, 1000): 32.0,
    (250, 2000): 63.5,
    (300, 100): 5.0,
    (300, 250): 11.8,
    (300, 500): 23.2,
    (300, 1000): 45.8,
    (350, 100): 6.7,
    (350, 250): 16.0,
    (350, 500): 31.4,
    (350, 1000): 62.2,
    (400, 100): 8.7,
    (400, 250): 20.7,
    (400, 500): 40.9,
    (450, 100): 10.9,
    (450, 250): 26.2,
    (450, 500): 51.6,
    (500, 100): 13.4,
    (500, 250): 32.2,
    (550, 100): 16.2,
    (550, 250): 38.9,
    (600, 100): 19.2,
    (600, 250): 46.3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export MSA statistics from data/zenodo_raw for both PANDIT and TreeBASE "
            "into a single CSV file."
        )
    )
    parser.add_argument(
        "--input-root",
        default="data/zenodo_raw",
        help="Root directory containing data_pandit and data_treebase.",
    )
    parser.add_argument(
        "--output",
        default="data/zenodo_raw/msa_stats.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--not-fit-output",
        default="data/zenodo_raw/msa_stats_pf2_not_fit.csv",
        help="Optional CSV path containing only rows flagged as not fitting the PF2 memory profile.",
    )
    return parser.parse_args()


def ceil_to_bin(value: int, bins: list[int]) -> int | None:
    for candidate in bins:
        if value <= candidate:
            return candidate
    return None


def annotate_pf2_memory(row: dict[str, object]) -> dict[str, object]:
    num_sequences = int(row["num_sequences"])
    alignment_length = int(row["alignment_length"])
    seq_bin = ceil_to_bin(num_sequences, PF2_SEQ_BINS)
    len_bin = ceil_to_bin(alignment_length, PF2_LEN_BINS)

    row["pf2_seq_bin"] = seq_bin if seq_bin is not None else ""
    row["pf2_length_bin"] = len_bin if len_bin is not None else ""

    if seq_bin is None or len_bin is None:
        row["pf2_estimated_max_gpu_rss_gb"] = ""
        row["pf2_fits_memory_limit"] = "no"
        row["pf2_memory_note"] = "out_of_profile_range"
        return row

    estimated_rss = PF2_RSS_GB.get((seq_bin, len_bin))
    max_supported_length = PF2_MAX_LEN_BY_SEQ_BIN[seq_bin]
    fits_memory_limit = len_bin <= max_supported_length and estimated_rss is not None

    row["pf2_estimated_max_gpu_rss_gb"] = f"{estimated_rss:.1f}" if estimated_rss is not None else ""
    row["pf2_fits_memory_limit"] = "yes" if fits_memory_limit else "no"
    row["pf2_memory_note"] = "within_profile" if fits_memory_limit else "exceeds_profile_limit"
    return row


def read_phylip_header(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()

    if not first_line:
        raise ValueError(f"Empty file: {path}")

    parts = first_line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid PHYLIP header in {path}: {first_line!r}")

    try:
        num_sequences = int(parts[0])
        alignment_length = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid PHYLIP header in {path}: {first_line!r}") from exc

    return num_sequences, alignment_length


def collect_pandit_rows(input_root: Path) -> list[dict[str, object]]:
    rows = []
    pandit_root = input_root / "data_pandit"
    for seq_type in ("aa", "dna"):
        type_root = pandit_root / seq_type
        if not type_root.exists():
            continue
        for family_dir in sorted(path for path in type_root.iterdir() if path.is_dir()):
            msa_id = family_dir.name
            msa_path = family_dir / f"data.{msa_id}"
            if not msa_path.exists():
                continue
            num_sequences, alignment_length = read_phylip_header(msa_path)
            rows.append(
                {
                    "dataset": "pandit",
                    "seq_type": seq_type,
                    "msa_id": msa_id,
                    "file_name": msa_path.name,
                    "relative_path": str(msa_path.relative_to(input_root)),
                    "num_sequences": num_sequences,
                    "alignment_length": alignment_length,
                }
            )
    return rows


def collect_treebase_rows(input_root: Path) -> list[dict[str, object]]:
    rows = []
    treebase_root = input_root / "data_treebase"
    if not treebase_root.exists():
        return rows

    for msa_path in sorted(treebase_root.glob("*.phy")):
        if msa_path.name.startswith("prot_"):
            seq_type = "aa"
        elif msa_path.name.startswith("dna_"):
            seq_type = "dna"
        else:
            continue

        num_sequences, alignment_length = read_phylip_header(msa_path)
        rows.append(
            {
                "dataset": "treebase",
                "seq_type": seq_type,
                "msa_id": msa_path.stem,
                "file_name": msa_path.name,
                "relative_path": str(msa_path.relative_to(input_root)),
                "num_sequences": num_sequences,
                "alignment_length": alignment_length,
            }
        )
    return rows


def get_fieldnames() -> list[str]:
    return [
        "dataset",
        "seq_type",
        "msa_id",
        "file_name",
        "relative_path",
        "num_sequences",
        "alignment_length",
        "pf2_seq_bin",
        "pf2_length_bin",
        "pf2_estimated_max_gpu_rss_gb",
        "pf2_fits_memory_limit",
        "pf2_memory_note",
    ]


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = get_fieldnames()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_path = Path(args.output)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    rows = collect_pandit_rows(input_root)
    rows.extend(collect_treebase_rows(input_root))
    rows = [annotate_pf2_memory(row) for row in rows]
    rows.sort(key=lambda row: (row["dataset"], row["seq_type"], row["msa_id"]))

    if not rows:
        raise FileNotFoundError(f"No MSA files found under {input_root}")

    write_csv(rows, output_path)
    not_fit_rows = [row for row in rows if row["pf2_fits_memory_limit"] == "no"]
    write_csv(not_fit_rows, Path(args.not_fit_output))
    print(
        f"[OK] Wrote {len(rows)} MSA records to {output_path} "
        f"and {len(not_fit_rows)} not-fit records to {args.not_fit_output}"
    )


if __name__ == "__main__":
    main()
