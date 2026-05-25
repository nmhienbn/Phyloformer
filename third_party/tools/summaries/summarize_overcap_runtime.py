#!/usr/bin/env python3
"""Summarize over-cap benchmark runtime for selected partition methods.

The primary runtime source is each partition directory's runtime_summary.csv.
Merge time is read from <base>/merge_timing.csv when present. If a selected
partition/merge row is missing there, the script falls back to IQ-TREE refit
logs in merge_<partition>_<merge>/final_trees and estimates wall-clock runtime
as max(end timestamp) - min(start timestamp).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


METHODS = [
    ("Window-rate", "window_rate"),
    ("Window-site", "window_pos"),
    ("SoftBioBlock", "softbioblock"),
]

DATASETS = [
    ("Cherry over-2GB", Path("runs/benchmarks/cherry_over2gb")),
    ("PANDIT over-1GB", Path("runs/benchmarks/pandit_over1gb")),
    ("Superaln 30k", Path("runs/benchmarks/superaln_16gb/30k")),
    ("Superaln 60k", Path("runs/benchmarks/superaln_16gb/60k")),
    ("Superaln 100k", Path("runs/benchmarks/superaln_16gb/100k")),
]
DATASET_ORDER = {name: idx for idx, (name, _) in enumerate(DATASETS)}
METHOD_ORDER = {
    "Window-rate": 0,
    "Window-site": 1,
    "SoftBioBlock": 2,
    "FastTree direct": 3,
    "IQ-TREE LG+GC": 4,
    "IQ-TREE MF": 5,
    "IQ-TREE model.txt": 6,
}

TIME_RE = re.compile(r"^Time:\s+(.*)$")
DATE_RE = re.compile(r"^Date and Time:\s+(.*)$")
ELAPSED_RE = re.compile(r"Elapsed \(wall clock\) time .*:\s+([0-9:.]+)")
TS_FORMAT = "%a %b %d %H:%M:%S %Y"


@dataclass
class RuntimeRow:
    dataset: str
    method: str
    n_cases: int
    partition_sec: float | None
    merge_sec: float | None
    merge_source: str
    total_override_sec: float | None = None

    @property
    def total_sec(self) -> float | None:
        if self.total_override_sec is not None:
            return self.total_override_sec
        if self.partition_sec is None:
            return None
        if self.merge_sec is None:
            return None
        return self.partition_sec + self.merge_sec

    @property
    def mean_total_sec(self) -> float | None:
        if self.total_sec is None or self.n_cases <= 0:
            return None
        return self.total_sec / self.n_cases


def read_partition_runtime(base: Path, partition: str) -> tuple[int, float]:
    path = base / f"partition_{partition}" / "runtime_summary.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    return int(row["n_alignments"]), float(row["total_sec"])


def read_merge_timing(base: Path, partition: str, merge: str) -> float | None:
    path = base / "merge_timing.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["partition"] == partition and row["merge"] == merge:
                return float(row["elapsed_sec"])
    return None


def parse_iqtree_timestamp(line: str, regex: re.Pattern[str]) -> datetime | None:
    match = regex.match(line.strip())
    if not match:
        return None
    return datetime.strptime(" ".join(match.group(1).split()), TS_FORMAT)


def estimate_merge_from_iqtree_logs(base: Path, partition: str, merge: str) -> float | None:
    log_dir = base / f"merge_{partition}_{merge}" / "final_trees"
    starts: list[datetime] = []
    ends: list[datetime] = []
    for log_path in sorted(log_dir.glob("*.iqtree_refit.log")):
        start = end = None
        with log_path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if start is None:
                    start = parse_iqtree_timestamp(line, TIME_RE)
                parsed_end = parse_iqtree_timestamp(line, DATE_RE)
                if parsed_end is not None:
                    end = parsed_end
        if start is not None and end is not None:
            starts.append(start)
            ends.append(end)

    if not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds()


def stems_from_tree_dir(path: Path) -> set[str]:
    return {p.stem for p in path.glob("*.nwk")}


def parse_gnu_time_elapsed(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = ELAPSED_RE.search(text)
    if not match:
        raise ValueError(f"elapsed wall-clock line not found in {path}")
    parts = match.group(1).split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"unsupported elapsed format in {path}: {match.group(1)}")


def sum_gnu_time_files(times_dir: Path, stems: set[str]) -> tuple[int, float]:
    total = 0.0
    n = 0
    for stem in sorted(stems):
        path = times_dir / f"{stem}.time"
        if not path.exists():
            continue
        total += parse_gnu_time_elapsed(path)
        n += 1
    return n, total


def sum_runtime_per_alignment(path: Path, ids: set[str]) -> tuple[int, float]:
    total = 0.0
    n = 0
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("id") not in ids:
                continue
            if row.get("status") != "done":
                continue
            total += float(row["elapsed_sec"])
            n += 1
    return n, total


def collect_classical_rows(root: Path) -> list[RuntimeRow]:
    rows: list[RuntimeRow] = []

    cherry_stems = stems_from_tree_dir(root / "runs/benchmarks/cherry_over2gb/pf2_direct")
    cherry_sources = [
        ("FastTree direct", root / "data/cherry_test_data/FastTree/times"),
        ("IQ-TREE LG+GC", root / "data/cherry_test_data/IQTree_LG+GC/times"),
        ("IQ-TREE MF", root / "data/cherry_test_data/IQTree_MF/times"),
    ]
    for method, times_dir in cherry_sources:
        n_cases, total_sec = sum_gnu_time_files(times_dir, cherry_stems)
        rows.append(
            RuntimeRow(
                dataset="Cherry over-2GB",
                method=method,
                n_cases=n_cases,
                partition_sec=None,
                merge_sec=None,
                merge_source="gnu_time_files",
                total_override_sec=total_sec,
            )
        )

    pandit_ids = stems_from_tree_dir(root / "runs/benchmarks/pandit_over1gb/pf2_direct")
    pandit_sources = [
        (
            "FastTree direct",
            root / "runs/completed/topology/fasttree_pandit_aa/runtime_per_alignment.csv",
        ),
        (
            "IQ-TREE model.txt",
            root / "runs/completed/topology/iqtree_modeltxt_pandit_aa/runtime_per_alignment.csv",
        ),
    ]
    for method, runtime_csv in pandit_sources:
        n_cases, total_sec = sum_runtime_per_alignment(runtime_csv, pandit_ids)
        rows.append(
            RuntimeRow(
                dataset="PANDIT over-1GB",
                method=method,
                n_cases=n_cases,
                partition_sec=None,
                merge_sec=None,
                merge_source="runtime_per_alignment.csv",
                total_override_sec=total_sec,
            )
        )

    return rows


def collect_rows(root: Path, merge: str) -> list[RuntimeRow]:
    rows: list[RuntimeRow] = []
    for dataset, rel_base in DATASETS:
        base = root / rel_base
        for method, partition in METHODS:
            n_cases, partition_sec = read_partition_runtime(base, partition)
            merge_sec = read_merge_timing(base, partition, merge)
            merge_source = "merge_timing.csv"
            if merge_sec is None:
                merge_sec = estimate_merge_from_iqtree_logs(base, partition, merge)
                merge_source = "iqtree_log_span" if merge_sec is not None else "missing"
            rows.append(RuntimeRow(dataset, method, n_cases, partition_sec, merge_sec, merge_source))
    rows = rows + collect_classical_rows(root)
    return sorted(
        rows,
        key=lambda r: (
            DATASET_ORDER.get(r.dataset, 999),
            METHOD_ORDER.get(r.method, 999),
            r.method,
        ),
    )


def fmt_sec(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "--"
    return f"{value:.2f} s"


def fmt_hours(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value / 3600:.2f} h"


def write_csv(rows: list[RuntimeRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "method",
                "n_cases",
                "partition_pf2_sec",
                "merge_sec",
                "total_sec",
                "mean_total_sec_per_case",
                "total_hours",
                "merge_source",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "method": row.method,
                    "n_cases": row.n_cases,
                    "partition_pf2_sec": ""
                    if row.partition_sec is None
                    else f"{row.partition_sec:.6f}",
                    "merge_sec": "" if row.merge_sec is None else f"{row.merge_sec:.6f}",
                    "total_sec": "" if row.total_sec is None else f"{row.total_sec:.6f}",
                    "mean_total_sec_per_case": ""
                    if row.mean_total_sec is None
                    else f"{row.mean_total_sec:.6f}",
                    "total_hours": "" if row.total_sec is None else f"{row.total_sec / 3600:.6f}",
                    "merge_source": row.merge_source,
                }
            )


def markdown_table(rows: list[RuntimeRow]) -> str:
    lines = [
        "| Bộ dữ liệu | Phương pháp | Số case | Partition+PF2 | Merge weighted | Tổng | Mean tổng/case |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.dataset,
                    row.method,
                    str(row.n_cases),
                    fmt_sec(row.partition_sec),
                    fmt_sec(row.merge_sec),
                    fmt_sec(row.total_sec),
                    fmt_sec(row.mean_total_sec),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def latex_table(rows: list[RuntimeRow]) -> str:
    body: list[str] = []
    prev_dataset = None
    for row in rows:
        if prev_dataset is not None and row.dataset != prev_dataset:
            body.append(r"        \midrule")
        body.append(
            "        "
            + " & ".join(
                [
                    row.dataset,
                    row.method,
                    str(row.n_cases),
                    fmt_sec(row.partition_sec),
                    fmt_sec(row.merge_sec),
                    fmt_sec(row.total_sec),
                    fmt_sec(row.mean_total_sec),
                ]
            )
            + r" \\"
        )
        prev_dataset = row.dataset

    return "\n".join(
        [
            r"\begin{table}[H]",
            r"    \centering",
            r"    \small",
            r"    \caption{Tổng hợp thời gian chạy end-to-end trên ba benchmark dài}",
            r"    \label{tab:runtime-summary-overcap}",
            r"    \begin{tabular}{llrrrrr}",
            r"        \toprule",
            r"        Bộ dữ liệu & Phương pháp & Số case & Partition+PF2 & Merge weighted & Tổng & Mean tổng/case \\",
            r"        \midrule",
            *body,
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )


def write_markdown(rows: list[RuntimeRow], csv_path: Path, path: Path, merge: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted({row.merge_source for row in rows})
    text = f"""# Tổng hợp runtime cho benchmark over-cap

Sinh lại file này bằng lệnh:

```bash
python third_party/tools/summaries/summarize_overcap_runtime.py
```

Summary này lấy ba pipeline partition-PF2 chính (`Window-rate`, `Window-site`,
`SoftBioBlock`) và thêm baseline trực tiếp `FastTree`/`IQ-TREE` cho Cherry và
PANDIT. Phương pháp merge cho các pipeline PF2 là `{merge}`. Cột
`Partition+PF2` được đọc từ từng file `partition_*/runtime_summary.csv`; cột
merge được đọc từ `merge_timing.csv` nếu có. Nếu thiếu row tương ứng, script
suy ra wall-clock từ timestamp đầu/cuối trong các log IQ-TREE refit ở
`merge_*_{merge}/final_trees/*.iqtree_refit.log`.

Với baseline trực tiếp, `Partition+PF2` và `Merge weighted` để trống (`--`) vì
runtime đã là tổng runtime của chính method đó. Cherry đọc từ các file GNU
`time` trong `data/cherry_test_data/*/times`; PANDIT đọc từ
`runtime_per_alignment.csv` trong `runs/completed/topology`.

CSV output: `{csv_path}`.

Nguồn merge timing trong lần chạy này: `{", ".join(sources)}`.

## Bảng Markdown

{markdown_table(rows)}

## Bảng LaTeX

```latex
{latex_table(rows)}
```

## Ghi chú

- Với pipeline PF2, `Tổng` là `Partition+PF2 + Merge weighted`.
- Với baseline trực tiếp, `Tổng` là runtime trực tiếp của method đó.
- `Mean tổng/case` là `Tổng / Số case`.
- Các dòng có nguồn merge `iqtree_log_span` được tái dựng từ timestamp trong
  log vì `merge_timing.csv` không có sẵn row tương ứng.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--merge", default="weighted", choices=["weighted", "fastrfs"])
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("runs/benchmarks/overcap_runtime_summary.csv"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("docs/benchmarks/overcap_runtime_summary.md"),
    )
    args = parser.parse_args()

    rows = collect_rows(args.root, args.merge)
    write_csv(rows, args.csv_out)
    write_markdown(rows, args.csv_out, args.md_out, args.merge)

    print(markdown_table(rows))
    print(f"\n[OK] wrote {args.csv_out}")
    print(f"[OK] wrote {args.md_out}")


if __name__ == "__main__":
    main()
