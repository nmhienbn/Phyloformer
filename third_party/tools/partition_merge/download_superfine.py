#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


SUPERFINE_URLS = [
    "https://datadryad.org/downloads/file_stream/94912",
    "http://datadryad.org/downloads/file_stream/94912",
]
SUPERFINE_MD5 = "9e06d3002e926fd5c794a771897fdb18"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_python(url: str, output: Path) -> None:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://datadryad.org/stash/dataset/doi:10.5061/dryad.879st",
        },
    )
    with urlopen(request, timeout=60) as response, output.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def download_with_curl(url: str, output: Path) -> None:
    subprocess.run(
        [
            "curl",
            "-L",
            "-A",
            "Mozilla/5.0",
            "-e",
            "https://datadryad.org/stash/dataset/doi:10.5061/dryad.879st",
            "-o",
            str(output),
            url,
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SuperFine source from the Dryad public dataset.")
    parser.add_argument("--outdir", default="third_party/tools/superfine")
    parser.add_argument("--extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    zip_path = outdir / "SuperFineSource.zip"

    errors = []
    for url in SUPERFINE_URLS:
        for downloader in (download_with_python, download_with_curl):
            try:
                downloader(url, zip_path)
                if zip_path.exists() and md5sum(zip_path) == SUPERFINE_MD5:
                    print(f"[OK] downloaded {zip_path}")
                    if args.extract:
                        extract_dir = outdir / "source"
                        extract_dir.mkdir(parents=True, exist_ok=True)
                        with zipfile.ZipFile(zip_path) as archive:
                            archive.extractall(extract_dir)
                        print(f"[OK] extracted {extract_dir}")
                    return
                got = md5sum(zip_path) if zip_path.exists() else "missing"
                errors.append(f"{url}: md5={got}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url}: {exc}")

    if zip_path.exists():
        zip_path.unlink()
    print("[ERROR] Could not download a valid SuperFineSource.zip", file=sys.stderr)
    print("Expected MD5:", SUPERFINE_MD5, file=sys.stderr)
    print("Tried:", *errors, sep="\n  ", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
