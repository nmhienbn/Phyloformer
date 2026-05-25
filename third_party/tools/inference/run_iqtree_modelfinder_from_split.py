#!/usr/bin/env python3

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "classical_methods" / "run_iqtree_modelfinder_from_split.py"),
    run_name="__main__",
)
