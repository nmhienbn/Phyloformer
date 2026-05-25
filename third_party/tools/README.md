# Third-Party Tool Wrappers

This directory contains repo-local wrappers around external phylogenetics tools.
They are kept outside `experiments/` because they orchestrate third-party
binaries or legacy toolchains rather than implement experiment-specific model
logic.

## Layout

- `benchmark_data/`: dataset preparation/export helpers for benchmark inputs.
- `evaluation/`: PhyloCompare and topology comparison wrappers.
- `inference/`: benchmark runners that call PF/PF2, IQ-TREE, FastME, FastTree,
  and MPBoot.
- `plots/`: plotting utilities for benchmark, topology, VRAM, and paper figures.
- `summaries/`: runtime/result summarizers.
- `vram/`: PF2 VRAM cap utilities shared by partitioning and plotting scripts.
- `partition_merge/`: PF2 partition tree merge/refit helpers that call or wrap
  IQ-TREE, FastRFS, MRP, and SuperFine workflows.
- `superfine/`: local SuperFine source bundle/extraction area used by the
  SuperFine wrapper.
