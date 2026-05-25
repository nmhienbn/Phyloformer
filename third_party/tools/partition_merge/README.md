# Partition Merge Tool Wrappers

Wrappers used by experiment 04 after PF2 block inference:

- `run_fastrfs_merge.py`: FastRFS merge wrapper.
- `run_mrp_merge.py`: MRP matrix generation plus IQ-TREE morphology backend.
- `run_superfine_merge.py`: SuperFine/ReUP wrapper.
- `run_merge_all.py`: batch runner for weighted/FastRFS merges.
- `run_supertree_merge_all.py`: batch runner for MRP/SuperFine merges.
- `run_direct_iqtree_refit.py` and `iqtree_refit.py`: IQ-TREE branch-length refit helpers.
- `download_superfine.py`: helper for fetching the SuperFine source bundle.

Weighted split-consensus is experiment-owned code at
`experiments/04_pf2_partition_merge_refit/src/merge/run_consensus_merge.py`.
These third-party wrappers import it for fallback/shared merge behavior.
