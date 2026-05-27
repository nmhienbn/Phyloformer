# SuperFine/MRP benchmark

## Tool cần có

- IQ-TREE: `bin/bin_linux/iqtree_2.2.0`
- MRP: `third_party/tools/partition_merge/run_mrp_merge.py`
- SuperFine: `third_party/tools/partition_merge/run_superfine_merge.py`
- Batch: `third_party/tools/partition_merge/run_supertree_merge_all.py`

## Chạy merge

Ví dụ với `cherry_over2gb`:

```bash
RUN_ROOT=runs/benchmarks/cherry_over2gb

python third_party/tools/partition_merge/run_supertree_merge_all.py \
  --base "$RUN_ROOT" \
  --partitions window_pos window_rate softbioblock \
  --merges mrp superfine \
  --superfine-cmd "$SUPERFINE_CMD" \
  --fit-branch-lengths \
  --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
  --iqtree-model LG+G4 \
  --iqtree-threads 2 \
  --workers 10 \
  --skip-existing
```

## Benchmark

```bash
ALIGN_DIR=data/cherry_test_data/alignments_over2gb
TRUE_TREE_DIR=data/cherry_test_data/trees
RUN_ROOT=runs/benchmarks/cherry_over2gb

python third_party/tools/evaluation/run_tree_dir_benchmarks.py \
  --reference cherry_over2gb="$TRUE_TREE_DIR" \
  --pred mrp_window_site\|cherry_over2gb="$RUN_ROOT/merge_window_pos_mrp/final_trees" \
  --pred mrp_window_rate\|cherry_over2gb="$RUN_ROOT/merge_window_rate_mrp/final_trees" \
  --pred mrp_softbioblock\|cherry_over2gb="$RUN_ROOT/merge_softbioblock_mrp/final_trees" \
  --pred superfine_window_site\|cherry_over2gb="$RUN_ROOT/merge_window_pos_superfine/final_trees" \
  --pred superfine_window_rate\|cherry_over2gb="$RUN_ROOT/merge_window_rate_superfine/final_trees" \
  --pred superfine_softbioblock\|cherry_over2gb="$RUN_ROOT/merge_softbioblock_superfine/final_trees" \
  --outdir "$RUN_ROOT/benchmark_results_supertree" \
  --bin-dir bin/bin_linux \
  --workers 10 \
  --threads 2 \
  --impute-missing-branch-lengths 1e-6
```

Vẽ boxplot vào thư mục mới:

```bash
python third_party/tools/plots/plot_topology_boxplots.py \
  --cmp-topo mrp_window_site\|cherry_over2gb="$RUN_ROOT/benchmark_results_supertree/cmp/mrp_window_site__cherry_over2gb_topo.csv" \
  --cmp-topo mrp_window_rate\|cherry_over2gb="$RUN_ROOT/benchmark_results_supertree/cmp/mrp_window_rate__cherry_over2gb_topo.csv" \
  --cmp-topo mrp_softbioblock\|cherry_over2gb="$RUN_ROOT/benchmark_results_supertree/cmp/mrp_softbioblock__cherry_over2gb_topo.csv" \
  --cmp-topo superfine_window_site\|cherry_over2gb="$RUN_ROOT/benchmark_results_supertree/cmp/superfine_window_site__cherry_over2gb_topo.csv" \
  --cmp-topo superfine_window_rate\|cherry_over2gb="$RUN_ROOT/benchmark_results_supertree/cmp/superfine_window_rate__cherry_over2gb_topo.csv" \
  --cmp-topo superfine_softbioblock\|cherry_over2gb="$RUN_ROOT/benchmark_results_supertree/cmp/superfine_softbioblock__cherry_over2gb_topo.csv" \
  --outdir experiments/04_pf2_partition_merge_refit/results/cherry_over2gb_superfine_mrp
```
