SuperFine source chính thức nằm ở Dryad dataset `doi:10.5061/dryad.879st`, file `SuperFineSource.zip`, MD5 `9e06d3002e926fd5c794a771897fdb18`.

Có thể thử tải bằng script:

```bash
python third_party/tools/partition_merge/download_superfine.py --extract
```

```text
third_party/tools/superfine/SuperFineSource.zip
```


Nếu Dryad chặn tải trực tiếp trong shell, tải file bằng browser rồi giải nén. Zip này chỉ bung ra 4 tarball `.tar.gz`, cần giải nén tiếp:

```bash
mkdir -p third_party/tools/superfine/source
unzip -o third_party/tools/superfine/SuperFineSource.zip -d third_party/tools/superfine/source
mkdir -p third_party/tools/superfine/source/extracted
for f in third_party/tools/superfine/source/*.tar.gz; do
  tar -xzf "$f" -C third_party/tools/superfine/source/extracted
done
```

Entrypoint SuperFine nằm trong package `reup`, không có file tên `SuperFine.py`:

```bash
third_party/tools/superfine/source/extracted/reup-1.0/reup/scripts/runReup.py
```

Lưu ý: code SuperFine/ReUP này là Python 2 cũ. Chế độ mặc định `-r qmc` cần binary `qmc`; chế độ `-r gmrp`/`-r rmrp` cần PAUP*. Máy hiện tại chưa có `python2`, `qmc`, hoặc `paup` trong `PATH`, nên chưa chạy SuperFine thật được. Nếu có đủ dependency, set `SUPERFINE_CMD` như sau:

```bash
SUPERFINE_PYTHONPATH="third_party/tools/superfine/source/extracted/reup-1.0:third_party/tools/superfine/source/extracted/spruce-1.0:third_party/tools/superfine/source/extracted/newick_modified-1.3.1:third_party/tools/superfine/source/extracted/DendroPy-v2.4.0"
SUPERFINE_CMD="PYTHONPATH=$SUPERFINE_PYTHONPATH python2 third_party/tools/superfine/source/extracted/reup-1.0/reup/scripts/runReup.py -r qmc {source_list}"
```

Runner `run_superfine_merge.py` sẽ lấy Newick từ stdout của `runReup.py` và ghi vào `{output_tree}`. Nếu chưa có dependency legacy, dùng `MERGES="mrp"` để chạy phần MRP hiện đại bằng IQ-TREE trước.


## Lệnh chạy tất cả dataset

Set `SUPERFINE_CMD` trước nếu muốn chạy cả SuperFine. Nếu chưa có môi trường Python 2 + `qmc`/PAUP*, đổi `MERGES="mrp"` để chạy MRP trước:

```bash
SUPERFINE_PYTHONPATH="third_party/tools/superfine/source/extracted/reup-1.0:third_party/tools/superfine/source/extracted/spruce-1.0:third_party/tools/superfine/source/extracted/newick_modified-1.3.1:third_party/tools/superfine/source/extracted/DendroPy-v2.4.0"
SUPERFINE_CMD="PYTHONPATH=$SUPERFINE_PYTHONPATH python2 third_party/tools/superfine/source/extracted/reup-1.0/reup/scripts/runReup.py -r qmc {source_list}"
MERGES="mrp superfine"
PARTITIONS="window_pos window_rate softbioblock"
```

### `runs/benchmarks/pandit_over1gb`

```bash
RUN_ROOT=runs/benchmarks/pandit_over1gb
TRUE_TREE_DIR=data/benchmarks/pandit_aa_zenodo/trees
LABEL=pandit_over1gb

python third_party/tools/partition_merge/run_supertree_merge_all.py \
  --base "$RUN_ROOT" \
  --partitions $PARTITIONS \
  --merges $MERGES \
  --superfine-cmd "$SUPERFINE_CMD" \
  --fit-branch-lengths \
  --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
  --iqtree-model LG+G4 \
  --iqtree-threads 2 \
  --workers 10 \
  --skip-existing

python third_party/tools/evaluation/run_tree_dir_benchmarks.py \
  --reference "$LABEL=$TRUE_TREE_DIR" \
  --pred mrp_window_site\|"$LABEL=$RUN_ROOT/merge_window_pos_mrp/final_trees" \
  --pred mrp_window_rate\|"$LABEL=$RUN_ROOT/merge_window_rate_mrp/final_trees" \
  --pred mrp_softbioblock\|"$LABEL=$RUN_ROOT/merge_softbioblock_mrp/final_trees" \
  --pred superfine_window_site\|"$LABEL=$RUN_ROOT/merge_window_pos_superfine/final_trees" \
  --pred superfine_window_rate\|"$LABEL=$RUN_ROOT/merge_window_rate_superfine/final_trees" \
  --pred superfine_softbioblock\|"$LABEL=$RUN_ROOT/merge_softbioblock_superfine/final_trees" \
  --outdir "$RUN_ROOT/benchmark_results_supertree" \
  --bin-dir bin/bin_linux \
  --workers 10 \
  --threads 2 \
  --impute-missing-branch-lengths 1e-6

python third_party/tools/plots/plot_topology_boxplots.py \
  --cmp-topo mrp_window_site\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/mrp_window_site__${LABEL}_topo.csv" \
  --cmp-topo mrp_window_rate\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/mrp_window_rate__${LABEL}_topo.csv" \
  --cmp-topo mrp_softbioblock\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/mrp_softbioblock__${LABEL}_topo.csv" \
  --cmp-topo superfine_window_site\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/superfine_window_site__${LABEL}_topo.csv" \
  --cmp-topo superfine_window_rate\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/superfine_window_rate__${LABEL}_topo.csv" \
  --cmp-topo superfine_softbioblock\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/superfine_softbioblock__${LABEL}_topo.csv" \
  --outdir experiments/04_pf2_partition_merge_refit/results/pandit_over1gb_superfine_mrp
```

### `runs/benchmarks/cherry_over2gb`

```bash
RUN_ROOT=runs/benchmarks/cherry_over2gb
TRUE_TREE_DIR=data/cherry_test_data/trees
LABEL=cherry_over2gb

python third_party/tools/partition_merge/run_supertree_merge_all.py \
  --base "$RUN_ROOT" \
  --partitions $PARTITIONS \
  --merges $MERGES \
  --superfine-cmd "$SUPERFINE_CMD" \
  --fit-branch-lengths \
  --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
  --iqtree-model LG+G4 \
  --iqtree-threads 2 \
  --workers 10 \
  --skip-existing

python third_party/tools/evaluation/run_tree_dir_benchmarks.py \
  --reference "$LABEL=$TRUE_TREE_DIR" \
  --pred mrp_window_site\|"$LABEL=$RUN_ROOT/merge_window_pos_mrp/final_trees" \
  --pred mrp_window_rate\|"$LABEL=$RUN_ROOT/merge_window_rate_mrp/final_trees" \
  --pred mrp_softbioblock\|"$LABEL=$RUN_ROOT/merge_softbioblock_mrp/final_trees" \
  --pred superfine_window_site\|"$LABEL=$RUN_ROOT/merge_window_pos_superfine/final_trees" \
  --pred superfine_window_rate\|"$LABEL=$RUN_ROOT/merge_window_rate_superfine/final_trees" \
  --pred superfine_softbioblock\|"$LABEL=$RUN_ROOT/merge_softbioblock_superfine/final_trees" \
  --outdir "$RUN_ROOT/benchmark_results_supertree" \
  --bin-dir bin/bin_linux \
  --workers 10 \
  --threads 2 \
  --impute-missing-branch-lengths 1e-6

python third_party/tools/plots/plot_topology_boxplots.py \
  --cmp-topo mrp_window_site\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/mrp_window_site__${LABEL}_topo.csv" \
  --cmp-topo mrp_window_rate\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/mrp_window_rate__${LABEL}_topo.csv" \
  --cmp-topo mrp_softbioblock\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/mrp_softbioblock__${LABEL}_topo.csv" \
  --cmp-topo superfine_window_site\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/superfine_window_site__${LABEL}_topo.csv" \
  --cmp-topo superfine_window_rate\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/superfine_window_rate__${LABEL}_topo.csv" \
  --cmp-topo superfine_softbioblock\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/superfine_softbioblock__${LABEL}_topo.csv" \
  --outdir experiments/04_pf2_partition_merge_refit/results/cherry_over2gb_superfine_mrp
```

### `runs/benchmarks/bigaln_16gb`

Chạy cả `30k`, `60k`, `100k`. Dữ liệu này dùng model refit `LG+G8`.

```bash
for SIZE in 30k 60k 100k; do
  RUN_ROOT="runs/benchmarks/bigaln_16gb/$SIZE"
  TRUE_TREE_DIR="data/bigaln_benchmark/$SIZE/trees"
  LABEL="bigaln_16gb_${SIZE}"

  python third_party/tools/partition_merge/run_supertree_merge_all.py \
    --base "$RUN_ROOT" \
    --partitions $PARTITIONS \
    --merges $MERGES \
    --superfine-cmd "$SUPERFINE_CMD" \
    --fit-branch-lengths \
    --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
    --iqtree-model LG+G8 \
    --iqtree-threads 2 \
    --workers 10 \
    --skip-existing

  python third_party/tools/evaluation/run_tree_dir_benchmarks.py \
    --reference "$LABEL=$TRUE_TREE_DIR" \
    --pred mrp_window_site\|"$LABEL=$RUN_ROOT/merge_window_pos_mrp/final_trees" \
    --pred mrp_window_rate\|"$LABEL=$RUN_ROOT/merge_window_rate_mrp/final_trees" \
    --pred mrp_softbioblock\|"$LABEL=$RUN_ROOT/merge_softbioblock_mrp/final_trees" \
    --pred superfine_window_site\|"$LABEL=$RUN_ROOT/merge_window_pos_superfine/final_trees" \
    --pred superfine_window_rate\|"$LABEL=$RUN_ROOT/merge_window_rate_superfine/final_trees" \
    --pred superfine_softbioblock\|"$LABEL=$RUN_ROOT/merge_softbioblock_superfine/final_trees" \
    --outdir "$RUN_ROOT/benchmark_results_supertree" \
    --bin-dir bin/bin_linux \
    --workers 10 \
    --threads 2 \
    --impute-missing-branch-lengths 1e-6

  python third_party/tools/plots/plot_topology_boxplots.py \
    --cmp-topo mrp_window_site\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/mrp_window_site__${LABEL}_topo.csv" \
    --cmp-topo mrp_window_rate\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/mrp_window_rate__${LABEL}_topo.csv" \
    --cmp-topo mrp_softbioblock\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/mrp_softbioblock__${LABEL}_topo.csv" \
    --cmp-topo superfine_window_site\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/superfine_window_site__${LABEL}_topo.csv" \
    --cmp-topo superfine_window_rate\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/superfine_window_rate__${LABEL}_topo.csv" \
    --cmp-topo superfine_softbioblock\|"$LABEL=$RUN_ROOT/benchmark_results_supertree/cmp/superfine_softbioblock__${LABEL}_topo.csv" \
    --outdir "experiments/04_pf2_partition_merge_refit/results/bigaln_16gb/$SIZE/superfine_mrp"
done
```

Nếu `MERGES="mrp"` thì bỏ các dòng `--pred superfine_*` và `--cmp-topo superfine_*` trước khi benchmark/plot.
