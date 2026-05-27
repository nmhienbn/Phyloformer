# Thí nghiệm 04: PF2 partition - merge - refit

Doc này chỉ giữ pipeline được mô tả trong chương 3: chia alignment dài thành block, chạy PF2 trên từng block, gộp cây block, rồi refit branch length bằng IQ-TREE trên alignment gốc.

Các phần audit dữ liệu, benchmark nháp, biến thể consensus cũ và script refit riêng đã được bỏ khỏi thí nghiệm này.

## Yêu cầu môi trường

- PF2 source: [`third_party/phyloformer2`](../../../third_party/phyloformer2)
- PF2 checkpoint: [`models/phyloformer2/pf2.tch`](../../../models/phyloformer2/pf2.tch)
- `torch`, `numpy`, `biopython`, `scikit-learn`, `tqdm`
- IQ-TREE, ví dụ [`bin/bin_linux/iqtree_2.2.0`](../../../bin/bin_linux/iqtree_2.2.0)
- FastRFS, ví dụ [`bin/bin_linux/FastRFS`](../../../bin/bin_linux/FastRFS)
- `phylocompare` nếu cần benchmark topology

## Code chính

- Chia block theo window-site/window-rate: [`../src/partition/run_window_pf2.py`](../src/partition/run_window_pf2.py)
- Chạy window-site/window-rate cho cả thư mục test: [`../src/partition/run_window_pf2_testset.py`](../src/partition/run_window_pf2_testset.py)
- Chia block SoftBioBlock: [`../src/partition/run_softbioblock_pf2.py`](../src/partition/run_softbioblock_pf2.py)
- Chạy SoftBioBlock cho cả thư mục test: [`../src/partition/run_softbioblock_pf2_testset.py`](../src/partition/run_softbioblock_pf2_testset.py)
- Runner chung cho các thư mục test: [`../src/partition/testset_runner.py`](../src/partition/testset_runner.py)
- Mô phỏng big alignment dài: [`../src/simulation/simulate_bigaln.py`](../src/simulation/simulate_bigaln.py)

Tool wrappers gọi binary/phần mềm ngoài nằm ở [`../../../third_party/tools/partition_merge`](../../../third_party/tools/partition_merge):

- Gộp weighted split consensus: `run_consensus_merge.py`
- Gộp FastRFS: `run_fastrfs_merge.py`
- Chạy toàn bộ partition x merge: `run_merge_all.py`
- Refit IQ-TREE dùng chung: `iqtree_refit.py`

## Input

Mỗi benchmark cần một thư mục alignment FASTA và một thư mục true tree tương ứng. Ví dụ:

```bash
ALIGN_DIR=data/sample/alignments
TRUE_TREE_DIR=data/sample/trees
RUN_ROOT=runs/benchmarks/sample
```

Các bộ dữ liệu đang dùng:

| Bộ | `ALIGN_DIR` | `TRUE_TREE_DIR` | `RUN_ROOT` | Ghi chú |
|---|---|---|---|---|
| CHERRY over-2GB | `data/cherry_test_data/alignments_over2gb` | `data/cherry_test_data/trees` | `runs/benchmarks/cherry_over2gb` | 1500 alignment, `alignments_over2gb` là symlink tới full CHERRY |
| PANDIT over-1GB | `data/benchmarks/pandit_aa_zenodo/alignments_over1gb` | `data/benchmarks/pandit_aa_zenodo/trees` | `runs/benchmarks/pandit_over1gb` | 340 alignment, `alignments_over1gb` là symlink tới full PANDIT |
| BigAln 30k | `data/bigaln_benchmark/30k` | `data/bigaln_benchmark/30k/trees` | `runs/benchmarks/bigaln_16gb/30k` | alignment nằm ở `rep_*/big.fa`; dùng thêm `--include-glob "*/big.fa"` |
| BigAln 60k | `data/bigaln_benchmark/60k` | `data/bigaln_benchmark/60k/trees` | `runs/benchmarks/bigaln_16gb/60k` | alignment nằm ở `rep_*/big.fa`; dùng thêm `--include-glob "*/big.fa"` |
| BigAln 100k | `data/bigaln_benchmark/100k` | `data/bigaln_benchmark/100k/trees` | `runs/benchmarks/bigaln_16gb/100k` | alignment nằm ở `rep_*/big.fa`; dùng thêm `--include-glob "*/big.fa"` |

Lưu ý: không có thư mục `data/cherry_over2gb`. Với CHERRY/PANDIT over-size, các alignment là symlink nên kiểm tra số lượng bằng `find ... \( -type f -o -type l \)`, không dùng mỗi `-type f`.

Với dữ liệu mô phỏng big alignment, sinh từ tập cây đầu vào:

```bash
python experiments/04_pf2_partition_merge_refit/src/simulation/simulate_bigaln.py \
  --trees data/bigaln_seed_trees \
  --outdir runs/benchmarks/bigaln_16gb/30k \
  --target-sites 30000 \
  --iqtree bin/bin_linux/iqtree_2.2.0 \
  --processes 4
```

Các replicate sinh ra dùng `big.fa` làm alignment và `true.nwk` làm cây thật.
Khi chạy các wrapper `*_testset.py` trên thư mục replicate này, thêm `--include-glob "*/big.fa"`.

## Bước 1: partition và chạy PF2 block

Window-site:

```bash
python experiments/04_pf2_partition_merge_refit/src/partition/run_window_pf2_testset.py \
  "$ALIGN_DIR" \
  "$RUN_ROOT/partition_window_pos" \
  --sort-by position \
  --checkpoint models/phyloformer2/pf2.tch \
  --gpu 0 \
  --cpu-threads 20 \
  --vram-gb 80 \
  --overwrite
```

Window-rate:

```bash
python experiments/04_pf2_partition_merge_refit/src/partition/run_window_pf2_testset.py \
  "$ALIGN_DIR" \
  "$RUN_ROOT/partition_window_rate" \
  --sort-by rate \
  --checkpoint models/phyloformer2/pf2.tch \
  --gpu 0 \
  --cpu-threads 20 \
  --vram-gb 80 \
  --overwrite
```

SoftBioBlock:

```bash
python experiments/04_pf2_partition_merge_refit/src/partition/run_softbioblock_pf2_testset.py \
  "$ALIGN_DIR" \
  "$RUN_ROOT/partition_softbioblock" \
  --checkpoint models/phyloformer2/pf2.tch \
  --gpu 7 \
  --cpu-threads 20 \
  --vram-gb 80 \
  --overwrite
```

Mỗi case sẽ có:

- `block_manifest.csv`: danh sách block và trọng số dùng khi merge
- `blocks/*.fa`: alignment block đã sinh
- `pf2_blocks/*.nwk`: cây PF2 của từng block
- `metadata/pipeline.json`: lưu alignment gốc để bước refit tự tìm lại

## Bước 2: merge và refit branch length

Chạy toàn bộ 3 partition method x 2 merge method:

```bash
python third_party/tools/partition_merge/run_merge_all.py \
  --base "$RUN_ROOT" \
  --partitions window_pos window_rate softbioblock \
  --merges weighted fastrfs \
  --fit-branch-lengths \
  --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
  --iqtree-model LG+G4 \
  --iqtree-threads 2 \
  --workers 20 \
  --skip-existing
```

Kết quả của mỗi tổ hợp nằm trong:

```text
$RUN_ROOT/merge_<partition>_<merge>/final_trees/*.nwk
```

Ví dụ:

```text
$RUN_ROOT/merge_softbioblock_weighted/final_trees/*.nwk
$RUN_ROOT/merge_window_rate_fastrfs/final_trees/*.nwk
```

## Ghi chú về IQ-TREE model

Nếu muốn dùng một model cố định cho tất cả case, dùng `--iqtree-model LG+G4` hoặc `--iqtree-model LG+G8`. Với dữ liệu mô phỏng bằng `simulate_bigaln.py`, generator dùng `LG+G8`, nên refit có thể chạy:

```bash
python third_party/tools/partition_merge/run_merge_all.py \
  --base runs/benchmarks/bigaln_16gb/30k \
  --partitions window_pos window_rate softbioblock \
  --merges weighted fastrfs \
  --fit-branch-lengths \
  --iqtree-model LG+G8
```

Nếu muốn để IQ-TREE chọn model/gamma cho từng case trên topology đã merge, chạy trực tiếp:

```bash
bin/bin_linux/iqtree_2.2.0 \
  -s <alignment.fa> \
  -te <merged_topology.nwk> \
  -m MFP \
  -nt 4 \
  -pre <output_prefix> \
  -redo
```

Sau đó dùng `<output_prefix>.treefile` làm cây cuối để benchmark. Cách này chậm hơn, nhưng phù hợp nếu cần kiểm tra việc chọn `G4`, `G8`, hoặc model thay thế khác theo từng alignment.

## Baseline PF2 direct + IQ-TREE refit

Phần này dùng để kiểm tra riêng baseline `PF2_direct` nhưng refit lại branch length bằng IQ-TREE trên full alignment. Topology vẫn là cây PF2 direct; IQ-TREE chỉ nhận topology qua `-te` và tối ưu branch length theo alignment gốc.

Không cần chạy lại PF2 direct nếu dùng các kết quả đã có trong project:

| Dataset | Alignment dir | PF2 direct topology dir | Reference trees | Số cây |
|---|---|---|---|---:|
| PANDIT full | `data/benchmarks/pandit_aa_zenodo/alignments` | `runs/pf2_direct/pandit` | `data/benchmarks/pandit_aa_zenodo/trees` | 6165 |
| PANDIT over-1GB | `data/benchmarks/pandit_aa_zenodo/alignments_over1gb` | `runs/benchmarks/pandit_over1gb/pf2_direct` | `data/benchmarks/pandit_aa_zenodo/trees` | 340 |
| PANDIT over-2GB | `data/benchmarks/pandit_aa_zenodo/alignments_over2gb` | `runs/benchmarks/pandit_over2gb/pf2_direct` | `data/benchmarks/pandit_aa_zenodo/trees` | 133 |
| CHERRY over-2GB | `data/cherry_test_data/alignments_over2gb` | `runs/benchmarks/cherry_over2gb/pf2_direct` | `data/cherry_test_data/trees` | 1500 |

Code batch refit cho PF2 direct:

```text
third_party/tools/partition_merge/run_direct_iqtree_refit.py
```

### PANDIT over-1GB direct refit

```bash
ALIGN_DIR=data/benchmarks/pandit_aa_zenodo/alignments_over1gb
TRUE_TREE_DIR=data/benchmarks/pandit_aa_zenodo/trees
RUN_ROOT=runs/benchmarks/pandit_over1gb
TOPO_DIR="$RUN_ROOT/pf2_direct"
REFIT_DIR="$RUN_ROOT/pf2_direct_iqtree_refit/trees"

python third_party/tools/partition_merge/run_direct_iqtree_refit.py \
  "$ALIGN_DIR" \
  "$TOPO_DIR" \
  "$REFIT_DIR" \
  --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
  --iqtree-model LG+G4 \
  --iqtree-threads 2 \
  --workers 40 \
  --skip-existing
```

### CHERRY over-2GB direct refit

```bash
ALIGN_DIR=data/cherry_test_data/alignments_over2gb
TRUE_TREE_DIR=data/cherry_test_data/trees
RUN_ROOT=runs/benchmarks/cherry_over2gb
TOPO_DIR="$RUN_ROOT/pf2_direct"
REFIT_DIR="$RUN_ROOT/pf2_direct_iqtree_refit/trees"

python third_party/tools/partition_merge/run_direct_iqtree_refit.py \
  "$ALIGN_DIR" \
  "$TOPO_DIR" \
  "$REFIT_DIR" \
  --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
  --iqtree-model LG+G4 \
  --iqtree-threads 2 \
  --workers 40 \
  --skip-existing
```

Kết quả refit nằm ở:

```text
$RUN_ROOT/pf2_direct_iqtree_refit/trees/*.nwk
```

Benchmark topology cho baseline direct và direct+IQ-TREE-refit:

```bash
python third_party/tools/evaluation/run_tree_dir_benchmarks.py \
  --reference direct_refit="$TRUE_TREE_DIR" \
  --pred PF2_direct\|direct_refit="$TOPO_DIR" \
  --pred PF2_direct_IQTree_refit\|direct_refit="$RUN_ROOT/pf2_direct_iqtree_refit/trees" \
  --outdir "$RUN_ROOT/benchmark_results_direct_refit" \
  --bin-dir bin/bin_linux \
  --workers 40 \
  --threads 2 \
  --impute-missing-branch-lengths 1e-6
```

Vẽ boxplot:

```bash
python third_party/tools/plots/plot_topology_boxplots.py \
  --cmp-topo PF2_direct\|direct_refit="$RUN_ROOT/benchmark_results_direct_refit/cmp/PF2_direct__direct_refit_topo.csv" \
  --cmp-topo PF2_direct_IQTree_refit\|direct_refit="$RUN_ROOT/benchmark_results_direct_refit/cmp/PF2_direct_IQTree_refit__direct_refit_topo.csv" \
  --outdir "$RUN_ROOT/figures_direct_refit"
```

## Bước 3: benchmark topology

So sánh các thư mục `final_trees` với cây thật:

```bash
python third_party/tools/evaluation/run_tree_dir_benchmarks.py \
  --reference cherry_over2gb="$TRUE_TREE_DIR" \
  --pred weighted_window_site\|cherry_over2gb="$RUN_ROOT/merge_window_pos_weighted/final_trees" \
  --pred weighted_window_rate\|cherry_over2gb="$RUN_ROOT/merge_window_rate_weighted/final_trees" \
  --pred weighted_softbioblock\|cherry_over2gb="$RUN_ROOT/merge_softbioblock_weighted/final_trees" \
  --pred fastrfs_window_site\|cherry_over2gb="$RUN_ROOT/merge_window_pos_fastrfs/final_trees" \
  --pred fastrfs_window_rate\|cherry_over2gb="$RUN_ROOT/merge_window_rate_fastrfs/final_trees" \
  --pred fastrfs_softbioblock\|cherry_over2gb="$RUN_ROOT/merge_softbioblock_fastrfs/final_trees" \
  --outdir "$RUN_ROOT/benchmark_results" \
  --bin-dir bin/bin_linux \
  --workers 20 \
  --threads 2 \
  --impute-missing-branch-lengths 1e-6
```

Vẽ boxplot:

```bash
python third_party/tools/plots/plot_topology_boxplots.py \
  --cmp-topo weighted_window_site\|cherry_over2gb="$RUN_ROOT/benchmark_results/cmp/weighted_window_site__cherry_over2gb_topo.csv" \
  --cmp-topo weighted_window_rate\|cherry_over2gb="$RUN_ROOT/benchmark_results/cmp/weighted_window_rate__cherry_over2gb_topo.csv" \
  --cmp-topo weighted_softbioblock\|cherry_over2gb="$RUN_ROOT/benchmark_results/cmp/weighted_softbioblock__cherry_over2gb_topo.csv" \
  --cmp-topo fastrfs_window_site\|cherry_over2gb="$RUN_ROOT/benchmark_results/cmp/fastrfs_window_site__cherry_over2gb_topo.csv" \
  --cmp-topo fastrfs_window_rate\|cherry_over2gb="$RUN_ROOT/benchmark_results/cmp/fastrfs_window_rate__cherry_over2gb_topo.csv" \
  --cmp-topo fastrfs_softbioblock\|cherry_over2gb="$RUN_ROOT/benchmark_results/cmp/fastrfs_softbioblock__cherry_over2gb_topo.csv" \
  --outdir experiments/04_pf2_partition_merge_refit/results/cherry_over2gb_partition_merge
```

## Kết quả cần push

Code và doc:

```bash
git add experiments/04_pf2_partition_merge_refit/src
git add experiments/04_pf2_partition_merge_refit/docs/partition_merge_refit_c3.md
```

Kết quả đã dùng trong luận văn:

```bash
git add experiments/04_pf2_partition_merge_refit/results/cherry_over2gb_partition_merge
git add experiments/04_pf2_partition_merge_refit/results/pandit_over1gb_partition_merge
```

Chỉ push thêm `bigaln_16gb` nếu thư mục đó có bảng/hình đã được đưa vào luận văn:

```bash
git add experiments/04_pf2_partition_merge_refit/results/bigaln_16gb
```
