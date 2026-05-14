# Chạy ML

## Chuẩn bị dữ liệu

```bash
cd /raid/home/hiennguyen/Phyloformer

python tools/classical_methods/prepare_zenodo_phylo_benchmark.py \
  pandit \
  data/zenodo_raw/data_pandit/aa \
  data/benchmarks/pandit_aa_zenodo

python tools/classical_methods/prepare_zenodo_phylo_benchmark.py \
  treebase \
  data/zenodo_raw/data_treebase \
  data/benchmarks/treebase_aa_zenodo \
  --treebase-prefix prot_
```

## FastTree

```bash
rm -rf runs/completed/topology/fasttree_pandit_aa
mkdir -p runs/completed/topology/fasttree_pandit_aa/trees

for fa in data/benchmarks/pandit_aa_zenodo/alignments/*.fa; do
  stem=$(basename "$fa" .fa)
  taskset -c 100 nice -n 10 \
    bin/bin_linux/FastTree -lg \
    < "$fa" \
    > "runs/completed/topology/fasttree_pandit_aa/trees/${stem}.nwk"
done

rm -rf runs/completed/topology/fasttree_treebase_aa
mkdir -p runs/completed/topology/fasttree_treebase_aa/trees

for fa in data/benchmarks/treebase_aa_zenodo/alignments/*.fa; do
  stem=$(basename "$fa" .fa)
  taskset -c 101 nice -n 10 \
    bin/bin_linux/FastTree -lg \
    < "$fa" \
    > "runs/completed/topology/fasttree_treebase_aa/trees/${stem}.nwk"
done
```

## IQ-TREE

```bash
python tools/classical_methods/run_iqtree_batch_progress.py \
  data/benchmarks/pandit_aa_zenodo/alignments \
  runs/completed/topology/iqtree_pandit_aa \
  --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
  --model-dir data/benchmarks/pandit_aa_zenodo/models \
  --threads 2 \
  --cpu-set 102-105 \
  --nice 10 \
  --seqtype AA \
  --redo \
  --overwrite

python tools/classical_methods/run_iqtree_batch_progress.py \
  data/benchmarks/treebase_aa_zenodo/alignments \
  runs/completed/topology/iqtree_treebase_aa \
  --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
  --model LG+G4 \
  --threads 2 \
  --cpu-set 106-109 \
  --nice 10 \
  --seqtype AA \
  --redo \
  --overwrite
```

## IQ-TREE sử dụng model.txt cho PANDIT

```bash
python tools/classical_methods/run_iqtree_batch_progress.py \
  data/benchmarks/pandit_aa_zenodo/alignments \
  runs/completed/topology/iqtree_modeltxt_pandit_aa \
  --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
  --model-dir data/benchmarks/pandit_aa_zenodo/models \
  --threads 4 \
  --cpu-set 110-117 \
  --nice 10 \
  --seqtype AA \
  --redo \
  --overwrite
```
