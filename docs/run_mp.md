# Chạy MPBoot

## Cài MPBoot

```bash
tar xvzf mpboot-1.1.0-AVX.tar.gz
mkdir -p bin
cp mpboot-1.1.0-AVX/mpboot bin/mpboot
chmod +x bin/mpboot
bin/mpboot -h
```

## Chạy PANDIT/TreeBASE AA

```bash
cd /raid/home/hiennguyen/Phyloformer

run_mpboot_batch() {
  alignments=$1
  output=$2
  cpu=$3
  spr_rad=$4
  shift 4

  rm -rf "$output"
  mkdir -p "$output/trees" "$output/logs"

  for fa in "$alignments"/*.fa; do
    stem=$(basename "$fa" .fa)
    abs_fa=$(readlink -f "$fa")
    workdir="$output/work/${stem}"
    mkdir -p "$workdir"

    (
      cd "$workdir" || exit 1
      taskset -c "$cpu" nice -n 10 \
        /raid/home/hiennguyen/Phyloformer/bin/mpboot \
        -s "$abs_fa" \
        -cost e \
        -spr_rad "$spr_rad" \
        "$@" \
        > "../../logs/${stem}.mpboot.log" 2>&1

      if [ -f "$(basename "$fa").contree" ]; then
        cp "$(basename "$fa").contree" "../../trees/${stem}.nwk"
      elif [ -f "$(basename "$fa").treefile" ]; then
        cp "$(basename "$fa").treefile" "../../trees/${stem}.nwk"
      fi
    )
  done
}

run_mpboot_batch data/benchmarks/pandit_aa_zenodo/alignments runs/completed/topology/mpboot_spr3_uniform_pandit_aa 110 3
run_mpboot_batch data/benchmarks/treebase_aa_zenodo/alignments runs/completed/topology/mpboot_spr3_uniform_treebase_aa 111 3
run_mpboot_batch data/benchmarks/pandit_aa_zenodo/alignments runs/completed/topology/mpboot_spr6_uniform_pandit_aa 112 6
run_mpboot_batch data/benchmarks/treebase_aa_zenodo/alignments runs/completed/topology/mpboot_spr6_uniform_treebase_aa 113 6
```

## Nếu cần bootstrap 1000

```bash
run_mpboot_batch data/benchmarks/pandit_aa_zenodo/alignments runs/completed/topology/mpboot_spr3_uniform_pandit_aa_bb1000 110 3 -bb 1000
run_mpboot_batch data/benchmarks/treebase_aa_zenodo/alignments runs/completed/topology/mpboot_spr3_uniform_treebase_aa_bb1000 111 3 -bb 1000
run_mpboot_batch data/benchmarks/pandit_aa_zenodo/alignments runs/completed/topology/mpboot_spr6_uniform_pandit_aa_bb1000 112 6 -bb 1000
run_mpboot_batch data/benchmarks/treebase_aa_zenodo/alignments runs/completed/topology/mpboot_spr6_uniform_treebase_aa_bb1000 113 6 -bb 1000
```
