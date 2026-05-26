## 1. Chia train-test từ tập PANDIT tỷ lệ 70-30

```bash
python experiments/03_pf2_pandit_domain_adaptation/src/training/create_pandit_msa_splits.py \
  --stats-csv data/zenodo_raw/msa_stats.csv \
  --outdir experiments/03_pf2_pandit_domain_adaptation/results/01_pandit_split \
  --seed 42
```

Số lượng dùng trong Chương 3:

- train: `4315`
- test: `1850`

## 2. Profiling PANDIT train
### Profiling thông tin từ MSA PANDIT train

```bash
python experiments/03_pf2_pandit_domain_adaptation/src/ood_diagnostics/profile_pandit_msa_split.py \
  experiments/03_pf2_pandit_domain_adaptation/results/01_pandit_split/pandit_train.tsv \
  --input-root data/zenodo_raw \
  --out-tsv experiments/03_pf2_pandit_domain_adaptation/results/02_empirical_profile/pandit_train_profile.tsv
```

### Profiling theo best fitted models
Chạy IQTree MFP:
```bash
SPLIT=experiments/03_pf2_pandit_domain_adaptation/results/01_pandit_split/pandit_train.tsv
OUT=runs/pandit_domain_adaptation/iqtree_mfp_pandit_train
IQTREE=bin/bin_linux/iqtree_2.2.0

mkdir -p "$OUT/work" "$OUT/logs"
tail -n +2 "$SPLIT" | while IFS=$'\t' read -r msa_id relative_path num_sequences alignment_length seq_bin len_bin stratum split; do
  aln="data/zenodo_raw/$relative_path"
  prefix="$OUT/work/$msa_id"
  log="$OUT/logs/$msa_id.iqtree.log"

  if [ -s "$prefix.iqtree" ] && [ -s "$prefix.treefile" ]; then
    continue
  fi

  taskset -c 102-105 nice -n 10 "$IQTREE" \
    -s "$aln" \
    --seqtype AA \
    -m MFP \
    -T 4 \
    --prefix "$prefix" \
    --redo \
    --quiet \
    > "$log" 2>&1
done
```

Profiling từ kết quả IQTree

```bash
python experiments/03_pf2_pandit_domain_adaptation/src/ood_diagnostics/analyze_pandit_iqtree_models.py \
  experiments/03_pf2_pandit_domain_adaptation/results/01_pandit_split/pandit_train.tsv \
  --iqtree-work-dir runs/completed/topology/iqtree_modeltxt_pandit_aa/work \
  --profile-tsv experiments/03_pf2_pandit_domain_adaptation/results/02_empirical_profile/pandit_train_profile.tsv \
  --outdir experiments/03_pf2_pandit_domain_adaptation/results/03_iqtree_model_calibration
```

### Profiling gap

```bash
python experiments/03_pf2_pandit_domain_adaptation/src/ood_diagnostics/profile_empirical_gap_stats.py \
  experiments/03_pf2_pandit_domain_adaptation/results/01_pandit_split/pandit_train.tsv \
  --input-root data/zenodo_raw \
  --outdir experiments/03_pf2_pandit_domain_adaptation/results/05_ood_pca
```

