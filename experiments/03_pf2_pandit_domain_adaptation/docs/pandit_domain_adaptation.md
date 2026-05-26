## 1. Sinh synthetic PANDIT-like 200k

```bash
python experiments/03_pf2_pandit_domain_adaptation/src/simulation/run_pandit_like_synthetic.py \
  --profile-tsv experiments/03_pf2_pandit_domain_adaptation/results/03_iqtree_model_calibration/pandit_train_iqtree_per_msa_joined.tsv \
  --outdir runs/pandit_domain_adaptation/stage5_synthetic_pandit_like_realism_v3_raw_200k \
  --n-msas 200000 \
  --seed 2042 \
  --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
  --tree-simulator third_party/phyloformer1/simulate_trees.py \
  --memory-threshold-mib 70000 \
  --cpu-threads 4 \
  --max-attempts 50
```

## 2. Tạo layout multisize

Sử dụng `--base-batch-size` theo profiling bên trên.

```bash
python experiments/03_pf2_pandit_domain_adaptation/src/training/prepare_pf2_multisize_training_layout.py \
  runs/pandit_domain_adaptation/stage5_synthetic_pandit_like_realism_v3_raw_200k/manifest.tsv \
  runs/pandit_domain_adaptation/stage7_pf2_adapt_realism_v3_raw_200k_data \
  --val-frac 0.01 \
  --seed 2042 \
  --base-batch-size 4 \
  --validate-every 1000 \
  --project PF2_PANDIT_ADAPT_200K \
  --cpu-set 96-127
```

## 3. Fine-tune PF2

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash runs/pandit_domain_adaptation/stage7_pf2_adapt_realism_v3_raw_200k_data/run_pf2_adapt_finetune.sh
```

Kết quả checkpoint nằm trong:

```text
runs/PF2_PANDIT_ADAPT_200K/<run_id>/checkpoints/latest.ckpt
runs/PF2_PANDIT_ADAPT_200K/<run_id>/checkpoints/best_val_loss.ckpt
```

Resume:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 96-127 python third_party/phyloformer2/train.py resume \
  runs/PF2_PANDIT_ADAPT_200K/20260519-232717-sangria-lasting-tiffany/checkpoints/latest.ckpt \
  --epochs 10 \
  --early-stop-patience -1
```