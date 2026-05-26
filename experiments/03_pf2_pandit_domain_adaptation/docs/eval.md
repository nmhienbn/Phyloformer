## 1. In ra thông số dữ liệu synthetic :))

```bash
python experiments/03_pf2_pandit_domain_adaptation/src/simulation/evaluate_synthetic_dataset.py \
  runs/pandit_domain_adaptation/stage5_synthetic_pandit_like_realism_v3_raw_200k/manifest.tsv \
  --empirical-gap-summary experiments/03_pf2_pandit_domain_adaptation/results/04_gap_tree_calibration/pandit_train_gap_stats.bucket_summary.tsv \
  --empirical-model-counts experiments/03_pf2_pandit_domain_adaptation/results/03_iqtree_model_calibration/pandit_train_model_counts.tsv \
  --empirical-gamma-quantiles experiments/03_pf2_pandit_domain_adaptation/results/03_iqtree_model_calibration/pandit_train_gamma_alpha_quantiles.tsv \
  --empirical-tree-quantiles experiments/03_pf2_pandit_domain_adaptation/results/03_iqtree_model_calibration/pandit_train_tree_quantiles.tsv \
  --outdir experiments/03_pf2_pandit_domain_adaptation/results/06_stage5_synthetic_v3
```

## 2. Profiling peak VRAM => Tìm batch size

VÌ PF2 ấy, chỉ chạy trên taxa bằng nhau ==> khác length gộp chung batch được nhưng cần thêm gap.
Đây là test xem gộp được batch mấy (1 batch max 24)

```bash
CUDA_VISIBLE_DEVICES=7 python experiments/03_pf2_pandit_domain_adaptation/src/training/profile_pf2_batch_vram.py \
  runs/pandit_domain_adaptation/stage5_synthetic_pandit_like_realism_v3_raw_200k/manifest.tsv \
  --checkpoint models/phyloformer2/pf2.tch \
  --out-tsv runs/pandit_domain_adaptation/stage5_synthetic_pandit_like_realism_v3_raw_200k/pf2_batch_vram_profile.tsv \
  --base-batch-sizes 2 4 8 \
  --max-groups 24 \
  --cpu-threads 4
```