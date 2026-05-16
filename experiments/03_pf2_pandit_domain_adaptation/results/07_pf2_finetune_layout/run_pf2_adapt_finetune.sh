#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PF_DATALOADER_WORKERS="${PF_DATALOADER_WORKERS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PF_DISABLE_WANDB_LOGGER=true
export WANDB_MODE=disabled

TRAIN_TREES=($(cat runs/pandit_domain_adaptation/stage7_pf2_adapt_realism_v3_raw_20k_data_max170/args/train_trees.args))
TRAIN_ALNS=($(cat runs/pandit_domain_adaptation/stage7_pf2_adapt_realism_v3_raw_20k_data_max170/args/train_alns.args))
VAL_TREES=($(cat runs/pandit_domain_adaptation/stage7_pf2_adapt_realism_v3_raw_20k_data_max170/args/val_trees.args))
VAL_ALNS=($(cat runs/pandit_domain_adaptation/stage7_pf2_adapt_realism_v3_raw_20k_data_max170/args/val_alns.args))

taskset -c 102-105 python PF2/train.py finetune PF2/pretrained/pf2.tch \
  --batch-size 1 \
  --base-batch-size 1 \
  --epochs 5 \
  --warmup 0.005 \
  --learning-rate 1e-6 \
  --unambiguous-order \
  --project PF2_PANDIT_ADAPT \
  --project-root runs \
  --log-every 50 \
  --validate-every 500 \
  --early-stop-patience -1 \
  --cache-root runs/pandit_domain_adaptation/stage7_pf2_adapt_realism_v3_raw_20k_data_max170/cache \
  --train-trees "${TRAIN_TREES[@]}" \
  --train-alns "${TRAIN_ALNS[@]}" \
  --val-trees "${VAL_TREES[@]}" \
  --val-alns "${VAL_ALNS[@]}"
