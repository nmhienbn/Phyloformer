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
export PF_DISABLE_WANDB_LOGGER=true
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TRAIN_TREES=($(cat runs/pandit_domain_adaptation/stage7_pf2_adapt_realism_v3_raw_200k_data/args/train_trees.args))
TRAIN_ALNS=($(cat runs/pandit_domain_adaptation/stage7_pf2_adapt_realism_v3_raw_200k_data/args/train_alns.args))
VAL_TREES=($(cat runs/pandit_domain_adaptation/stage7_pf2_adapt_realism_v3_raw_200k_data/args/val_trees.args))
VAL_ALNS=($(cat runs/pandit_domain_adaptation/stage7_pf2_adapt_realism_v3_raw_200k_data/args/val_alns.args))

taskset -c 96-127 python third_party/phyloformer2/train.py finetune models/phyloformer2/pf2.tch \
  --batch-size 1 \
  --base-batch-size 4 \
  --epochs 5 \
  --warmup 0.005 \
  --learning-rate 1e-6 \
  --unambiguous-order \
  --project PF2_PANDIT_ADAPT_200K \
  --project-root runs \
  --log-every 50 \
  --validate-every 1000 \
  --early-stop-patience -1 \
  --cache-root runs/pandit_domain_adaptation/stage7_pf2_adapt_realism_v3_raw_200k_data/cache \
  --train-trees "${TRAIN_TREES[@]}" \
  --train-alns "${TRAIN_ALNS[@]}" \
  --val-trees "${VAL_TREES[@]}" \
  --val-alns "${VAL_ALNS[@]}"
