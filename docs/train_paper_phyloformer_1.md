# Replicate huấn luyện PhyloFormer 1 theo paper

Runbook này dùng layout refactor mới: source PF1 nằm trong `third_party/phyloformer1`, còn config thí nghiệm nằm trong `configs/`.

## Chuẩn bị môi trường

```bash
pip install -r third_party/phyloformer1/requirements.txt
```

IQ-TREE2 dùng binary có sẵn trong repo:

```text
bin/bin_linux/iqtree_2.2.0
bin/bin_macos/iqtree_2.2.0
```

## Sinh dữ liệu PFBase

Paper dùng alignment `50` taxa, length `500`, train khoảng `170,616` mẫu và validation khoảng `17,016` mẫu.

Sinh cây:

```bash
mkdir -p data/paper_pfbase/train/trees data/paper_pfbase/val/trees

python third_party/phyloformer1/simulate_trees.py \
  --ntips 50 \
  --ntrees 170616 \
  --output data/paper_pfbase/train/trees \
  --type birth-death

python third_party/phyloformer1/simulate_trees.py \
  --ntips 50 \
  --ntrees 17016 \
  --output data/paper_pfbase/val/trees \
  --type birth-death
```

Sinh MSA LG+GC:

```bash
mkdir -p data/paper_pfbase/train/msas data/paper_pfbase/val/msas

python third_party/phyloformer1/alisim.py \
  --outdir data/paper_pfbase/train/msas \
  --substitution LG \
  --gamma GC \
  --length 500 \
  --iqtree ./bin/bin_linux/iqtree_2.2.0 \
  --max-attempts 20000 \
  data/paper_pfbase/train/trees

python third_party/phyloformer1/alisim.py \
  --outdir data/paper_pfbase/val/msas \
  --substitution LG \
  --gamma GC \
  --length 500 \
  --iqtree ./bin/bin_linux/iqtree_2.2.0 \
  --max-attempts 20000 \
  data/paper_pfbase/val/trees
```

Tên cây và alignment phải khớp nhau. Ví dụ `data/paper_pfbase/train/trees/0_50_tips.nwk` cần có alignment tương ứng `data/paper_pfbase/train/msas/0_50_tips.fa`.

## Train PFBase bằng MAE

Paper PFBase dùng MAE, batch size `4`, LR `1e-3`, warmup `3000`, validation mỗi `3000` step, early stop patience `5`, kiến trúc `6` blocks / `64` dim / `4` heads / dropout `0`.

CLI trực tiếp:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python third_party/phyloformer1/train_distributed.py \
  --train-trees data/paper_pfbase/train/trees \
  --train-alignments data/paper_pfbase/train/msas \
  --val-trees data/paper_pfbase/val/trees \
  --val-alignments data/paper_pfbase/val/msas \
  --nb-blocks 6 \
  --embed-dim 64 \
  --nb-heads 4 \
  --n-seqs 50 \
  --dropout 0.0 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --warmup-steps 3000 \
  --nb-epochs 20 \
  --check-val-every 3000 \
  --no-improvement-stop 5 \
  --loss mae \
  --accelerator gpu \
  --devices 6 \
  --num-nodes 1 \
  --strategy ddp \
  --ignore-slurm-env \
  --output-dir runs/paper_pfbase \
  --run-name PFBASE_PAPER_PRETRAIN_6GPU
```

Chạy bằng config:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python third_party/phyloformer1/train_distributed.py \
  --config experiments/01_pf1_vram/configs/paper_pfbase_pretrain_6gpu_exact.json
```

Có thể override từng tham số của config từ CLI:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python third_party/phyloformer1/train_distributed.py \
  --config experiments/01_pf1_vram/configs/paper_pfbase_pretrain_6gpu_exact.json \
  --devices 4 \
  --run-name PFBASE_RETRY_4GPU \
  --output-dir runs/paper_pfbase_retry_4gpu
```

## Fine-tune PF bằng MRE

Paper fine-tune từ PFBase bằng MRE. CLI hiện tại đã hỗ trợ `--loss mae` và `--loss mre`; không còn cần sửa hardcode loss.

CLI trực tiếp:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python third_party/phyloformer1/train_distributed.py \
  --train-trees data/paper_pfbase/train/trees \
  --train-alignments data/paper_pfbase/train/msas \
  --val-trees data/paper_pfbase/val/trees \
  --val-alignments data/paper_pfbase/val/msas \
  --base-model runs/paper_pfbase/checkpoints_<pretrain_identifier>/last.ckpt \
  --nb-blocks 6 \
  --embed-dim 64 \
  --nb-heads 4 \
  --n-seqs 50 \
  --dropout 0.0 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --warmup-steps 3000 \
  --nb-epochs 4 \
  --check-val-every 3000 \
  --no-improvement-stop 5 \
  --loss mre \
  --accelerator gpu \
  --devices 6 \
  --num-nodes 1 \
  --strategy ddp \
  --ignore-slurm-env \
  --output-dir runs/paper_pf \
  --run-name PF_PAPER_FINETUNE_MRE_6GPU
```

Chạy bằng config:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python third_party/phyloformer1/train_distributed.py \
  --config experiments/01_pf1_vram/configs/paper_pfbase_finetune_mre_6gpu_exact.json \
  --base-model runs/paper_pfbase/checkpoints_<pretrain_identifier>/last.ckpt
```
