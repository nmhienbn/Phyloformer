Làm theo runbook này để bám sát paper nhất có thể bằng code hiện tại trong repo.

**1) Chuẩn bị môi trường**
1. Cài deps:
```bash
pip install -r requirements.txt
```
2. Dùng IQTree2 (repo có sẵn `bin/bin_linux/iqtree_2.2.0` / `bin/bin_macos/iqtree_2.2.0`).
3. Xác nhận luồng train chỉ đọc dữ liệu có sẵn: `train_distributed.py:215`, `train_distributed.py:218`, `phyloformer/data.py:70`.

**2) Sinh dữ liệu LG+GC giống paper (PFBase)**
Paper dùng cây/alignment 50 taxa, length 500, train ~170,616 và val ~17,016.

1. Tạo cây:
```bash
mkdir -p data/paper_pfbase/train/trees data/paper_pfbase/val/trees
python simulate_trees.py --ntips 50 --ntrees 170616 --output data/paper_pfbase/train/trees --type birth-death
python simulate_trees.py --ntips 50 --ntrees 17016  --output data/paper_pfbase/val/trees   --type birth-death
```
2. Sinh MSA LG+GC (không cho duplicate, để gần paper):
```bash
mkdir -p data/paper_pfbase/train/msas data/paper_pfbase/val/msas
python alisim.py --outdir data/paper_pfbase/train/msas --substitution LG --gamma GC --length 500 --iqtree ./bin/bin_linux/iqtree_2.2.0 --max-attempts 20000 data/paper_pfbase/train/trees
python alisim.py --outdir data/paper_pfbase/val/msas   --substitution LG --gamma GC --length 500 --iqtree ./bin/bin_linux/iqtree_2.2.0 --max-attempts 20000z data/paper_pfbase/val/trees
```
3. Kiểm tra số cặp match theo stem (`.nwk` ↔ `.fa`) như yêu cầu ở `README.md:207`.

**3) Train PFBase (tham số theo paper)**
Paper: batch=4, LR=1e-3, warmup=3000, check-val mỗi 3000 step, early stop patience=5 lần check, kiến trúc 6 blocks / 64 dim / 4 heads / dropout 0.

```bash
python train_distributed.py \
  --train-trees data/paper_pfbase/train/trees \
  --train-alignments data/paper_pfbase/train/msas \
  --val-trees data/paper_pfbase/val/trees \
  --val-alignments data/paper_pfbase/val/msas \
  --nb-blocks 6 \
  --embed-dim 64 \
  --nb-heads 4 \
  --dropout 0.0 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --warmup-steps 3000 \
  --nb-epochs 30 \
  --check-val-every 3000 \
  --no-improvement-stop 5 \
  --output-dir runs/paper_pfbase
```

Ghi chú quan trọng:
1. Để ra đúng target schedule ~213.2k steps như paper, cần đúng quy mô data và chạy 6 GPU A100 (paper).  
2. Nếu chạy SLURM multi-GPU, script tự đọc env (`train_distributed.py:413`).

**4) Fine-tune PF (bản trong paper)**
Paper fine-tune thêm từ PFBase với LR=1e-4 và loss MRE.

- Command fine-tune (khung):
```bash
python train_distributed.py \
  --train-trees data/paper_pf/train/trees \
  --train-alignments data/paper_pf/train/msas \
  --val-trees data/paper_pf/val/trees \
  --val-alignments data/paper_pf/val/msas \
  --base-model runs/paper_pfbase/checkpoints_.../last.ckpt \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --warmup-steps 3000 \
  --check-val-every 3000 \
  --no-improvement-stop 5 \
  --output-dir runs/paper_pf
```

- Lưu ý: `train_distributed.py` hiện hardcode train loss là MAE (`criterion = torch.nn.L1Loss()` ở `train_distributed.py:428`), nên **chưa đúng 100% paper cho bước PF (MRE)**.  
Nếu bạn muốn, mình có thể sửa script thêm cờ `--loss {mae,mre}` để chạy đúng hệt paper ở lượt tiếp theo.

## Chạy không cần SLURM (mới)

`train_distributed.py` giờ hỗ trợ:
- `--config <json>` để nạp cấu hình
- `--devices`, `--num-nodes`, `--strategy`, `--accelerator` để chạy multi-GPU mà không cần biến `SLURM_*`

Ví dụ config mẫu đã tạo sẵn:

`configs/paper_pfbase_train_6gpu.json`

Chạy:

```bash
python train_distributed.py --config configs/paper_pfbase_train_6gpu.json
```

Bạn vẫn có thể override từng tham số từ CLI:

```bash
python train_distributed.py \
  --config configs/paper_pfbase_train_6gpu.json \
  --run-name PFBASE_RETRY \
  --output-dir runs/paper_pfbase_retry
```

Có, chạy được không cần SLURM.

`train_distributed.py` của bạn giờ đã hỗ trợ multi-GPU trực tiếp qua Lightning DDP bằng các flag `--devices/--strategy/--num-nodes`, nên chỉ cần chạy trên 1 node có nhiều GPU.

Chạy luôn 6 GPU:
```bash
conda activate phylo
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_distributed.py --config configs/paper_pfbase_train_4gpu.json


CUDA_VISIBLE_DEVICES=1,2,3,4 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
python train_distributed.py --config configs/paper_pfbase_train_4gpu.json

```

Hoặc không dùng config:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python train_distributed.py \
  --train-trees data/paper_pfbase/train/trees \
  --train-alignments data/paper_pfbase/train/msas \
  --val-trees data/paper_pfbase/val/trees \
  --val-alignments data/paper_pfbase/val/msas \
  --accelerator gpu \
  --devices 6 \
  --num-nodes 1 \
  --strategy ddp \
  --ignore-slurm-env
```

Cách kiểm tra có thật sự chạy 6 GPU:
1. Log phải có `devices: 6` và `strategy: 'ddp'` ở phần `INIT TRAINER WITH ARGS`.
2. `nvidia-smi` sẽ thấy nhiều process Python (thường 6 process train).

# tạo cache
```
python - <<'PY'
from pathlib import Path
from phyloformer.data import precompute_distance_cache

def make_pairs(tree_dir, aln_dir):
    trees = [p for p in Path(tree_dir).iterdir() if p.suffix in {".nwk", ".newick"}]
    alns = {p.stem: p for p in Path(aln_dir).iterdir() if p.suffix in {".fa", ".fasta"}}
    pairs, missing = [], 0
    for t in trees:
        a = alns.get(t.stem)
        if a is None:
            missing += 1
            continue
        pairs.append((str(t), str(a)))
    print(f"{tree_dir}: pairs={len(pairs)}, missing={missing}")
    return pairs

train_pairs = make_pairs("data/paper_pfbase/train/trees", "data/paper_pfbase/train/msas")
val_pairs = make_pairs("data/paper_pfbase/val/trees", "data/paper_pfbase/val/msas")

precompute_distance_cache(train_pairs, "data/paper_pfbase/distance_cache/train", overwrite=False)
precompute_distance_cache(val_pairs, "data/paper_pfbase/distance_cache/val", overwrite=False)
print("cache done")
PY
```

```
# chạy 1 lần để tạo cache
python train_distributed.py \
  --config configs/paper_pfbase_train_4gpu.json \
  --distance-cache-dir data/paper_pfbase/distance_cache \
  --alignment-cache-dir data/paper_pfbase/alignment_cache \
  --precompute-distance-cache \
  --precompute-alignment-cache

# train thật (không precompute nữa)
python train_distributed.py \
  --config configs/paper_pfbase_train_4gpu.json \
  --distance-cache-dir data/paper_pfbase/distance_cache \
  --alignment-cache-dir data/paper_pfbase/alignment_cache
```