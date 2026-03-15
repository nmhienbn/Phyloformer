Làm theo runbook này để chạy `PF2_MAE` sát paper PF2 nhất có thể bằng code hiện có trong repo.

Ghi chú về độ chắc chắn:
1. Môi trường hiện tại không có `pdftotext`, nên mình không trích được nguyên văn từ [PhyloFormer 2.pdf](/raid/home/hiennguyen/Phyloformer/PhyloFormer 2.pdf).
2. Các điểm dưới đây được xác nhận từ code PF2:
   - `PF2_MAE` tương ứng với cờ `--l1-loss` trong [PF2/train.py](/raid/home/hiennguyen/Phyloformer/PF2/train.py#L1119).
   - `pf2_base.tch` là model nền, trained trên cây 50 taxa với `LG+G8+indels` theo [PF2/README.md](/raid/home/hiennguyen/Phyloformer/PF2/README.md#L46).
   - Kiến trúc mặc định của PF2 train là `6 blocks / 64 embed / 4 heads / pair_dim 256` theo [PF2/train.py](/raid/home/hiennguyen/Phyloformer/PF2/train.py#L1095).
3. Vì vậy runbook này bám sát code PF2 và phần mô tả model/data đã ghi trong repo.

**1) Chuẩn bị môi trường**
1. Tạo env riêng cho PF2:
```bash
cd /raid/home/hiennguyen/Phyloformer/PF2
conda create -n pf2 python=3.10 -y
conda activate pf2
pip install -r requirements.txt
pip install lightning wandb
```
2. Nếu muốn train, dùng luôn repo hiện tại:
```bash
cd /raid/home/hiennguyen/Phyloformer
```
3. Kiểm tra CLI:
```bash
python PF2/train.py --help
python PF2/infer.py --help
```

**2) Sinh dữ liệu giống PF2 base**
Theo [PF2/README.md](/raid/home/hiennguyen/Phyloformer/PF2/README.md#L46), `pf2_base` dùng:
1. cây `50 taxa`
2. alignment `LG+G8+indels`
3. chiều dài alignment `500`

Tạo cây:
```bash
mkdir -p data/pf2_mae/train/trees data/pf2_mae/val/trees
python src/simulate_trees.py --ntips 50 --ntrees 170616 --output data/pf2_mae/train/trees --type birth-death
python src/simulate_trees.py --ntips 50 --ntrees 17016  --output data/pf2_mae/val/trees   --type birth-death
```

Sinh alignment `LG+G8+indels`:
```bash
mkdir -p data/pf2_mae/train/msas data/pf2_mae/val/msas
python PF2/scripts/alisim.py data/pf2_mae/train/trees --outdir data/pf2_mae/train/msas --substitution LG --gamma G8 --length 500 --indels --processes 1 --max-attempts 20000
python PF2/scripts/alisim.py data/pf2_mae/val/trees   --outdir data/pf2_mae/val/msas   --substitution LG --gamma G8 --length 500 --indels --processes 1 --max-attempts 20000
```

Lưu ý:
1. `--indels` trong script PF2 sẽ trim alignment về đúng `length=500`.
2. Script PF2 mặc định gọi binary `iqtree2`, nên binary này phải nằm trong `PATH`. Nếu chưa có, bạn có thể symlink `bin/bin_linux/iqtree_2.2.0` thành `iqtree2`.

Ví dụ:
```bash
mkdir -p ~/bin
ln -sf /raid/home/hiennguyen/Phyloformer/bin/bin_linux/iqtree_2.2.0 ~/bin/iqtree2
export PATH="$HOME/bin:$PATH"
```

**3) Tạo cache cho PF2**
`PF2/train.py` yêu cầu `--cache-root` và dataset `NJDataset` sẽ đọc cache từ đó.

```bash
mkdir -p data/pf2_mae/cache/train data/pf2_mae/cache/val

PYTHONPATH=src python - <<'PY'
from pathlib import Path
from phyloformer.data import precompute_distance_cache

def make_pairs(tree_dir, aln_dir):
    trees = sorted(p for p in Path(tree_dir).iterdir() if p.suffix in {".nwk", ".newick"})
    alns = {p.stem: p for p in Path(aln_dir).iterdir() if p.suffix in {".fa", ".fasta"}}
    pairs = []
    missing = []
    for t in trees:
        a = alns.get(t.stem)
        if a is None:
            missing.append(t.stem)
            continue
        pairs.append((str(t), str(a)))
    print(tree_dir, "pairs=", len(pairs), "missing=", len(missing))
    if missing:
        print("first_missing=", missing[:10])
    return pairs

train_pairs = make_pairs("data/pf2_mae/train/trees", "data/pf2_mae/train/msas")
val_pairs = make_pairs("data/pf2_mae/val/trees", "data/pf2_mae/val/msas")

precompute_distance_cache(train_pairs, "data/pf2_mae/cache/train", overwrite=False)
precompute_distance_cache(val_pairs, "data/pf2_mae/cache/val", overwrite=False)
print("cache done")
PY
```

**4) Train PF2_MAE từ đầu**
`PF2_MAE` ở đây là PF2 chạy distance-matrix regression với `MAE`, tức:
1. dùng command `train`
2. bật `--l1-loss`
3. không bật `--evopf`
4. không bật `--unambiguous-order`
5. không bật `--branch-lengths`

Command khởi chạy:
```bash
conda activate pf2
cd /raid/home/hiennguyen/Phyloformer

python PF2/train.py train \
  --l1-loss \
  --project PF2_EXPERIMENTS \
  --project-root runs \
  --batch-size 4 \
  --epochs 30 \
  --warmup 3000 \
  --learning-rate 1e-4 \
  --validate-every 3000 \
  --log-every 50 \
  --train-trees data/pf2_mae/train/trees \
  --train-alns data/pf2_mae/train/msas \
  --val-trees data/pf2_mae/val/trees \
  --val-alns data/pf2_mae/val/msas \
  --cache-root data/pf2_mae/cache
```

Giải thích mapping:
1. `--l1-loss` bật nhánh MAE trong [PF2/train.py](/raid/home/hiennguyen/Phyloformer/PF2/train.py#L651).
2. `mae_train`, `mse_train`, `mre_train` sẽ được log trong train loop ở [PF2/train.py](/raid/home/hiennguyen/Phyloformer/PF2/train.py#L877).
3. Kiến trúc mặc định đã là `embed_dim=64`, `pair_dim=256`, `n_blocks=6`, `n_heads=4`.

**5) Multi-GPU / mixed precision**
PF2 hiện trong repo này không có CLI DDP kiểu `train_distributed.py` của Phyloformer cũ. Nó dựa vào `idr_torch` cho môi trường Jean-Zay trong [PF2/train.py](/raid/home/hiennguyen/Phyloformer/PF2/train.py#L372).

Vì vậy:
1. nếu bạn chạy local workstation thường, cứ bắt đầu bằng 1 GPU
2. mixed precision có thể bật bằng `--mixed-precision`
3. deepspeed/flexattention chỉ nên bật nếu bạn đã cài đúng stack

Ví dụ 1 GPU:
```bash
CUDA_VISIBLE_DEVICES=7 python PF2/train.py train \
  --l1-loss \
  --mixed-precision \
  --project PF2_EXPERIMENTS \
  --project-root runs \
  --batch-size 4 \
  --epochs 30 \
  --warmup 3000 \
  --learning-rate 1e-4 \
  --validate-every 3000 \
  --log-every 50 \
  --train-trees data/pf2_mae/train/trees \
  --train-alns data/pf2_mae/train/msas \
  --val-trees data/pf2_mae/val/trees \
  --val-alns data/pf2_mae/val/msas \
  --cache-root data/pf2_mae/cache
```

**6) Fine-tune từ `pf2_base.tch` nếu cần**
Nếu bạn không muốn train từ đầu mà muốn bắt đầu từ pretrained base:

```bash
python PF2/train.py finetune PF2/pretrained/pf2_base.tch \
  --l1-loss \
  --project PF2_EXPERIMENTS \
  --project-root runs \
  --batch-size 4 \
  --epochs 10 \
  --warmup 3000 \
  --learning-rate 1e-4 \
  --validate-every 3000 \
  --log-every 50 \
  --train-trees data/pf2_mae/train/trees \
  --train-alns data/pf2_mae/train/msas \
  --val-trees data/pf2_mae/val/trees \
  --val-alns data/pf2_mae/val/msas \
  --cache-root data/pf2_mae/cache
```

**7) Inference PF2_MAE -> distance matrices -> FastME**
Sau khi train, checkpoint tốt nhất thường nằm ở:
`runs/PF2_EXPERIMENTS/<run_name>/checkpoints/best_val_loss.ckpt`

Tạo ma trận khoảng cách:
```bash
CKPT="runs/PF2_EXPERIMENTS/<run_name>/checkpoints/best_val_loss.ckpt"
OUT="runs/pf2_mae/eval_final_test_set"
rm -rf "$OUT"
mkdir -p "$OUT/trees"

CUDA_VISIBLE_DEVICES=7 python PF2/infer.py \
  --mode dm \
  data/final_test_set/alignments \
  "$CKPT" \
  "$OUT/mats"
```

Suy cây + so với true tree:
```bash
python tools/run_fastme_compare.py \
  --bin-dir ./bin/bin_linux \
  --mats-dir "$OUT/mats" \
  --trees-dir "$OUT/trees" \
  --label "final_test_set_pf2_mae" \
  --threads 1 \
  --true-trees data/final_test_set/trees \
  --cmp-out "$OUT/cmp_pf2_mae" \
  --method-name PF2_MAE+FastME
```

**8) Ghi chú quan trọng về code hiện tại**
Mình đã vá một lỗi runtime trong [PF2/train.py](/raid/home/hiennguyen/Phyloformer/PF2/train.py): nhánh `--l1-loss` trước đó có thể dùng biến `mseloss` trước khi gán trong train loop, nên `PF2_MAE` có nguy cơ crash ngay khi train.

Nếu muốn, bước tiếp theo hợp lý là mình tạo luôn:
1. một shell script `tools/run_pf2_mae.sh` để launch train
2. một markdown `run_pf2_all_test_sets.md` để benchmark PF2_MAE trên toàn bộ test sets giống file benchmark hiện tại
