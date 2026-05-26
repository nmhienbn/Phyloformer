# PF1 Quartet Loss Fine-tune

Các hàm loss nằm trong:

```text
experiments/02_pf1_quartet_loss/src/losses/
```

| Biến thể | File code | Ý nghĩa |
| --- | --- | --- |
| `close` / `close-2` | [quartet_close.py](third_party/phyloformer1/phyloformer/losses/quartet_close.py) | ép hai tổng chéo không thuộc topology thật tiến lại gần nhau |
| `push` | [quartet_push.py](third_party/phyloformer1/phyloformer/losses/quartet_push.py) | buộc tổng của topology thật nhỏ hơn hai tổng chéo theo margin |
| `combined` | [quartet_combined.py](third_party/phyloformer1/phyloformer/losses/quartet_combined.py) | kết hợp `close` và `push` |

Config fine-tune tương ứng:

```text
experiments/02_pf1_quartet_loss/configs/finetune_pfbase_quartet_close_2gpu.json
experiments/02_pf1_quartet_loss/configs/finetune_pfbase_quartet_close_6gpu.json
experiments/02_pf1_quartet_loss/configs/finetune_pfbase_quartet_push_2gpu.json
experiments/02_pf1_quartet_loss/configs/finetune_pfbase_quartet_push_6gpu.json
experiments/02_pf1_quartet_loss/configs/finetune_pfbase_quartet_combined_2gpu.json
```

## Chạy fine-tune

Chạy từng biến thể bằng config:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python third_party/phyloformer1/train_distributed.py \
  --config experiments/02_pf1_quartet_loss/configs/finetune_pfbase_quartet_close_2gpu.json
```

Các config 2 GPU đang dùng cho thí nghiệm:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python third_party/phyloformer1/train_distributed.py \
  --config experiments/02_pf1_quartet_loss/configs/finetune_pfbase_quartet_close_2gpu.json

CUDA_VISIBLE_DEVICES=0,1 \
python third_party/phyloformer1/train_distributed.py \
  --config experiments/02_pf1_quartet_loss/configs/finetune_pfbase_quartet_push_2gpu.json

CUDA_VISIBLE_DEVICES=0,1 \
python third_party/phyloformer1/train_distributed.py \
  --config experiments/02_pf1_quartet_loss/configs/finetune_pfbase_quartet_combined_2gpu.json
```

Nếu chạy bản 6 GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python third_party/phyloformer1/train_distributed.py \
  --config experiments/02_pf1_quartet_loss/configs/finetune_pfbase_quartet_close_6gpu.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python third_party/phyloformer1/train_distributed.py \
  --config experiments/02_pf1_quartet_loss/configs/finetune_pfbase_quartet_push_6gpu.json
```

Có thể override checkpoint base hoặc output dir từ CLI:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python third_party/phyloformer1/train_distributed.py \
  --config experiments/02_pf1_quartet_loss/configs/finetune_pfbase_quartet_push_2gpu.json \
  --base-model runs/paper_pfbase/checkpoints_<pretrain_identifier>/last.ckpt \
  --output-dir runs/pfbase_quartet_push_retry \
  --run-name PFBASE_FINETUNE_QPUSH_RETRY
```

## Vẽ so sánh final test

```bash
python third_party/tools/plots/make_quartet_loss_plot.py
```
