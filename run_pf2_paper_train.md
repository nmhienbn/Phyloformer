Làm theo runbook này để train `PF2_MAE` và `PF2` bám sát [PhyloFormer 2.pdf](/raid/home/hiennguyen/Phyloformer/PhyloFormer%202.pdf) nhất có thể bằng code hiện có trong repo.

**Mục tiêu**
- `PF2_MAE`: evoPF encoder + MAE loss trên pairwise distances, đúng nhánh ablation trong paper.
- `PF2`: mô hình chính của paper, dùng BayesNJ loss để học posterior trên phylogeny, sau đó fine-tune multi-size.

**Nguồn paper đã map vào repo**
- `Table S.1`:
  - `PF2_MAE`: Random init, `(128|256)`, `BD, LG+GC`, `MAE`, batch `16`, epochs `30`, LR `5e-4`, warmup `1000`, selected step `51000`.
  - `PF2`: Random init, `(128|256)`, `BD, LG+G8`, `BayesNJ`, batch `16`, epochs `30`, LR `1e-4`, warmup `1000`, selected step `86000`.
  - multi-size fine-tune of `PF2`: start from `PF2`, `BD, LG+G8, multi`, `BayesNJ`, batch `1-40`, epochs `30`, LR `1e-6`, warmup `0.5%`, selected step `8000`.
- `Appendix A.3.1`:
  - main `PF2` dataset: `7,966,499` train trees + `10,000` val trees, `50` taxa, `LG + 8-category discrete Gamma + indels`.
  - multi-size fine-tune dataset: sizes `10..170` by step `10`, each size has `20,000` train + `1,000` val.
  - `PF2_MAE` uses the original `LG+GC` 200k dataset from Nesterenko et al. (2025).
- `Section 3` and checkpoint hparams:
  - `PF2` paper model uses evoPF with `12` blocks, `embed_dim=128`, `pair_dim=256`, `n_heads=4`.
  - In repo, điều này tương ứng với `--evopf -b 12 -e 128 -P 256 -H 4`.
  - BayesNJ path tương ứng với `--unambiguous-order --branch-lengths`.
  - `PF2_MAE` tương ứng với `--evopf --l1-loss`.

**Giới hạn cần nói rõ**
- Paper train trên `4 H100 GPUs` ở chế độ distributed data parallel.
- `PF2/train.py` trong repo chỉ tự bật DDP khi có môi trường `idr_torch` kiểu Jean-Zay. Nếu bạn chạy máy local bình thường thì command vẫn chạy, nhưng không phải đúng hệt hạ tầng paper.
- Paper không ghi đầy đủ mọi hyperparameter phụ như `weight_decay`, `validate_every`, `mixed_precision`. Trong runbook này, các giá trị không thấy trong paper được giữ theo default của code hoặc ghi chú rõ.

**1) Chuẩn bị môi trường**
```bash
cd /raid/home/hiennguyen/Phyloformer
source /raid/home/hiennguyen/miniconda3/etc/profile.d/conda.sh
conda activate pf2
```

Nếu vừa sửa env xong, kiểm tra nhanh:
```bash
python - <<'PY'
import torch, scipy, pandas
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
from torch.nn.attention.flex_attention import flex_attention
print('torch', torch.__version__)
print('scipy', scipy.__version__)
print('pandas', pandas.__version__)
print('deepspeed4science OK', DS4Sci_EvoformerAttention)
print('flex_attention OK', flex_attention)
PY
```

**2) Mapping paper sang CLI của `PF2/train.py`**
- `PF2_MAE`:
  - paper meaning: evoPF + regression on pairwise distances with MAE
  - CLI mapping: `train --evopf --l1-loss`
- `PF2`:
  - paper meaning: evoPF + BayesNJ topological/branch-length posterior loss
  - CLI mapping: `train --evopf --unambiguous-order --branch-lengths`
- Kiến trúc paper:
  - `-b 12 -e 128 -P 256 -H 4`
- Warmup:
  - `-w 1000` means exactly `1000` warmup steps because code interprets values `> 1` as step count.
  - `-w 0.005` means `0.5%` warmup because code interprets values `<= 1` as fraction of total training steps.

**3) Dữ liệu paper cho `PF2_MAE`**
Paper không dùng bộ `LG+G8+indels` cho `PF2_MAE`. Nó dùng lại bộ `BD, LG+GC` 200k của paper PF1.

Nếu bạn đã có sẵn bộ `LGGC` của PF1, chỉ cần trỏ vào đúng thư mục train/val.
Nếu chưa có, dùng runbook cũ của PF1 để sinh bộ `LG+GC` đó trước.

Giả sử cấu trúc như sau:
```text
data/pf2_paper_mae/
  train/trees
  train/msas
  val/trees
  val/msas
  cache/train
  cache/val
```

**4) Train `PF2_MAE` đúng theo paper**
Command bám `Table S.1`:
```bash
python PF2/train.py train \
  --evopf \
  --l1-loss \
  --wandb-sync \
  --embed-dim 128 \
  --pair-dim 256 \
  --n-blocks 12 \
  --n-heads 4 \
  --batch-size 16 \
  --epochs 30 \
  --warmup 1000 \
  --learning-rate 5e-4 \
  --project PF2_PAPER \
  --project-root runs \
  --log-every 50 \
  --train-trees data/paper_pfbase/train/trees \
  --train-alns data/paper_pfbase/train/msas \
  --val-trees data/paper_pfbase/val/trees \
  --val-alns data/paper_pfbase/val/msas \
  --cache-root data/paper_pfbase/cache
```

Ghi chú:
- Paper selected step là `51000`. Nếu bạn muốn chọn checkpoint đúng tinh thần paper, lấy checkpoint validation tốt nhất gần step đó nhất.
- `validate_every` không được nêu trong paper. Nếu muốn checkpoint dày hơn để match `selected step`, bạn có thể thêm `--validate-every 1000`.
- Không bật `--unambiguous-order` và không bật `--branch-lengths` cho nhánh này.

**5) Dữ liệu paper cho `PF2` pretraining**
Theo `Appendix A.3.1`:
- `7,966,499` train trees
- `10,000` validation trees
- `50` taxa
- `LG` substitution model
- `8-category discrete Gamma`
- thêm indels bằng `alisim`
- indel insertion/deletion rate đều là `2e-3`
- indel length theo default `zipfian exponent 1.7`, max size `100`

Giả sử dữ liệu đã có cấu trúc:
```text
data/pf2_paper_main/
  train/trees
  train/msas
  val/trees
  val/msas
  cache/train
  cache/val
```

**6) Train `PF2` đúng theo paper**
Command bám `Table S.1` và kiến trúc paper:
```bash
python PF2/train.py train \
  --evopf \
  --unambiguous-order \
  --branch-lengths \
  --embed-dim 128 \
  --pair-dim 256 \
  --n-blocks 12 \
  --n-heads 4 \
  --batch-size 16 \
  --epochs 30 \
  --warmup 1000 \
  --learning-rate 1e-4 \
  --project PF2_PAPER \
  --project-root runs \
  --log-every 50 \
  --train-trees data/pf2_paper_main/train/trees \
  --train-alns data/pf2_paper_main/train/msas \
  --val-trees data/pf2_paper_main/val/trees \
  --val-alns data/pf2_paper_main/val/msas \
  --cache-root data/pf2_paper_main/cache
```

Ghi chú:
- Paper selected step là `86000`.
- `PF2/pretrained/pf2_base.tch` trong repo tương ứng với model nền này.
- Nếu mục tiêu là reproduce inference chứ không phải retrain từ đầu 7.9M mẫu, bạn có thể dùng ngay `PF2/pretrained/pf2_base.tch`.

**7) Fine-tune multi-size cho `PF2` đúng theo paper**
Theo `Appendix A.3.1` và `A.4`:
- sizes: `10, 20, ..., 170`
- mỗi size: `20,000` train + `1,000` val
- dynamic batch size, range thực tế khoảng `1..40`
- schedule được define theo batch size tại `50` taxa
- maximum batch size bằng batch size tại `50` taxa
- target LR `1e-6`
- warmup `0.5%`
- selected step `8000`

Trong repo, multi-size fine-tune được kích hoạt bằng cách truyền nhiều thư mục `--train-trees/--train-alns/--val-trees/--val-alns` và thêm `--base-batch-size`.

Ví dụ bằng bash arrays:
```bash
TRAIN_TREES=()
TRAIN_ALNS=()
VAL_TREES=()
VAL_ALNS=()

for N in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170; do
  TRAIN_TREES+=("data/pf2_paper_multisize/train_${N}/trees")
  TRAIN_ALNS+=("data/pf2_paper_multisize/train_${N}/msas")
  VAL_TREES+=("data/pf2_paper_multisize/val_${N}/trees")
  VAL_ALNS+=("data/pf2_paper_multisize/val_${N}/msas")
done

python PF2/train.py finetune PF2/pretrained/pf2_base.tch \
  --batch-size 40 \
  --base-batch-size 40 \
  --epochs 30 \
  --warmup 0.005 \
  --learning-rate 1e-6 \
  --project PF2_PAPER \
  --project-root runs \
  --log-every 50 \
  --validate-every 500 \
  --cache-root data/pf2_paper_multisize/cache \
  --train-trees "${TRAIN_TREES[@]}" \
  --train-alns "${TRAIN_ALNS[@]}" \
  --val-trees "${VAL_TREES[@]}" \
  --val-alns "${VAL_ALNS[@]}"
```

Ghi chú quan trọng:
- Trong multi-size mode, `PF2/train.py` không dùng cache pairwise distances theo nhánh `NJDataset`; nó dùng `MultisizeUnambiguousMergeOrderDataset` trực tiếp từ trees/MSAs.
- `--base-batch-size 40` là cách gần nhất để map “batch size 1-40” trong paper sang code hiện tại.
- `--batch-size 40` được parser yêu cầu nhưng trong multi-size path, scheduler sẽ dựa vào `--base-batch-size`.
- `--validate-every 500` không được paper nêu; mình thêm để bạn có checkpoint đủ dày quanh selected step `8000`. Nếu muốn đúng tuyệt đối theo paper, đây là một hyperparameter chưa được công bố.

**8) Cách chọn checkpoint theo paper**
Paper báo `selected step`, không chỉ `best final epoch`.

Quy ước practical:
- `PF2_MAE`: checkpoint tốt nhất quanh `step 51000`
- `PF2` main pretrain: checkpoint tốt nhất quanh `step 86000`
- `PF2` multi-size fine-tune: checkpoint tốt nhất quanh `step 8000`

Repo lưu:
```text
runs/PF2_PAPER/<run_name>/checkpoints/
  best_val_loss.ckpt
  latest.ckpt
  checkpoints.tar
```

Nếu bạn muốn khớp sát paper nhất, nên:
1. bật validation định kỳ
2. xem các checkpoint trong `checkpoints.tar` hoặc `latest.ckpt` log history
3. lấy checkpoint gần `selected step` tương ứng nếu nó cũng có validation loss tốt

**9) Nếu bạn không retrain full-size từ đầu**
Có 2 shortcut hợp lý:

- Reproduce `PF2` inference từ pretrained:
```bash
CKPT=PF2/pretrained/pf2.tch
```

- Reproduce multi-size fine-tuned `PF2` gần paper nhất từ pretrained:
```bash
CKPT=PF2/pretrained/pf2.tch
```

- Reproduce base 50-taxa model gần paper nhất từ pretrained:
```bash
CKPT=PF2/pretrained/pf2_base.tch
```

**10) Những gì trong paper nhưng repo hiện không đóng gói sẵn**
- Pipeline sinh chính xác bộ `7.9M LG+G8+indels` của paper không có sẵn thành một lệnh duy nhất trong repo gốc này.
- `PF2_MAE` paper dùng bộ `LG+GC` 200k từ paper PF1, không phải bộ `LG+G8+indels`.
- DDP `4 H100` của paper không tự tái hiện trên máy local thường vì `PF2/train.py` gắn với `idr_torch` cho cluster Jean-Zay.

**11) Tóm tắt ngắn**
- `PF2_MAE` paper:
```bash
python PF2/train.py train --evopf --l1-loss -e 128 -P 256 -b 12 -H 4 -s 16 -E 30 -w 1000 -r 5e-4 ...
```
- `PF2` paper pretrain:
```bash
python PF2/train.py train --evopf --unambiguous-order --branch-lengths -e 128 -P 256 -b 12 -H 4 -s 16 -E 30 -w 1000 -r 1e-4 ...
```
- `PF2` paper multi-size fine-tune:
```bash
python PF2/train.py finetune PF2/pretrained/pf2_base.tch --base-batch-size 40 -E 30 -w 0.005 -r 1e-6 ...
```
