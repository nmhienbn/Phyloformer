# Train PF2 Theo Paper

PhyloFormer có 2 loại loss: MAE Loss và BayesNJ Loss có kiến trúc và cách train khác hẳn nhau. Do đó, muốn replicate, ta cần huấn luyện:

- `PF2_MAE`: EvoPF encoder + MAE trên pairwise distances.
- `PF2`: EvoPF + BayesNJ topology/branch-length objective.
- `PF2` multi-size: fine-tune từ `pf2_base.tch` trên nhiều số taxa.

Chú ý kiểm soát số GPU sử dụng bằng biến môi trường `CUDA_VISIBLE_DEVICES`.

## Chuẩn Bị môi trường

```bash
pip install -r third_party/phyloformer2/requirements.txt
pip install -r third_party/phyloformer2/requirements.train.txt
```

## Sinh dữ liệu mô phỏng
Tham khảo [train_paper_phyloformer_1.md](docs/train_paper_phyloformer_1.md)
Chú ý rằng cách sinh dữ liệu y như cũ nhưng chú ý số lượng theo [Figure 4](paper/PhyloFormer 2.pdf)

## Train PF2_MAE

Theo paper, `PF2_MAE` dùng lại bộ PF1 `BD, LG+GC`, 50 taxa, length 500.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python third_party/phyloformer2/train.py train \
  --evopf \
  --l1-loss \
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
  --validate-every 1000 \
  --train-trees data/paper_pfbase/train/trees \
  --train-alns data/paper_pfbase/train/msas \
  --val-trees data/paper_pfbase/val/trees \
  --val-alns data/paper_pfbase/val/msas \
  --cache-root data/paper_pfbase/cache
```

Chạy bằng file config JSON:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python third_party/tools/inference/run_pf2_from_json.py \
  configs/pf2_evopf_mae_4gpu.json
```

## Train PF2 Base

### Mô phỏng dữ liệu
Thông tin về bộ dữ liệu mô phỏng:
- `7,966,499` train trees
- `10,000` validation trees
- `50` taxa
- `LG + G8 + indels`
- alignment length `500`

Sinh dữ liệu sử dùng PF1 tree simulator và PF2 `alisim.py`:

```bash
mkdir -p data/pf2_paper_main/train/trees data/pf2_paper_main/val/trees

python third_party/phyloformer1/simulate_trees.py \
  --ntips 50 \
  --ntrees 8000000 \
  --output data/pf2_paper_main/train/trees \
  --type birth-death

python third_party/phyloformer1/simulate_trees.py \
  --ntips 50 \
  --ntrees 10000 \
  --output data/pf2_paper_main/val/trees \
  --type birth-death


mkdir -p data/pf2_paper_main/train/msas data/pf2_paper_main/val/msas

python third_party/phyloformer2/scripts/alisim.py \
  data/pf2_paper_main/train/trees \
  --outdir data/pf2_paper_main/train/msas \
  --substitution LG \
  --gamma G8 \
  --length 500 \
  --indels \
  --processes 1 \
  --max-attempts 200

python third_party/phyloformer2/scripts/alisim.py \
  data/pf2_paper_main/val/trees \
  --outdir data/pf2_paper_main/val/msas \
  --substitution LG \
  --gamma G8 \
  --length 500 \
  --indels \
  --processes 1 \
  --max-attempts 200
```

Huấn luyện:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python third_party/phyloformer2/train.py train \
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
  --validate-every 1000 \
  --train-trees data/pf2_paper_main/train/trees \
  --train-alns data/pf2_paper_main/train/msas \
  --val-trees data/pf2_paper_main/val/trees \
  --val-alns data/pf2_paper_main/val/msas \
  --cache-root data/pf2_paper_main/cache
```

Checkpoint pretrained tương ứng trong repo:

```text
models/phyloformer2/pf2_base.tch
```

## Fine-Tune PF2 Multi-Size

Paper fine-tune từ `PF2 base` trên sizes `10, 20, ..., 170`, mỗi size có `20,000` train và `1,000` val.

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

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python third_party/phyloformer2/train.py finetune \
  models/phyloformer2/pf2_base.tch \
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