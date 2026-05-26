# Đo VRAM thực tế của PF1/PF2 khi train và infer

Hiện tại những lúc cần đo VRAM bao gồm:

- `PF1` lúc `infer`
- `PF1` lúc `train`
- `PF2_MAE/evoPF` lúc `infer`
- `PF2_MAE/evoPF` lúc `train`
- `PF2_BayesNJ` trong `third_party/phyloformer2` lúc `infer`
- `PF2_BayesNJ` trong `third_party/phyloformer2` lúc `train`

Ý tưởng: script sẽ sinh MSA synthetic với kích thước (#sequences, sequence_length), chạy model, rồi chạy `nvidia-smi` theo PID để lấy GPU RSS thực tế. Ngoài ra script cũng tính toán của mức sử dụng GPU của pytorch với `torch.cuda.max_memory_allocated()` và `torch.cuda.max_memory_reserved()` để đối chiếu.

## Yêu cầu môi trường

- `torch`
- `matplotlib`
- `nvidia-smi`
- PF1 source: [`third_party/phyloformer1`](../../../third_party/phyloformer1)
- PF2 requirements: [`third_party/phyloformer2/requirements.txt`](../../../third_party/phyloformer2/requirements.txt)

## Hướng dẫn chạy

Các profile hỗ trợ

- `pf1-infer`
- `pf1-train`
- `pf2-mae-infer`
- `pf2-mae-train`
- `pf2-bayesnj-infer`
- `pf2-bayesnj-train`

```bash
python experiments/01_pf1_vram/src/measure_pf_memory.py \
  --profile <profile_name> \
  --seqs <num_taxa_seperated_by_commas> \
  --lengths <sequence_length_seperated_by_commas> \
  --batch-size <batch_size> \
  --dtype fp32 \
  --gpu 0 \
  --outdir <out_dir>
```

Ví dụ:

```bash
python experiments/01_pf1_vram/src/measure_pf_memory.py \
  --profile pf1-infer \
  --seqs 10,50,100,150,200,250,300,350,400,450,500,550,600 \
  --lengths 100,250,500,1000,2000,3000,4000,5000,10000 \
  --batch-size 1 \
  --dtype fp32 \
  --gpu 0 \
  --outdir experiments/01_pf1_vram/results/pf1_infer_fp32

python experiments/01_pf1_vram/src/measure_pf_memory.py \
  --profile pf2-bayesnj-infer \
  --seqs 10,50,100,150,200,250,300,350,400,450,500,550,600 \
  --lengths 100,250,500,1000,2000,3000,4000,5000,10000 \
  --batch-size 1 \
  --dtype fp32 \
  --gpu 0 \
  --outdir experiments/01_pf1_vram/results/pf2_bayesnj_infer_fp32
```

Mặc định sử dụng hai checkpoint để check:

- PF1 checkpoint: [`models/phyloformer1/pf_base.ckpt`](../../../models/phyloformer1/pf_base.ckpt)
- PF2 checkpoint: [`models/phyloformer2/pf2.tch`](../../../models/phyloformer2/pf2.tch)

Kết quả bao gồm `memory_grid_matrix.csv` là ma trận `số sequence` x `độ dài`
Để replicate được hình trong bài, sử dụng:

- [`experiments/01_pf1_vram/src/plot_pf1_pf2_memory_side_by_side.py`](../src/plot_pf1_pf2_memory_side_by_side.py)

Ví dụ:

```bash
python experiments/01_pf1_vram/src/plot_pf1_pf2_memory_side_by_side.py \
  --pf1-csv experiments/01_pf1_vram/results/pf1_infer_fp32/memory_grid_matrix.csv \
  --pf2-csv experiments/01_pf1_vram/results/pf2_bayesnj_infer_fp32/memory_grid_matrix.csv \
  --pf1-title "PF1 inference measured on A100" \
  --pf2-title "PF2 inference measured on A100" \
  --title "PF1 vs PF2 GPU memory scaling" \
  --copy-pf1-csv experiments/01_pf1_vram/results/pf1_vs_pf2_compare/pf1_measured_matrix.csv \
  --copy-pf2-csv experiments/01_pf1_vram/results/pf1_vs_pf2_compare/pf2_measured_matrix.csv \
  --out experiments/01_pf1_vram/results/pf1_vs_pf2_compare/pf1_vs_pf2_side_by_side.pdf
```
