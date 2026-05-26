# OOD embedding PCA-KDE contour

Có hai embedding space khác nhau:

- `pf2.tch`
- `pf2_pandit.ckpt`

## 1. Figure 1: fixed/size-matched vs PANDIT trong `pf2.tch`

```bash
PF2_PRETRAINED=models/phyloformer2/pf2.tch
```

Sinh synthetic PF2-native:

```bash
python experiments/03_pf2_pandit_domain_adaptation/src/ood_diagnostics/run_pf2_native_synthetic_pca.py \
  --outdir runs/pandit_domain_adaptation/stage3_pf2_native_pca_fixed \
  --n-msas 5000 \
  --seed 42 \
  --sampling pf2-native-fixed \
  --n-taxa 50 \
  --alignment-length 500 \
  --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
  --tree-simulator third_party/phyloformer1/simulate_trees.py \
  --alignment-simulator third_party/phyloformer1/alisim.py \
  --cpu-threads 4

python experiments/03_pf2_pandit_domain_adaptation/src/ood_diagnostics/run_pf2_native_synthetic_pca.py \
  --profile-tsv experiments/03_pf2_pandit_domain_adaptation/results/02_empirical_profile/pandit_train_profile.tsv \
  --outdir runs/pandit_domain_adaptation/stage3_pf2_native_pca_size_matched \
  --n-msas 5000 \
  --seed 42 \
  --sampling pf2-native-size-matched \
  --iqtree-bin bin/bin_linux/iqtree_2.2.0 \
  --tree-simulator third_party/phyloformer1/simulate_trees.py \
  --alignment-simulator third_party/phyloformer1/alisim.py \
  --cpu-threads 4
```

## 2. Figure 2: PANDIT-like 200k vs PANDIT trong `pf2_pandit.ckpt`

```bash
PF2_PANDIT=models/trained/pf2_pandit.ckpt
SYN200K=runs/pandit_domain_adaptation/stage5_synthetic_pandit_like_realism_v3_raw_200k/manifest.tsv
SYN200K_SAMPLE=runs/pandit_domain_adaptation/ood_200k_inputs/synthetic_pandit_like_200k_sample20k.tsv
```

Sampling 20k từ 200k:

```bash
mkdir -p runs/pandit_domain_adaptation/ood_200k_inputs

conda run --no-capture-output -n pf2 python - <<'PY'
import csv, random
from pathlib import Path

src = Path("runs/pandit_domain_adaptation/stage5_synthetic_pandit_like_realism_v3_raw_200k/manifest.tsv")
out = Path("runs/pandit_domain_adaptation/ood_200k_inputs/synthetic_pandit_like_200k_sample20k.tsv")
rng = random.Random(2042)

with src.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

rows = rng.sample(rows, 20000)
rows.sort(key=lambda row: row["msa_id"])

with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print(out, len(rows))
PY
```

Trích embedding synthetic 20k sample bằng `pf2_pandit`:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pf2 \
python experiments/03_pf2_pandit_domain_adaptation/src/ood_diagnostics/extract_evopf_embeddings.py \
  "$SYN200K_SAMPLE" \
  "$PF2_PANDIT" \
  runs/pandit_domain_adaptation/stage7_postft_evopf_embeddings/synthetic_pandit_like_200k_sample20k \
  --input-root . \
  --device cuda \
  --cpu-threads 4
```

Trích embedding full PANDIT bằng cùng `pf2_pandit`:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pf2 \
python experiments/03_pf2_pandit_domain_adaptation/src/ood_diagnostics/extract_evopf_embeddings.py \
  experiments/03_pf2_pandit_domain_adaptation/results/01_pandit_split/pandit_all.tsv \
  "$PF2_PANDIT" \
  runs/pandit_domain_adaptation/stage7_postft_evopf_embeddings/pandit_all_pf2_pandit \
  --input-root data/zenodo_raw \
  --device cuda \
  --cpu-threads 4
```

Vẽ lại contour và copy vào paper figures:

```bash
OUT=runs/pandit_domain_adaptation/pca_contour_pf2_pandit_synthetic_200k_sample20k_vs_pandit_all

conda run --no-capture-output -n pf2 \
python experiments/03_pf2_pandit_domain_adaptation/src/ood_diagnostics/plot_embedding_pca_contours.py \
  --outdir "$OUT" \
  --simulated-pandit-like-manifest runs/pandit_domain_adaptation/stage7_postft_evopf_embeddings/synthetic_pandit_like_200k_sample20k/embedding_manifest.tsv \
  --pandit-stage7-manifest runs/pandit_domain_adaptation/stage7_postft_evopf_embeddings/pandit_all_pf2_pandit/embedding_manifest.tsv \
  --no-also-write-scatter-dirs \
  --cpu-threads 4

cp "$OUT/pca_contour_simulated_pandit_like_vs_pandit.pdf" \
  experiments/03_pf2_pandit_domain_adaptation/results/00_paper_figures/pca_contour_simulated_pandit_like_vs_pandit.pdf

cp "$OUT/pca_contour_simulated_pandit_like_vs_pandit.png" \
  experiments/03_pf2_pandit_domain_adaptation/results/00_paper_figures/pca_contour_simulated_pandit_like_vs_pandit.png
```

Kết quả lần chạy hiện tại:

```bash
Simulated PANDIT-like: 20000 embeddings, sampled from synthetic 200k
PANDIT data: 6165 embeddings
PC1 explained variance: 0.4498108327
PC2 explained variance: 0.2077771723
```
