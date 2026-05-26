# Sinh dữ liệu benchmark big alignment protein (LG+G8)

Pipeline sinh protein big alignment có true tree để benchmark phương pháp chia-gộp MSA dài với PF2. Pipeline bám theo PF1: sinh cây bằng mô hình birth-death + empirical diameter scaling từ HOGENOM/RaxML, sinh từng gene protein bằng AliSim (LG+G8, gene_len ~ lognormal từ data Fungi, alpha sample từ
phân phối HOGENOM empirical), concatenate gene cho tới khi đạt target sites.

---

## Cấu hình

```bash
N_REPS_PER_SIZE=30        # số replicate mỗi site size (= tổng số cây)
TREES_PER_NTIPS=10        # số cây mỗi giá trị ntips
NTIPS_VALUES="50 75 100"  # 3 giá trị × 10 cây = 30 cây, range [50, 100]
SITE_SIZES="30k 60k 100k"
IQTREE=bin/bin_linux/iqtree_2.2.0

# gene_len ~ lognormal(mu=6.215, sigma=0.8) clip [100, 4500]
#   → mean ≈ 557 aa, mode ≈ 368 aa  (Fungi protein data)
# model: LG+G8, alpha ~ HOGENOM empirical
# sinh gene liên tiếp tới khi tổng sites ≥ target_sites
```

## Bước 1 — Sinh cây birth-death

Dùng script PF1 (`simulate_trees.py`) với diameter scaling từ HOGENOM + RaxML empirical.
Chạy lặp qua 3 giá trị ntips (50, 75, 100), mỗi giá trị sinh 10 cây → tổng 30 cây.

```bash
mkdir -p data/bigaln_benchmark/trees

for NTIPS in 50 75 100; do
  python third_party/phyloformer1/simulate_trees.py \
    --ntrees 10 \
    --ntips $NTIPS \
    --type birth-death \
    --output data/bigaln_benchmark/trees
done
```

Mỗi cây được lưu dạng `{i}_{ntips}_tips.nwk` (ví dụ `0_50_tips.nwk`, `9_100_tips.nwk`).

## Bước 2 — Sinh big alignment cho từng replicate

Script `third_party/phyloformer1/simulate_bigaln.py` sinh gene liên tiếp bằng AliSim cho tới khi tổng sites ≥ target_sites, mỗi gene dùng gene_len riêng sample từ lognormal.

```bash
declare -A TARGET_SITES=( [30k]=30000 [60k]=60000 [100k]=100000 )

for SIZE in 30k 60k 100k; do
  python experiments/04_pf2_partition_merge_refit/src/simulation/simulate_bigaln.py \
    --trees data/bigaln_benchmark/trees \
    --outdir data/bigaln_benchmark/$SIZE \
    --target-sites ${TARGET_SITES[$SIZE]} \
    --iqtree $IQTREE \
    --processes 8
done
```

Mỗi replicate tạo ra:
- `big.fa` — big alignment concatenate, taxa × total_sites
- `partition.tsv` — `gene_id  start  end  model  gamma  alpha  gene_len`
- `meta.json` — `n_taxa, n_genes, target_sites, total_sites, gene_len_{mean,min,max}`