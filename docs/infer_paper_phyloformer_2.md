# Infer PhyloFormer 2

Doc này dùng cho pipeline PF2 hiện tại:

```text
MSA -> PF2 --mode max-sample -> predicted .nwk -> phylocompare -> plots
```

Không dùng FastME trong pipeline PF2 chính. `PF2 --mode dm -> FastME` chỉ giữ như thử nghiệm cũ, không dùng để báo cáo chính.

## 1. Tải test datasets từ Zenodo

```bash
cd /raid/home/hiennguyen/Phyloformer
mkdir -p downloads

for f in \
  paper_test_sets.tar.xz \
  results.tar.gz \
  cherry_fine_tune.tar.xz \
  LG_fine_tune_mre.tar.xz \
  pastek_fine_tune.tar.xz
do
  curl -L "https://zenodo.org/records/13742527/files/${f}?download=1" -o "downloads/${f}"
done
```

```bash
tar -xJf downloads/paper_test_sets.tar.xz -C data
tar -xzf downloads/results.tar.gz -C .
```

Expected layout:

```text
data/final_test_set/alignments/*.fa
data/final_test_set/trees/*.nwk
data/LGGC+gaps/alignments/*.fa
data/LGGC+gaps/trees/*.nwk
data/cherry_test_data/alignments/*.fa
data/cherry_test_data/trees/*.nwk
data/pastek_test_data/alignments/*.fa
data/pastek_test_data/trees/*.nwk
```

## 2. Checkpoint và môi trường

Chạy bằng env `pf2`:

```bash
conda activate pf2
```

Checkpoint mặc định:

```bash
CKPT="models/phyloformer2/pf2.tch"
MODEL_NAME="pf2"
PLOT_MODEL_NAME="PF2"
BIN="./bin/bin_linux"
```

Checkpoint fine-tuned theo dataset nếu muốn chạy riêng:

```bash
PF2_BASE="models/phyloformer2/pf2.tch"
PF2_CHERRY="models/phyloformer2/pf2_cherry.tch"
PF2_PASTEK="models/phyloformer2/pf2_pastek.tch"
```

## 3. Infer một test set

`third_party/phyloformer2/infer.py` yêu cầu output dir chưa tồn tại, nên luôn `rm -rf` trước khi chạy.

```bash
DATASET="data/final_test_set"
NAME="$(basename "$DATASET")"
OUT="runs/$MODEL_NAME/eval_${NAME}"

rm -rf "$OUT"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n pf2 \
  python third_party/phyloformer2/infer.py \
    --mode max-sample \
    "$DATASET/alignments" \
    "$CKPT" \
    "$OUT/trees"
```

So sánh với true trees:

```bash
python third_party/tools/evaluation/run_phylocompare.py \
  --bin-dir "$BIN" \
  --pred-trees "$OUT/trees" \
  --true-trees "$DATASET/trees" \
  --cmp-out "$OUT/cmp_pf2" \
  --method-name "$PLOT_MODEL_NAME" \
  --label "$NAME"
```

## 4. Infer toàn bộ paper test sets

Chạy PF2 base trên cả 4 bộ:

```bash
TOTAL_START=$(date +%s)

for D in data/final_test_set data/LGGC+gaps data/cherry_test_data data/pastek_test_data; do
  NAME="$(basename "$D")"
  OUT="runs/$MODEL_NAME/eval_${NAME}"
  SET_START=$(date +%s)

  rm -rf "$OUT"
  mkdir -p "$OUT"

  CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pf2 \
    python third_party/phyloformer2/infer.py \
      --mode max-sample \
      "$D/alignments" \
      "$CKPT" \
      "$OUT/trees"

  python third_party/tools/evaluation/run_phylocompare.py \
    --bin-dir "$BIN" \
    --pred-trees "$OUT/trees" \
    --true-trees "$D/trees" \
    --cmp-out "$OUT/cmp_pf2" \
    --method-name "$PLOT_MODEL_NAME" \
    --label "$NAME"

  SET_END=$(date +%s)
  echo "[TIME] [$NAME] PF2 infer+compare: $((SET_END - SET_START))s"
done

TOTAL_END=$(date +%s)
echo "[TIME] [PF2] ALL DATASETS TOTAL: $((TOTAL_END - TOTAL_START))s"
```

## 5. Infer với checkpoint fine-tuned

Nếu muốn dùng checkpoint tương ứng cho từng domain:

```bash
TOTAL_START=$(date +%s)

declare -A CKPTS
CKPTS["final_test_set"]="models/phyloformer2/pf2.tch"
CKPTS["LGGC+gaps"]="models/phyloformer2/pf2.tch"
CKPTS["cherry_test_data"]="models/phyloformer2/pf2_cherry.tch"
CKPTS["pastek_test_data"]="models/phyloformer2/pf2_pastek.tch"

MODEL_NAME="pf2_domain_ckpt"
PLOT_MODEL_NAME="PF2_domain_ckpt"

for D in data/final_test_set data/LGGC+gaps data/cherry_test_data data/pastek_test_data; do
  NAME="$(basename "$D")"
  CKPT="${CKPTS[$NAME]}"
  OUT="runs/$MODEL_NAME/eval_${NAME}"
  SET_START=$(date +%s)

  rm -rf "$OUT"
  mkdir -p "$OUT"

  CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n pf2 \
    python third_party/phyloformer2/infer.py \
      --mode max-sample \
      "$D/alignments" \
      "$CKPT" \
      "$OUT/trees"

  python third_party/tools/evaluation/run_phylocompare.py \
    --bin-dir "$BIN" \
    --pred-trees "$OUT/trees" \
    --true-trees "$D/trees" \
    --cmp-out "$OUT/cmp_pf2" \
    --method-name "$PLOT_MODEL_NAME" \
    --label "$NAME"

  SET_END=$(date +%s)
  echo "[TIME] [$NAME] PF2 domain ckpt infer+compare: $((SET_END - SET_START))s"
done

TOTAL_END=$(date +%s)
echo "[TIME] [PF2 domain ckpt] ALL DATASETS TOTAL: $((TOTAL_END - TOTAL_START))s"
```

## 6. Plot

Plot kết quả của một model trên 4 test sets:

```bash
python third_party/tools/plots/make_plots2.py \
  --new-model-only \
  --new-model-name "$PLOT_MODEL_NAME" \
  --new-topo "runs/$MODEL_NAME/eval_final_test_set/cmp_pf2_topo.csv" \
  --new-dist "runs/$MODEL_NAME/eval_final_test_set/cmp_pf2_dist.csv" \
  --new-topo "runs/$MODEL_NAME/eval_LGGC+gaps/cmp_pf2_topo.csv" \
  --new-dist "runs/$MODEL_NAME/eval_LGGC+gaps/cmp_pf2_dist.csv" \
  --new-topo "runs/$MODEL_NAME/eval_cherry_test_data/cmp_pf2_topo.csv" \
  --new-dist "runs/$MODEL_NAME/eval_cherry_test_data/cmp_pf2_dist.csv" \
  --new-topo "runs/$MODEL_NAME/eval_pastek_test_data/cmp_pf2_topo.csv" \
  --new-dist "runs/$MODEL_NAME/eval_pastek_test_data/cmp_pf2_dist.csv" \
  --outdir "figures/$MODEL_NAME"
```

Nếu chỉ cần boxplot topology:

```bash
python third_party/tools/plots/plot_topology_boxplots.py \
  --cmp-topo "PF2|final_test_set=runs/$MODEL_NAME/eval_final_test_set/cmp_pf2_topo.csv" \
  --cmp-topo "PF2|LGGC+gaps=runs/$MODEL_NAME/eval_LGGC+gaps/cmp_pf2_topo.csv" \
  --cmp-topo "PF2|cherry_test_data=runs/$MODEL_NAME/eval_cherry_test_data/cmp_pf2_topo.csv" \
  --cmp-topo "PF2|pastek_test_data=runs/$MODEL_NAME/eval_pastek_test_data/cmp_pf2_topo.csv" \
  --outdir "figures/$MODEL_NAME/topology_boxplots"
```

## 7. Chạy qua wrapper benchmark

Infer, compare, runtime summary và optional plot:

```bash
CUDA_VISIBLE_DEVICES=7 python third_party/tools/inference/run_pf2_pure_benchmark.py \
  models/phyloformer2/pf2.tch \
  data/final_test_set/alignments \
  runs/pf2/eval_final_test_set_wrapper \
  --true-trees data/final_test_set/trees \
  --method-name PF2 \
  --plot \
  --overwrite
```

Wrapper này gọi `third_party/phyloformer2/infer.py --mode max-sample`, nên kết quả topology tương đương pipeline thủ công ở trên.

## 8. Ghi chú

- `--mode max-sample` xuất trực tiếp cây `.nwk`; đây là pipeline PF2 chính.
- `--mode samples` xuất nhiều cây sample cho mỗi MSA, dùng khi cần phân tích bất định topology.
- `--mode dm` xuất distance matrix `.phy`; không dùng cho benchmark PF2 chính vì sẽ kéo thêm FastME.
- Nếu gặp OOM, giảm subset bằng số taxa/alignment length hoặc dùng pipeline partition trong `experiments/04_pf2_partition_merge_refit`.
