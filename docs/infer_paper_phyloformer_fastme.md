# Tải test datasets từ Zenodo
- raw tree, MSA

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
# Infer PhyloFormer 1 + FastME
last.ckpt -> .phy -> .nwk -> cmp_pf1_topo/dist.csv

- Setting checkpoint, tên model lưu và tên model trong plot:
    ```bash
    CKPT="runs/pfbase_quartet_mre/checkpoints_LR_0.0001_O_Adam_L_QMRE_L0.2_M0.05_Q2000_E_4_BS_1_ACC_12_NB_6_NH_4_HD_64_D_0.0_W3000/last.ckpt"
    MODEL_NAME="pfbase_quartet_mre"
    PLOT_MODEL_NAME="PF_PUSH+FastME"
    BIN="./bin/bin_linux"
    ```

- Chạy toàn bộ test set:
    ```bash
    TOTAL_START=$(date +%s)
    for D in data/final_test_set; do
        NAME=$(basename "$D")
        OUT="runs/$MODEL_NAME/eval_${NAME}"
        mkdir -p "$OUT/mats" "$OUT/trees"
        SET_START=$(date +%s)

        CUDA_VISIBLE_DEVICES=7 python third_party/phyloformer1/infer_alns.py "$CKPT" "$D/alignments" -o "$OUT/mats"

        python third_party/benchmark/run_fastme_compare.py \
          --bin-dir "$BIN" \
          --mats-dir "$OUT/mats" \
          --trees-dir "$OUT/trees" \
          --label "$NAME" \
          --threads 1 \
          --true-trees "$D/trees" \
          --cmp-out "$OUT/cmp_pf1" \
          --method-name "$PLOT_MODEL_NAME"
        SET_END=$(date +%s)
        echo "[TIME] [$NAME] END-TO-END TOTAL: $((SET_END - SET_START))s"
    done
    TOTAL_END=$(date +%s)
    echo "[TIME] ALL DATASETS TOTAL: $((TOTAL_END - TOTAL_START))s"
    ```

- Plot cần chú ý truyền `--new-dist`/`--new-topo` của những bộ cần benchmark:
    ```bash
    python third_party/benchmark/make_plots2.py \
    --new-model-only \
    --new-model-name "$PLOT_MODEL_NAME" \
    --new-topo "runs/$MODEL_NAME/eval_final_test_set/cmp_pf1_topo.csv" \
    --new-dist "runs/$MODEL_NAME/eval_final_test_set/cmp_pf1_dist.csv" \
    --new-topo "runs/$MODEL_NAME/eval_LGGC+gaps/cmp_pf1_topo.csv" \
    --new-dist "runs/$MODEL_NAME/eval_LGGC+gaps/cmp_pf1_dist.csv" \
    --new-topo "runs/$MODEL_NAME/eval_cherry_test_data/cmp_pf1_topo.csv" \
    --new-dist "runs/$MODEL_NAME/eval_cherry_test_data/cmp_pf1_dist.csv" \
    --new-topo "runs/$MODEL_NAME/eval_pastek_test_data/cmp_pf1_topo.csv" \
    --new-dist "runs/$MODEL_NAME/eval_pastek_test_data/cmp_pf1_dist.csv" \
    --outdir figures/$MODEL_NAME
    ```

# Chạy PF2_MAE: evoPF + FastME

```bash
EVO_CKPT="runs/PF2_PAPER/20260321-223929-tan-dry-woodlouse/checkpoints/best_val_loss.ckpt"
MODEL_NAME="evopf"
PLOT_MODEL_NAME="evoPF"
BIN="./bin/bin_linux"
```

- Chạy toàn bộ test set
    ```bash
    TOTAL_START=$(date +%s)
    for D in data/final_test_set data/LGGC+gaps data/cherry_test_data data/pastek_test_data; do
        NAME=$(basename "$D")
        OUT="runs/$MODEL_NAME/eval_${NAME}"
        rm -rf "$OUT"
        mkdir -p "$OUT/trees"
        SET_START=$(date +%s)

        if CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n pf2 python third_party/phyloformer2/infer.py \
          --mode dm \
          "$D/alignments" \
          "$EVO_CKPT" \
          "$OUT/mats"; then
            N_MATS=$(find "$OUT/mats" -maxdepth 1 -name '*.phy' | wc -l)
        else
            N_MATS=0
        fi

        if [ "$N_MATS" -eq 0 ]; then
            echo "[WARN] [${NAME}_evoPF] No .phy files produced in $OUT/mats, skip FASTME/phylocompare"
            SET_END=$(date +%s)
            echo "[TIME] [${NAME}_evoPF] END-TO-END TOTAL: $((SET_END - SET_START))s"
            continue
        fi

        python third_party/benchmark/run_fastme_compare.py \
          --bin-dir "$BIN" \
          --mats-dir "$OUT/mats" \
          --trees-dir "$OUT/trees" \
          --label "${NAME}_evoPF" \
          --threads 1 \
          --true-trees "$D/trees" \
          --cmp-out "$OUT/cmp_evopf" \
          --method-name evoPF+FastME

        SET_END=$(date +%s)
        echo "[TIME] [${NAME}_evoPF] END-TO-END TOTAL: $((SET_END - SET_START))s"
    done
    TOTAL_END=$(date +%s)
    echo "[TIME] [evoPF] ALL DATASETS TOTAL: $((TOTAL_END - TOTAL_START))s"
    ```

- Plot cho PF2: evoPF + FastME
    ```bash
    python third_party/benchmark/make_plots2.py \
      --new-model-only \
      --new-model-name "$PLOT_MODEL_NAME" \
      --new-topo "runs/$MODEL_NAME/eval_final_test_set/cmp_evopf_topo.csv" \
      --new-dist "runs/$MODEL_NAME/eval_final_test_set/cmp_evopf_dist.csv" \
      --new-topo "runs/$MODEL_NAME/eval_LGGC+gaps/cmp_evopf_topo.csv" \
      --new-dist "runs/$MODEL_NAME/eval_LGGC+gaps/cmp_evopf_dist.csv" \
      --new-topo "runs/$MODEL_NAME/eval_cherry_test_data/cmp_evopf_topo.csv" \
      --new-dist "runs/$MODEL_NAME/eval_cherry_test_data/cmp_evopf_dist.csv" \
      --new-topo "runs/$MODEL_NAME/eval_pastek_test_data/cmp_evopf_topo.csv" \
      --new-dist "runs/$MODEL_NAME/eval_pastek_test_data/cmp_evopf_dist.csv" \
      --outdir figures/evopf/cmp_evopf
    ```
