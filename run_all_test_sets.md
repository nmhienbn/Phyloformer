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
# Chạy mô hình infer
last.ckpt -> .phy -> .nwk -> cmp_qsiam_topo/dist.csv

```bash
CKPT="runs/pfbase_quartet_push_close/checkpoints_LR_0.0001_O_Adam_L_QPUSH_L1_M0.05_Q2000_E_4_BS_1_ACC_12_NB_6_NH_4_HD_64_D_0.0_W3000/last.ckpt"
MODEL_NAME="pfbase_quartet_push_close"
BIN="./bin/bin_linux"
```

- chạy 1 test set
    ```bash
    OUT="runs/$MODEL_NAME/eval_final_test_set_qsiam"
    mkdir -p "$OUT/mats" "$OUT/trees"

    # 1) MSA -> distance matrices
    python src/infer_alns.py "$CKPT" data/final_test_set/alignments -o "$OUT/mats"

    # 2) FASTME + 3) so sánh với true trees (có tqdm + tổng thời gian)
    python tools/run_fastme_compare.py \
      --bin-dir "$BIN" \
      --mats-dir "$OUT/mats" \
      --trees-dir "$OUT/trees" \
      --label final_test_set \
      --threads 1 \
      --true-trees data/final_test_set/trees \
      --cmp-out "$OUT/cmp_qsiam" \
      --method-name PF_QSIAM+FastME
    ```

- chạy toàn bộ test set:
    ```bash
    TOTAL_START=$(date +%s)
    for D in data/final_test_set data/LGGC+gaps data/cherry_test_data data/pastek_test_data; do
        NAME=$(basename "$D")
        OUT="runs/$MODEL_NAME/eval_${NAME}_qsiam"
        mkdir -p "$OUT/mats" "$OUT/trees"
        SET_START=$(date +%s)

        CUDA_VISIBLE_DEVICES=7 python src/infer_alns.py "$CKPT" "$D/alignments" -o "$OUT/mats"

        python tools/run_fastme_compare.py \
          --bin-dir "$BIN" \
          --mats-dir "$OUT/mats" \
          --trees-dir "$OUT/trees" \
          --label "$NAME" \
          --threads 1 \
          --true-trees "$D/trees" \
          --cmp-out "$OUT/cmp_qsiam" \
          --method-name PF_MRE_QSIAM+FastME
        SET_END=$(date +%s)
        echo "[TIME] [$NAME] END-TO-END TOTAL: $((SET_END - SET_START))s"
    done
    TOTAL_END=$(date +%s)
    echo "[TIME] ALL DATASETS TOTAL: $((TOTAL_END - TOTAL_START))s"
    ```

# Plot
```bash
python tools/plots/make_plots2.py --full \
  --qsiam-topo "runs/$MODEL_NAME/eval_final_test_set_qsiam/cmp_qsiam_topo.csv" \
  --qsiam-dist "runs/$MODEL_NAME/eval_final_test_set_qsiam/cmp_qsiam_dist.csv" \
  --outdir figures/$MODEL_NAME
```

# Chạy PF2 / evoPF
- `PF2/infer.py` có hỗ trợ xuất trực tiếp distance matrix `.phy` bằng `--mode dm`.
- Script này tự nhận biết checkpoint có phải `evoPF` hay không qua `hparams["evopf"]` trong checkpoint.
- Lưu ý: output directory truyền cho `PF2/infer.py` phải chưa tồn tại, vì script dùng `os.makedirs(outroot)` không có `exist_ok=True`.
- Nên chạy `PF2/infer.py` bằng env `pf2`, và dùng `conda run --no-capture-output -n pf2 ...` để thấy `tqdm`/stdout realtime.

```bash
EVO_CKPT="PF2/pretrained/pf2.tch"
MODEL_NAME="evopf"
BIN="./bin/bin_linux"
```

- chạy 1 test set
    ```bash
    OUT="runs/$MODEL_NAME/eval_final_test_set"
    rm -rf "$OUT"
    mkdir -p "$OUT/trees"
    SET_START=$(date +%s)

    # 1) MSA -> distance matrices bằng PF2/evoPF
    if CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n pf2 python PF2/infer.py \
      --mode dm \
      data/final_test_set/alignments \
      "$EVO_CKPT" \
      "$OUT/mats"; then
      N_MATS=$(find "$OUT/mats" -maxdepth 1 -name '*.phy' | wc -l)
    else
      N_MATS=0
    fi

    if [ "$N_MATS" -eq 0 ]; then
      echo "[WARN] [final_test_set_evoPF] No .phy files produced in $OUT/mats, skip FASTME/phylocompare"
    else
      # 2) FASTME + 3) so sánh với true trees
      python tools/run_fastme_compare.py \
        --bin-dir "$BIN" \
        --mats-dir "$OUT/mats" \
        --trees-dir "$OUT/trees" \
        --label "final_test_set_evoPF" \
        --threads 1 \
        --true-trees data/final_test_set/trees \
        --cmp-out "$OUT/cmp_evopf" \
        --method-name evoPF+FastME
    fi

    SET_END=$(date +%s)
    echo "[TIME] [final_test_set_evoPF] END-TO-END TOTAL: $((SET_END - SET_START))s"
    ```

- chạy toàn bộ test set
    ```bash
    TOTAL_START=$(date +%s)
    for D in data/final_test_set data/LGGC+gaps data/cherry_test_data data/pastek_test_data; do
        NAME=$(basename "$D")
        OUT="runs/$MODEL_NAME/eval_${NAME}"
        rm -rf "$OUT"
        mkdir -p "$OUT/trees"
        SET_START=$(date +%s)

        if CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n pf2 python PF2/infer.py \
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

        python tools/run_fastme_compare.py \
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

- nếu chưa chắc checkpoint nào là evoPF:
    ```bash
    python - <<'PY'
    import torch
    ckpt = torch.load("PF2/pretrained/<checkpoint>.tch", map_location="cpu")
    print(ckpt["hparams"].get("evopf", False))
    PY
    ```
  Kết quả `True` nghĩa là checkpoint đó sẽ được `PF2/infer.py` load bằng lớp `EvoPF`.

- plot cho PF2 / evoPF
    ```bash
    python tools/plots/make_plots2.py --full \
      --cmp-topo "evoPF=runs/$MODEL_NAME/eval_final_test_set/cmp_evopf_topo.csv" \
      --cmp-dist "evoPF=runs/$MODEL_NAME/eval_final_test_set/cmp_evopf_dist.csv" \
      --outdir figures/cmp_evopf
    ```

- nếu muốn so trực tiếp `evoPF` với `qsiam`
    ```bash
    python tools/plots/make_plots2.py \
      --cmp-topo "evoPF=runs/evopf/eval_final_test_set/cmp_evopf_topo.csv" \
      --cmp-dist "evoPF=runs/evopf/eval_final_test_set/cmp_evopf_dist.csv" \
      --cmp-topo "QSIAM=runs/pfbase_quartet_siamese/eval_final_test_set_qsiam/cmp_qsiam_topo.csv" \
      --cmp-dist "QSIAM=runs/pfbase_quartet_siamese/eval_final_test_set_qsiam/cmp_qsiam_dist.csv" \
      --outdir figures/cmp_evopf_vs_qsiam
    ```
