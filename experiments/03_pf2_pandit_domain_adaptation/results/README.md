# PF2 PANDIT Domain Adaptation Results

Thư mục này gom lại các artifact nhỏ cần cho paper/thesis. Raw MSA, tree,
embedding `.npy`, checkpoint và log huấn luyện lớn vẫn để ngoài Git.

## Layout theo mạch paper

| Thư mục | Nội dung |
| --- | --- |
| `00_paper_figures/` | Hình đã dùng trực tiếp trong thesis: benchmark PF2 gốc trên PANDIT, PCA scatter và PCA contour |
| `01_pandit_split/` | Split PANDIT train/test và thống kê split |
| `02_empirical_profile/` | Profile PANDIT train: số taxa, độ dài alignment, gap ratio, informative-site ratio |
| `03_iqtree_model_calibration/` | Bảng IQ-TREE ModelFinder trên PANDIT train, gồm model, gamma alpha và tree stats |
| `04_gap_tree_calibration/` | Gap/tree bucket calibration dùng để kiểm tra synthetic Stage 5 v3 |
| `05_ood_pca/` | PCA OOD trước fine-tune: fixed PF2-native và size-matched PF2-native, gồm scatter và contour |
| `06_stage5_synthetic_v3/` | Manifest, memory risk và eval summary của synthetic PANDIT-like Stage 5 v3 |
| `07_pf2_finetune_layout/` | Layout train/validation synthetic 200k multisize và script fine-tune |
| `08_pf2_finetune_eval/` | Kết quả PANDIT test so sánh FastTree, IQ-TREE ModelFinder, PF2 gốc và PF2-Adapt 200k latest |

## Artifact không copy

- `stage5_synthetic_pandit_like_realism_v3_raw_200k/msas/`
- `stage5_synthetic_pandit_like_realism_v3_raw_200k/trees_by_taxa/`
- `stage3_evopf_embeddings/**/pooled/*.npy`
- `stage7_postft_evopf_embeddings/**/pooled/*.npy`
- `runs/PF2_PANDIT_ADAPT_200K/**/checkpoints/*.ckpt`
- raw run logs trong `runs/PF2_PANDIT_ADAPT_200K/**`

Các file trên lớn và không cần thiết để đọc bảng/hình trong paper. Nếu cần tái
lập từ đầu, dùng runbook trong `../docs/pandit_domain_adaptation_c3.md`.

## PCA contour plots

Script:

- `../src/ood_diagnostics/plot_embedding_pca_contours.py`

Output chính:

- `00_paper_figures/pca_contour_training_vs_pandit.png`
- `00_paper_figures/pca_contour_training_vs_pandit.pdf`
- `00_paper_figures/pca_contour_simulated_pandit_like_vs_pandit.png`
- `00_paper_figures/pca_contour_simulated_pandit_like_vs_pandit.pdf`
- `05_ood_pca/fixed/embedding_pca_contour.png`
- `05_ood_pca/fixed/embedding_pca_contour.pdf`
- `05_ood_pca/size_matched/embedding_pca_contour.png`
- `05_ood_pca/size_matched/embedding_pca_contour.pdf`

PANDIT được gộp `train` và `test` trước khi vẽ contour. Các contour được
vẽ thành subplot riêng trong cùng PCA space và được ghi cạnh các scatter cũ
trong cùng thư mục.

## Ghi chú Stage 5

Artifact Stage 5 trong thư mục này đã được thay bằng bản synthetic 200k. Các
file eval phân bố vẫn giữ prefix `pilot_*` do tên output của script:

```text
06_stage5_synthetic_v3/eval/pilot_*.tsv
06_stage5_synthetic_v3/eval/pilot_pass_report.json
```

Không có các file `synthetic_*.tsv` trong workspace hiện tại.
