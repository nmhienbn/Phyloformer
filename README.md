# Phyloformer Experiments

Repo này chứa code, dữ liệu cấu hình, kết quả và tài liệu phục vụ các thí nghiệm
Phyloformer/PF2 trong thesis.

## Thư mục chính

- `experiments/`: các thí nghiệm chính, chia theo chương/nhóm việc.
  - `01_pf1_vram/`: đo và so sánh VRAM của PF1/PF2.
  - `02_pf1_quartet_loss/`: benchmark PF1 với quartet loss.
  - `03_pf2_pandit_domain_adaptation/`: synthetic PANDIT-like data, fine-tune PF2, OOD/PCA diagnostics.
  - `04_pf2_partition_merge_refit/`: partition alignment, merge/supertree, IQ-TREE branch-length refit.
- `thesis/`: nội dung thesis.
- `docs/`: hướng dẫn chạy các workflow chung pf1, pf2, pp truyền thống.
- `models/`: pretrained và trained models
- `data/`: dữ liệu đầu vào: msa, cây...
- `runs/`: output từ training/benchmark runs.
- `third_party/`: code/tool bên thứ ba và wrappers.
  - `phyloformer1/`, `phyloformer2/`: upstream/model code.
  - `tools/`: wrappers cho inference, evaluation, plotting, VRAM, partition merge, SuperFine.
- `bin/`: binary tools local như IQ-TREE, FastME, FastRFS, PhyloCompare.
- `unused/`: code/file cũ đã tách khỏi workflow hiện tại.

## Hướng dẫn sử dụng
Cài môi trường phù hợp với thí nghiệm:
- [requirements.txt](experiments/04_pf2_partition_merge_refit/requirements.txt)
- [requirements.txt](third_party/phyloformer1/requirements.txt)
- [requirements.txt](third_party/phyloformer2/requirements.txt)
- [requirements.train.txt](third_party/phyloformer2/requirements.train.txt)