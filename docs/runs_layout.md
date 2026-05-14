# Runs Layout

Tài liệu này ghi lại cách `runs/` đang được chia lại để bớt lẫn giữa:

- output của thí nghiệm đang chạy
- artifact trung gian của pipeline hiện tại
- benchmark / eval cũ đã chạy xong

## Current Policy

Giữ ở `runs/` root chỉ các nhóm lớn cấp cao:

- `runs/pandit_domain_adaptation` cho toàn bộ pipeline adaptation hiện tại
- `runs/completed/*` cho benchmark / eval cũ đã hoàn tất
- các nhóm paper / legacy chưa dọn vì còn phụ thuộc workflow cũ

Đã gom các thí nghiệm cũ vào:

- `runs/completed/topology`
- `runs/completed/gpartition`
- `runs/completed/eval`

## Current PANDIT Adaptation Layout

Các artifact adaptation hiện tại đã được gom dưới `runs/pandit_domain_adaptation`:

- `runs/PF2_PANDIT_ADAPT`
- `runs/pandit_domain_adaptation/stage1_msa_splits`
- `runs/pandit_domain_adaptation/stage2_profiles`
- `runs/pandit_domain_adaptation/stage3_evopf_embeddings`
- `runs/pandit_domain_adaptation/stage3_pca_empirical_only`
- `runs/pandit_domain_adaptation/stage3_pca_ood_fixed`
- `runs/pandit_domain_adaptation/stage3_pca_ood_size_matched`
- `runs/pandit_domain_adaptation/stage3_pf2_native_pca_*`
- `runs/pandit_domain_adaptation/stage4_gpartition_train_prepare*`
- `runs/pandit_domain_adaptation/stage5_synthetic_pandit_like_raw_20k`
- `runs/pandit_domain_adaptation/stage7_pf2_adapt_raw_20k_data_max170`
- `runs/pandit_domain_adaptation/eval_subsets`
- `runs/pandit_domain_adaptation/archive/*` cho smoke/debug/legacy one-off của riêng pipeline này

Lý do: đây là output đang dùng trong runbook hiện tại, hoặc còn liên quan đến run
fine-tune đang chạy.

## Completed Topology

Đã move sang `runs/completed/topology`:

- `bench_*`
- `bootstrap_*`  archive / historical only
- `fasttree_*`
- `iqtree_*`
- `mpboot_*`
- `phase1_pandit_clean_benchmark`
- `topology_benchmark_pandit_aa*`

Các docs benchmark/classical đã được update sang path mới này; bootstrap workflow hiện nằm ở archive note trong `unused/`.

## Completed GPartition

Đã move sang `runs/completed/gpartition`:

- `gpf2_pandit_1470`
- `gpf2_treebase_M3114`
- `gpf2_zenodo_aa_full`

Các docs benchmark/gpartition dùng output cũ đã được update.

## Completed Eval

Đã move sang `runs/completed/eval`:

- `evopf`
- `hybrid_a`
- các thư mục `evopf_*` kiểu env-fix / smoke / one-off debug

Các docs eval so sánh với `evoPF` hoặc `HybridA` đã được update.

## Intentionally Left At Root

Các thư mục sau chưa move:

- `runs/PF2_PAPER`
- `runs/paper_pfbase`
- `runs/pfbase_quartet_*`

Lý do: chúng còn dính với training / paper reproduction setup và một số command mặc
định đang dùng trực tiếp các path này. Nếu muốn dọn tiếp nhóm này thì nên làm sau,
khi không còn cần giữ tương thích với workflow paper hiện tại.
