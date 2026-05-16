Benchmark các cấu hình về phần cứng khác nhau khi train:

| Loại Model | Số lượng GPU | Batch Size (mỗi GPU) | Accumulate Batch | Effective Batch Size | File Config (JSON) |
| --- | --- | --- | --- | --- | --- |
| PF1 | 6 | 4 | 1 | 24 | [paper_pfbase_finetune_mre_6gpu_exact.json](configs/paper_pfbase_finetune_mre_6gpu_exact.json) |
| PF1 Pretrained | 6 | 4 | 1 | 24 | [paper_pfbase_pretrain_6gpu_exact.json](configs/paper_pfbase_pretrain_6gpu_exact.json) |
| PF1 Pretrained | 6 | 1 | 4 | 24 | [paper_pfbase_pretrain_6gpu_effective.json](configs/paper_pfbase_pretrain_6gpu_effective.json) |
| PF1 Pretrained | 1 | 1 | 24 | 24 | [paper_pfbase_pretrain_1gpu_effective.json](configs/paper_pfbase_pretrain_1gpu_effective.json) |

So sánh 4 cấu hình trên:

```bash
python third_party/benchmark/plot_pfbase_gpu_compare.py
```