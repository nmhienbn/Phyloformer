#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_2d(vec: torch.Tensor) -> torch.Tensor:
    if vec.ndim == 1:
        return vec.unsqueeze(0)
    if vec.ndim == 2:
        return vec
    raise ValueError(
        f"Expected [pairs] or [batch, pairs], got shape {tuple(vec.shape)}."
    )


def _infer_num_leaves(num_pairs: int) -> int:
    disc = 1 + 8 * num_pairs
    n = int((1 + math.isqrt(disc)) // 2)
    if n * (n - 1) // 2 != num_pairs:
        raise ValueError(
            f"Cannot infer a valid number of leaves from num_pairs={num_pairs}."
        )
    return n


def get_1d_index(i: torch.Tensor, j: torch.Tensor, num_leaves: int) -> torch.Tensor:
    i_min = torch.minimum(i, j)
    j_max = torch.maximum(i, j)
    return (
        i_min * num_leaves
        - (i_min * (i_min + 1)) // 2
        + j_max
        - i_min
        - 1
    )


def sample_random_quartets(
    num_leaves: int,
    num_quartets: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.rand((num_quartets, num_leaves), device=device).topk(4, dim=1).indices


class QuartetPushCloseMRELoss(nn.Module):
    component_names = ("additivity_loss", "mre_loss")

    def __init__(self, lambda_q: float = 1.0, margin: float = 0.05, num_quartets: int = 2000, eps: float = 1e-8):
        """
        lambda_q: Trọng số của hàm Additivity Loss (Tương đương lambda trong bài báo ICML).
        margin: Khoảng cách m_0 bắt buộc giữa topology đúng và các tổng chéo.
        """
        super().__init__()
        self.lambda_q = lambda_q
        self.margin = margin
        self.num_quartets = num_quartets
        self.eps = eps

    def forward(
        self,
        y_pred_vec: torch.Tensor,
        y_true_vec: torch.Tensor,
        num_leaves: Optional[int] = None,
    ):
        y_pred_vec = _ensure_2d(y_pred_vec)
        y_true_vec = _ensure_2d(y_true_vec).type_as(y_pred_vec)
        if y_pred_vec.shape != y_true_vec.shape:
            raise ValueError(
                "Prediction/target shape mismatch: "
                f"{tuple(y_pred_vec.shape)} vs {tuple(y_true_vec.shape)}."
            )

        if num_leaves is None:
            num_leaves = _infer_num_leaves(y_pred_vec.shape[1])
        if num_leaves < 4:
            raise ValueError("Quartet loss requires num_leaves >= 4.")
        expected_pairs = num_leaves * (num_leaves - 1) // 2
        if y_pred_vec.shape[1] != expected_pairs:
            raise ValueError(
                f"Expected {expected_pairs} pairwise distances for {num_leaves} leaves, "
                f"got {y_pred_vec.shape[1]}."
            )

        batch_size = y_pred_vec.shape[0]

        # ==========================================
        # 1. TRÍCH XUẤT QUARTET VÀ TÍNH 3 TỔNG CHÉO
        # ==========================================
        quartets = sample_random_quartets(
            num_leaves=num_leaves,
            num_quartets=self.num_quartets,
            device=y_pred_vec.device,
        )
        a, b, c, d = quartets.unbind(dim=1)
        pair_i = torch.stack((a, c, a, b, a, b), dim=0)
        pair_j = torch.stack((b, d, c, d, d, c), dim=0)
        pair_idx = get_1d_index(pair_i, pair_j, num_leaves)
        flat_idx = pair_idx.reshape(-1)

        pred_pairs = y_pred_vec.index_select(1, flat_idx).reshape(
            batch_size, 6, self.num_quartets
        )
        true_pairs = y_true_vec.index_select(1, flat_idx).reshape(
            batch_size, 6, self.num_quartets
        )

        # pred_sums / true_sums shape: [batch, 3, num_quartets]
        pred_sums = pred_pairs.reshape(batch_size, 3, 2, self.num_quartets).sum(dim=2)
        true_sums = true_pairs.reshape(batch_size, 3, 2, self.num_quartets).sum(dim=2)
        
        # Tìm topology thật (index có tổng khoảng cách nhỏ nhất trên ma trận thực tế)
        min_idx = torch.argmin(true_sums, dim=1) # [batch, num_quartets]

        # ==========================================
        # 2. BÓC TÁCH CÁC TỔNG (S_true và S_cross)
        # ==========================================
        # S3: Tổng dự đoán ứng với topology thật (Tương đương S3 trong ICML)
        S3 = pred_sums.gather(1, min_idx.unsqueeze(1)).squeeze(1) # [batch, num_quartets]

        # Lọc lấy 2 tổng chéo còn lại (S1 và S2)
        mask = torch.ones_like(pred_sums, dtype=torch.bool)
        mask.scatter_(1, min_idx.unsqueeze(1), False)
        
        # Permute để nhóm các phần tử của cùng 1 quartet lại gần nhau trong bộ nhớ trước khi reshape
        pred_sums_perm = pred_sums.permute(0, 2, 1) # [batch, num_quartets, 3]
        mask_perm = mask.permute(0, 2, 1)           # [batch, num_quartets, 3]
        
        # Rút 2 tổng chéo và gán thành S_cross1, S_cross2
        cross_sums = pred_sums_perm[mask_perm].reshape(batch_size, self.num_quartets, 2)
        S_cross1 = cross_sums[:, :, 0] # [batch, num_quartets]
        S_cross2 = cross_sums[:, :, 1] # [batch, num_quartets]

        # ==========================================
        # 3. ICML ADDITIVITY LOSS (L_close + L_push)
        # ==========================================
        # L_close: Ép 2 tổng lớn nhất phải tiệm cận bằng nhau
        L_close = torch.abs(S_cross1 - S_cross2)

        # L_push: Đẩy tổng thật ra xa trung bình 2 tổng chéo một khoảng margin
        S_cross_avg = (S_cross1 + S_cross2) / 2.0
        L_push = F.relu(S3 - S_cross_avg + self.margin)

        additivity_loss = (L_close + L_push).mean()

        # ==========================================
        # 4. GLOBAL MRE (DEVIATION LOSS)
        # ==========================================
        # Tính sai số tương đối trên toàn bộ cây thay vì chỉ trên mẫu
        denom = y_true_vec.clamp_min(self.eps)
        mre_loss = (torch.abs(y_pred_vec - y_true_vec) / denom).mean()
        
        # ==========================================
        # 5. TỔNG HỢP LOSS
        # ==========================================
        loss = mre_loss + self.lambda_q * additivity_loss

        return loss, additivity_loss, mre_loss
