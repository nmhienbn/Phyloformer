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
    """
    Map pair indices (i, j) to flattened upper-triangle index used by seq2pair.
    """
    i_min = torch.minimum(i, j)
    j_max = torch.maximum(i, j)
    return i_min * num_leaves - (i_min * (i_min + 1)) // 2 + j_max - i_min - 1


def sample_random_quartets(
    num_leaves: int,
    num_quartets: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.rand((num_quartets, num_leaves), device=device).topk(4, dim=1).indices


class QuartetSoftmaxLoss(nn.Module):

    def __init__(
        self, temperature: float = 1.0, sigma: float = 1.0, num_quartets: int = 50
    ):
        super().__init__()
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))
        self.inv_temperature = 1.0 / float(temperature)
        self.num_quartets = num_quartets
        self.mae = nn.L1Loss()

    def forward(
        self,
        y_pred_vec: torch.Tensor,
        y_true_vec: torch.Tensor,
        num_leaves: Optional[int] = None,
    ) -> torch.Tensor:
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

        quartets = sample_random_quartets(
            num_leaves=num_leaves,
            num_quartets=self.num_quartets,
            device=y_pred_vec.device,
        )
        a, b, c, d = quartets.unbind(dim=1)
        pair_i = torch.stack((a, c, a, b, a, b), dim=0)
        pair_j = torch.stack((b, d, c, d, d, c), dim=0)
        pair_idx = get_1d_index(pair_i, pair_j, num_leaves)  # [6, num_quartets]
        flat_idx = pair_idx.reshape(-1)

        batch_size = y_pred_vec.shape[0]
        pred_pairs = y_pred_vec.index_select(1, flat_idx).reshape(
            batch_size, 6, self.num_quartets
        )
        true_pairs = y_true_vec.index_select(1, flat_idx).reshape(
            batch_size, 6, self.num_quartets
        )

        pred_sums = pred_pairs.reshape(batch_size, 3, 2, self.num_quartets).sum(dim=2)
        true_sums = true_pairs.reshape(batch_size, 3, 2, self.num_quartets).sum(dim=2)
        min_idx = torch.argmin(true_sums, dim=1)  # [batch, num_quartets]

        logits = -(pred_sums.transpose(1, 2).reshape(-1, 3)) * self.inv_temperature
        target = min_idx.reshape(-1)

        quartet_loss = F.cross_entropy(logits, target)

        return quartet_loss + self.sigma * self.mae(y_pred_vec, y_true_vec)
