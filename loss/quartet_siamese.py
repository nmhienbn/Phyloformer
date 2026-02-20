# https://github.com/VivianBrandenburg/qtools/blob/main/qtools/src/qtools/lossfunctions.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

# =============================================================================
# distance calculations
# =============================================================================


def computedistancematrix(input, norm):
    v0, v1, v2, v3 = input[..., 0], input[..., 1], input[..., 2], input[..., 3]
    Dij = norm(v0 - v1)
    Dik = norm(v0 - v2)
    Dil = norm(v0 - v3)
    Djk = norm(v1 - v2)
    Djl = norm(v1 - v3)
    Dkl = norm(v2 - v3)
    return torch.stack([Dij, Dik, Dil, Djk, Djl, Dkl], dim=0)


def get_distance_matrix(y_pred):
    ypred_dist = computedistancematrix(
        y_pred, lambda x: torch.linalg.vector_norm(x, ord=2, dim=-1)
    )
    return ypred_dist


# =============================================================================
# quartet loss and siamese regulation
# =============================================================================


class QuartetSiameseLoss(nn.Module):

    def __init__(self, sigma: float = 0.0):
        """
        Initialize the quartet loss function and set sigma.
        Sigma is the relation between quartet loss and siamese regulation.

        Loss = Quartet loss + (Siamese Regulation * Sigma)

        """
        super().__init__()
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

    def forward(self, input, target):
        # calculate distances between output feature vectors
        dist = get_distance_matrix(input)

        #####  e-loss  ####
        _, Dik, Dil, Djk, Djl, _ = dist

        e = (Dil + Djk) - (Dik + Djl)
        e_loss = e**2

        ##### siamese regulation  ####
        if self.sigma != 0:
            # difference between predicted distance and pairwise sequence distance
            siam_loss = torch.sum(torch.abs(target - dist))
            loss = e_loss + self.sigma * siam_loss
        else:
            loss = e_loss

        return loss
