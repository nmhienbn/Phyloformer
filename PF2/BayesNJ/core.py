# pyright: basic

import math
import os
import warnings
from collections import defaultdict
from glob import glob
from itertools import combinations
from pathlib import Path
from typing import Optional, Union, override

import numpy as np
import torch
import torch.nn.functional as F
from dendropy import Node, TaxonNamespace, Tree

# import torch.distributions as distr
from torch.utils.data import Dataset, DistributedSampler, Sampler

from BayesNJ.evopf import EvoPF

from .pf_sdk.pf.data import load_alignment, load_distance_matrix

ROOT_2PI = torch.tensor([2 * math.pi]).sqrt()


def load_splits(treepath, taxa_order):
    # This namespace ensures that the leaf order is the same
    # in the splits matrix and the aln
    namespace = TaxonNamespace([x.replace("_", " ") for x in taxa_order])
    tree = Tree.get_from_path(treepath, schema="newick", taxon_namespace=namespace)

    splits, brlens = [], []
    for bipartition, edge in tree.bipartition_edge_map.items():
        b_s = str(bipartition)  # binary bitmask

        # Dendropy includes the most trivial bipartitions of
        # All the leaves on one side
        if "1" not in b_s or "0" not in b_s:
            continue
        # In dendropy the 1st taxa is the least significant bit
        # so we need to reverse the bipartition bitmask
        splits.append([float(x) for x in b_s[::-1]])
        brlens.append(edge.length)

    return torch.tensor(splits), torch.tensor(brlens)


def get_split_bitmask(bitstring):
    """Makes sure the least significant bit is always 0"""
    bitmask = int(bitstring, base=2)
    if bitstring[-1] == "1":
        flipper = 2 ** len(bitstring) - 1
        return bitmask ^ flipper
    return bitmask


def build_tree_from_splits(splits, brlens, ids):
    warnings.warn(DeprecationWarning("Please use `splits_to_tree` instead"))

    ns = TaxonNamespace(ids[::-1])
    bitmasks = [get_split_bitmask("".join(f"{x}" for x in row)) for row in splits.int()]
    blens = {k: brlens[i].item() for i, k in enumerate(bitmasks)}
    tiplens = {taxa: ln.item() for taxa, ln in zip(ns[::-1], brlens)}

    t = Tree.from_split_bitmasks(
        bitmasks, ns, split_edge_lengths=blens, is_rooted=False
    )
    t.update_bipartitions()  # make sure that it is well and truly un-rooted before setting tip lengths

    # For some reason Tree.from_split_bitmasks does not use tip edge lengths so we have to set them manually
    for tip_edge in t.leaf_edge_iter():
        tip_edge.length = tiplens[tip_edge.head_node.taxon]

    return t


def splits_to_tree(splits, brlens, ids):
    def as_int(t: torch.Tensor):
        return int("".join(t.int().cpu().numpy().astype("str").tolist()), 2)

    split_ints = [as_int(split) for split in splits.squeeze()]
    edge_lengths = {
        split: length.item()
        for (split, length) in zip(split_ints, brlens.squeeze().cpu())
    }
    current = {1 << i: x for i, x in enumerate(ids[::-1])}  # Star tree
    potential_merges = {
        k1 | k2: (k1, k2) for (k1, k2) in combinations(current, 2)
    }  # possible merges

    for split in split_ints:
        if split in current:  # Trivial tip split
            continue

        # get merge
        child1, child2 = potential_merges[split]

        # remove merged nodes
        node1 = current.pop(child1)
        node2 = current.pop(child2)

        # Update potential merges
        for other in current:
            potential_merges[split | other] = (split, other)

        # Add new parent
        e1 = edge_lengths[child1]
        e2 = edge_lengths[child2]
        current[split] = f"({node1}:{e1},{node2}:{e2})"

    # Un-root no matter what
    tree = Tree.get_from_string(
        "("
        + ",".join([f"{subtree}:{edge_lengths[k]}" for k, subtree in current.items()])
        + ");",
        schema="newick",
        rooting="force-unrooted",
    )
    tree.is_rooted = False
    tree.update_bipartitions()

    return tree.as_string(schema="newick", suppress_rooting=True)


def merges_to_tree(merges, brlens, ids):
    nodes: list[str | None] = [f"{n}:{blen}" for (n, blen) in zip(ids, brlens)]
    for k, (i, j) in enumerate(merges.int().cpu().numpy()):
        u = len(ids) + k
        nodes.append(f"({nodes[i]},{nodes[j]})I_{k}:{brlens[u]}")
        nodes[i], nodes[j] = None, None
    return "(" + ",".join(f"{n}" for n in nodes if n is not None) + ");"


def argmin(x, axis=-1):
    return F.one_hot(torch.argmin(x, dim=axis), list(x.shape)[axis]).float()


def argmax(x, axis=-1):
    return F.one_hot(torch.argmax(x, dim=axis), list(x.shape)[axis]).float()


def sample_oh(x, axis=-1):
    return F.one_hot(x.multinomial(num_samples=1), list(x.shape)[axis]).float()


def mask_from_ignored(ignored):
    return torch.outer(~ignored, ~ignored).bool()


def batch_mask_from_ignored(ignored):
    return torch.matmul(
        (~ignored[:, :, None]).float(), (~ignored[:, None, :]).float()
    ).bool()


def compute_Q(D, ignored):
    """
    Compute the corrected distance matrix Q from the evolutionary
    distance matrix D.
    """
    # False if ignored, True if included
    mask_2d = mask_from_ignored(ignored)

    # We can probably get away with just the torch.sum and broadcasting
    # Instead of building the full nxn matrix...
    R = torch.repeat_interleave(  # type: ignore
        torch.sum(D.masked_fill(~mask_2d, 0), axis=1, keepdim=True),  # type: ignore
        len(D),
        dim=-1,
    )

    # Compute Q and make sure it is symmetric
    # n_active = sum(~ignored)
    n_active = torch.sum(~ignored)
    Q = (n_active - 2) * D - R - R.T
    Q = 0.5 * (Q + Q.T)

    # Fill masked values with inf
    Q.masked_fill_(~mask_2d, torch.inf)
    Q.fill_diagonal_(torch.inf)

    return Q


def batch_compute_Q(D, ignored):
    """
    Compute the corrected distance matrix Q from the evolutionary
    distance matrix D.
    """

    batch_size, n_nodes, _ = D.shape

    # False if ignored, True if included
    mask_2d = batch_mask_from_ignored(ignored)

    # We can probably get away with just the torch.sum and broadcasting
    # Instead of building the full nxn matrix...
    R = torch.repeat_interleave(
        torch.sum(D.masked_fill(~mask_2d, 0), axis=-1, keepdim=True),  # type: ignore
        n_nodes,
        dim=-1,  # type: ignore
    )

    # Compute Q and make sure it is symmetric
    n_active = torch.sum(~ignored) / batch_size
    Q = (n_active - 2) * D - R - R.transpose(-1, -2)
    Q = 0.5 * (Q + Q.transpose(-1, -2))

    # Fill masked values + diagonal with inf
    mask = (~mask_2d) | torch.eye(n_nodes, device=mask_2d.device).bool()
    Q.masked_fill_(mask, torch.inf)

    return Q


def get_new_dist(D, ignored, H_i, H_j, force_nonneg):
    """
    Computes branch lengths from merged nodes i and j to parent node u.
    This also compute the distances from u to all other un-merged taxa
    in the distance matrix D.
    """

    D_masked = D.masked_fill(~mask_from_ignored(ignored), 0)

    # Get distances to merged nodes
    D_i = H_i @ D_masked
    D_j = H_j @ D_masked
    d_ij = H_j @ D_i

    # Compute distances to new parent node
    D_u = 0.5 * (D_i + D_j - d_ij)

    R = torch.sum(D_masked, dim=-1)

    # Compute branch lengths from u to i and j
    # n_active = max(sum(~ignored), 3.0)  # ?
    n_active = torch.max(
        torch.sum(~ignored), torch.tensor([3.0], device=ignored.device)
    )  # ?
    delta_iu = 0.5 * d_ij + (R @ H_i - R @ H_j) / (2 * (n_active - 2))
    delta_ju = d_ij - delta_iu

    if force_nonneg:
        # Force negative branches to 0 and add absolute value to the other branch
        new_delta_iu = torch.clamp_min(delta_iu, min=0) - torch.clamp_max(delta_ju, 0)
        new_delta_ju = torch.clamp_min(delta_ju, min=0) - torch.clamp_max(delta_iu, 0)

        delta_iu = new_delta_iu
        delta_ju = new_delta_ju

    return D_u, delta_iu, delta_ju


def batch_get_new_dist(D, ignored, H_i, H_j, force_nonneg):
    """
    Computes branch lengths from merged nodes i and j to parent node u.
    This also compute the distances from u to all other un-merged taxa
    in the distance matrix D.
    """

    batch_size, _, _ = D.shape

    mask_2d = batch_mask_from_ignored(ignored)
    D_masked = D.masked_fill(~mask_2d, 0)

    # Get distances to merged nodes
    D_i = (H_i[:, None, :] @ D_masked).squeeze((1, 2))
    D_j = (H_j[:, None, :] @ D_masked).squeeze((1, 2))
    d_ij = (H_j[:, None, :] @ D_i[:, :, None]).squeeze((1, 2))

    # Compute distances to new parent node
    D_u = 0.5 * (D_i + D_j - d_ij[:, None])

    R = torch.sum(D_masked, dim=-1)

    # Compute branch lengths from u to i and j
    n_active = torch.max(
        torch.sum(~ignored) / batch_size,
        torch.tensor([3.0], device=ignored.device),
    )
    delta_iu = 0.5 * d_ij + (
        R[:, None, :] @ H_i[:, :, None] - R[:, None, :] @ H_j[:, :, None]
    ).squeeze((1, 2)) / (2 * (n_active - 2))
    delta_ju = d_ij - delta_iu

    if force_nonneg:
        new_delta_iu = torch.clamp_min(delta_iu, min=0) - torch.clamp_max(delta_ju, 0)
        new_delta_ju = torch.clamp_min(delta_ju, min=0) - torch.clamp_max(delta_iu, 0)

        delta_iu = new_delta_iu
        delta_ju = new_delta_ju

    return D_u, delta_iu, delta_ju


def select_merge(Q):
    """Gets One-Hot vectors corresponding to the row and col of the pair to merge (smallest Q value)"""
    flat_onehot = argmin(Q.view(-1))
    square_onehot = flat_onehot.view(Q.shape)
    ones = torch.ones((Q.shape[0], 1), device=Q.device, dtype=torch.float)
    H_i = torch.matmul(square_onehot, ones).squeeze()
    H_j = torch.matmul(ones.transpose(0, 1), square_onehot).transpose(0, 1).squeeze()

    return H_i, H_j


def batch_select_merge(Q):
    """Gets One-Hot vectors corresponding to the row and col of the pair to merge (smallest Q value)"""

    batch_size, node_count, _ = Q.shape

    flat_onehot = argmin(Q.reshape(batch_size, -1))
    square_onehot = flat_onehot.view(Q.shape)
    ones = torch.ones((batch_size, node_count, 1), device=Q.device, dtype=torch.float)
    H_i = torch.matmul(square_onehot, ones).squeeze((1, 2))
    H_j = (
        torch.matmul(ones.transpose(-1, -2), square_onehot)
        .transpose(-1, -2)
        .squeeze((1, 2))
    )

    return H_i, H_j


def compute_merge_logproba(Q, H_i, H_j, ignored, temp=None):
    """
    Given a Q matrix and the 2 nodes to merge, compute the logprobability of merge given Q
    Optional temprature parameter that will multiply the input to the softmin
    """
    mask_2d = mask_from_ignored(ignored)

    # We want 1 version of the Q value for the softmax
    mask_tril = ~(torch.ones_like(Q).triu(diagonal=1) != 0)
    mask = torch.logical_or(~mask_2d, mask_tril)

    Q_masked = Q.clone().contiguous()
    Q_masked.masked_fill_(mask, torch.inf)

    if temp is not None:  # Apply temperature
        Q_masked = temp * Q_masked

    all_logprobas = (-Q_masked.view(-1)).log_softmax(dim=-1).view(Q.shape)

    row, col = H_i.argmax(-1), H_j.argmax(-1)

    return all_logprobas[row, col]


def sample_merge(Q, ignored, temp=None, use_max_proba: bool = False):
    mask_2d = mask_from_ignored(ignored)

    # We want 1 version of the Q value for the softmax
    mask_tril = ~(torch.ones_like(Q).triu(diagonal=1) != 0)
    mask = torch.logical_or(~mask_2d, mask_tril)

    Q_masked = Q.clone().contiguous()
    Q_masked.masked_fill_(mask, torch.inf)

    if temp is not None:  # Apply temperature
        Q_masked = temp * Q_masked

    all_probas = (-Q_masked.view(-1)).softmax(dim=-1)

    if use_max_proba:
        flat_onehot = argmax(all_probas)
    else:
        flat_onehot = sample_oh(all_probas)

    square_onehot = flat_onehot.view(Q.shape)
    ones = torch.ones((Q.shape[0], 1), device=Q.device, dtype=torch.float)
    H_i = torch.matmul(square_onehot, ones).squeeze()
    H_j = torch.matmul(ones.transpose(0, 1), square_onehot).transpose(0, 1).squeeze()

    return H_i, H_j


def batch_sample_merge(Q, ignored, temp=None, use_max_proba: bool = False):
    batch_size, _, _ = Q.shape

    mask_2d = batch_mask_from_ignored(ignored)
    # We want 1 version of the Q value for the softmax
    mask_tril = ~(torch.ones_like(Q[0, :]).triu(diagonal=1) != 0)
    mask = torch.logical_or(~mask_2d, mask_tril.unsqueeze(0))

    Q_masked = Q.clone().contiguous()
    Q_masked.masked_fill_(mask, torch.inf)

    if temp is not None:  # Apply temperature
        Q_masked = temp * Q_masked

    all_probas = (-Q_masked.view(batch_size, -1)).softmax(dim=-1)  # .view(Q.shape)

    if use_max_proba:
        flat_onehot = argmax(all_probas)
    else:
        flat_onehot = sample_oh(all_probas)

    batch_size, node_count, _ = Q.shape

    # flat_onehot = argmin(Q.reshape(batch_size, -1))
    square_onehot = flat_onehot.view(Q.shape)
    ones = torch.ones((batch_size, node_count, 1), device=Q.device, dtype=torch.float)
    H_i = torch.matmul(square_onehot, ones).squeeze((1, 2))
    H_j = (
        torch.matmul(ones.transpose(-1, -2), square_onehot)
        .transpose(-1, -2)
        .squeeze((1, 2))
    )

    return H_i, H_j


def batch_compute_merge_logproba(Q, H_i, H_j, ignored, temp=None):
    """
    Given a Q matrix and the 2 nodes to merge, compute the logprobability of merge given Q.
    Optional temprature parameter that will multiply the input to the softmin
    """

    batch_size, _, _ = Q.shape

    mask_2d = batch_mask_from_ignored(ignored)
    # We want 1 version of the Q value for the softmax
    mask_tril = ~(torch.ones_like(Q[0, :]).triu(diagonal=1) != 0)
    mask = torch.logical_or(~mask_2d, mask_tril.unsqueeze(0))

    Q_masked = Q.clone().contiguous()
    Q_masked.masked_fill_(mask, torch.inf)

    if temp is not None:  # Apply temperature
        Q_masked = temp * Q_masked

    all_logprobas = (-Q_masked.view(batch_size, -1)).log_softmax(dim=-1).view(Q.shape)

    merge_logprobas = (
        H_i.unsqueeze(1) @ all_logprobas.masked_fill(mask, 0)
    ) @ H_j.unsqueeze(-1)

    return merge_logprobas.squeeze()


def init_NJ(distances):
    device = distances.device
    dists = distances.type(torch.float).to(device=device)
    # n_leaves = len(dists)
    n_leaves = dists.shape[0]
    node_count = 2 * n_leaves - 3  # Unrooted splits representation

    # Init splits matrix
    splits = torch.concatenate(
        [
            torch.eye(n_leaves, dtype=torch.float, device=device),
            torch.zeros((n_leaves - 3, n_leaves), dtype=torch.float, device=device),
        ]
    )  # .requires_grad_(True)
    brlens = torch.zeros(
        node_count,
        dtype=torch.float,
        device=device,
    )

    # Pad distance matrix to add future merged nodes
    dists = F.pad(
        dists,
        pad=(0, n_leaves - 3, 0, n_leaves - 3),
        mode="constant",
        value=torch.inf,
    ).fill_diagonal_(0)

    # Mask to track ignored nodes
    ignored_nodes = torch.full(
        [
            node_count,
        ],
        False,
        dtype=torch.bool,
        device=device,
    )
    ignored_nodes[n_leaves:] = True

    return (dists, splits, brlens, ignored_nodes, n_leaves, node_count)


def batch_init_NJ(distances, rooted: bool = False):
    device = distances.device
    dists = distances.type(torch.float).to(device=device)
    batch_size, n_leaves, _ = dists.shape

    internal_nodes = n_leaves - 3
    if rooted:
        internal_nodes += 1

    node_count = n_leaves + internal_nodes

    # Init splits matrix
    splits = torch.concatenate(
        [
            torch.eye(n_leaves, dtype=torch.float, device=device),
            torch.zeros((internal_nodes, n_leaves), dtype=torch.float, device=device),
        ]
    ).repeat(batch_size, 1, 1)

    brlens = torch.zeros(
        node_count,
        dtype=torch.float,
        device=device,
    ).repeat(batch_size, 1)

    # Pad distance matrix to add future merged nodes
    dists = F.pad(
        dists,
        pad=(0, internal_nodes, 0, internal_nodes),
        mode="constant",
        value=torch.inf,
    ).masked_fill_(torch.eye(node_count, device=device).bool(), 0.0)

    # Mask to track ignored nodes
    ignored_nodes = torch.full(
        (batch_size, node_count),
        False,
        dtype=torch.bool,
        device=device,
    )
    ignored_nodes[:, n_leaves:] = True

    return (dists, splits, brlens, ignored_nodes, n_leaves, node_count)


def update_splits(splits, H_i, H_j, n_leaves, iter, node_count):
    hotsplit = H_i + H_j
    splitupdate = torch.zeros(
        (node_count, node_count),
        device=splits.device,
        dtype=torch.float,
    )
    splitupdate[: n_leaves + iter, : n_leaves + iter] = torch.eye(n_leaves + iter)
    splitupdate[n_leaves + iter] = hotsplit
    return splitupdate @ splits


def batch_update_splits(splits, H_i, H_j, n_leaves, iter, node_count):
    hotsplit = H_i + H_j
    splitupdateb = torch.zeros(
        (H_i.shape[0], node_count, node_count),
        device=splits.device,
        dtype=torch.float,
    )
    splitupdateb[:, : n_leaves + iter, : n_leaves + iter] = torch.eye(n_leaves + iter)
    splitupdateb[:, n_leaves + iter] = hotsplit

    return splitupdateb @ splits


def update_brlens(brlens, H_i, H_j, delta_iu, delta_ju):
    return brlens + (H_i * delta_iu) + (H_j * delta_ju)


def batch_update_brlens(brlens, H_i, H_j, delta_iu, delta_ju):
    return brlens + (H_i * delta_iu[:, None]) + (H_j * delta_ju[:, None])


def update_dists(dists, D_u, u):
    D_u[u] = 0.0

    # Set parent distance row
    dists = torch.vstack((dists[:u, :], D_u.unsqueeze(0), dists[u + 1 :, :]))
    # Set parent distance col
    dists = torch.hstack((dists[:, :u], D_u.unsqueeze(-1), dists[:, u + 1 :]))

    return dists


def batch_update_dists(dists, D_u, u):
    D_u[:, u] = 0.0

    # Set parent distance row
    dists = torch.vstack(
        (
            dists[:, :u, :].transpose(0, 1),
            D_u.unsqueeze(0),
            dists[:, u + 1 :, :].transpose(0, 1),
        )
    ).transpose(0, 1)
    # Set parent distance col
    dists = torch.hstack(
        (
            dists[:, :, :u].transpose(1, 2),
            D_u.unsqueeze(1),
            dists[:, :, u + 1 :].transpose(1, 2),
        )
    ).transpose(1, 2)

    return dists


def update_ignored(ignored_nodes, H_i, H_j, u):
    i, j = int(torch.argmax(H_i)), int(torch.argmax(H_j))
    ignored_nodes[i] = True
    ignored_nodes[j] = True
    ignored_nodes[u] = False

    return ignored_nodes


def batch_update_ignored(ignored_nodes, H_i, H_j, u):
    # Udate masked values
    i, j = (
        torch.argmax(H_i, dim=-1),
        torch.argmax(H_j, dim=-1),
    )
    b_idx = torch.arange(ignored_nodes.shape[0])
    ignored_nodes[b_idx, i] = True
    ignored_nodes[b_idx, j] = True
    if u is not None:
        ignored_nodes[b_idx, u] = False

    return ignored_nodes


def soft_nj(
    distances: torch.Tensor,
    force_nonneg: bool = False,
    temp: Optional[float] = None,
):
    """
    Soft version of NJ (very similar to dodonaphy implementation)
    """
    dists, splits, brlens, ignored_nodes, n_leaves, node_count = init_NJ(distances)
    merge_order = []

    logsum = torch.zeros(1, device=dists.device)
    for iter in range(n_leaves - 2):
        # Compute Q matrix
        Q = compute_Q(dists, ignored_nodes)

        # Convert the one hot argmin of the flattened Q to 2 one_hot vectors (row & col)
        H_i, H_j = select_merge(Q)

        # Compute probability of merge
        logsum = logsum + compute_merge_logproba(Q, H_i, H_j, ignored_nodes, temp)

        merge_order.append(torch.stack((H_i, H_j)))

        # Compute new distances
        D_u, delta_iu, delta_ju = get_new_dist(
            dists, ignored_nodes, H_i, H_j, force_nonneg
        )

        u = iter + n_leaves
        if u < 2 * n_leaves - 3:  # We are not done
            splits = update_splits(splits, H_i, H_j, n_leaves, iter, node_count)
            brlens = update_brlens(brlens, H_i, H_j, delta_iu, delta_ju)
            dists = update_dists(dists, D_u, u)
            ignored_nodes = update_ignored(ignored_nodes, H_i, H_j, u)

        else:  # We cluster the last remaining 3 nodes into the "root"
            i, j = int(torch.argmax(H_i)), int(torch.argmax(H_j))
            ignored_nodes[i] = True
            ignored_nodes[j] = True

            H_k = (~ignored_nodes).float()
            delta_ku = H_i @ dists @ H_k - delta_iu

            brlens = brlens + (H_i * delta_iu) + (H_j * delta_ju) + (H_k * delta_ku)

    return splits, brlens, torch.stack(merge_order), logsum


def batch_soft_nj(
    distances: torch.Tensor,
    force_nonneg: bool = False,
    temp: Optional[float] = None,
):
    """
    Soft version of NJ (very similar to dodonaphy implementation)
    """

    dists, splits, brlens, ignored_nodes, n_leaves, node_count = batch_init_NJ(
        distances
    )
    merge_order = []

    batch_size = dists.shape[0]

    logsum = torch.zeros(batch_size, device=dists.device)
    for iter in range(n_leaves - 2):
        # Compute Q matrix
        Q = batch_compute_Q(dists, ignored_nodes)

        # Convert the one hot argmin of the flattened Q to 2 one_hot vectors (row & col)
        H_i, H_j = batch_select_merge(Q)

        # Compute probability of merge
        logsum = logsum + batch_compute_merge_logproba(Q, H_i, H_j, ignored_nodes, temp)

        merge_order.append(torch.stack((H_i, H_j)))  # TODO: CHECK THIS

        # Compute new distances
        D_u, delta_iu, delta_ju = batch_get_new_dist(
            dists, ignored_nodes, H_i, H_j, force_nonneg
        )

        u = iter + n_leaves
        if u < 2 * n_leaves - 3:  # We are not done
            splits = batch_update_splits(splits, H_i, H_j, n_leaves, iter, node_count)
            brlens = batch_update_brlens(brlens, H_i, H_j, delta_iu, delta_ju)
            dists = batch_update_dists(dists, D_u, u)
            ignored_nodes = batch_update_ignored(ignored_nodes, H_i, H_j, u)

        else:  # We cluster the last remaining 3 nodes into the "root"
            i, j = (
                torch.argmax(H_i, dim=-1),
                torch.argmax(H_j, dim=-1),
            )
            b_idx = torch.arange(batch_size)
            ignored_nodes[b_idx, i] = True
            ignored_nodes[b_idx, j] = True

            H_k = (~ignored_nodes).float()
            delta_ku = (H_i.unsqueeze(1) @ dists @ H_k.unsqueeze(-1)).squeeze(
                (1, 2)
            ) - delta_iu

            brlens = (
                brlens
                + (H_i * delta_iu[:, None])
                + (H_j * delta_ju[:, None])
                + (H_k * delta_ku[:, None])
            )

    return splits, brlens, torch.stack(merge_order).permute(2, 0, 1, 3), logsum


def tree_probability_with_merges(
    distances: torch.Tensor, merge_order: torch.Tensor, temp: Optional[float] = None
) -> tuple[torch.Tensor, bool]:
    dists, _, _, ignored_nodes, n_leaves, _ = init_NJ(distances)

    logsum = torch.zeros(1, device=dists.device)
    has_neg = False
    for iter, merge in enumerate(merge_order.squeeze()):
        # Get merge from predefined order
        (H_i, H_j) = merge

        # Compute Q matrix
        Q = compute_Q(dists, ignored_nodes)

        # Compute probability of merge
        logsum = logsum + compute_merge_logproba(Q, H_i, H_j, ignored_nodes, temp)

        # Compute new distances
        D_u, _, _ = get_new_dist(dists, ignored_nodes, H_i, H_j, False)
        if (D_u.masked_fill(ignored_nodes, 0) < 0).any():
            has_neg = True

        u = iter + n_leaves
        if u < 2 * n_leaves - 3:  # Not last merge
            dists = update_dists(dists, D_u, u)
            ignored_nodes = update_ignored(ignored_nodes, H_i, H_j, u)

    return logsum, has_neg


def tree_probability_with_merges_branches(
    distances: torch.Tensor,
    merge_order: torch.Tensor,
    temp: Optional[float] = None,
):
    dists, _, brlens, ignored_nodes, n_leaves, _ = init_NJ(distances)

    logsum = torch.zeros(1, device=dists.device)
    for iter, merge in enumerate(merge_order.squeeze()):
        # Get merge from predefined order
        (H_i, H_j) = merge

        # Compute Q matrix
        Q = compute_Q(dists, ignored_nodes)
        # Compute new distances
        D_u, delta_iu, delta_ju = get_new_dist(dists, ignored_nodes, H_i, H_j, False)

        # Compute probability of merge
        logsum = logsum + compute_merge_logproba(Q, H_i, H_j, ignored_nodes, temp)

        u = iter + n_leaves
        if u < 2 * n_leaves - 3:  # Not last merge
            dists = update_dists(dists, D_u, u)
            brlens = batch_update_brlens(brlens, H_i, H_j, delta_iu, delta_ju)
            ignored_nodes = update_ignored(ignored_nodes, H_i, H_j, u)
        else:
            i, j = int(torch.argmax(H_i)), int(torch.argmax(H_j))
            ignored_nodes[i] = True
            ignored_nodes[j] = True

            H_k = (~ignored_nodes).float()
            delta_ku = H_i @ dists @ H_k - delta_iu

            brlens = brlens + (H_i * delta_iu) + (H_j * delta_ju) + (H_k * delta_ku)

    return logsum, brlens


def batch_tree_probability_with_merges(
    distances: torch.Tensor, merges: torch.Tensor, temp: Optional[float] = None
) -> tuple[torch.Tensor, bool]:
    dists, _, _, ignored_nodes, n_leaves, _ = batch_init_NJ(distances)

    batch_size = dists.shape[0]
    logsum = torch.zeros(batch_size, device=dists.device)
    has_neg = False

    # bs, n_iter, 2, n_splits -> n_iter, 2, bs, n_splits
    merges = merges.permute(1, 2, 0, 3)
    for iter, merge in enumerate(merges):
        # Get merge from predefined order
        (H_i, H_j) = merge

        # Compute Q matrix
        Q = batch_compute_Q(dists, ignored_nodes)

        # Compute probability of merge
        logsum = logsum + batch_compute_merge_logproba(Q, H_i, H_j, ignored_nodes, temp)

        # Compute new distances
        D_u, _, _ = batch_get_new_dist(dists, ignored_nodes, H_i, H_j, False)
        if (D_u.masked_fill(ignored_nodes, 0) < 0).any():
            has_neg = True

        u = iter + n_leaves
        if u < 2 * n_leaves - 3:  # not last merge
            dists = batch_update_dists(dists, D_u, u)
            ignored_nodes = batch_update_ignored(ignored_nodes, H_i, H_j, u)

    return logsum, has_neg


def batch_tree_probability_with_merges_branches(
    distances: torch.Tensor, merges: torch.Tensor, temp: Optional[float] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    dists, _, brlens, ignored_nodes, n_leaves, _ = batch_init_NJ(distances)

    batch_size = dists.shape[0]
    logsum = torch.zeros(batch_size, device=dists.device)

    # bs, n_iter, 2, n_splits -> n_iter, 2, bs, n_splits
    merges = merges.permute(1, 2, 0, 3)
    for iter, merge in enumerate(merges):
        # Get merge from predefined order
        (H_i, H_j) = merge

        # Compute Q matrix
        Q = batch_compute_Q(dists, ignored_nodes)

        # Compute probability of merge
        logsum = logsum + batch_compute_merge_logproba(Q, H_i, H_j, ignored_nodes, temp)

        # Compute new distances
        D_u, delta_iu, delta_ju = batch_get_new_dist(
            dists, ignored_nodes, H_i, H_j, False
        )

        u = iter + n_leaves
        if u < 2 * n_leaves - 3:  # not last merge
            brlens = batch_update_brlens(brlens, H_i, H_j, delta_iu, delta_ju)
            dists = batch_update_dists(dists, D_u, u)
            ignored_nodes = batch_update_ignored(ignored_nodes, H_i, H_j, u)
        else:  # We cluster the last remaining 3 nodes into the "root"
            i, j = (
                torch.argmax(H_i, dim=-1),
                torch.argmax(H_j, dim=-1),
            )
            b_idx = torch.arange(batch_size)
            ignored_nodes[b_idx, i] = True
            ignored_nodes[b_idx, j] = True

            H_k = (~ignored_nodes).float()
            delta_ku = (H_i.unsqueeze(1) @ dists @ H_k.unsqueeze(-1)).squeeze(
                (1, 2)
            ) - delta_iu

            brlens = (
                brlens
                + (H_i * delta_iu[:, None])
                + (H_j * delta_ju[:, None])
                + (H_k * delta_ku[:, None])
            )

    return logsum, brlens


def sample_tree_topology(
    distances: torch.Tensor, taxa_ids: list[str], use_max_proba: bool = False
):
    dists, _, _, ignored_nodes, n_leaves, _ = init_NJ(distances)

    nodes = [x for x in taxa_ids] + [
        "",
    ] * (n_leaves - 2)
    tree = ""
    for iter in range(n_leaves - 2):
        # Compute Q matrix
        Q = compute_Q(dists, ignored_nodes)

        # Convert the one hot argmin of the flattened Q to 2 one_hot vectors (row & col)
        H_i, H_j = sample_merge(Q, ignored_nodes, None, use_max_proba)
        i, j = int(torch.argmax(H_i)), int(torch.argmax(H_j))

        # Compute new distances
        D_u, _, _ = get_new_dist(dists, ignored_nodes, H_i, H_j, False)

        u = iter + n_leaves
        if u < 2 * n_leaves - 3:  # We are not done
            dists = update_dists(dists, D_u, u)
            ignored_nodes = update_ignored(ignored_nodes, H_i, H_j, u)
            nodes[u] = f"({nodes[i]},{nodes[j]})"
        else:  # We cluster the last remaining 3 nodes into the "root"
            ignored_nodes[i] = True
            ignored_nodes[j] = True
            k = int(torch.argmax((~ignored_nodes).float()))
            tree = f"({nodes[i]},{nodes[j]},{nodes[k]});"

    return tree


def batch_compute_tree_probability_with_gamma_brlens(
    model: EvoPF,
    sequence_embeddings: torch.Tensor,  # From EvoPF
    distances: torch.Tensor,  # From EvoPF
    brlens: torch.Tensor,  # From the real tree
    merges: torch.Tensor,  # From the real tree
    temp: Optional[float] = None,
    topo_only: bool = False,
    ignore_topo: bool = False,
    upper_beta_c: Optional[float] = None,
    lgnrl_mu_x_min: Optional[float] = None,
    lgnrl_mu_x_max: Optional[float] = None,
    lgnrl_sigma_x_min: Optional[float] = None,
    lgnrl_sigma_x_max: Optional[float] = None,
    lgnrl_lsigma_div: Optional[float] = None,
    verbose: bool = True,
):
    dists, _, _, ignored_nodes, n_leaves, node_count = batch_init_NJ(
        distances, rooted=True
    )
    constraints = torch.zeros_like(dists)

    # [B,d_m,n] -> [B,2n-2,d_m]
    embs = F.pad(
        sequence_embeddings,
        (0, node_count - n_leaves),
        mode="constant",
        value=0.0,
    ).transpose(-1, -2)
    batch_size = dists.size(0)

    topo_logsum = torch.zeros(batch_size, device=dists.device)
    gamma_logsum = torch.zeros(batch_size, device=dists.device)
    beta_logsum = torch.zeros(batch_size, device=dists.device)

    # bs, n_iter, 2, n_splits -> n_iter, 2, bs, n_splits
    merges = merges.permute(1, 2, 0, 3)
    for iter, merge in enumerate(merges):
        # Get merge from predefined order
        (H_i, H_j) = merge

        # Compute Q matrix
        Q = batch_compute_Q(dists, ignored_nodes)

        # Compute probability of merge
        if not ignore_topo:
            topo_logsum = topo_logsum + batch_compute_merge_logproba(
                Q, H_i, H_j, ignored_nodes, temp
            )

        # Compute new distances
        D_u, _, _ = batch_get_new_dist(dists, ignored_nodes, H_i, H_j, False)

        # BRANCH LENGTH OPTIM
        if not topo_only:
            # Compute parent embedding
            E_u = model.embed_parent_node(
                H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
            ).squeeze(1)
            print("E_u", E_u.shape)
            # Apply normalization to parent embedding
            E_u = model.e_norm(E_u)

            # Compute branch probabilities
            l_i = (H_i.unsqueeze(1) @ brlens.unsqueeze(-1)).squeeze((1, 2))
            l_j = (H_j.unsqueeze(1) @ brlens.unsqueeze(-1)).squeeze((1, 2))
            s = l_i + l_j
            balance = l_i / s
            M_ij = (H_i.unsqueeze(1) @ constraints @ H_j.unsqueeze(-1)).squeeze((1, 2))

            stdn = torch.distributions.normal.Normal(
                torch.zeros_like(s), torch.ones_like(s)
            )

            # params: [l_mu, l_mu/l_sigma]
            l_params = model.predict_lognormal_params(E_u).squeeze(1)
            # l_params = model.s_batch_norm(l_params)

            l_mux = l_params[:, 0]
            l_ratio = l_params[:, 1]
            l_sigmax = l_mux - l_ratio

            # l_mu = bounded_output_log_exp(ll_mu)
            # l_sigma = bounded_output_log_exp(ll_sigma, min_val=1e-2)
            mu_x = torch.exp(l_mux)
            sigma_x = torch.exp(l_sigmax)

            # l_params = model.s_batch_norm(l_params)
            # l_mu = l_params[:, 0]
            # l_sigma = l_params[:, 0] / (l_params[:, 1] + 1e-3)
            # l_sigma = 1e-4 + F.softplus(l_sigma)

            # We predict mu_x and sigma^2_x
            # mu_x = empirical_params[:, 0].clamp(min=lgnrl_mu_x_min, max=lgnrl_mu_x_max)
            # sigma_x = (
            #     empirical_params[:, 1].clamp(
            #         min=lgnrl_sigma_x_min, max=lgnrl_sigma_x_max
            #     )
            #     * empirical_params[:, 0]
            # )

            l_mu, l_sigma = lognormal_params_from_empirical(mu_x, sigma_x)
            if verbose:
                print("l_mu", l_mu)
                print("l_sigma", l_sigma)

                print("s", s)
                print("Mode s", torch.exp(l_mu - l_sigma**2))
                print("s MRE", (s - torch.exp(l_mu - l_sigma**2)).abs() / s)

            # make sure mu/sigma < 3 with hard eps
            # l_sigma = l_sigma.maximum(l_mu.abs() / (lgnrl_lsigma_div or 1.0))
            # print("l_sigma_corr", l_sigma)

            # BALANCE version
            # beta_params = model.predict_beta_params(
            #     H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
            # ).squeeze(
            #     1
            # )  # both in [0,1]

            # beta_mode = F.sigmoid(
            #     model.b_w_batch_norm(beta_params[:, 0].unsqueeze(-1)).squeeze(-1)
            # )
            # beta_concentration = F.sigmoid(
            #     model.b_c_batch_norm(beta_params[:, 1].unsqueeze(-1)).squeeze(-1)
            # )

            # if verbose:
            #     print("balance", balance)
            #     print("b_w", beta_mode)
            #     print("b_c", beta_concentration)
            #     print("b MRE", (balance - beta_mode).abs() / balance)
            # beta_a = 1 + beta_concentration * beta_mode
            # beta_b = 1 + beta_concentration * (1 - beta_mode)

            # DIFF version
            # params_b = F.sigmoid(model.brlen_beta(
            #     H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
            # ).squeeze(1))
            params_b = model.brlen_beta(
                H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
            ).squeeze(1)
            # print('pb shape', params_b.shape)
            # logit_b = model.b_w_batch_norm(params_b[:,].unsqueeze(1)).squeeze(1)
            # logit_b_ratio = model.b_c_batch_norm(params_b[:,1].unsqueeze(1)).squeeze(1)
            logit_b_sg = model.b_c_batch_norm(params_b[:, 1].unsqueeze(1)).squeeze(1)
            # print('logit_b', logit_b)

            # mu_b = params_b[:,0]
            # mu_b = s * torch.tanh(logit_b)
            # mu_b = bounded_output_log_exp(logit_b, min_val=1e-8, max_val=s-1e-8)
            # mu_b = bounded_output_log_exp(params_b[:, 0], min_val=1e-8, max_val=s-1e-8)
            mu_b = bounded_output(params_b[:, 0], min_val=1e-8, max_val=s - 1e-8)
            sg_b = bounded_output_log_exp(logit_b_sg, min_val=1e-1, max_val=1e1)
            # ratio_b = bounded_output_log_exp(params_b[:, 1], min_val=1e-2, max_val=1e2)
            # sg_b = mu_b.abs() / ratio_b
            # sg_b = s

            # mu_b = (params_b[:, 0] - 0.5) * (2 * s)  # in [-s,s]
            # mu_b = torch.clamp(10**params_b[:, 0], min=-s, max=s)  # in [0,1]
            # sg_b = torch.ones_like(mu_b)
            # sg_b = torch.clamp(params_b[:, 1], min=1e-4)
            # if verbose:
            #     print('li-lj', l_i - l_j)
            #     print('mu_b', mu_b)
            #     print('diff MRE', ((l_i - l_j - mu_b)/(l_i - l_j)).abs())

            if verbose:
                # print("diff", l_i - l_j)
                print("li", l_i)
                print("mu_b", mu_b)
                print("logit_b_sg", logit_b_sg)
                print("sg_b", sg_b)
                # print("diff MRE", ((l_i - l_j - mu_b)/(l_i - l_j)).abs())
                # print("diff MRE", ((l_i - l_j - mu_b)/s).abs())
                print("li MSE", (l_i - mu_b) ** 2)

            gamma_logprob = lognormal_log_pdf(s - M_ij, l_mu, l_sigma)
            # gamma_logprob = lognormal_log_pdf(s, l_mu, l_sigma)
            # beta_logprob = truncated_normal_log_pdf(l_i - l_j, mu_b, sg_b, stdn, -s, s)
            beta_logprob = truncated_normal_log_pdf(l_i, mu_b, sg_b, stdn, 0, s)

            # beta_logprob = beta_log_pdf(
            #     balance, torch.cat([beta_a.unsqueeze(1), beta_b.unsqueeze(1)], dim=1)
            # )

            beta_logsum = beta_logsum + beta_logprob
            gamma_logsum = gamma_logsum + gamma_logprob

        u = iter + n_leaves
        if u < node_count:  # not last merge
            # Update "distances", ignored_nodes and sequence embeddings
            dists = batch_update_dists(dists, D_u, u)
            ignored_nodes = batch_update_ignored(ignored_nodes, H_i, H_j, u)

            if not topo_only:
                embs[:, u] = E_u

                # update constraints
                constraints = constraints.max(s[:, None, None])
                mask2d = batch_mask_from_ignored(ignored_nodes)
                constraints.masked_fill_(~mask2d, 0)
                constraints[:, u, :] = 0
                constraints[:, :, u] = 0

    return topo_logsum, gamma_logsum, beta_logsum


def bounded_output_log_exp(z, min_val=1e-4, max_val=10.0):
    log_min = torch.log(torch.tensor(min_val))
    log_max = torch.log(torch.tensor(max_val))
    return torch.exp(log_min + (log_max - log_min) * torch.sigmoid(z))


def bounded_output(z, min_val=1e-4, max_val=10.0):
    return min_val + (max_val - min_val) * torch.sigmoid(z)


def scaling_to_sum_constraint(xp, yp, s, min_val=1e-4):
    """
    Projects (xp, yp) onto the region:
        x > 0, y > 0, x + y >= s
    using a differentiable approximation.

    Returns:
        (x, y): projected values
    """

    # Step 1: Soft-clamp to ensure positivity
    x = min_val + torch.nn.functional.softplus(xp)
    y = min_val + torch.nn.functional.softplus(yp)

    sum_xy = x + y
    scale = torch.where(sum_xy < s, s / sum_xy, torch.ones_like(s))

    # Step 2: Rescale only when sum is too small
    x_proj = x * scale
    y_proj = y * scale

    return x_proj, y_proj


def project_xy_lagrange(xp, yp, s):
    """
    Project (xp, yp) onto the feasible set x + y >= s
    using Lagrange projection on the boundary x + y = s when needed.

    Args:
        xp: Tensor of shape (B,) - initial x values
        yp: Tensor of shape (B,) - initial y values
        s: Tensor of shape (B,) - lower bound on x + y

    Returns:
        x_proj, y_proj: Tensors of shape (B,) - projected values
    """
    sum_xy = xp + yp
    mask = sum_xy < s  # Need to project only if the constraint is violated

    # Compute projection only where needed
    delta = xp - yp
    x_proj = (s + delta) / 2
    y_proj = (s - delta) / 2

    # Only apply projection where needed
    x_out = torch.where(mask, x_proj, xp)
    y_out = torch.where(mask, y_proj, yp)

    return x_out, y_out


def batch_compute_tree_probability_with_mse_brlens(
    model: EvoPF,
    sequence_embeddings: torch.Tensor,  # From EvoPF
    distances: torch.Tensor,  # From EvoPF
    brlens: torch.Tensor,  # From the real tree
    merges: torch.Tensor,  # From the real tree
    temp: Optional[float] = None,
    topo_only: bool = False,
    ignore_topo: bool = False,
    verbose: bool = True,
):
    dists, _, _, ignored_nodes, n_leaves, node_count = batch_init_NJ(
        distances, rooted=True
    )
    constraints = torch.zeros_like(dists)

    # [B,d_m,n] -> [B,2n-2,d_m]
    embs = F.pad(
        sequence_embeddings,
        (0, node_count - n_leaves),
        mode="constant",
        value=0.0,
    ).transpose(-1, -2)
    batch_size = dists.size(0)

    topo_logsum = torch.zeros(batch_size, device=dists.device)
    gamma_logsum = torch.zeros(batch_size, device=dists.device)
    beta_logsum = torch.zeros(batch_size, device=dists.device)

    # bs, n_iter, 2, n_splits -> n_iter, 2, bs, n_splits
    merges = merges.permute(1, 2, 0, 3)
    for iter, merge in enumerate(merges):
        # Get merge from predefined order
        (H_i, H_j) = merge

        # Compute Q matrix
        Q = batch_compute_Q(dists, ignored_nodes)

        # Compute probability of merge
        if not ignore_topo:
            topo_logsum = topo_logsum + batch_compute_merge_logproba(
                Q, H_i, H_j, ignored_nodes, temp
            )

        # Compute new distances
        D_u, _, _ = batch_get_new_dist(dists, ignored_nodes, H_i, H_j, False)

        # BRANCH LENGTH OPTIM
        if not topo_only:
            # Compute parent embedding
            E_u = model.embed_parent_node(
                H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
            ).squeeze(1)
            E_u = model.e_norm(E_u)

            # Compute branch probabilities
            l_i = (H_i.unsqueeze(1) @ brlens.unsqueeze(-1)).squeeze((1, 2))
            l_j = (H_j.unsqueeze(1) @ brlens.unsqueeze(-1)).squeeze((1, 2))
            s = l_i + l_j
            balance = l_i / s
            M_ij = (H_i.unsqueeze(1) @ constraints @ H_j.unsqueeze(-1)).squeeze((1, 2))

            stdn = torch.distributions.normal.Normal(
                torch.zeros_like(s), torch.ones_like(s)
            )

            # params_s = F.sigmoid(model.brlen_lognormal(E_u).squeeze(1))
            # params_s = model.brlen_lognormal(E_u)
            # mu_s = 1e-4 + F.softplus(params_s[:, 0])
            # logit_s = model.s_batch_norm(params_s[:,0].unsqueeze(1)).squeeze(1)
            # print("ps shape", params_s.shape)
            # logit_s = model.s_batch_norm(params_s).squeeze(1)

            # mu_s = bounded_output(logit_s[:, 0], min_val=(M_ij + 1e-8), max_val=20)
            # mu_s = bounded_output(params_s[:, 0], min_val=(M_ij + 1e-8), max_val=20)
            # sg_s = bounded_output_log_exp(logit_s[:, 1], min_val=1e-1, max_val=1e1)

            # sg_s = bounded_output_log_exp(logit_s[:, 1])
            # mu_s = 10**params_s[:, 0]
            # mu_s = params_s[:, 0] * 10  # in [0,10]
            # sg_s = torch.clamp(torch.sqrt(mu_s), min=1)
            # sg_s = torch.clamp(params_s[:, 1] * 2, min=5e-1) # in [1e-1,2]
            # sg_s = torch.ones_like(mu_s)

            # if verbose:
            #     print("s", s)
            #     print("mu_s", mu_s)
            #     print("logit_s", logit_s[:,0])
            #     print("sg_s", sg_s)
            #     print("s MSE", (s - mu_s)**2)

            params_b = model.brlen_beta(
                H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
            ).squeeze(1)

            # logit_li_sg = model.b_c_batch_norm(params_b[:,1].unsqueeze(1)).squeeze(1)
            # logit_li_mu = model.b_w_batch_norm(params_b[:,0].unsqueeze(1)).squeeze(1)
            # mu_li = bounded_output(params_b[:, 0], min_val=1e-4, max_val=M_ij - 1e-4)
            # mu_lj = bounded_output(params_b[:, 1], min_val=1e-4, max_val=M_ij - 1e-4)
            # mu_li = bounded_output(logit_li_mu, min_val=1e-8, max_val=s-1e-8)
            # mu_li = 1e-4 + F.softplus(params_b[:, 0])
            # mu_lj = 1e-4 + F.softplus(params_b[:, 1])
            # mu_li, mu_lj = project_xy_lagrange(mu_li, mu_lj, M_ij)
            mu_li, mu_lj = scaling_to_sum_constraint(
                params_b[:, 0], params_b[:, 1], M_ij
            )

            # sg_li = bounded_output_log_exp(logit_li_sg, min_val=1e-1, max_val=1e1)
            # sg_li = torch.ones_like(mu_li)

            if verbose:
                # print("diff", l_i - l_j)
                print("li", l_i)
                print("mu_li", mu_li)
                # print("logit_li_mu", logit_li_mu)
                # print("logit_li_sg", logit_li_sg)
                # print("sg_li", sg_li)
                print("li MSE", (l_i - mu_li) ** 2)
                print("lj", l_j)
                print("mu_lj", mu_lj)
                print("lj MSE", (l_j - mu_lj) ** 2)

            # beta_mode = F.sigmoid(
            #     model.b_w_batch_norm(beta_params[:, 0].unsqueeze(-1)).squeeze(-1)
            # )
            # beta_concentration = F.sigmoid(
            #     model.b_c_batch_norm(beta_params[:, 1].unsqueeze(-1)).squeeze(-1)
            # )

            # if verbose:
            #     print("balance", balance)
            #     print("b_w", beta_mode)
            #     print("b_c", beta_concentration)
            #     print("b MRE", (balance - beta_mode).abs() / balance)
            # beta_a = 1 + beta_concentration * beta_mode
            # beta_b = 1 + beta_concentration * (1 - beta_mode)

            # params_b = F.sigmoid(model.brlen_beta(
            #     H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
            # ).squeeze(1))
            # params_b = model.brlen_beta(
            #     H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
            # ).squeeze(1)
            # print('pb shape', params_b.shape)
            # # logit_b = model.b_batch_norm(params_b[:,0].unsqueeze(1)).squeeze(1)
            # logit_b = model.b_batch_norm(params_b)
            #
            # mu_b = s * torch.tanh(logit_b[:, 0])
            # sg_b = bounded_output_log_exp(logit_b[:,1])

            # mu_b = (params_b[:, 0] - 0.5) * (2 * s)  # in [-s,s]
            # mu_b = torch.clamp(10**params_b[:, 0], min=-s, max=s)  # in [0,1]
            # sg_b = torch.ones_like(mu_b)
            # sg_b = torch.clamp(params_b[:, 1] * 2, min=5e-1)  # in [1e-1, 2]
            # if verbose:
            #     print('li-lj', l_i - l_j)
            #     print('mu_b', mu_b)
            #     print('diff MRE', ((l_i - l_j - mu_b)/(l_i - l_j)).abs())

            li_logprob = -((l_i - mu_li) ** 2)
            lj_logprob = -((l_j - mu_lj) ** 2)

            # cherry_logprob = truncated_normal_log_pdf(s , mu_s, sg_s, stdn, a=M_ij)
            # cherry_logprob = truncated_normal_log_pdf(s, mu_s, sg_s, stdn, a=0)
            # balance_logprob = truncated_normal_log_pdf(l_i - l_j, mu_b, sg_b, stdn, -s, s)
            # balance_logprob = truncated_normal_log_pdf(balance, mu_b, sg_b, stdn, 0, 1)
            # balance_logprob = beta_log_pdf(
            #     balance, torch.cat([beta_a.unsqueeze(1), beta_b.unsqueeze(1)], dim=1)
            # )
            # li_logprob = truncated_normal_log_pdf(l_i , mu_li, sg_li, stdn, a=1e-8, b=(s - 1e-8))

            # # cherry length contrained in [0,+inf[
            # cherry_logprob = (-(((s - M_ij) - mu_s) ** 2) / 2.0) - torch.log(
            #     1.0 - stdn.cdf(0 - mu_s)
            # )
            # balance_logprob = ((-(l_i - l_j) - mu_b) ** 2 / 2.0) - torch.log(
            #     stdn.cdf(s - mu_b) - stdn.cdf(-s - mu_b)
            # )

            gamma_logsum = gamma_logsum + lj_logprob
            beta_logsum = beta_logsum + li_logprob

        u = iter + n_leaves
        if u < node_count:  # not last merge
            # Update "distances", ignored_nodes and sequence embeddings
            dists = batch_update_dists(dists, D_u, u)
            ignored_nodes = batch_update_ignored(ignored_nodes, H_i, H_j, u)

            if not topo_only:
                embs[:, u] = E_u

                # update constraints
                constraints = constraints.max(s[:, None, None])
                mask2d = batch_mask_from_ignored(ignored_nodes)
                constraints.masked_fill_(~mask2d, 0)
                constraints[:, u, :] = 0
                constraints[:, :, u] = 0

    return topo_logsum, gamma_logsum, beta_logsum


def truncated_normal_log_pdf(x, mu, sg, stdn, a=None, b=None):
    """
    Normal distribution truncated in to [a,b]
    if a or b are None then the corresponding bound is infinite
    """
    a_Phi = stdn.cdf((a - mu) / sg) if a is not None else torch.zeros_like(mu)
    b_Phi = stdn.cdf((b - mu) / sg) if b is not None else torch.ones_like(mu)
    log_x_phi = -0.5 * (((x - mu) / sg).pow(2.0))

    # return log_x_phi - sg.log() - torch.log(b_Phi - a_Phi)
    return log_x_phi - sg.log()


def batch_sample_tree_with_brlens(
    model: EvoPF,
    sequence_embeddings: torch.Tensor,  # From EvoPF
    distances: torch.Tensor,  # From EvoPF
    temp: Optional[torch.Tensor] = None,
    use_max_proba: bool = False,
):
    dists, splits, brlens, ignored_nodes, n_leaves, node_count = batch_init_NJ(
        distances, rooted=True
    )
    constraints = torch.zeros_like(dists)

    # [B,d_m,n] -> [B,2n-2,d_m]
    embs = F.pad(
        sequence_embeddings,
        (0, node_count - n_leaves),
        mode="constant",
        value=0.0,
    ).transpose(-1, -2)
    batch_size = dists.size(0)

    topo_logsum = torch.zeros(batch_size, device=dists.device)
    brlen_logsum = torch.zeros(batch_size, device=dists.device)

    # merges = []
    # brlens

    for iter in range(node_count):
        # Compute Q matrix
        Q = batch_compute_Q(dists, ignored_nodes)

        # Sample merge
        (H_i, H_j) = batch_sample_merge(Q, ignored_nodes, temp, use_max_proba)

        # Compute probability of merge
        topo_logsum = topo_logsum + batch_compute_merge_logproba(
            Q, H_i, H_j, ignored_nodes, temp
        )

        # Compute new distances
        D_u, _, _ = batch_get_new_dist(dists, ignored_nodes, H_i, H_j, False)

        # Compute parent embedding
        E_u = model.embed_parent_node(
            H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
        ).squeeze(1)

        E_u = model.e_norm(E_u).squeeze(1)

        # Sample branch lengths
        # gamma_params = model.predict_lognormal_params(E_u).squeeze(1)
        # l_params = model.predict_lognormal_params(E_u).squeeze(1)

        # l_mux = l_params[:, 0]
        # l_ratio = l_params[:, 1]
        # l_sigmax = l_mux - l_ratio

        # mu_x = torch.exp(l_mux)
        # sigma_x = torch.exp(l_sigmax)

        # l_mu, l_sigma = lognormal_params_from_empirical(mu_x,sigma_x)

        # print("l_params", l_params)
        # print("l_mu", l_mu)
        # print("l_sigma", l_sigma)

        # beta_params = model.predict_beta_params(
        #     H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
        # ).squeeze(
        #     1
        # )

        # print("beta_params", beta_params)

        # beta_mode = model.b_w_batch_norm(beta_params[:, 0].unsqueeze(-1)).squeeze(-1)
        # print("b_w_logit", beta_mode)
        # beta_mode = F.sigmoid(beta_mode)
        # print("b_w", beta_mode)
        # beta_concentration = model.b_c_batch_norm(
        #     beta_params[:, 1].unsqueeze(-1)
        # ).squeeze(-1)
        # print("b_c_logit", beta_mode)
        # beta_concentration = F.sigmoid(beta_concentration)
        # print("b_c", beta_mode)

        # print("beta_a", 1 + beta_concentration * beta_mode)
        # print("beta_a", 1 + beta_concentration * (1 - beta_mode))

        # beta_a = torch.clamp(1 + beta_concentration * beta_mode, min=1.0 + 1e-6)
        # beta_b = torch.clamp(1 + beta_concentration * (1 - beta_mode), min=1.0 + 1e-6)

        # print("clamped beta_a", beta_a)
        # print("clamped beta_a", beta_b)

        # mu_b = beta_params[:,0]

        # sg_b = torch.ones_like(mu_b)

        # gamma = torch.distributions.Gamma(gamma_params[:, 0], gamma_params[:, 1])
        # gamma = torch.distributions.LogNormal(l_mu, l_sigma)
        # gamma = torch.distributions.LogNormal(
        #     gamma_params[:, 0], 1e-4 + F.softplus(gamma_params[:0,] / (gamma_params[:, 1] + 1e-3))
        # )
        # print(f'alpha={beta_a}, beta={beta_b}')
        # beta = torch.distributions.Beta(beta_a, beta_b)

        M_ij = (H_i.unsqueeze(1) @ constraints @ H_j.unsqueeze(-1)).squeeze((1, 2))

        # params_s = model.brlen_lognormal(E_u)
        # mu_s = bounded_output(params_s[:, 0], min_val=(M_ij + 1e-8), max_val=20)

        # if use_max_proba:
        #     s = mu_s
        #     #s = M_ij + mu_s#gamma.mode
        #     # s = M_ij + gamma.mean
        #     #balance = beta_mode  # beta.mode
        # else:
        #     s = M_ij + gamma.sample()
        #     #balance = beta.sample()

        params_b = model.brlen_beta(
            H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
        ).squeeze(1)

        mu_li = 1e-4 + F.softplus(params_b[:, 0])
        mu_lj = 1e-4 + F.softplus(params_b[:, 1])
        mu_li, mu_lj = project_xy_lagrange(mu_li, mu_lj, M_ij)

        beta_logprob = 0  # truncated_normal_log_pdf(l_i, mu_b, sg_b, stdn, 0, s)

        # mu_b = bounded_output_log_exp(beta_params[:, 0], min_val=1e-8, max_val=s-1e-8)
        # mu_b = bounded_output(beta_params[:, 0], min_val=1e-8, max_val=s-1e-8)
        # logit_b_sg = model.b_c_batch_norm(beta_params[:,1].unsqueeze(1)).squeeze(1)
        # sg_b = bounded_output_log_exp(logit_b_sg, min_val=1e-1, max_val=1e1)
        # mu_li = bounded_output(params_b[:, 0], min_val=1e-8, max_val=s-1e-8)

        print("mu_li", mu_li, flush=True)
        # mu_b = torch.clamp(mu_b, min=1e-8*torch.ones_like(mu_b), max=(s-1e-8))

        l_i = mu_li
        l_j = mu_lj
        s = l_i + l_j
        # l_j = s - l_i
        print("li", l_i, flush=True)
        print("lj", l_j, flush=True)
        # l_i = balance * s
        # l_j = (1 - balance) * s

        # Compute probabilities for branches
        # gamma_logprob = gamma_log_pdf(s, gamma_params[:, 0], 1e-4 + F.softplus(gamma_params[:, 1]))
        gamma_logprob = 0  # lognormal_log_pdf(s, l_mu, l_sigma)
        # print("balance", balance)
        # beta_logprob = beta_log_pdf(
        #     balance, torch.cat([beta_a.unsqueeze(1), beta_b.unsqueeze(1)], dim=1)
        # )

        brlen_logsum = brlen_logsum + gamma_logprob + beta_logprob

        u = iter + n_leaves
        if u < node_count:  # not last merge
            # Update "distances", ignored_nodes and sequence embeddings
            dists = batch_update_dists(dists, D_u, u)
            ignored_nodes = batch_update_ignored(ignored_nodes, H_i, H_j, u)
            embs[:, u] = E_u
            splits = batch_update_splits(splits, H_i, H_j, n_leaves, iter, node_count)
            brlens = batch_update_brlens(brlens, H_i, H_j, l_i, l_j)

            # update constraints
            constraints = constraints.max(s[:, None, None])
            mask2d = batch_mask_from_ignored(ignored_nodes)
            constraints.masked_fill_(~mask2d, 0)
            constraints[:, u, :] = 0
            constraints[:, :, u] = 0

    return splits, brlens, topo_logsum, brlen_logsum


def batch_sample_tree_with_mse_brlens(
    model: EvoPF,
    sequence_embeddings: torch.Tensor,  # From EvoPF
    distances: torch.Tensor,  # From EvoPF
    temp: Optional[torch.Tensor] = None,
    use_max_proba: bool = False,
):
    dists, splits, brlens, ignored_nodes, n_leaves, node_count = batch_init_NJ(
        distances, rooted=True
    )
    constraints = torch.zeros_like(dists)

    # [B,d_m,n] -> [B,2n-2,d_m]
    embs = F.pad(
        sequence_embeddings,
        (0, node_count - n_leaves),
        mode="constant",
        value=0.0,
    ).transpose(-1, -2)
    batch_size = dists.size(0)

    topo_logsum = torch.zeros(batch_size, device=dists.device)
    brlen_logsum = torch.zeros(batch_size, device=dists.device)

    # merges = []
    # brlens

    for iter in range(node_count):
        # Compute Q matrix
        Q = batch_compute_Q(dists, ignored_nodes)

        # Sample merge
        (H_i, H_j) = batch_sample_merge(Q, ignored_nodes, temp, use_max_proba)

        # Compute probability of merge
        topo_logsum = topo_logsum + batch_compute_merge_logproba(
            Q, H_i, H_j, ignored_nodes, temp
        )

        # Compute new distances
        D_u, _, _ = batch_get_new_dist(dists, ignored_nodes, H_i, H_j, False)

        # Compute parent embedding
        E_u = model.embed_parent_node(
            H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
        ).squeeze(1)

        E_u = model.e_norm(E_u).squeeze(1)

        # Sample branch lengths
        # gamma_params = model.predict_lognormal_params(E_u).squeeze(1)
        def bounded_output_log_exp(z, min_val=1e-4, max_val=10.0):
            log_min = torch.log(torch.tensor(min_val))
            log_max = torch.log(torch.tensor(max_val))
            return torch.exp(log_min + (log_max - log_min) * torch.sigmoid(z))

        params_s = model.brlen_lognormal(E_u)
        logit_s = model.s_batch_norm(params_s).squeeze(1)

        mu_s = bounded_output_log_exp(logit_s[:, 0])
        sg_s = torch.ones_like(mu_s)
        print("mu_s", mu_s)

        beta_params = model.predict_beta_params(
            H_i.unsqueeze(1) @ embs, H_j.unsqueeze(1) @ embs
        ).squeeze(1)  # both in [0,1]

        print("beta_params", beta_params)

        beta_mode = model.b_w_batch_norm(beta_params[:, 0].unsqueeze(-1)).squeeze(-1)
        print("b_w_logit", beta_mode)
        beta_mode = F.sigmoid(beta_mode)
        print("b_w", beta_mode)
        beta_concentration = model.b_c_batch_norm(
            beta_params[:, 1].unsqueeze(-1)
        ).squeeze(-1)
        print("b_c_logit", beta_mode)
        beta_concentration = F.sigmoid(beta_concentration)
        print("b_c", beta_mode)

        print("beta_a", 1 + beta_concentration * beta_mode)
        print("beta_a", 1 + beta_concentration * (1 - beta_mode))

        beta_a = torch.clamp(1 + beta_concentration * beta_mode, min=1.0 + 1e-6)
        beta_b = torch.clamp(1 + beta_concentration * (1 - beta_mode), min=1.0 + 1e-6)

        print("clamped beta_a", beta_a)
        print("clamped beta_a", beta_b)

        beta = torch.distributions.Beta(beta_a, beta_b)
        normal = torch.distributions.Normal(mu_s, sg_s)

        M_ij = (H_i.unsqueeze(1) @ constraints @ H_j.unsqueeze(-1)).squeeze((1, 2))

        if use_max_proba:
            s = M_ij + mu_s
            balance = beta_mode  # beta.mode
        else:
            samp = -torch.ones_like(mu_s)
            while (samp < 0).any():
                samp = normal.sample()
            s = M_ij + samp
            balance = beta.sample()

        l_i = balance * s
        l_j = (1 - balance) * s

        # Compute probabilities for branches
        stdn = torch.distributions.Normal(torch.zeros_like(mu_s), torch.ones_like(sg_s))
        gamma_logprob = truncated_normal_log_pdf(s - M_ij, mu_s, sg_s, stdn, a=0)

        print("balance", balance)
        beta_logprob = beta_log_pdf(
            balance, torch.cat([beta_a.unsqueeze(1), beta_b.unsqueeze(1)], dim=1)
        )

        brlen_logsum = brlen_logsum + gamma_logprob + beta_logprob

        u = iter + n_leaves
        if u < node_count:  # not last merge
            # Update "distances", ignored_nodes and sequence embeddings
            dists = batch_update_dists(dists, D_u, u)
            ignored_nodes = batch_update_ignored(ignored_nodes, H_i, H_j, u)
            embs[:, u] = E_u
            splits = batch_update_splits(splits, H_i, H_j, n_leaves, iter, node_count)
            brlens = batch_update_brlens(brlens, H_i, H_j, l_i, l_j)

            # update constraints
            constraints = constraints.max(s[:, None, None])
            mask2d = batch_mask_from_ignored(ignored_nodes)
            constraints.masked_fill_(~mask2d, 0)
            constraints[:, u, :] = 0
            constraints[:, :, u] = 0

    return splits, brlens, topo_logsum, brlen_logsum


def joint_gamma_beta_raw_params_to_params(
    raw_alpha_t, raw_beta_t, raw_alpha_z, raw_beta_z, eps=1e-4
):
    alpha_t = 1.0 + F.softplus(raw_alpha_t) + eps
    beta_t = F.softplus(raw_beta_t) + eps
    alpha_z = 1.0 + F.softplus(raw_alpha_z) + eps
    beta_z = 1.0 + F.softplus(raw_beta_z) + eps

    return alpha_t, beta_t, alpha_z, beta_z


def log_joint_shifted_gamma_beta(
    x, y, s, raw_alpha_t, raw_beta_t, raw_alpha_z, raw_beta_z, eps=1e-4
):
    """
    Batched log-density of (x,y) under:
      T = x + y >= s
      U = T - s ~ Gamma(alpha_t, rate=beta_t)
      Z = x / T ~ Beta(alpha_z, beta_z)

    Parameters are predicted by NN as unconstrained reals:
      raw_alpha_t, raw_beta_t, raw_alpha_z, raw_beta_z
    These are mapped to positive reals via softplus + eps.

    Args:
        x, y, s : tensors of shape (B,)
        raw_alpha_t, raw_beta_t, raw_alpha_z, raw_beta_z : tensors of shape (B,)
        eps : numerical stability epsilon

    Returns:
        log_fU : tensor of shape (B,) with Gamma log density values for U = T - s
        log_fZ : tensor of shape (B,) with Beta log density values for Z = x / T
        logp : tensor of shape (B,) with log density values for T = x + y
    """

    # ensure tensors
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    s = torch.as_tensor(s, device=x.device, dtype=x.dtype)

    # Transform NN outputs into valid params
    alpha_t = 1.0 + F.softplus(raw_alpha_t) + eps
    beta_t = F.softplus(raw_beta_t) + eps
    alpha_z = 1.0 + F.softplus(raw_alpha_z) + eps
    beta_z = 1.0 + F.softplus(raw_beta_z) + eps

    T = x + y
    U = T - s

    U = torch.clamp(U, min=eps)
    T = torch.clamp(T, min=eps)

    Z = torch.clamp(x / T, min=eps, max=1.0 - eps)
    one_minus_Z = torch.clamp(1.0 - Z, min=eps)

    # log f_U(u): Gamma density in rate parameterization
    log_fU = (
        alpha_t * torch.log(beta_t)
        - torch.lgamma(alpha_t)
        + (alpha_t - 1.0) * torch.log(U)
        - beta_t * U
    )

    # log f_Z(z): Beta density
    log_fZ = (
        torch.lgamma(alpha_z + beta_z)
        - torch.lgamma(alpha_z)
        - torch.lgamma(beta_z)
        + (alpha_z - 1.0) * torch.log(Z)
        + (beta_z - 1.0) * torch.log(one_minus_Z)
    )

    logp = log_fU + log_fZ - torch.log(T)

    return log_fU, log_fZ, logp


def mean_shifted_gamma_beta(
    raw_alpha_t, raw_beta_t, raw_alpha_z, raw_beta_z, s, eps=1e-4
):
    """
    raw_*: tensors shape (B,) from NN (unconstrained)
    s: tensor shape (B,) or scalar
    Returns: mean_x, mean_y (tensors shape (B,))
    """

    # map to positive params (and >1 if you want)
    alpha_t = 1.0 + F.softplus(raw_alpha_t) + eps  # shape > 1
    beta_t = 1.0 + F.softplus(raw_beta_t) + eps  # rate > 1
    alpha_z = F.softplus(raw_alpha_z) + eps  # >0
    beta_z = 1.0 + F.softplus(raw_beta_z) + eps  # > 1

    # expectations
    E_T = s + (alpha_t / beta_t)  # E[T] = s + alpha/beta
    E_Z = alpha_z / (alpha_z + beta_z)

    E_X = E_Z * E_T
    E_Y = (1.0 - E_Z) * E_T
    return E_X, E_Y, E_T


def lognormal_params_from_empirical(
    empirical_mean: torch.Tensor, empirical_var: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mu_x_sq = empirical_mean.pow(2.0)
    sigma_x_sq = empirical_var

    l_mu = torch.log(mu_x_sq / torch.sqrt(mu_x_sq + sigma_x_sq))
    l_sigma = torch.log(1.0 + (sigma_x_sq / mu_x_sq))

    return l_mu, l_sigma.sqrt()


def lognormal_log_pdf(input: torch.Tensor, l_mu: torch.Tensor, l_sigma: torch.Tensor):
    """
    l_mu: log of loc
    l_sigma: log of scale ( > 0)

    computes log(PDF(x))

    with PDF(x) = 1/(xσ√(2)) * exp(-(ln(x) - μ)^2 / 2σ^2)
    c.f. https://en.wikipedia.org/wiki/Log-normal_distribution
    """

    assert (l_sigma > 0.0).all(), "log of scale must be positive"

    return -torch.log(input * l_sigma * ROOT_2PI.to(input)) - (
        1 / (2 * l_sigma.pow(2))
    ) * torch.pow(torch.log(input) - l_mu, 2)


def kumaraswamy_log_pdf(input: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """
    a: > 0
    b: > 0
    input: [0,1]

    computes log(PDF(x))

    with PDF(x) = abx^(a-1) * (1-x^a)^(b-1)
    c.f. https://en.wikipedia.org/wiki/Kumaraswamy_distribution
    """

    assert (a > 0).all() and (b > 0).all(), "Parameters must be > 0"
    assert (input >= 0).all() and (input <= 1.0).all(), "Beta is supported on [0,1]"

    return (
        torch.log(a * b)
        + torch.xlogy(a - 1, input)
        + torch.xlogy(b - 1, 1 - torch.pow(input, a))
    )


def gamma_log_pdf(input: torch.Tensor, shape: torch.Tensor, rate: torch.Tensor):
    """
    shape: α shape parameter (> 0).
    rate: λ rate parameter (> 0).

    computes log(PDF(x)) with:

    PDF(x) = (λ^α / Γ(α)) * x^(α-1) * exp(-λ*x)
    c.f. https://en.wikipedia.org/wiki/Gamma_distribution
    """
    assert (shape > 0).all() and (rate > 0).all(), (
        "Gamma parameters α and λ must be > 0 \n"
        f"Got α = {shape} \n"
        f"and λ = {rate} \n"
    )

    assert (input >= 0).all(), "Gamma distribution only defined on R+"

    return (
        torch.xlogy(shape, rate)
        + torch.xlogy(shape - 1, input)
        - rate * input
        - torch.lgamma(shape)
    )


def beta_log_pdf(input: torch.Tensor, params: torch.Tensor):
    """
    input: x (potentially batched)
    params: vector of concentration parameters [α, b] (potentially batched)

    computes log(PDF(x)) with:

    PDF(x) = (x^(α-1) * (1-x)^(β-1)) / B(α,β)
    B(α,β) = Γ(α)Γ(β) / Γ(α + β)
    """
    assert params.size(-1) == 2, "There must be only 2 parameters for Beta"
    assert (
        params > 0.0
    ).all(), f"Beta parameters must be positive, was alpha={params[:, 0]} and beta={params[:, 1]}"
    assert (input >= 0).all() and (
        input <= 1.0
    ).all(), f"Beta is supported on [0,1], input was {input}"

    x = torch.stack([input, 1 - input], -1)

    return (
        torch.xlogy(params - 1.0, x).sum(-1)
        + torch.lgamma(params.sum(-1))
        - torch.lgamma(params).sum(-1)
    )


class NJDataset(Dataset):
    """
    Dataset that loads:
     - X: the one hot encoded MSA
     - S: the ordered merged resulting from NJ on the tree
     - Y: the log probability of the successive merges given the tree distances
     and Optionally:
     - The split matrix
     - The branch length vector
    """

    def __init__(
        self,
        tree_dir: str,
        msa_dir: str,
        cache_dir: Optional[str] = None,
        get_branch_lengths: bool = False,
        get_all_data: bool = False,
    ) -> None:
        super().__init__()
        self.tree_dir = tree_dir
        self.msa_dir = msa_dir
        self.cache_dir = cache_dir or os.path.join(
            Path(self.tree_dir).parent, "cached_tensors"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self.get_branch_lengths = get_branch_lengths
        self.get_all_data = get_all_data
        self.tree_msa_pairs = self.find_pairs()

    def find_pairs(self) -> list[tuple[str, str]]:
        trees = {
            Path(treepath).stem: treepath
            for ext in ["nwk", "newick"]
            for treepath in glob(f"{self.tree_dir}/*{ext}")
        }
        msas = {
            Path(msapath).stem: msapath
            for ext in ["fa", "fasta"]
            for msapath in glob(f"{self.msa_dir}/*{ext}")
        }

        pairs = []
        for id_, treepath in trees.items():
            if (msapath := msas.get(id_)) is not None:
                pairs.append((treepath, msapath))

        return pairs

    def __len__(self) -> int:
        return len(self.tree_msa_pairs)

    def __getitem__(
        self, index
    ) -> Union[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]],
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            list[str],
        ],
    ]:
        treepath, msapath = self.tree_msa_pairs[index]
        id_ = Path(treepath).stem
        cachedpath = os.path.join(self.cache_dir, f"{id_}.tch")

        msa, ids = load_alignment(msapath)

        if os.path.exists(cachedpath):
            splits, brlens, merge_order, logprob = torch.load(cachedpath)
        else:
            dm = load_distance_matrix(treepath, ids=ids, get_vec=False)
            splits, brlens, merge_order, logprob = soft_nj(dm)

            # Cache computed value
            torch.save((splits, brlens, merge_order, logprob), cachedpath)

        if self.get_all_data:
            return msa, splits, brlens, merge_order, logprob, ids
        elif self.get_branch_lengths:
            return msa, brlens, merge_order, logprob, ids
        else:
            return msa, merge_order, logprob, ids


class MultisizeUnambiguousMergeOrderDataset(Dataset):
    def __init__(
        self,
        tree_dirs: list[str],
        msa_dirs: list[str],
    ):
        super().__init__()
        self.tree_dirs = tree_dirs
        self.msa_dirs = msa_dirs

        self.pairs = {
            (p := self.find_pairs(trees, msas))[0]: p[1]
            for trees, msas in zip(self.tree_dirs, self.msa_dirs)
        }

        self.merged_indices = [
            (s, i) for s, p in self.pairs.items() for i in range(len(p))
        ]

    def find_pairs(self, tree_dir, msa_dir) -> tuple[int, list[tuple[str, str]]]:
        trees = {
            Path(treepath).stem: treepath
            for ext in ["nwk", "newick"]
            for treepath in glob(f"{tree_dir}/*{ext}")
        }
        msas = {
            Path(msapath).stem: msapath
            for ext in ["fa", "fasta"]
            for msapath in glob(f"{msa_dir}/*{ext}")
        }

        pairs = []
        for id_, treepath in trees.items():
            if (msapath := msas.get(id_)) is not None:
                pairs.append((treepath, msapath))

        with open(pairs[0][1], "rb") as msa:
            size = sum(line.startswith(b">") for line in msa)

        return (size, pairs)

    def __len__(self):
        return sum(len(v) for v in self.pairs.values())

    @override
    def __getitem__(self, index):
        s, i = self.merged_indices[index]
        treepath, msapath = self.pairs[s][i]
        msa, ids = load_alignment(msapath)
        merge_order, brlens = get_merge_order(treepath, ids)

        return msa, merge_order, brlens, ids


class UnambiguousMergeOrderDataset(Dataset):
    def __init__(
        self,
        tree_dir: str,
        msa_dir: str,
    ) -> None:
        super().__init__()
        self.tree_dir = tree_dir
        self.msa_dir = msa_dir

        self.tree_msa_pairs = self.find_pairs()

    def find_pairs(self) -> list[tuple[str, str]]:
        trees = {
            Path(treepath).stem: treepath
            for ext in ["nwk", "newick"]
            for treepath in glob(f"{self.tree_dir}/*{ext}")
        }
        msas = {
            Path(msapath).stem: msapath
            for ext in ["fa", "fasta"]
            for msapath in glob(f"{self.msa_dir}/*{ext}")
        }

        pairs = []
        for id_, treepath in trees.items():
            if (msapath := msas.get(id_)) is not None:
                pairs.append((treepath, msapath))

        return pairs

    def __len__(self) -> int:
        return len(self.tree_msa_pairs)

    def __getitem__(self, index):
        treepath, msapath = self.tree_msa_pairs[index]
        msa, ids = load_alignment(msapath)
        merge_order, brlens = get_merge_order(treepath, ids)

        return msa, merge_order, brlens, ids


def get_merge_order(treepath, ids):
    ns = TaxonNamespace(x.replace("_", " ") for x in ids)
    tip_order = {t: i for i, t in enumerate(ns)}

    n_leaves = len(ids)
    n_nodes = 2 * n_leaves - 1

    brlens = [None for _ in ids]

    tree: Tree = Tree.get_from_path(treepath, schema="newick", taxon_namespace=ns)
    assert len(tree.seed_node.child_nodes()) == 2, "Only applicaple to rooted trees"

    # Pre-compute cherry lengths
    nodes = {}
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            node.label = tip_order[node.taxon]
            brlens[tip_order[node.taxon]] = node.edge_length
            nodes[node] = NodeWrapper(0.0, is_cherry=False, is_merged=True)
        else:
            is_cherry = all(n.is_leaf() for n in node.child_node_iter())
            cherry_len = sum(e.length for e in node.child_edge_iter())
            nodes[node] = NodeWrapper(cherry_len, is_cherry=is_cherry, is_merged=False)

    # Compute merge order
    merge_order = []
    for i in range(n_nodes - n_leaves):
        u = len(ids) + i

        node_to_merge = find_node_to_merge(nodes)

        merge_order.append(
            tuple(sorted([n.label for n in node_to_merge.child_node_iter()]))
        )

        node_to_merge.label = u

        nodes[node_to_merge].is_cherry = False
        nodes[node_to_merge].is_merged = True

        # Check if parent is new cherry
        parent = node_to_merge.parent_node
        if parent is not None:
            nodes[parent].is_cherry = all(
                nodes[k].is_merged for k in parent.child_node_iter()
            )
            brlens.append(node_to_merge.edge_length)
        else:
            assert i == n_leaves - 2, "Merging root before other nodes..."

    return (
        F.one_hot(torch.tensor(merge_order), num_classes=n_nodes - 1),
        torch.tensor(brlens),
    )


class NodeWrapper:
    def __init__(
        self, cherry_len: float, is_cherry: bool = False, is_merged: bool = False
    ) -> None:
        self.len = cherry_len
        self.is_cherry = is_cherry
        self.is_merged = is_merged

    def __repr__(self) -> str:
        if self.is_cherry:
            s = f"C({self.len})"
        else:
            s = f"N({self.len})"
        if self.is_merged:
            s += "|X"
        return s


def find_node_to_merge(nodes: dict[Node, NodeWrapper]) -> Node:
    min_node = None
    min_info = None

    for node, info in nodes.items():
        if not info.is_cherry:
            continue

        if min_node is None:
            min_node = node
            min_info = info
            continue

        if info.len < min_info.len:  # type: ignore
            min_node = node
            min_info = info

    assert min_node is not None

    return min_node


class BatchTuple:
    def __init__(self, t):
        self.t = t

    def unwrap(self):
        return self.t


class SameSizeBatchSampler(Sampler):
    def __init__(
        self,
        dataset: MultisizeUnambiguousMergeOrderDataset,
        base_batch_size: int,
        base_size: int,
        shuffle: bool = True,
        indices: list[int] | None = None,
        seed: int = 0,
    ):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.base_batch_size = base_batch_size
        self.base_size = base_size
        self.start_iter = 0

        if not isinstance(dataset, MultisizeUnambiguousMergeOrderDataset):
            raise ValueError(
                "This Sampler only works for MultisizeUnambiguousMergeOrderDataset"
            )

        # Get indices of data examples, along with the size of the dataset
        self.indices = np.array(
            [(s, i) for i, (s, _) in enumerate(self.dataset.merged_indices)]
        )
        if indices is not None:  # e.g witha distributed sampler
            self.indices = self.indices[indices]

        self.grouped_indices = defaultdict(list)
        for k, i in self.indices:
            self.grouped_indices[k].append(i)
        self.grouped_indices = {k: np.array(v) for k, v in self.grouped_indices.items()}

    def set_starting_step(self, step: int):
        self.start_iter = step

    def _get_batch_size(self, size):
        ratio = (self.base_size / size) ** 2
        batch_size = math.floor(ratio * self.base_batch_size)
        return min(max(batch_size, 1), self.base_batch_size)

    def __len__(self):
        n = 0
        for size, items in self.grouped_indices.items():
            batch_size = self._get_batch_size(size)
            n += len(items) // batch_size
        return n

    def __iter__(self):
        # Set reproducible rng
        rng = np.random.default_rng(self.seed)

        # Shuffle indices if needed
        grouped_indices = self.grouped_indices
        if self.shuffle:
            grouped_indices = {
                k: rng.permutation(v) for k, v in self.grouped_indices.items()
            }

        batches = []
        for size, items in grouped_indices.items():
            batch_size = self._get_batch_size(size)
            samplers = [iter(items.tolist())] * batch_size
            for batch_indices in zip(*samplers, strict=False):
                batches.append(BatchTuple([*batch_indices]))

        if self.shuffle:
            batches = rng.permutation(batches)

        for b in batches[self.start_iter :]:
            yield b.unwrap()


class DistributedSameSizeSampler(DistributedSampler):
    def __init__(
        self,
        dataset: MultisizeUnambiguousMergeOrderDataset,
        base_size: int,
        base_batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset, shuffle=shuffle, seed=seed, drop_last=True)
        self.base_batch_size = base_batch_size
        self.base_size = base_size
        self.start_iter = 0

    def set_starting_step(self, step: int):
        self.start_iter = step

    def __iter__(self):
        sampler = SameSizeBatchSampler(
            self.dataset,
            base_batch_size=self.base_batch_size,
            base_size=self.base_size,
            shuffle=self.shuffle,
            seed=self.seed + self.epoch,
            indices=list(super().__iter__()),
        )
        sampler.set_starting_step(self.start_iter)
        return iter(sampler)

    def __len__(self):
        return len(
            SameSizeBatchSampler(
                self.dataset,
                base_batch_size=self.base_batch_size,
                base_size=self.base_size,
                shuffle=self.shuffle,
                seed=self.seed + self.epoch,
                indices=list(super().__iter__()),
            )
        )
