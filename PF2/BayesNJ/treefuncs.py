# pyright: basic

import os
from typing import Optional

import torch
import torch.nn.functional as F

from .core import (  # common
    batch_compute_merge_logproba,
    batch_get_new_dist,
    batch_init_NJ,
    batch_mask_from_ignored,
    batch_sample_merge,
    batch_update_dists,
    batch_update_ignored,
    joint_gamma_beta_raw_params_to_params,
    log_joint_shifted_gamma_beta,
    mean_shifted_gamma_beta,
)
from .evopf import EvoPF

CONSTRAIN = os.environ.get("PF_CONSTRAIN", "False").lower() in ["true", "t", "1"]


def batch_sample_Q(model: EvoPF, embs: torch.Tensor, ignored):
    batch_size, n_nodes, _ = embs.shape
    Q = torch.zeros(batch_size, n_nodes, n_nodes).to(embs)
    ix, jx = torch.tril_indices(n_nodes, n_nodes, -1)
    mask_2d = batch_mask_from_ignored(ignored)

    # Compute Q matrix
    vals = model.q_embedder(embs[:, ix, :], embs[:, jx, :])
    Q[:, ix, jx] = vals.squeeze(-1)
    Q = Q + Q.transpose(-1, -2)

    # Mask ignored ignored_nodes
    mask = (~mask_2d) | torch.eye(n_nodes, device=mask_2d.device).bool()
    Q.masked_fill_(mask, torch.inf)

    return Q


def batch_compute_tree_logprob(
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
    if verbose:
        print(f"TOPO ONLY: {topo_only}")

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

    # Init return values
    topo_logsum = torch.zeros(batch_size, device=dists.device)
    brlen_comp1_logsum = torch.zeros(batch_size, device=dists.device)
    brlen_comp2_logsum = torch.zeros(batch_size, device=dists.device)
    brlen_joint_logsum = torch.zeros(batch_size, device=dists.device)
    brlen_mse_sum = torch.zeros(batch_size, device=dists.device)

    # Initialize container
    Q = torch.zeros_like(dists)

    # bs, n_iter, 2, n_splits -> n_iter, 2, bs, n_splits
    merges = merges.permute(1, 2, 0, 3)
    for iter, merge in enumerate(merges):
        # Get merge from predefined order
        (H_i, H_j) = merge

        # Compute Q matrix
        Q = batch_sample_Q(model, embs, ignored_nodes)

        # Compute probability of merge
        if not ignore_topo:
            topo_logsum = topo_logsum + batch_compute_merge_logproba(
                Q, H_i, H_j, ignored_nodes, temp
            )

        u = iter + n_leaves

        if u < node_count:  # not last merge
            # Update "distances", ignored_nodes and sequence embeddings
            # dists = batch_update_dists(dists, D_u, u)
            ignored_nodes = batch_update_ignored(ignored_nodes, H_i, H_j, u)

            # Compute parent embedding
            E_i = (H_i.unsqueeze(1) @ embs).squeeze(1)
            E_j = (H_j.unsqueeze(1) @ embs).squeeze(1)
            E_u = model.embed_parent_node(E_i, E_j).squeeze(1)
            E_u = model.e_norm(E_u)
            embs[:, u] = E_u

            if not topo_only:
                # Compute branch probabilities
                l_i = (H_i.unsqueeze(1) @ brlens.unsqueeze(-1)).squeeze((1, 2))
                l_j = (H_j.unsqueeze(1) @ brlens.unsqueeze(-1)).squeeze((1, 2))
                s = l_i + l_j
                balance = l_i / s

                #          FILL IN THIS PART            #
                #########################################

                # infer Branch lengths
                M_ij = (H_i.unsqueeze(1) @ constraints @ H_j.unsqueeze(-1)).squeeze(
                    (1, 2)
                )

                # Joint (Gamma, Beta) version
                raw_t_params = model.gamma_sampler(E_i, E_j).squeeze(1)
                raw_z_params = model.beta_sampler(E_i, E_j).squeeze(1)

                gamma_logprob, beta_logprob, joint_logprob = (
                    log_joint_shifted_gamma_beta(
                        l_i,
                        l_j,
                        M_ij,
                        raw_t_params[:, 0],
                        raw_t_params[:, 1],
                        raw_z_params[:, 0],
                        raw_z_params[:, 1],
                        eps=1e-4,
                    )
                )

                El_i, El_j, _ = mean_shifted_gamma_beta(
                    raw_t_params[:, 0],
                    raw_t_params[:, 1],
                    raw_z_params[:, 0],
                    raw_z_params[:, 1],
                    M_ij,
                    eps=1e-4,
                )
                bl_mse = -((l_i - El_i) ** 2) - ((l_j - El_j) ** 2)

                brlen_comp1_logsum = brlen_comp1_logsum + gamma_logprob
                brlen_comp2_logsum = brlen_comp2_logsum + beta_logprob
                brlen_joint_logsum = brlen_joint_logsum + joint_logprob
                brlen_mse_sum = brlen_mse_sum + bl_mse
                #########################################
                #                 END                   #
                #########################################

                # update constraints
                constraints = constraints.max(s[:, None, None])
                mask2d = batch_mask_from_ignored(ignored_nodes)
                constraints.masked_fill_(~mask2d, 0)
                constraints[:, u, :] = 0
                constraints[:, :, u] = 0
        else:
            if not topo_only:
                # Get last 2 indices
                E_i = (H_i.unsqueeze(1) @ embs).squeeze(1)
                E_j = (H_j.unsqueeze(1) @ embs).squeeze(1)

                l_i = (H_i.unsqueeze(1) @ brlens.unsqueeze(-1)).squeeze((1, 2))
                l_j = (H_j.unsqueeze(1) @ brlens.unsqueeze(-1)).squeeze((1, 2))
                s = l_i + l_j

                M_ij = (H_i.unsqueeze(1) @ constraints @ H_j.unsqueeze(-1)).squeeze(
                    (1, 2)
                )

                # Joint (Gamma, Beta) version
                raw_t_params = model.gamma_sampler(E_i, E_j).squeeze(1)
                raw_z_params = model.beta_sampler(E_i, E_j).squeeze(1)

                gamma_logprob, _, _ = log_joint_shifted_gamma_beta(
                    l_i,
                    l_j,
                    M_ij,
                    raw_t_params[:, 0],
                    raw_t_params[:, 1],
                    raw_z_params[:, 0],
                    raw_z_params[:, 1],
                    eps=1e-4,
                )

                _, _, E_s = mean_shifted_gamma_beta(
                    raw_t_params[:, 0],
                    raw_t_params[:, 1],
                    raw_z_params[:, 0],
                    raw_z_params[:, 1],
                    M_ij,
                    eps=1e-4,
                )

                bl_mse = -((s - E_s) ** 2)

                brlen_comp1_logsum = brlen_comp1_logsum + gamma_logprob
                brlen_joint_logsum = brlen_joint_logsum + gamma_logprob
                brlen_mse_sum = brlen_mse_sum + bl_mse

    return (
        topo_logsum,
        brlen_comp1_logsum,
        brlen_comp2_logsum,
        brlen_joint_logsum,
        brlen_mse_sum,
    )


def batch_sample_trees(
    model: EvoPF,
    sequence_embeddings: torch.Tensor,  # From EvoPF
    distances: torch.Tensor,  # From EvoPF
    temp: Optional[torch.Tensor] = None,
    use_max_proba: bool = False,
    verbose: bool = True,
):
    # def sample_branch_length(emb1, emb2):
    #     mu = model.brlen_sampler(emb1, emb2).squeeze(1)
    #     if use_max_proba:
    #         return mu
    #     return mu  # Here would sample a gaussian

    dists, _, _, ignored_nodes, n_leaves, node_count = batch_init_NJ(
        distances, rooted=False
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
    bidx = torch.arange(batch_size)

    topo_logsum = torch.zeros(batch_size, device=dists.device)
    brlen_logsum = torch.zeros(batch_size, device=dists.device)

    n_merges = n_leaves - 3
    merges = torch.empty((batch_size, n_merges, 2), device=dists.device).long()
    brlens = -torch.ones((batch_size, node_count), device=dists.device)

    for iter in range(n_merges):
        # Compute Q matrix
        Q = batch_sample_Q(model, embs, ignored_nodes)
        if verbose:
            print("Q:", Q)

        # Sample merge
        (H_i, H_j) = batch_sample_merge(Q, ignored_nodes, temp, use_max_proba)

        # Compute probability of merge
        topo_logsum = topo_logsum + batch_compute_merge_logproba(
            Q, H_i, H_j, ignored_nodes, temp
        )

        # Child embeddings
        E_i = (H_i.unsqueeze(1) @ embs).squeeze(1)
        E_j = (H_j.unsqueeze(1) @ embs).squeeze(1)

        # Compute parent embedding
        u = iter + n_leaves
        E_u = model.embed_parent_node(E_i, E_j).squeeze(1)
        E_u = model.e_norm(E_u)

        # Extract constraint
        M_ij = (H_i.unsqueeze(1) @ constraints @ H_j.unsqueeze(-1)).squeeze((1, 2))

        #########################################
        #          FILL IN THIS PART            #
        #########################################

        # Joint (Gamma, Beta) version
        raw_t_params = model.gamma_sampler(E_i, E_j).squeeze(1)
        raw_z_params = model.beta_sampler(E_i, E_j).squeeze(1)

        l_i, l_j, U, Z, _ = batch_sample_branch_lengths(
            raw_t_params, raw_z_params, M_ij, use_max_proba
        )

        if verbose:
            print(f"Iter {u}")
            print("M_ij", M_ij)
            print("U", U)
            print("Z", Z)
            print("li", l_i, flush=True)
            print("lj", l_j, flush=True)

        # Write merge
        i, j = H_i.argmax(-1), H_j.argmax(-1)
        merges[bidx, iter, 0] = i
        merges[bidx, iter, 1] = j
        brlens[bidx, i] = l_i
        brlens[bidx, j] = l_j

        #########################################
        #                 END                   #
        #########################################

        if iter < node_count:
            # Update "distances", ignored_nodes and sequence embeddings
            D_u, _, _ = batch_get_new_dist(dists, ignored_nodes, H_i, H_j, False)
            dists = batch_update_dists(dists, D_u, u)
            ignored_nodes = batch_update_ignored(ignored_nodes, H_i, H_j, u)
            embs[:, u] = E_u

            # update constraints
            s = l_i + l_j
            constraints = constraints.max(s[:, None, None])
            mask2d = batch_mask_from_ignored(ignored_nodes)
            constraints.masked_fill_(~mask2d, 0)
            constraints[:, u, :] = 0
            constraints[:, :, u] = 0

    # Here we compute last 3 branch lengths
    Q = batch_sample_Q(model, embs, ignored_nodes)
    (H_i, H_j) = batch_sample_merge(Q, ignored_nodes, temp, use_max_proba)
    topo_logsum = topo_logsum + batch_compute_merge_logproba(
        Q, H_i, H_j, ignored_nodes, temp
    )
    E_i = (H_i.unsqueeze(1) @ embs).squeeze(1)
    E_j = (H_j.unsqueeze(1) @ embs).squeeze(1)
    E_u = model.embed_parent_node(E_i, E_j).squeeze(1)
    E_u = model.e_norm(E_u)
    M_ij = (H_i.unsqueeze(1) @ constraints @ H_j.unsqueeze(-1)).squeeze((1, 2))

    # Update ignored and constraints
    ignored_nodes = batch_update_ignored(ignored_nodes, H_i, H_j, None)

    H_k = (~ignored_nodes).float()
    E_k = (H_k.unsqueeze(1) @ embs).squeeze(1)

    raw_t_params = model.gamma_sampler(E_i, E_j).squeeze(1)
    raw_z_params = model.beta_sampler(E_i, E_j).squeeze(1)
    raw_t_params2 = model.gamma_sampler(E_u, E_k).squeeze(1)
    raw_z_params2 = model.beta_sampler(E_u, E_k).squeeze(1)

    l_i, l_j, U, Z, _ = batch_sample_branch_lengths(
        raw_t_params, raw_z_params, M_ij, use_max_proba
    )
    s = l_i + l_j
    *_, l_k = batch_sample_branch_lengths(
        raw_t_params2, raw_z_params2, M_ij, use_max_proba
    )

    if verbose:
        print("U", U)
        print("Z", Z)
        print("li", l_i, flush=True)
        print("lj", l_j, flush=True)
        print("lk", l_k, flush=True)

    i, j, k = H_i.argmax(-1), H_j.argmax(-1), H_k.argmax(-1)
    brlens[bidx, i] = l_i
    brlens[bidx, j] = l_j
    brlens[bidx, k] = l_k

    return merges, brlens, topo_logsum, brlen_logsum


def batch_sample_branch_lengths(raw_t_params, raw_z_params, M_ij, use_max_proba):
    alpha_t, beta_t, alpha_z, beta_z = joint_gamma_beta_raw_params_to_params(
        raw_t_params[:, 0],
        raw_t_params[:, 1],
        raw_z_params[:, 0],
        raw_z_params[:, 1],
    )

    gamma = torch.distributions.Gamma(alpha_t, beta_t)
    beta = torch.distributions.Beta(alpha_z, beta_z)

    if use_max_proba:
        U = gamma.mode
        Z = beta.mode
    else:
        U = gamma.sample()
        Z = beta.sample()

    T = M_ij + U
    l_i = Z * T
    l_j = T - l_i

    return l_i, l_j, U, Z, T
