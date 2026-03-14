# pyright: basic

import math
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parametrize

from .memory import make_mem_hook
from .pf_sdk.pf.modules import MSAEmbedder, seq2pairs

try:
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

except ImportError:
    DEEPSPEED_LOADED = False
else:
    DEEPSPEED_LOADED = True


try:
    from torch.nn.attention.flex_attention import flex_attention

except ImportError:
    COMPILED_FLEX = None
    FLEX_LOADED = False
else:
    COMPILED_FLEX = torch.compile(flex_attention)
    FLEX_LOADED = True

USE_Q_MLP = os.environ.get("PF_USE_Q_MLP", "False").lower() in ["true", "t", "1"]

def lognormal_params_from_empirical(
    empirical_mean: float, empirical_var: float
) -> tuple[float, float]:
    mu_x_sq = empirical_mean**2
    sigma_x_sq = empirical_var

    loc = math.log(mu_x_sq / math.sqrt(mu_x_sq + sigma_x_sq))
    scale = math.log(1 + (sigma_x_sq / mu_x_sq))

    return loc, scale


# Computed on validation set with 10k trees
CHERRY_MEAN = 2 * 0.1791756450803857
CHERRY_VAR = 4 * 0.17247928274940227
CHERRY_MIN = 2 * 0.000100031
CHERRY_MAX = 2 * 9.99393

# We want (α/λ) == 2brlen_avg and (α/λ^2) = 10 * brlen_var
GAMMA_INIT_SHAPE = CHERRY_MEAN**2 / (10 * CHERRY_VAR)
GAMMA_INIT_RATE = CHERRY_MEAN / (10 * CHERRY_VAR)

# Initial parameters of the log-normal distribution to match empirical mean and variance
# LOGNORMAL_INIT_LOC, LOGNORMAL_INIT_SCALE = lognormal_params_from_empirical(CHERRY_MEAN, 10 * CHERRY_VAR)
# LOGNORMAL_MIN_LOC, _ = lognormal_params_from_empirical(CHERRY_MIN, CHERRY_VAR)
# LOGNORMAL_MAX_LOC, _ = lognormal_params_from_empirical(CHERRY_MAX, CHERRY_VAR)


class Symmetric(nn.Module):
    def forward(self, X: Tensor):
        return (X + X.transpose(-1, -2)) / 2


class ColAttenPairBias(nn.Module):
    def __init__(
        self,
        msa_dim: int,
        pairs_dim: int,
        n_heads: int,
        symmetric: bool = False,
        use_deepspeed: bool = False,
        use_flexattention: bool = False,
    ) -> None:
        if use_deepspeed:
            assert DEEPSPEED_LOADED, "DeepSpeed is not available"

        if use_flexattention:
            assert FLEX_LOADED, "Flex attention is not available"

        super().__init__()
        self.embed_dim = msa_dim
        self.pairs_dim = pairs_dim
        self.n_heads = n_heads
        self.symmetric = symmetric
        self.deepspeed = use_deepspeed
        self.flexattention = use_flexattention

        self.head_dim = msa_dim // n_heads
        self.sqrt_dim = math.sqrt(self.head_dim)

        # Layer norms
        self.msa_norm = nn.LayerNorm(msa_dim)
        self.pairs_norm = nn.LayerNorm(pairs_dim)

        # Input projectors
        self.q_proj = nn.Linear(msa_dim, msa_dim, bias=False)
        self.k_proj = nn.Linear(msa_dim, msa_dim, bias=False)
        self.v_proj = nn.Linear(msa_dim, msa_dim, bias=False)
        self.g_proj = nn.Linear(msa_dim, msa_dim, bias=True)
        self.b_proj = nn.Linear(pairs_dim, n_heads, bias=False)

        # Output projector
        self.out_proj = nn.Linear(msa_dim, msa_dim)

        self.pre_qkdiv = lambda _: None
        self.pre_bias = lambda _: None
        self.pre_attout = lambda _: None
        self.pre_reshape = lambda _: None
        self.qkdiv = lambda _: None
        self.bias = lambda _: None
        self.attout = lambda _: None
        self.reshape = lambda _: None

    def register_hooks(self, prefix, memfile, steps, dtypes):
        self.pre_qkdiv = make_mem_hook(
            memfile, f"{prefix}.qkdiv", "pre_fwd", steps, dtypes
        )
        self.qkdiv = make_mem_hook(memfile, f"{prefix}.qkdiv", "fwd", steps, dtypes)
        self.pre_attout = make_mem_hook(
            memfile, f"{prefix}.attout", "pre_fwd", steps, dtypes
        )
        self.attout = make_mem_hook(memfile, f"{prefix}.attout", "fwd", steps, dtypes)
        self.pre_bias = make_mem_hook(
            memfile, f"{prefix}.bias", "pre_fwd", steps, dtypes
        )
        self.bias = make_mem_hook(memfile, f"{prefix}.bias", "fwd", steps, dtypes)
        self.pre_reshape = make_mem_hook(
            memfile, f"{prefix}.reshape", "pre_fwd", steps, dtypes
        )
        self.reshape = make_mem_hook(memfile, f"{prefix}.reshape", "fwd", steps, dtypes)

    def forward(self, msa, pairs):
        if self.flexattention:
            return self.forward_flexattention(msa, pairs)
        elif self.deepspeed:
            return self.forward_deepspeed(msa, pairs)
        else:
            return self.forward_torch(msa, pairs)

    def forward_torch(self, msa, pairs):
        # MSA: [B,d_m,n,L]
        # Pairs: [B,n,n,d_p] (or [B,nC2,d_p] is symmetric)
        B, d_m, n, L = msa.size()
        H = self.n_heads
        d_h = self.head_dim
        sqrt_c = self.sqrt_dim

        # Normalize
        msa = self.msa_norm(msa.transpose(-1, -3))  # [B,L,n,d_m]
        pairs = self.pairs_norm(pairs)  # [B,n,n,d_p] ([B,nC2,d_p])

        # pair bias: [B,n,n,H] ([B,nC2,H])
        b = self.b_proj(pairs)

        # Input projections
        # projections: [b,L,H,n,d_h]
        q = self.q_proj(msa).reshape(B, L, n, H, d_h).transpose(2, 3)
        k = self.k_proj(msa).reshape(B, L, n, H, d_h).transpose(2, 3)
        v = self.v_proj(msa).reshape(B, L, n, H, d_h).transpose(2, 3)

        # gating: [b,L,H,n,d_h]
        g = F.sigmoid(self.g_proj(msa).reshape(B, L, n, H, d_h).transpose(2, 3))

        # Attention
        # [B,L,H,n,n]
        self.pre_qkdiv(None)
        qkdiv = (q @ k.transpose(-1, -2)) / sqrt_c
        self.qkdiv(None)

        self.pre_bias(None)
        if self.symmetric:
            biased = qkdiv
            b = b.transpose(-1, -2).unsqueeze(1)
            ix, jx = torch.tril_indices(n, n, -1)

            # Add to lower tri
            biased[:, :, :, ix, jx] += b
            # Add to upper tri
            biased = biased.transpose(-1, -2)
            biased[:, :, :, ix, jx] += b
            # Undo transpose
            biased = biased.transpose(-1, -2)
        else:
            biased = qkdiv + b.transpose(-1, -3).unsqueeze(1)
        self.bias(None)

        self.pre_attout(None)
        attn_weights = biased.softmax(-1)
        attn_out = g * (attn_weights @ v)
        self.attout(None)

        self.pre_reshape(None)
        attn_out = attn_out.transpose(2, 3).reshape(B, L, n, d_m)
        self.reshape(None)

        # [B,L,n,d_m]
        out = self.out_proj(attn_out)

        # [B,d_m,n,L]
        return out.permute(0, 3, 2, 1)

    def forward_deepspeed(self, msa, pairs):
        # MSA: [B,d_m,n,L]
        # Pairs: [B,n,n,d_p] (or [B,nC2,d_p] is symmetric)

        B, d_m, n, L = msa.size()
        H = self.n_heads
        d_h = self.head_dim

        # Normalize
        msa = self.msa_norm(msa.transpose(-1, -3))  # [B,L,n,d_m]
        pairs = self.pairs_norm(pairs)  # [B,n,n,d_p] ([B,nC2,d_p])

        # pair bias: [B,n,n,H] ([B,nC2,H])
        b = self.b_proj(pairs)
        if self.symmetric:
            ix, jx = torch.tril_indices(n, n, -1)
            b_temp = torch.zeros(B, n, n, H, device=b.device)
            # Lower tri
            b_temp[:, ix, jx, :] += b
            # Upper tri
            b_temp = b_temp.transpose(1, 2)
            b_temp[:, ix, jx, :] += b
            # Move to b
            b = b_temp

        # [B,1,H,n,n]
        b = b.permute(0, 3, 1, 2).unsqueeze(1)

        # Input projections
        # projections: [B,L,n,H,d_h]
        q = self.q_proj(msa).reshape(B, L, n, H, d_h)
        k = self.k_proj(msa).reshape(B, L, n, H, d_h)
        v = self.v_proj(msa).reshape(B, L, n, H, d_h)

        # gating: [B,L,n,H,d_h]
        g = F.sigmoid(self.g_proj(msa).reshape(B, L, n, H, d_h))

        self.pre_attout(None)
        orig_dtype = q.dtype
        if orig_dtype not in [torch.bfloat16, torch.float16]:
            biases = [None, b.to(torch.bfloat16)]  # DeepSpeed needs a "res_mask"
            attn_out = g * DS4Sci_EvoformerAttention(  # type: ignore
                q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), biases
            )
            attn_out = attn_out.to(orig_dtype)
        else:
            biases = [None, b.to(orig_dtype)]
            attn_out = g * DS4Sci_EvoformerAttention(q, k, v, biases)  # type: ignore
        self.attout(None)

        self.pre_reshape(None)
        # [B,L,n,H,d_h] -> [B,L,n,d_m]
        attn_out = attn_out.reshape(B, L, n, d_m)
        self.reshape(None)

        # [B,L,n,d_m]
        out = self.out_proj(attn_out)

        # [B,d_m,n,L]
        return out.permute(0, 3, 2, 1)

    def forward_flexattention(self, msa, pairs):
        # MSA: [B,d_m,n,L]
        # Pairs: [B,n,n,d_p] (or [B,nC2,d_p] is symmetric)
        B, d_m, n, L = msa.size()
        H = self.n_heads
        d_h = self.head_dim

        # Normalize
        msa = self.msa_norm(msa.transpose(-1, -3))  # [B,L,n,d_m]
        pairs = self.pairs_norm(pairs)  # [B,n,n,d_p] ([B,nC2,d_p])

        # Input projections
        # projections: [B,L,H,n,d_h]
        q = self.q_proj(msa).reshape(B, L, n, H, d_h).transpose(2, 3)
        k = self.k_proj(msa).reshape(B, L, n, H, d_h).transpose(2, 3)
        v = self.v_proj(msa).reshape(B, L, n, H, d_h).transpose(2, 3)

        # gating: [b,L,H,n,d_h]
        g = F.sigmoid(self.g_proj(msa).reshape(B, L, n, H, d_h).transpose(2, 3))

        # pair bias: [B,n,n,H] ([B,nC2,H])
        pb = self.b_proj(pairs)
        if self.symmetric:
            # [B,1,H,nC2]
            pb = pb.transpose(-1, -2).unsqueeze(1)
            ix, jx = torch.tril_indices(n, n, -1)
            b_temp = torch.zeros(B, 1, H, n, n, device=pb.device)

            # Add to lower tri
            b_temp[:, :, :, ix, jx] += pb
            # Add to upper tri
            b_temp = b_temp.transpose(-1, -2)
            b_temp[:, :, :, ix, jx] += pb
            # Undo transpose
            pb = b_temp.transpose(-1, -2)
        else:
            pb = pb.transpose(-1, -3).unsqueeze(1)

        self.pre_attout(None)
        # [B,L,H,n,d_h] -> [B*L,H,n,d_h]
        q = q.reshape(B * L, H, n, d_h)
        k = k.reshape(B * L, H, n, d_h)
        v = v.reshape(B * L, H, n, d_h)

        # [B,1,H,n,n] -> [B*L,H,n,n]
        pb = pb.expand(B, L, H, n, n).reshape(B * L, H, n, n)

        def pair_bias_score_mod(score, b, h, q_idx, kv_idx):
            bias = pb[b, h, q_idx, kv_idx]
            return score + bias

        # [B*L,H,n,d_h]
        attn_out: Tensor = COMPILED_FLEX(q, k, v, pair_bias_score_mod)  # type: ignore
        attn_out = g * attn_out.reshape(B, L, H, n, d_h)
        self.attout(None)

        self.pre_reshape(None)
        attn_out = attn_out.transpose(2, 3).reshape(B, L, n, d_m)
        self.reshape(None)

        # [B,L,n,d_m]
        out = self.out_proj(attn_out)

        # [B,d_m,n,L]
        return out.permute(0, 3, 2, 1)


class RowAttenGated(nn.Module):
    def __init__(self, msa_dim: int, n_heads: int) -> None:
        super().__init__()

        self.msa_dim = msa_dim
        self.n_heads = n_heads

        self.mha = nn.MultiheadAttention(msa_dim, n_heads, batch_first=True, bias=False)
        self.norm = nn.LayerNorm(msa_dim)
        self.g_proj = nn.Linear(msa_dim, msa_dim)
        self.out_proj = nn.Linear(msa_dim, msa_dim)

    def forward(self, msa):
        B, d_m, n, L = msa.size()

        # [B,L,n,d_m] -> [B*n,L,d_m]
        msa = self.norm(msa.transpose(-1, -3))
        r = msa.transpose(-2, -3).reshape(B * n, L, d_m)
        # gating: [B*n,L,d_m]
        g = F.sigmoid(self.g_proj(r))

        attn_out, _ = self.mha(r, r, r, need_weights=False)
        attn_out = (g * attn_out).reshape(B, n, L, d_m)

        # [B,n,L,d_m]
        out = self.out_proj(attn_out)

        # [B,d_m,n,L]
        return out.permute(0, 3, 1, 2)


class Transition(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, dim * 4)
        self.relu = nn.ReLU()
        self.contract = nn.Linear(dim * 4, dim)

    def default_fwd(self, input):
        return self.contract(self.relu(self.expand(self.norm(input))))

    def forward(self, input):
        return self.default_fwd(input)


class MSATransition(Transition):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)

    def forward(self, input):
        # [B,d_m,n,L] -> [B,n,L,d_m]
        msa = input.transpose(-1, -3)
        msa = self.default_fwd(msa)

        # back to [B,d_m,n,L]
        return msa.transpose(-1, -3)


class OuterProductMean(nn.Module):
    def __init__(
        self, msa_dim: int, pair_dim: int, inner_dim: int, symmetric: bool = False
    ) -> None:
        super().__init__()
        self.symmetric = symmetric
        self.norm = nn.LayerNorm(msa_dim)
        self.a_proj = nn.Linear(msa_dim, inner_dim)
        if not symmetric:
            self.b_proj = nn.Linear(msa_dim, inner_dim)
        self.o_proj = nn.Linear(inner_dim**2, pair_dim)

        self.pre_einsum = lambda _: None
        self.einsum = lambda _: None
        self.pre_reshape = lambda _: None
        self.reshape = lambda _: None
        self.pre_indexab = lambda _: None
        self.indexab = lambda _: None

    def register_hooks(self, prefix, memfile, steps, dtypes):
        self.pre_einsum = make_mem_hook(
            memfile, f"{prefix}.einsum", "pre_fwd", steps, dtypes
        )
        self.einsum = make_mem_hook(memfile, f"{prefix}.einsum", "fwd", steps, dtypes)
        self.pre_reshape = make_mem_hook(
            memfile, f"{prefix}.reshape", "pre_fwd", steps, dtypes
        )
        self.reshape = make_mem_hook(memfile, f"{prefix}.reshape", "fwd", steps, dtypes)
        self.pre_indexab = make_mem_hook(
            memfile, f"{prefix}.indexab", "pre_fwd", steps, dtypes
        )
        self.indexab = make_mem_hook(memfile, f"{prefix}.indexab", "fwd", steps, dtypes)

    def forward(self, msa):
        # MSA: [B,d_m,n,L]
        _, _, n, L = msa.size()

        # [B,L,n,d_m]
        msa = self.norm(msa.transpose(-1, -3))

        # [B,n,L,d_in]
        a = self.a_proj(msa).transpose(-2, -3)

        self.pre_einsum(None)
        if self.symmetric:
            # [B,n,n,d_in,d_in]
            outer_prod = torch.einsum("...nld,...NlD->...nNdD", a, a) / L
            i, j = torch.tril_indices(n, n, -1)
            # [B,nC2,d_in,d_in]
            outer_prod = outer_prod[:, i, j, :, :]
        else:
            # [B,n,n,d_in,d_in]
            b = self.b_proj(msa).transpose(-2, -3)
            outer_prod = torch.einsum("...nld,...NlD->...nNdD", a, b) / L
        self.einsum(None)

        self.pre_reshape(None)
        # [B,n,n,d_in*d_in] (or [B,nC2,d_in*d_in])
        o = outer_prod.reshape(outer_prod.shape[:-2] + (-1,))
        self.reshape(None)

        return self.o_proj(o)


class ConcatenationPairUpdate(nn.Module):
    def __init__(
        self, msa_dim: int, pair_dim: int, inner_dim: int, symmetric: bool = False
    ) -> None:
        super().__init__()
        self.symmetric = symmetric
        self.norm = nn.LayerNorm(msa_dim)
        self.a_proj = nn.Linear(msa_dim, inner_dim)
        self.b_proj = nn.Linear(msa_dim, inner_dim)
        self.o_mlp = nn.Sequential(
            nn.Linear(2 * inner_dim, pair_dim),
            nn.ReLU(),
            nn.Linear(pair_dim, pair_dim),
        )

    def forward(self, msa):
        # MSA: [B,d_m,n,L]
        _, _, n, _ = msa.size()

        # [B,L,n,d_m]
        msa = self.norm(msa.transpose(-1, -3))

        # [B,n,d_in]
        a = self.a_proj(msa).mean(dim=1)
        b = self.b_proj(msa).mean(dim=1)

        if self.symmetric:
            ia, ib = torch.tril_indices(n, n, -1)
            a = a[:, ia, :]
            b = b[:, ib, :]
        else:
            a = a[:, :, None, :]
            b = b[:, None, :, :]

        # [B,n,n,2*d_in]
        c = torch.cat([a + b, (a - b).abs()], dim=-1)

        # [B,n,n,d_p]
        out = self.o_mlp(c)

        return out


class PairAttention(nn.Module):
    def __init__(self, pair_dim: int, n_heads: int, symmetric: bool = False) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        self.n_heads = n_heads
        self.symmetric = symmetric

        self.norm = nn.LayerNorm(pair_dim)
        self.mha = nn.MultiheadAttention(
            pair_dim, n_heads, batch_first=True, bias=False
        )
        self.g_proj = nn.Linear(pair_dim, pair_dim)
        self.out_proj = nn.Linear(pair_dim, pair_dim)

    def forward_sym(self, pairs):
        # Pairs: [B,nC2,d_p]

        r = self.norm(pairs)
        g = self.g_proj(r)
        out, _ = self.mha(r, r, r, need_weights=False)
        out = g * out

        return self.out_proj(out)

    def forward_assym(self, pairs):
        # Pairs: [B,n,n,d_p]
        B, n, _, d_p = pairs.size()

        r = self.norm(pairs).reshape(B * n, n, d_p)
        g = self.g_proj(r)
        out, _ = self.mha(r, r, r, need_weights=False)
        out = (g * out).reshape(B, n, n, d_p)

        return self.out_proj(out)

    def forward(self, pairs):
        if self.symmetric:
            # [B,nC2,d_p]
            return self.forward_sym(pairs)
        else:
            # [B,n,n,d_p]
            return self.forward_assym(pairs)


class EvoPFBlock(nn.Module):
    def __init__(
        self,
        msa_dim: int,
        pair_dim: int,
        n_heads: int,
        use_opm: bool = True,
        symmetric: bool = False,
        use_deepspeed: bool = False,
        use_flexattention: bool = False,
    ) -> None:
        super().__init__()
        self.msa_dim = msa_dim
        self.pairs_dim = pair_dim
        self.n_heads = n_heads
        self.use_opm = use_opm

        # MSA stack
        self.row_atten = RowAttenGated(msa_dim, n_heads)
        self.col_atten = ColAttenPairBias(
            msa_dim, pair_dim, n_heads, symmetric, use_deepspeed, use_flexattention
        )
        self.msa_trans = MSATransition(msa_dim)

        # Pair stack
        if self.use_opm:
            self.pair_update = OuterProductMean(msa_dim, pair_dim, 32, symmetric)
        else:
            self.pair_update = ConcatenationPairUpdate(msa_dim, pair_dim, 32, symmetric)
        self.pair_atten = PairAttention(pair_dim, n_heads, symmetric)
        self.pair_trans = Transition(pair_dim)

    def forward(self, msa, pairs):
        # MSA: [B,d_m,n,L]
        # Pairs: [B,n,n,d_p]

        ## MSA STACK
        # Col-Wise attention w/ pair biases
        res = msa
        msa = self.col_atten(msa, pairs) + res

        # Row-size attention
        res = msa
        msa = self.row_atten(msa) + res

        # MSA Transition
        res = msa
        msa = self.msa_trans(msa) + res

        ## PAIR STACK
        # Outer product mean to update pairs
        res = pairs
        pairs = self.pair_update(msa) + res

        # Apply self attention to pairs
        res = pairs
        pairs = self.pair_atten(pairs) + res

        # Pair transition
        res = pairs
        pairs = self.pair_trans(pairs) + res

        return msa, pairs


class PairEmbedder(nn.Module):
    def __init__(self, pair_dim: int, symmetric: bool = False) -> None:
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Conv2d(in_channels=23, out_channels=pair_dim, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.symmetric = symmetric

    def forward_sym(self, msa):
        msa_emb = self.embedder(msa)  # [b,d_p,L,n]
        s2p = seq2pairs(msa.size(-1), lower=True, diff=False).to(msa)  # [nC2,n]
        pair_sums = msa_emb @ s2p.T  # [b,d_p,L,nC2]
        pairs = pair_sums.mean(dim=-2)  # [b,d_p,nC2]

        return pairs.permute(0, 2, 1)  # [d,nC2,d_p]

    def forward_assym(self, msa):
        msa_emb = self.embedder(msa)  # [b,d_p,L,n]
        outer_sum = msa_emb.unsqueeze(-1) + msa_emb.unsqueeze(-2)  # [b,d_p,L,n,n]
        pairs = outer_sum.mean(dim=-3)  # [b,d_p,n,n]

        return pairs.permute(0, 2, 3, 1)  # [b,n,n,d_p]

    def forward(self, msa):
        if self.symmetric:
            # [d,nC2,d_p]
            return self.forward_sym(msa)
        else:
            # [b,n,n,d_p]
            return self.forward_assym(msa)


class DistanceMLP(nn.Module):
    def __init__(self, pair_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.ReLU(),
            nn.Linear(pair_dim, 1),
            nn.Softplus(),
        )

    def forward(self, pairs):
        # N = (n,n) or nC2
        # [b,N,d_p] -> [b,N,1]
        return self.mlp(pairs)


class ParamsFromSeq(nn.Module):
    def __init__(self, d_msa: int, n_params: int, pos: bool = True, eps: float = 1e-8):
        super().__init__()
        self.pos = pos
        self.eps = eps
        self.lin = nn.Linear(in_features=d_msa, out_features=n_params)

    def forward(self, seq: Tensor) -> Tensor:
        if self.pos:
            return F.softplus(self.lin(seq)) + self.eps
        return self.lin(seq)


class ParamsFromPair(nn.Module):
    def __init__(
        self,
        d_msa: int,
        n_params: int,
        symmetric: bool = False,
        pos: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.pos = pos
        self.eps = eps
        self.bil = nn.Bilinear(
            in1_features=d_msa, in2_features=d_msa, out_features=n_params
        )
        if symmetric:
            parametrize.register_parametrization(self.bil, "weight", Symmetric())

    def forward(self, seq1: Tensor, seq2: Tensor) -> Tensor:
        if self.pos:
            return F.softplus(self.bil(seq1, seq2)) + self.eps
        return self.bil(seq1, seq2)


class QSampler(nn.Module):
    def __init__(self, h_dim: int, use_mlp: bool):
        super().__init__()
        self.h_dim = h_dim
        self.use_mlp = use_mlp

        self.bil = ParamsFromPair(h_dim, h_dim if self.use_mlp else 1, pos=False, symmetric=True)
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4 * h_dim),
                nn.ReLU(),
                nn.Linear(4 * h_dim, 1),
            )

        nn.init.xavier_uniform_(self.bil.bil.parametrizations.weight.original)
        nn.init.constant_(self.bil.bil.bias, 0.0)

    def forward(self, seqs1: Tensor, seqs2: Tensor):
        if self.use_mlp:
            return self.mlp(self.bil(seqs1, seqs2))
        else:
            return self.bil(seqs1, seqs2)


class EvoPF(nn.Module):
    def __init__(
        self,
        n_blocks: int = 6,
        n_heads: int = 4,
        h_dim: int = 64,
        pair_dim: int = 256,
        use_opm: bool = True,
        distance_mlp: bool = False,
        symmetric: bool = False,
        use_deepspeed: bool = False,
        use_flexattention: bool = False,
        use_bilinear_embedder: bool = False,
        use_brlens: bool = True,
    ):
        super().__init__()
        self.symmetric = symmetric
        self.msa_embedder = MSAEmbedder(h_dim)
        self.pair_embedder = PairEmbedder(pair_dim, symmetric)
        self.evoblocks = nn.ModuleList(
            [
                EvoPFBlock(
                    h_dim,
                    pair_dim,
                    n_heads,
                    use_opm,
                    symmetric,
                    use_deepspeed,
                    use_flexattention,
                )
                for _ in range(n_blocks)
            ]
        )

        self.pairs_to_dm = (
            DistanceMLP(pair_dim) if distance_mlp else nn.Linear(pair_dim, 1)
        )

        self.output_msa_emb = use_bilinear_embedder
        self.use_brlens = use_brlens

        if use_bilinear_embedder:

            self.e_norm = nn.GroupNorm(8, h_dim)
            self.parent_embedder = ParamsFromPair(
                h_dim, h_dim, pos=False, symmetric=True
            )
            self.q_embedder = QSampler(h_dim, use_mlp=USE_Q_MLP)

            if self.use_brlens:
                # self.brlen_sampler = ParamsFromPair(h_dim, 1, pos=True, symmetric=True)
                # nn.init.xavier_uniform_(self.brlen_sampler.bil.parametrizations.weight.original, gain=1)
                # nn.init.constant_(self.brlen_sampler.bil.bias, 0.0)
                
                # self.predict_lognormal_params = ParamsFromPair(h_dim, 2, pos=False, symmetric=True)
                # nn.init.xavier_uniform_(self.predict_lognormal_params.bil.parametrizations.weight.original, gain=1)
                # nn.init.constant_(self.predict_lognormal_params.bil.bias, 0.0)

                # self.predict_beta_params = ParamsFromPair(h_dim, 2, pos=False, symmetric=False)
                # nn.init.xavier_uniform_(self.predict_beta_params.bil.weight, gain=1)
                # nn.init.constant_(self.predict_beta_params.bil.bias, 0.0)

                self.gamma_sampler = ParamsFromPair(h_dim, 2, pos=False, symmetric=True)
                nn.init.xavier_uniform_(self.gamma_sampler.bil.parametrizations.weight.original, gain=1)
                nn.init.constant_(self.gamma_sampler.bil.bias, 0.0)

                self.beta_sampler = ParamsFromPair(h_dim, 2, pos=False, symmetric=False)
                nn.init.xavier_uniform_(self.beta_sampler.bil.weight, gain=1)
                nn.init.constant_(self.beta_sampler.bil.bias, 0.0)

                

    def embed_parent_node(self, left: Tensor, right: Tensor):
        # [B,d_m] x 2 -> [B,d_m]
        return self.parent_embedder(left, right)

    def embed_input(self, input: Tensor):
        # [b,d_m,n,L]
        msa_emb = self.msa_embedder(input)
        # [b,n,n,d_p]
        pairs = self.pair_embedder(input)

        return msa_emb, pairs

    def forward(self, input):
        msa, pairs = self.embed_input(input)

        for block in self.evoblocks:
            msa, pairs = block(msa, pairs)

        # Compute distances
        dm = self.pairs_to_dm(pairs).squeeze(-1)
        if not self.symmetric:
            dm = (dm + dm.transpose(-1, -2)) / 2.0

        if self.output_msa_emb:
            msa_emb = msa.mean(-1)
            # If we don't use dm in the loss (e.g. with learnable Q), then torch.Distributed will complain
            # Here I am making sure that dm is used with no consequence on the computation hopefully.
            return dm, msa_emb + dm.sum() * 0. 

        return dm
