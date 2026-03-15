import math
import sys
from typing import Any, Dict, Optional

import torch  # type:ignore
from torch import nn

EPS = 1e-8
SOFTPLUS_EPS = 1e-5


def seq2pairs(n, lower=False, diff=False):
    """
    Gives a matrix which will transform from sequence space to pair space:

     - If lower=True, then the order of the pairs corresponds to the order
     of the lower triangular vector of the pairwise matrix. By default the
     order corresponds to the upper triangular vector

     - If diff=True, then multiplying this matrix with the MSA embedding
     will yield the difference between embeddings instead of the sum by
     default (e.g. useful if computing pairwise distances)
    """
    n_pairs = n * (n - 1) // 2
    seq2pair = torch.zeros(n_pairs, n)

    # Compute where to put 1
    row_idx = torch.arange(n_pairs).long()

    idx = torch.triu_indices(n, n, offset=1)
    if lower:
        idx = torch.tril_indices(n, n, offset=-1)

    # Define fill values
    fill = torch.Tensor([1])
    fill2 = fill * (-1 if diff else 1)

    # Fill the matrix and return
    seq2pair = seq2pair.index_put((row_idx, idx[0]), fill)
    return seq2pair.index_put((row_idx, idx[1]), fill2)


class DistanceModule(nn.Module):
    def __init__(self, use_cls: bool = False):
        super().__init__()
        self.use_cls = use_cls

    def reduce_seqdim(self, input, mask=None):
        """
        Goes from [BxHxLxN] to [BxHxN]
        if using CLS just subsets to the CLS position.
        if not averages over the sequence lenght (minus CLS position)
        taking into accound padded sites
        """
        # Input shape: (batch_size, embed_dim, seq_len, n_seqs)

        # Return only CLS position, leave all other dimensions untouched
        if self.use_cls:
            return input[:, :, 0, :]

        # If we are not usign the CLS remove it from sequence
        out = input[:, :, 1:, :]
        if mask is not None:
            mask = mask[:, None, 1:, :]  # remove CLS from mask + add broadcast dim
            out = out.masked_fill(~mask, 0)
            # Count number of un-masked sites in each sequence.
            # When the whole sequence is masked then the numerator will be 0
            # so we can safely replace the 0 values in the denominator
            # with a 1 to avoid dividing by 0
            denom = (s := mask.sum(dim=-2)).masked_fill(s == 0, 1)
            return out.sum(dim=-2) / denom
        else:
            # Simply average over the sequence length
            return out.mean(dim=-2)


class SymmetricBilinearPairwise(DistanceModule):
    """
    Implementing symmetric bilinear to compute distances between
    sequence embeddings (à la DEDAL).
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.0,
        use_bias: bool = True,
        use_sqrt_norm: bool = True,
        factorized_kernel: bool = True,
        use_cls: bool = True,
        ensure_positive: bool = True,
        eps: float = SOFTPLUS_EPS,
    ):
        super().__init__(use_cls=use_cls)

        self.embed_dim = embed_dim
        self.dropout = dropout
        self.use_bias = use_bias
        self.use_sqrt_norm = use_sqrt_norm
        self.factorized_kernel = factorized_kernel
        self.ensure_positive = ensure_positive
        self.eps = eps

        self.w = nn.Parameter(self.init_kernel())

        if self.use_bias:
            self.b = nn.Parameter(torch.zeros((1)))

        self.drop = nn.Dropout(dropout)

    def init_kernel(self):
        w = torch.empty((self.embed_dim, self.embed_dim))
        nn.init.xavier_uniform_(w)
        if self.factorized_kernel:
            return torch.matmul(w.T, w)
        else:
            return 0.5 * (w + w.T)

    def forward(self, input, mask=None):
        _, dim, _, n_seqs = input.shape

        # From (batch_size, embed_dim, seq_len, n_seqs) -> (batch_size, n_seqs, embed_dim)
        x = self.drop(self.reduce_seqdim(input, mask))
        x = x.transpose(-2, -1)

        # Compute bilinear form -> (batch_size, n_seqs, n_seqs)
        w = 0.5 * (self.w + self.w.T)
        out = torch.einsum("bir,rs,bjs->bij", x, w, x)

        if self.use_sqrt_norm:
            dim = torch.tensor(dim)
            out = out / dim.sqrt()

        if self.use_bias:
            out = out + self.b

        if self.ensure_positive:
            # Adding the eps to avoid vanishing gradients for negative values of out
            # see https://github.com/tensorflow/probability/issues/751
            out = torch.nn.functional.softplus(out) + self.eps

        # Symmetrize (Although I'm pretty sure this is redundant...)
        dm = 0.5 * (out + out.mT)

        def tril_indices(rows: int, cols: int, offset):
            return torch.ones(rows, cols).tril(offset).nonzero().T

        # Move to lower triangle -> (batch_size, n_pairs)
        # idx = torch.tril_indices(n_seqs, n_seqs, offset=-1)
        idx = tril_indices(n_seqs, n_seqs, offset=-1)
        dists = dm[:, idx[0], idx[1]]

        return dists


class EuclideanPairwiseDistances(DistanceModule):
    """
    Compute pairwise euclidean distances on between sequences
    on the average emebdding
    """

    def __init__(
        self,
        dropout: float = 0.0,
        eps: float = EPS,
        use_cls: bool = False,
        learnable: bool = False,
        use_bias: bool = False,
        embed_dim: Optional[int] = None,  # Needed if `learnable == True`
    ):
        super().__init__(use_cls=use_cls)
        self.dropout = dropout
        self.eps = eps
        self.learnable = learnable
        self.embed_dim = embed_dim
        self.use_bias = use_bias

        if self.learnable:
            if self.embed_dim is None:
                raise ValueError(
                    f"If {self.__class__} is set to learnable, then the embeding dimension must be specified"
                )
            self.weights = nn.Linear(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                bias=self.use_bias,
            )

    def forward(self, input, mask=None):
        _, _, _, n_seqs = input.shape

        # From (batch_size, embed_dim, seq_len, n_seqs) -> (batch_size, n_seqs, embed_dim)
        x = self.reduce_seqdim(input, mask)
        x = x.transpose(-2, -1)

        # Get index of lower triangle of the distance matrix -> (n_pairs,n_seqs)
        s2p = seq2pairs(n_seqs, lower=True, diff=True).to(x.device)

        # Compute pairwise differences -> (batch_size, n_pairs, embed_dim)
        diffs = torch.matmul(s2p, x)
        if self.learnable:
            diffs = self.weights(diffs)

        # Square, sum over embed dim + sqrt for euclidean distance
        dists = ((diffs**2).sum(dim=-1) + self.eps).sqrt()

        return dists


class BaseCrossAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embed_dim_q: int,
        embed_dim_kv: int,
        qk_dim: int,
        out_dim: int,
        dropout: float,
    ):
        super().__init__()

        if qk_dim % n_heads != 0 or out_dim % n_heads != 0:
            raise ValueError(
                "Dimensions  must be divisible by the number of heads.\n"
                f"QK: {qk_dim}, Out: {out_dim} -> n_heads: {n_heads}"
            )

        self.n_heads = n_heads
        self.embed_dim_q = embed_dim_q
        self.embed_dim_kv = embed_dim_kv
        self.qk_dim = qk_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.qk_head_dim = qk_dim // n_heads
        self.out_head_dim = out_dim // n_heads

        self.q_proj = nn.Linear(embed_dim_q, qk_dim)
        self.k_proj = nn.Linear(embed_dim_kv, qk_dim)
        self.v_proj = nn.Linear(embed_dim_kv, out_dim)

        self.out_proj = nn.Linear(out_dim, out_dim)

        self.atten_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)


class LinearKernelCrossAttention(BaseCrossAttention):
    def __init__(
        self,
        n_heads: int,
        embed_dim_q: int,
        embed_dim_kv: int,
        qk_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__(n_heads, embed_dim_q, embed_dim_kv, qk_dim, out_dim, dropout)

        self.elu = nn.ELU()
        self.eps = eps

    def forward(self, input_q, input_kv, mask=None):
        batch_size_q, spect_dim_q, atten_dim_q, embed_dim_q = input_q.size()  # CLS
        batch_size_kv, spect_dim_kv, atten_dim_kv, embed_dim_kv = input_kv.size()  # MSA

        # Sanity check
        assert batch_size_q == batch_size_kv
        assert atten_dim_q == atten_dim_kv
        assert embed_dim_q == self.embed_dim_q
        assert embed_dim_kv == self.embed_dim_kv

        if mask is not None:
            raise NotImplementedError("Masking not yet implemented")

        k = (
            self.k_proj(input_kv)
            .view(
                batch_size_kv,
                spect_dim_kv,
                atten_dim_kv,
                self.n_heads,
                self.qk_head_dim,
            )
            .transpose(2, 3)
        )
        v = (
            self.v_proj(input_kv)
            .view(
                batch_size_kv,
                spect_dim_kv,
                atten_dim_kv,
                self.n_heads,
                self.out_head_dim,
            )
            .transpose(2, 3)
        )
        q = (
            self.q_proj(input_q)
            .view(
                batch_size_q, spect_dim_q, atten_dim_q, self.n_heads, self.qk_head_dim
            )
            .transpose(2, 3)
        )

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        KtV = k.transpose(-1, -2) @ v

        Z = 1 / (q @ k.transpose(-1, -2).sum(dim=-1, keepdim=True) + self.eps)
        # Z = Z.expand(
        #     batch_size_q, spect_dim_kv, self.n_heads, atten_dim_q, self.qk_head_dim
        # ) # Why do we need this expand when we can just braodcast ?...

        V = Z * (q @ KtV)
        V = (
            V.transpose(2, 3)
            .contiguous()
            .view(batch_size_q, -1, atten_dim_q, self.out_dim)
        )

        out = self.proj_drop(self.out_proj(V))

        return out


class ScaledDotProductCrossAttention(BaseCrossAttention):
    def __init__(
        self,
        n_heads: int,
        embed_dim_q: int,
        embed_dim_kv: int,
        qk_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(n_heads, embed_dim_q, embed_dim_kv, qk_dim, out_dim, dropout)

    def forward(self, input_q, input_kv, mask=None):
        batch_size_q, spect_dim_q, atten_dim_q, embed_dim_q = input_q.size()  # CLS
        batch_size_kv, spect_dim_kv, atten_dim_kv, embed_dim_kv = input_kv.size()  # MSA

        # Sanity check
        assert batch_size_q == batch_size_kv
        assert atten_dim_q == atten_dim_kv
        assert embed_dim_q == self.embed_dim_q
        assert embed_dim_kv == self.embed_dim_kv

        if mask is not None:
            raise NotImplementedError("Masking not yet implemented")

        k = (
            self.k_proj(input_kv)
            .view(
                batch_size_kv,
                spect_dim_kv,
                atten_dim_kv,
                self.n_heads,
                self.qk_head_dim,
            )
            .transpose(2, 3)
        )
        v = (
            self.v_proj(input_kv)
            .view(
                batch_size_kv,
                spect_dim_kv,
                atten_dim_kv,
                self.n_heads,
                self.out_head_dim,
            )
            .transpose(2, 3)
        )
        q = (
            self.q_proj(input_q)
            .view(
                batch_size_q, spect_dim_q, atten_dim_q, self.n_heads, self.qk_head_dim
            )
            .transpose(2, 3)
        )

        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = torch.softmax(attn_scores / (self.qk_head_dim**0.5), dim=-1)

        z = torch.matmul(attn_weights, v)
        z = (
            z.contiguous()
            .transpose(2, 3)
            .view(batch_size_q, spect_dim_kv, atten_dim_q, self.out_dim)
        )

        out = self.proj_drop(self.out_proj(z))

        return out


class BaseAttention(nn.Module):
    """
    Base module to implement various self-attention mechanisms
    Allows for (Q,K) and V to have different dimensions
    """

    def __init__(
        self,
        nb_heads: int,
        embed_dim: int,
        qk_dim: Optional[int] = None,
        dropout: float = 0.0,
        # eps: float = 1e-6,
    ):
        super().__init__()

        # By default all matrices have the same shape
        if qk_dim is None:
            qk_dim = embed_dim

        if embed_dim % nb_heads != 0 or qk_dim % nb_heads != 0:
            raise ValueError(
                "Embed dim and QK dim (if specified) mus tbe divisible by the number of heads.\n"
                f"Embed: {embed_dim}, QK: {qk_dim} -> n_heads: {nb_heads}"
            )

        # Dimensions and parameters
        self.embed_dim = embed_dim
        self.qk_dim = qk_dim
        self.nb_heads = nb_heads
        self.dropout = dropout

        self.head_dim = embed_dim // nb_heads
        self.head_qk_dim = qk_dim // nb_heads

        # Projectors
        self.k_proj = nn.Linear(embed_dim, qk_dim)
        self.q_proj = nn.Linear(embed_dim, qk_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.atten_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)


class ScaledDotProductAttention(nn.Module):
    """
    Multi-head scaled dot product wrapper around pytorch's efficient implementation
    """

    def __init__(self, nb_heads: int, embed_dim: int, dropout: float = 0):
        super().__init__()

        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.dropout = dropout

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=nb_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, input, mask=None):
        batch_size, spect_dim, atten_dim, embed_dim = input.size()
        if mask is not None:
            expected_shape = (batch_size, spect_dim, atten_dim)
            assert (
                mask.shape == expected_shape
            ), f"Mask has incorrect dimensions, expeceted {expected_shape} but got {mask.shape}"

        # Pytorch's multihead attention expects Q,K and V of shape [B,L,d] with L being the
        # attention dimension. So we need to merge the spectator dimension with the batch dimension
        reshaped = input.reshape(batch_size * spect_dim, atten_dim, embed_dim)
        if mask is not None:
            mask = mask.reshape(batch_size * spect_dim, atten_dim)

        # Not returning weigts makes PT use the optimized `scaled_dot_product_attention` function
        # which can use Flash attention and other optimized attention implementations
        out, _ = self.mha(
            reshaped, reshaped, reshaped, key_padding_mask=mask, need_weights=False
        )

        # Reshaped to the original shape
        out = out.reshape(batch_size, spect_dim, atten_dim, embed_dim)

        # Final output projection
        out = self.out_proj(self.proj_drop(out))

        return out


class LinearKernelAttention(BaseAttention):
    """
    Implementation of the Linear Kernel Attention from:
    doi.org/10.48550/arXiv.2006.16236
    """

    def __init__(
        self, nb_heads: int, embed_dim: int, dropout: float = 0, eps: float = 1e-6
    ):
        super().__init__(nb_heads, embed_dim, None, dropout)

        self.elu = nn.ELU()
        self.eps = eps

    def forward(self, input, mask=None):
        """
        mask should be of shape (batch_size, nb_row, nb_col)
        """
        batch_size, nb_row, nb_col, embed_dim = input.size()
        if mask is not None:
            expected_shape = (batch_size, nb_row, nb_col)
            assert (
                mask.shape == expected_shape
            ), f"Mask has incorrect dimensions, expeceted {expected_shape} but got {mask.shape}"

        k = (
            self.k_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim)
            .transpose(2, 3)
        )
        q = (
            self.q_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim)
            .transpose(2, 3)
        )
        v = (
            self.v_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim)
            .transpose(2, 3)
        )

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        if mask is not None:
            # batch_size, nb_row, (nb_heads), nb_col, (head_dim)
            k = k.masked_fill(~mask[:, :, None, :, None], 0)
            q = q.masked_fill(~mask[:, :, None, :, None], 0)

        KtV = k.transpose(-1, -2) @ v

        Z = 1 / (q @ k.transpose(-1, -2).sum(dim=-1, keepdim=True) + self.eps)
        Z = Z.expand(batch_size, nb_row, self.nb_heads, nb_col, self.head_dim)

        V = Z * (q @ KtV)
        V = V.transpose(2, 3).contiguous().view(batch_size, -1, nb_col, embed_dim)

        out = self.proj_drop(self.out_proj(V))

        return out


class ScaledLinearAttention(BaseAttention):
    """Custom Rank-1 Linear Attention used in Phyloformer"""

    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__(nb_heads, embed_dim, nb_heads, dropout)

    def forward(self, input, mask=None):
        batch_size, nb_row, nb_col, embed_dim = input.size()

        k = (
            self.k_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_qk_dim)
            .transpose(2, 3)
        )
        q = (
            self.q_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_qk_dim)
            .transpose(2, 3)
        )
        v = (
            self.v_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim)
            .transpose(2, 3)
        )

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        # Scale Q to keep amplitude under control
        q = q / q.mean(dim=-2, keepdim=True)

        # Normalize K
        k = k / k.sum(
            dim=-2, keepdim=True
        )  # Sum directly on -2 instead of transposing an summing

        KtV = k.transpose(-1, -2) @ v

        V = q @ KtV
        V = V.transpose(2, 3).contiguous().view(batch_size, -1, nb_col, embed_dim)

        out = self.proj_drop(self.out_proj(V))

        return out


class SingleAxialAttention(nn.Module):
    def __init__(
        self,
        nb_heads: int,
        embed_dim: int,
        atten_type: str,
        dropout: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__()

        # Params
        self.nb_heads = nb_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.verbose = verbose

        atten_kwargs = dict(nb_heads=nb_heads, embed_dim=embed_dim, dropout=dropout)

        self.norm = nn.LayerNorm(self.embed_dim)
        if atten_type == "ScaledDotProduct":
            self.atten = ScaledDotProductAttention(**atten_kwargs)  # type: ignore
        elif atten_type == "LinearKernel":
            self.atten = LinearKernelAttention(**atten_kwargs)  # type: ignore
        elif atten_type == "ScaledLinear":
            self.atten = ScaledLinearAttention(**atten_kwargs)  # type: ignore
        else:
            raise ValueError(
                f"Unknown attention type: {atten_type}. "
                "Allowed: 'ScaledDotProduct', 'LinearKernel' and 'ScaledLinear'."
            )

    def forward(self, *_args, **_kwargs):
        raise NotImplementedError(
            "This function must be implemented by descendant classes..."
        )

    def check_shapes(self, input, mask):
        (batch_size, embed_dim, nb_pairs, seq_len) = input.size()

        if self.verbose:
            print(
                f"{self} input:\n\t-b: {batch_size}\n\t-D: {embed_dim}\n\t-n: {nb_pairs}\n\t-L: {seq_len}",
                file=sys.stderr,
            )

        if mask is not None:
            expected_shape = (batch_size, nb_pairs, seq_len)
            assert (
                mask.shape == expected_shape
            ), f"Mask has incorrect dimensions, expeceted {expected_shape} but got {mask.shape}"


class RowWiseAttention(SingleAxialAttention):
    """Does within-sequence attention updates"""

    def __init__(
        self,
        nb_heads: int,
        embed_dim: int,
        atten_type: str,
        dropout: float = 0,
        verbose: bool = False,
    ):
        super().__init__(nb_heads, embed_dim, atten_type, dropout, verbose)

    def forward(self, input, mask=None):
        self.check_shapes(input, mask)

        # Keep residual
        res = input

        # Apply LayerNorm to embedding dimension
        out = self.norm(input.transpose(-1, -3)).transpose(-1, -3)

        # (b, D, n, L) -> (b, n, L, D)
        out = out.permute(0, 2, 3, 1)

        # Apply self-attention along `L`
        out = self.atten(out, mask)

        # (b, n, L, D) -> (b, D, n, L)
        out = out.permute(0, 3, 1, 2)

        return out + res


class ColWiseAttention(SingleAxialAttention):
    """Does within-sequence attention updates"""

    def __init__(
        self,
        nb_heads: int,
        embed_dim: int,
        atten_type: str,
        dropout: float = 0,
        verbose: bool = False,
    ):
        super().__init__(nb_heads, embed_dim, atten_type, dropout, verbose)

    def forward(self, input, mask=None):
        self.check_shapes(input, mask)

        # Keep residual
        res = input

        # Apply LayerNorm to embedding dimension
        out = self.norm(input.transpose(-1, -3)).transpose(-1, -3)

        # (b, D, n, L) -> (b, L, n, D)
        out = out.permute(0, 3, 2, 1)

        if mask is not None:
            # (b, n, L) -> (b, L, n)
            mask = mask.transpose(-1, -2)

        # Apply self-attention along `n`
        out = self.atten(out, mask)

        # (b, L, n, D) -> (b, D, n, L)
        out = out.permute(0, 3, 2, 1)

        return out + res


class AxialFFN(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.norm = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim * 4,
                kernel_size=1,
                stride=1,
            ),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.embed_dim * 4,
                out_channels=self.embed_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.Dropout(self.dropout),
        )

    def forward(self, input):
        res = input

        # Apply LayerNorm to embedding dimension
        out = self.norm(input.transpose(-1, -3)).transpose(-1, -3)

        # Run through FFN
        out = self.ffn(out)

        return out + res


class AxialAttention(nn.Module):
    def __init__(self, nb_heads, embed_dim, atten_type, dropout: float = 0):
        super().__init__()

        self.row_atten = RowWiseAttention(nb_heads, embed_dim, atten_type, dropout)
        self.col_atten = ColWiseAttention(nb_heads, embed_dim, atten_type, dropout)
        self.ffn = AxialFFN(embed_dim, dropout)

    def forward(self, input, mask=None):
        out = self.row_atten(input, mask)
        out = self.col_atten(out, mask)
        return self.ffn(out)


class ReverseAxialAttention(AxialAttention):
    def __init__(self, nb_heads, embed_dim, atten_type, dropout: float = 0):
        super().__init__(nb_heads, embed_dim, atten_type, dropout)

    def forward(self, input, mask=None):
        out = self.col_atten(input, mask)
        out = self.row_atten(out, mask)
        return self.ffn(out)


class MixedAxialAttention(nn.Module):
    def __init__(self, nb_heads: int, embed_dim: int, dropout: float = 0):
        super().__init__()

        self.row_atten = RowWiseAttention(nb_heads, embed_dim, "LinearKernel", dropout)
        self.col_atten = ColWiseAttention(
            nb_heads, embed_dim, "ScaledDotProduct", dropout
        )
        self.ffn = AxialFFN(embed_dim, dropout)

    def forward(self, input, mask=None):
        out = self.row_atten(input, mask)
        out = self.col_atten(out, mask)
        return self.ffn(out)


class ReverseMixedAxialAttention(MixedAxialAttention):
    def __init__(self, nb_heads: int, embed_dim: int, dropout: float = 0):
        super().__init__(nb_heads, embed_dim, dropout)

    def forward(self, input, mask=None):
        out = self.col_atten(input, mask)
        out = self.row_atten(out, mask)
        return self.ffn(out)


class PhyloformerLayer(nn.Module):
    """Phyloformer's Transformer Layer"""

    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        dropout: float,
        use_sla: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.dropout = dropout
        self.use_sla = use_sla

        if self.use_sla:
            self.row_attention = ScaledLinearAttention(
                embed_dim=self.embed_dim, nb_heads=self.nb_heads
            )
            self.col_attention = ScaledLinearAttention(
                embed_dim=self.embed_dim, nb_heads=self.nb_heads
            )
        else:
            self.row_attention = LinearKernelAttention(
                embed_dim=self.embed_dim, nb_heads=self.nb_heads
            )
            self.col_attention = LinearKernelAttention(
                embed_dim=self.embed_dim, nb_heads=self.nb_heads
            )

        # Normalization layers
        self.row_norm = nn.LayerNorm(self.embed_dim)
        self.col_norm = nn.LayerNorm(self.embed_dim)
        self.ffn_norm = nn.LayerNorm(self.embed_dim)

        # Feed forward NN
        self.ffn = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim * 4,
                kernel_size=1,
                stride=1,
            ),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.embed_dim * 4,
                out_channels=self.embed_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.Dropout(self.dropout),
        )

    def forward(self, input, mask=None):
        # Row attention sub-block (batch_size, embed_dim, seq_len, n_seqs)
        res_row = input
        out = self.row_norm(input.transpose(-1, -3)).transpose(-1, -3)
        # permute to (batch_size, seq_len, n_seqs, embed_dim)
        out = self.row_attention(out.permute(0, 2, 3, 1), mask).permute(0, 3, 1, 2)
        out = out + res_row  # residual connection

        # Col attention sub-block
        res_col = out
        out = self.col_norm(out.transpose(-1, -3)).transpose(-1, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)
        out = self.col_attention(out.permute(0, 3, 2, 1), mask).permute(0, 3, 2, 1)
        out = out + res_col

        # FFN sub-block
        res_ffn = out
        out = self.ffn_norm(out.transpose(-1, -3)).transpose(-1, -3)
        out = self.ffn(out)
        out = out + res_ffn

        return out


class Phyloformer(nn.Module):
    """Phyloformer implementation (published)"""

    def __init__(
        self,
        n_blocks: int = 6,
        n_heads: int = 4,
        h_dim: int = 64,
        dropout: float = 0.0,
        n_seqs: int = 20,
        seq_len: int = 200,
        **kwargs,
    ):
        super().__init__()
        self.nb_blocks = n_blocks
        self.nb_heads = n_heads
        self.embed_dim = h_dim
        self.dropout = dropout

        self.embedding_block = nn.Sequential(
            nn.Conv2d(
                in_channels=22, out_channels=self.embed_dim, kernel_size=1, stride=1
            ),
            nn.ReLU(),
        )

        self.seq2pair = seq2pairs(self.n_seqs, lower=False, diff=False)

        self.attention_blocks = nn.ModuleList(
            [
                PhyloformerLayer(
                    embed_dim=self.embed_dim,
                    nb_heads=self.nb_heads,
                    dropout=self.dropout,
                    use_sla=True,
                )
                for _ in range(self.nb_blocks)
            ]
        )

        self.pwFNN = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=1,
                kernel_size=1,
                stride=1,
            ),
            nn.Dropout(self.dropout),
            nn.Softplus(),
        )

    def update_seq2pair(self, new_n_seqs):
        if new_n_seqs != self.n_seqs:
            self.n_seqs = new_n_seqs
            self.seq2pair = seq2pairs(new_n_seqs, lower=False, diff=False)

    def forward(self, input):
        # input: (batch_size, 22, seq_len, n_seqs)
        self.update_seq2pair(input.shape[-1])

        # Embed alignment to embed_dim
        out = self.embedding_block(input)
        # Pair representation -> (batch_size, embed_dim, nb_pairs, seq_len)
        out = torch.matmul(self.seq2pair, out.transpose(-1, -2))

        # Attention
        for block in self.attention_blocks:
            out = block(out)

        # Convolution -> (batch_size, 1, nb_pairs, seq_len)
        out = self.pwFNN(out)

        # Average of sequence length -> (batch_size, nb_pairs)
        out = torch.squeeze(torch.mean(out, dim=-1))

        return out

    def _get_hyperparams(self):
        return dict(
            n_blocks=self.nb_blocks,
            n_heads=self.nb_heads,
            h_dim=self.embed_dim,
            dropout=self.dropout,
            n_seqs=self.n_seqs,
            seq_len=self.seq_len,
        )


class PhyloformerSeq(nn.Module):
    """PF-seq implementation"""

    def __init__(
        self,
        n_blocks: int = 6,
        n_heads: int = 4,
        h_dim: int = 64,
        dropout: float = 0.0,
        use_sla: bool = False,
        use_bilinear: bool = True,
        use_shortcuts: bool = False,
        bilinear_kwargs: Optional[Dict[str, Any]] = None,
        euclidean_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.nb_blocks = n_blocks
        self.nb_heads = n_heads
        self.embed_dim = h_dim
        self.dropout = dropout
        self.use_sla = use_sla
        self.use_bilinear = use_bilinear
        self.use_shortcuts = use_shortcuts
        self.bilinear_kwargs = bilinear_kwargs
        self.euclidean_kwargs = euclidean_kwargs

        self.embedding_block = nn.Sequential(
            nn.Conv2d(
                in_channels=23, out_channels=self.embed_dim, kernel_size=1, stride=1
            ),
            nn.ReLU(),
        )

        self.attention_blocks = nn.ModuleList(
            [
                PhyloformerLayer(
                    embed_dim=self.embed_dim,
                    nb_heads=self.nb_heads,
                    dropout=self.dropout,
                    use_sla=self.use_sla,
                )
                for _ in range(self.nb_blocks)
            ]
        )

        dim = self.embed_dim * self.nb_blocks if self.use_shortcuts else self.embed_dim
        if self.use_bilinear:
            kws = bilinear_kwargs or dict()
            self.distances = SymmetricBilinearPairwise(dim, dropout=self.dropout, **kws)
        else:
            kws = euclidean_kwargs or dict()
            self.distances = EuclideanPairwiseDistances(
                self.dropout, embed_dim=dim, **kws
            )

    def forward(self, input, mask=None):
        # Embed alignment to embed_dim -> (batch_size, embed_dim, seq_len, n_seqs)
        out = self.embedding_block(input)
        if mask is not None:
            out = out.masked_fill(~mask[:, None, :, :], 0)

        # Attention blocks -> (batch_size, embed_dim, seq_len, n_seqs)
        shortcuts = []
        for block in self.attention_blocks:
            out = block(out, mask)
            shortcuts.append(out)

        # Compute lower triangle of pairwise distance matrix -> (batch_size, n_pairs)
        if self.use_shortcuts:
            stacked = torch.cat(shortcuts, dim=1)  # Stack along embed_dim
            dists = self.distances(stacked, mask)
        else:
            dists = self.distances(out, mask)

        return dists

    def _get_hyperparams(self):
        return dict(
            n_blocks=self.nb_blocks,
            n_heads=self.nb_heads,
            h_dim=self.embed_dim,
            dropout=self.dropout,
            use_sla=self.use_sla,
            use_bilinear=self.use_bilinear,
            use_shortcuts=self.use_shortcuts,
            bilinear_kwargs=self.bilinear_kwargs,
            euclidean_kwargs=self.euclidean_kwargs,
        )


class MSAEmbedder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=23, out_channels=embed_dim, kernel_size=1,stride=1),
            nn.ReLU()
        )
    def forward(self, input):
        # [B,d,L,n]
        out = self.block(input)

        # [B,D,n,L]
        return out.transpose(-1,-2)
        

class AltPFSeqBase(nn.Module):
    def __init__(
        self,
        n_blocks: int = 6,
        n_heads: int = 4,
        h_dim: int = 64,
        dropout: float = 0.0,
        use_bilinear: bool = True,
        use_shortcuts: bool = False,
        bilinear_kwargs: Optional[Dict[str, Any]] = None,
        euclidean_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.nb_blocks = n_blocks
        self.nb_heads = n_heads
        self.embed_dim = h_dim
        self.dropout = dropout
        self.use_bilinear = use_bilinear
        self.use_shortcuts = use_shortcuts
        self.bilinear_kwargs = bilinear_kwargs
        self.euclidean_kwargs = euclidean_kwargs

        self.embedding_block = MSAEmbedder(self.embed_dim)

        dim = self.embed_dim * self.nb_blocks if self.use_shortcuts else self.embed_dim
        if self.use_bilinear:
            kws = bilinear_kwargs or dict()
            self.distances = SymmetricBilinearPairwise(dim, dropout=self.dropout, **kws)
        else:
            kws = euclidean_kwargs or dict()
            self.distances = EuclideanPairwiseDistances(
                self.dropout, embed_dim=dim, **kws
            )

    def embed_msa(self, input, mask=None):
        # [B,d,L,n] -> [B,d,n,L]
        out = self.embedding_block(input)
        if mask is not None:
            mask = mask.transpose(-1,-2)
            out = out.masked_fill(~mask[:, None, :, :], 0)

        return out, mask

    def compute_dists(self, input, shortcuts, mask):
        # Compute lower triangle of pairwise distance matrix -> (batch_size, n_pairs)
        if self.use_shortcuts:
            stacked = torch.cat(shortcuts, dim=1)  # Stack along embed_dim
            dists = self.distances(stacked.transpose(-1, -2), mask)
        else:
            dists = self.distances(input.transpose(-1, -2), mask)

        return dists

    def _get_hyperparams(self):
        return dict(
            n_blocks=self.nb_blocks,
            n_heads=self.nb_heads,
            h_dim=self.embed_dim,
            dropout=self.dropout,
            use_sla=self.use_sla,
            use_bilinear=self.use_bilinear,
            use_shortcuts=self.use_shortcuts,
            bilinear_kwargs=self.bilinear_kwargs,
            euclidean_kwargs=self.euclidean_kwargs,
        )


class PhyloformerSeq2(AltPFSeqBase):
    """
    This should be exactly the same as the PhyloformerSeq implementation
    but using the new RowWiseAttention and ColWiseAttention modules.
    This means that it actually does column-wise attention first...
    """

    def __init__(
        self,
        n_blocks: int = 6,
        n_heads: int = 4,
        h_dim: int = 64,
        dropout: float = 0,
        use_sla: bool = False,
        use_bilinear: bool = True,
        use_shortcuts: bool = False,
        bilinear_kwargs: Optional[Dict[str, Any]] = None,
        euclidean_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            n_blocks,
            n_heads,
            h_dim,
            dropout,
            use_bilinear,
            use_shortcuts,
            bilinear_kwargs,
            euclidean_kwargs,
        )

        atten_type = "ScaledLinear" if use_sla else "LinearKernel"
        self.attention_blocks = nn.ModuleList(
            [
                ReverseAxialAttention(
                    embed_dim=self.embed_dim,
                    nb_heads=self.nb_heads,
                    dropout=self.dropout,
                    atten_type=atten_type,
                )
                for _ in range(self.nb_blocks)
            ]
        )

    def forward(self, input, mask=None):
        # Embedding:
        # (batch_size, 23, seq_len, n_seqs) -> (batch_size, embed_dim, n_seqs, seq_len)
        out, mask = self.embed_msa(input, mask)

        # Attention blocks -> (batch_size, embed_dim, n_seqs, seq_len)
        shortcuts = []
        for block in self.attention_blocks:
            out = block(out, mask)
            shortcuts.append(out)

        # Distance:
        # (batch_size, embed_dim, n_seqs, seq_len) -> (batch_size, n_pairs)
        dists = self.compute_dists(out, shortcuts, mask)

        return dists

    def _from_pfseqdict(self, pfseq_dict):
        """
        Load a PhyloformerSeq state dict into a PhyloformerSeq2 object
        """

        def renamer(key: str):
            key = key.replace(".row_attention.", ".col_atten.atten.")
            key = key.replace(".col_attention.", ".row_atten.atten.")
            key = key.replace(".ffn.", ".ffn.ffn.")
            key = key.replace(".row_norm.", ".col_atten.norm.")
            key = key.replace(".col_norm.", ".row_atten.norm.")
            key = key.replace(".ffn_norm.", ".ffn.norm.")
            return key

        self.load_state_dict({renamer(k): v for k, v in pfseq_dict.items()})


class PhyloformerSeqMixed(AltPFSeqBase):
    """
    This is the same as PFSeq implementation using new axial attention modules.
    However the attention types are mixed with linear kernel attention for row-wise attention
    and scaled dot product attention for col-wise attention.
    Here we also have col-attention before row-attention
    """

    def __init__(
        self,
        n_blocks: int = 6,
        n_heads: int = 4,
        h_dim: int = 64,
        dropout: float = 0,
        use_bilinear: bool = True,
        use_shortcuts: bool = False,
        bilinear_kwargs: Optional[Dict[str, Any]] = None,
        euclidean_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            n_blocks,
            n_heads,
            h_dim,
            dropout,
            use_bilinear,
            use_shortcuts,
            bilinear_kwargs,
            euclidean_kwargs,
        )

        self.attention_blocks = nn.ModuleList(
            [
                ReverseMixedAxialAttention(
                    embed_dim=self.embed_dim,
                    nb_heads=self.nb_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.nb_blocks)
            ]
        )

    def forward(self, input, mask=None):
        # Embedding:
        # (batch_size, 23, seq_len, n_seqs) -> (batch_size, embed_dim, n_seqs, seq_len)
        out, mask = self.embed_msa(input, mask)

        # Attention blocks -> (batch_size, embed_dim, n_seqs, seq_len)
        shortcuts = []
        for block in self.attention_blocks:
            out = block(out, mask)
            shortcuts.append(out)

        # Distance:
        # (batch_size, embed_dim, n_seqs, seq_len) -> (batch_size, n_pairs)
        dists = self.compute_dists(out, shortcuts, mask)

        return dists
