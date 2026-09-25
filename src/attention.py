"""Attention Mechanisms: Numerically Stable Softmax, Scaled Dot-Product Attention, and Causal GQA.

===============================================================================
WHAT NEEDS TO BE IMPLEMENTED:
===============================================================================
1. softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
   - Numerically stable softmax along dimension `dim`.
   - Subtract the maximum value along `dim` (keepdim=True) before exponentiating.
   - Sum exponentiated values along `dim` and divide.
   - RESTRICTION: Do NOT use `torch.softmax`, `torch.log_softmax`,
     `torch.nn.functional.softmax`, or `torch.nn.functional.log_softmax`.

2. scaled_dot_product_attention(
       queries: torch.Tensor,
       keys: torch.Tensor,
       values: torch.Tensor,
       mask: torch.Tensor | None = None,
   ) -> torch.Tensor:
   - queries: (..., n_q, d_k)
   - keys:    (..., n_kv, d_k)
   - values:  (..., n_kv, d_v)
   - result:  (..., n_q, d_v)
   - Leading dimensions must broadcast. n_q and n_kv may differ, and d_v need not equal d_k.
   - Scores: S = (queries @ keys.transpose(-2, -1)) / sqrt(d_k).
   - Mask: Optional boolean mask where True means allowed and False means masked.
     * Replace False entries with -infinity (float("-inf")).
     * Validation: For each query position, check if at least one key is allowed.
       If an entire row is masked (all False), raise a ValueError.
   - Probabilities: Apply custom stable `softmax(scores, dim=-1)`.
   - Output: probs @ values.
   - RESTRICTION: Do NOT use `torch.nn.functional.scaled_dot_product_attention`.

3. CausalGroupedQueryAttention:
   - Subclass `torch.nn.Module`.
   - Constructor parameters:
     * d_model: int
     * n_q_heads: int
     * n_kv_heads: int
     * context_length: int
     * rope_theta: float
     * device: torch.device | None = None
     * dtype: torch.dtype | None = None
   - Checks:
     * Require d_model % n_q_heads == 0.
     * Require n_q_heads % n_kv_heads == 0.
     * head_dim = d_model // n_q_heads.
     * group_size g = n_q_heads // n_kv_heads.
   - Linear Projections (bias-free using custom Linear):
     * q_proj: Linear(d_model, n_q_heads * head_dim)
     * k_proj: Linear(d_model, n_kv_heads * head_dim)
     * v_proj: Linear(d_model, n_kv_heads * head_dim)
     * out_proj: Linear(n_q_heads * head_dim, d_model)
   - Positional Encoding:
     * RotaryPositionalEmbedding(rope_theta, head_dim, context_length, device=device)
   - Forward pass:
     forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor
     * x shape: (batch_size, sequence_length, d_model)
     * token_positions: None or (..., sequence_length).
       If None, default to torch.arange(sequence_length, device=x.device).
     * Project x to Q, K, V.
     * Grouped arrangement:
       - Q: [B, n, h_q * dh] -> [B, n, h_kv, g, dh] -> transpose to [B, h_kv, g, n, dh]
       - K: [B, n, h_kv * dh] -> [B, n, h_kv, dh] -> transpose to [B, h_kv, n, dh]
       - V: [B, n, h_kv * dh] -> [B, n, h_kv, dh] -> transpose to [B, h_kv, n, dh]
     * Apply RoPE to Q and K (broadcast over head axes). Values are NOT rotated.
     * Compute scaled dot-product attention:
       - Multiply Q [B, h_kv, g, n, dh] and K [B, h_kv, n, dh] over dh -> scores [B, h_kv, g, n, n].
       - Broadcast K across the g queries in each group without materializing repeats.
       - Scale by 1 / sqrt(dh).
       - Apply causal mask: j <= i is True, j > i is False (-inf).
       - Stable softmax over key dimension.
       - Contract with V [B, h_kv, n, dh] -> output [B, h_kv, g, n, dh].
     * Merge heads back preserving a = (h - 1)*g + r:
       - [B, h_kv, g, n, dh] -> [B, n, h_kv, g, dh] -> [B, n, h_q * dh] = [B, n, d_model]
     * Apply out_proj.
   - RESTRICTION: Do NOT use `torch.nn.MultiheadAttention`.

===============================================================================
PSEUDOCODE:
===============================================================================

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 1. max_val = torch.max(x, dim=dim, keepdim=True).values
    # 2. exp_x = torch.exp(x - max_val)
    # 3. sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    # 4. return exp_x / sum_exp

def scaled_dot_product_attention(queries, keys, values, mask=None):
    # 1. d_k = queries.shape[-1]
    # 2. scores = torch.matmul(queries, keys.transpose(-2, -1)) / (d_k ** 0.5)
    # 3. if mask is not None:
    #        if not mask.any(dim=-1).all():
    #            raise ValueError("Every query must have at least one unmasked key")
    #        scores = scores.masked_fill(~mask, float("-inf"))
    # 4. weights = softmax(scores, dim=-1)
    # 5. return torch.matmul(weights, values)

class CausalGroupedQueryAttention(torch.nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, context_length, rope_theta, device=None, dtype=None):
        super().__init__()
        # 1. Store hyperparameters and compute head_dim, group_size
        # 2. self.q_proj = Linear(d_model, n_q_heads * head_dim, device=device, dtype=dtype)
        # 3. self.k_proj = Linear(d_model, n_kv_heads * head_dim, device=device, dtype=dtype)
        # 4. self.v_proj = Linear(d_model, n_kv_heads * head_dim, device=device, dtype=dtype)
        # 5. self.out_proj = Linear(n_q_heads * head_dim, d_model, device=device, dtype=dtype)
        # 6. self.rope = RotaryPositionalEmbedding(rope_theta, head_dim, context_length, device=device)

    def forward(self, x, token_positions=None):
        # B, n, _ = x.shape
        # if token_positions is None:
        #     token_positions = torch.arange(n, device=x.device)
        # q = self.q_proj(x).view(B, n, self.n_kv_heads, self.group_size, self.head_dim)
        # k = self.k_proj(x).view(B, n, self.n_kv_heads, self.head_dim)
        # v = self.v_proj(x).view(B, n, self.n_kv_heads, self.head_dim)
        # q = self.rope(q, token_positions).permute(0, 2, 3, 1, 4)  # [B, h_kv, g, n, dh]
        # k = self.rope(k, token_positions).permute(0, 2, 1, 3)     # [B, h_kv, n, dh]
        # v = v.permute(0, 2, 1, 3)                                # [B, h_kv, n, dh]
        # scores = torch.einsum("bhgnd,bhmd->bhgnm", q, k) / (self.head_dim ** 0.5)
        # causal_mask = torch.tril(torch.ones((n, n), dtype=torch.bool, device=x.device))
        # scores = scores.masked_fill(~causal_mask, float("-inf"))
        # probs = softmax(scores, dim=-1)
        # out = torch.einsum("bhgnm,bhmd->bhgnd", probs, v)
        # out = out.permute(0, 3, 1, 2, 4).reshape(B, n, self.d_model)
        # return self.out_proj(out)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from src.layers import Linear
from src.rope import RotaryPositionalEmbedding


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_val)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("feature dimension of queries and keys must match")
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError("sequence length of keys and values must match")
    if mask is not None and mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")

    d_k = queries.shape[-1]
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        if not mask.any(dim=-1).all():
            raise ValueError("Every query must have at least one unmasked key")
        scores = scores.masked_fill(~mask, -float("inf"))
    weights = softmax(scores, dim=-1)
    return torch.matmul(weights, values)


class CausalGroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        context_length: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_q_heads != 0:
            raise ValueError("d_model must be divisible by n_q_heads")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")

        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.context_length = context_length
        self.rope_theta = float(rope_theta)
        self.head_dim = d_model // n_q_heads
        self.group_size = n_q_heads // n_kv_heads

        self.q_proj = Linear(d_model, n_q_heads * self.head_dim, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.out_proj = Linear(n_q_heads * self.head_dim, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(self.rope_theta, self.head_dim, context_length, device=device)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape
        if token_positions is None:
            token_positions = torch.arange(sequence, device=x.device)

        q = self.q_proj(x).view(batch, sequence, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, sequence, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, sequence, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        q = q.view(batch, self.n_kv_heads, self.group_size, sequence, self.head_dim)
        scores = torch.einsum("bhgqd,bhkd->bhgqk", q, k) / math.sqrt(self.head_dim)

        causal_mask = torch.tril(torch.ones((sequence, sequence), dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal_mask, -float("inf"))
        probs = softmax(scores, dim=-1)

        out = torch.einsum("bhgqk,bhkd->bhgqd", probs, v)
        out = out.permute(0, 3, 1, 2, 4).reshape(batch, sequence, self.d_model)
        return self.out_proj(out)
