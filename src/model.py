"""Transformer Block and Full Transformer LM.

===============================================================================
WHAT NEEDS TO BE IMPLEMENTED:
===============================================================================
1. TransformerBlock:
   - Subclass `torch.nn.Module`.
   - Constructor parameters:
     * d_model: int
     * n_q_heads: int
     * n_kv_heads: int
     * d_ff: int
     * context_length: int
     * rope_theta: float
     * norm_eps: float = 1e-5
     * device: torch.device | None = None
     * dtype: torch.dtype | None = None
   - Submodules:
     * attention_norm: RMSNorm(d_model, norm_eps)
     * attention: CausalGroupedQueryAttention(d_model, n_q_heads, n_kv_heads, context_length, rope_theta)
     * ffn_norm: RMSNorm(d_model, norm_eps)
     * ffn: SwiGLU(d_model, d_ff)
   - Pre-norm arrangement:
     u = x + attention(attention_norm(x), token_positions=token_positions)
     y = u + ffn(ffn_norm(u))
   - Input shape: (batch_size, sequence_length, d_model), returns same shape.

2. TransformerLM:
   - Subclass `torch.nn.Module`.
   - Constructor parameters:
     * vocab_size: int
     * context_length: int
     * d_model: int
     * num_layers: int
     * n_q_heads: int
     * n_kv_heads: int
     * d_ff: int
     * rope_theta: float
     * norm_eps: float = 1e-5
     * device: torch.device | None = None
     * dtype: torch.dtype | None = None
   - Store: self.context_length = context_length
   - Submodules:
     * token_embedding: Embedding(vocab_size, d_model)
     * blocks: ModuleList of num_layers TransformerBlock instances
     * final_norm: RMSNorm(d_model, norm_eps)
     * lm_head: Linear(d_model, vocab_size)  (untied weights!)
   - Sequence length validation:
     * Require 1 <= sequence_length <= self.context_length.
     * Reject empty sequences or sequences longer than context_length.
   - Forward pass:
     forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor
     * x = token_embedding(token_ids)
     * for block in blocks: x = block(x, token_positions=token_positions)
     * x = final_norm(x)
     * logits = lm_head(x)
     * return logits  # unnormalized logits, NO softmax inside the model
   - Total parameter count with fixed assignment configuration: exactly 19,272,192.
   - RESTRICTION: Do NOT use `torch.nn.Transformer`, `torch.nn.TransformerEncoder`,
     or `torch.nn.TransformerEncoderLayer`.

===============================================================================
PSEUDOCODE:
===============================================================================

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, d_ff, context_length, rope_theta, norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        # self.attention_norm = RMSNorm(d_model, norm_eps=norm_eps, device=device, dtype=dtype)
        # self.attention = CausalGroupedQueryAttention(d_model, n_q_heads, n_kv_heads, context_length, rope_theta, device=device, dtype=dtype)
        # self.ffn_norm = RMSNorm(d_model, norm_eps=norm_eps, device=device, dtype=dtype)
        # self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        # x = x + self.attention(self.attention_norm(x), token_positions=token_positions)
        # x = x + self.ffn(self.ffn_norm(x))
        # return x


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, n_q_heads, n_kv_heads, d_ff, rope_theta, norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        # self.context_length = context_length
        # self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # self.blocks = nn.ModuleList([
        #     TransformerBlock(d_model, n_q_heads, n_kv_heads, d_ff, context_length, rope_theta, norm_eps=norm_eps, device=device, dtype=dtype)
        #     for _ in range(num_layers)
        # ])
        # self.final_norm = RMSNorm(d_model, norm_eps=norm_eps, device=device, dtype=dtype)
        # self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids, token_positions=None):
        # seq_len = token_ids.shape[-1]
        # if seq_len < 1 or seq_len > self.context_length:
        #     raise ValueError(f"Sequence length {seq_len} outside [1, {self.context_length}]")
        # x = self.token_embedding(token_ids)
        # for block in self.blocks:
        #     x = block(x, token_positions=token_positions)
        # x = self.final_norm(x)
        # return self.lm_head(x)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import CausalGroupedQueryAttention
from src.layers import Embedding, Linear, RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        context_length: int,
        rope_theta: float,
        norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_ff = d_ff
        self.context_length = context_length
        self.norm_eps = norm_eps

        self.attention_norm = RMSNorm(d_model=d_model, norm_eps=norm_eps, device=device, dtype=dtype)
        self.attention = CausalGroupedQueryAttention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            context_length=context_length,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )
        self.ffn_norm = RMSNorm(d_model=d_model, norm_eps=norm_eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), token_positions=token_positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        rope_theta: float,
        norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                d_ff=d_ff,
                context_length=context_length,
                rope_theta=rope_theta,
                norm_eps=norm_eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model=d_model, norm_eps=norm_eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        if seq_len < 1 or seq_len > self.context_length:
            raise ValueError(f"Sequence length {seq_len} outside [1, {self.context_length}]")
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, token_positions=token_positions)
        x = self.final_norm(x)
        return self.lm_head(x)