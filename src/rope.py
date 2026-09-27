"""Rotary Position Embeddings (RoPE) — Adjacent-Pair Convention.

===============================================================================
WHAT NEEDS TO BE IMPLEMENTED:
===============================================================================
RotaryPositionalEmbedding:
- Subclass `torch.nn.Module`.
- Constructor arguments:
    * rope_theta: float (frequency base, e.g. 10,000.0)
    * head_dim: int (dimension of each query/key head; must be even)
    * context_length: int (maximum supported sequence length)
    * device: torch.device | None
- Validation:
    * Raise ValueError if head_dim is odd.
- Precomputed buffers:
    * Compute frequencies omega_k = rope_theta ** (-2*k / head_dim) for k in [0, head_dim//2 - 1].
    * Compute angles phi_{i, k} = i * omega_k for i in [0, context_length - 1].
    * Precompute cosine and sine tables of shape [context_length, head_dim // 2].
    * Store as non-persistent buffers using:
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
- Forward method:
    forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
    * x: shape (..., sequence_length, head_dim)
    * token_positions: integer tensor with final dimension sequence_length.
      Leading dimensions must broadcast over batch-like dimensions of x.
    * Validation:
        - Check x.shape[-1] == head_dim.
        - Check token_positions is integer type (torch.is_floating_point(token_positions) is False).
        - Check bounds: all positions must satisfy 0 <= pos < context_length.
    * Apply adjacent-pair rotation:
        - Pairs are (x1, x2), (x3, x4), ..., (x_{dh-1}, x_{dh}).
        - rotate_pair(x) = (-x2, x1, -x4, x3, ..., -x_{dh}, x_{dh-1}).
        - Efficient pairwise realization:
            R_i x = x * cos_i + rotate_pair(x) * sin_i
        - Broadcast compact head_dim // 2 sine/cosine values across the two coordinates
          of each coordinate pair.
    * Return tensor with unchanged shape and floating dtype.

===============================================================================
PSEUDOCODE:
===============================================================================

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, rope_theta: float, head_dim: int, context_length: int, device=None):
        super().__init__()
        # 1. if head_dim % 2 != 0: raise ValueError("head_dim must be even")
        # 2. self.head_dim = head_dim
        # 3. self.context_length = context_length
        # 4. k = torch.arange(0, head_dim // 2, dtype=torch.float32, device=device)
        # 5. omega = rope_theta ** (-2.0 * k / head_dim)
        # 6. positions = torch.arange(0, context_length, dtype=torch.float32, device=device)
        # 7. phi = positions[:, None] * omega[None, :]  # shape: [context_length, head_dim // 2]
        # 8. cos = torch.cos(phi)
        # 9. sin = torch.sin(phi)
        # 10. self.register_buffer("cos_cached", cos, persistent=False)
        # 11. self.register_buffer("sin_cached", sin, persistent=False)

    def _rotate_pair(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to pair structure:
        # x_paired = x.view(*x.shape[:-1], self.head_dim // 2, 2)
        # x1 = x_paired[..., 0]
        # x2 = x_paired[..., 1]
        # rotated = torch.stack((-x2, x1), dim=-1)
        # return rotated.view_as(x)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. Validate x.shape[-1] == self.head_dim
        # 2. Validate token_positions is integer and within [0, self.context_length)
        # 3. Index cos and sin tables:
        #    cos = self.cos_cached[token_positions]  # [..., seq, head_dim // 2]
        #    sin = self.sin_cached[token_positions]  # [..., seq, head_dim // 2]
        # 4. Repeat/broadcast cos and sin along the coordinate pair dimension:
        #    cos = torch.repeat_interleave(cos, 2, dim=-1)  # [..., seq, head_dim]
        #    sin = torch.repeat_interleave(sin, 2, dim=-1)  # [..., seq, head_dim]
        # 5. Align dimensions if x has head dimensions (e.g. unsqueeze before head dimension)
        # 6. return (x * cos + self._rotate_pair(x) * sin).to(x.dtype)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        rope_theta: float,
        head_dim: int,
        context_length: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.rope_theta = float(rope_theta)
        self.head_dim = head_dim
        self.context_length = context_length
        k = torch.arange(0, head_dim // 2, dtype=torch.float32, device=device)
        self.omega = self.rope_theta ** (-2.0 * k / head_dim)
        positions = torch.arange(0, context_length, dtype=torch.float32, device=device)
        phi = positions[:, None] * self.omega[None, :]  # [context_length, head_dim // 2]
        cos = torch.cos(phi)
        sin = torch.sin(phi)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _rotate_pair(self, x: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-1] != self.head_dim:
            raise ValueError(f"Expected final dimension {self.head_dim} (head_dim), got {x.shape[-1]}")
        if token_positions.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            raise TypeError("token_positions must have integer dtype")
        if (token_positions < 0).any() or (token_positions >= self.context_length).any():
            raise ValueError("token_positions contains values outside [0, context_length)")
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(f"token_positions sequence length {token_positions.shape[-1]} does not match x sequence length {x.shape[-2]}")

        pos_long = token_positions.to(torch.long)
        cos = self.cos_cached[pos_long]  # [*token_positions.shape, head_dim // 2]
        sin = self.sin_cached[pos_long]
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [*token_positions.shape, head_dim]
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        batch_dims = x.ndim - 2
        pos_batch_dims = token_positions.ndim - 1
        new_shape = (
            *token_positions.shape[:-1],
            *((1,) * max(0, batch_dims - pos_batch_dims)),
            token_positions.shape[-1],
            self.head_dim,
        )
        cos = cos.reshape(new_shape).to(x.dtype)
        sin = sin.reshape(new_shape).to(x.dtype)

        rotated = self._rotate_pair(x)
        return x * cos + rotated * sin
