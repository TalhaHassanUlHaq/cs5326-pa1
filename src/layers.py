"""Primitive Building Blocks: Linear, Embedding, RMSNorm, SiLU, and SwiGLU.

===============================================================================
WHAT NEEDS TO BE IMPLEMENTED:
===============================================================================
1. Linear:
   - Subclass `torch.nn.Module`.
   - Store weight as `torch.nn.Parameter` of shape `(out_features, in_features)`.
   - Bias-free (no bias parameter or bias calculation).
   - Initialization: Truncated normal distribution with mean=0.0 and
     std = sqrt(2.0 / (in_features + out_features)), truncated to [-3*std, 3*std].
   - Forward pass: Computes y = x @ W^T, accepting shape (..., in_features)
     and producing (..., out_features).
   - RESTRICTION: Do not use `torch.nn.Linear` or `torch.nn.functional.linear`.

2. Embedding:
   - Subclass `torch.nn.Module`.
   - Store embedding table as `torch.nn.Parameter` of shape `(num_embeddings, embedding_dim)`.
   - Initialization: Truncated normal distribution with mean=0.0 and std=1.0,
     truncated to [-3.0, 3.0].
   - Forward pass: Direct integer indexing returning shape (*token_ids.shape, embedding_dim).
   - RESTRICTION: Do not use `torch.nn.Embedding` or `torch.nn.functional.embedding`.

3. RMSNorm:
   - Subclass `torch.nn.Module`.
   - Parameter: Learnable gain gamma initialized to ones of shape (d_model,). No bias.
   - Numerical constant norm_eps (default 1e-5).
   - Precision: If input is float16 or bfloat16, temporarily upcast to float32
     for RMS computation, then cast result back to input dtype. If float32 or higher,
     preserve that dtype throughout.
   - Formula:
       RMS(x) = sqrt(mean(x^2, dim=-1, keepdim=True) + norm_eps)
       RMSNorm(x) = gamma * (x / RMS(x))
   - RESTRICTION: Do not use `torch.nn.RMSNorm` or `torch.nn.functional.rms_norm`.

4. silu:
   - Activation function: silu(x) = x * sigmoid(x).
   - RESTRICTION: Do not use `torch.nn.SiLU` or `torch.nn.functional.silu`.
     You MAY use `torch.sigmoid`.

5. SwiGLU:
   - Subclass `torch.nn.Module`.
   - Contains three bias-free Linear layers:
     * gate_proj: Linear(d_model, d_ff)
     * up_proj: Linear(d_model, d_ff)
     * down_proj: Linear(d_ff, d_model)
   - Forward pass:
       SwiGLU(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
   - Accepts input of shape (..., d_model) and returns (..., d_model).

===============================================================================
PSEUDOCODE:
===============================================================================

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 1. Allocate uninitialized parameter of shape (out_features, in_features)
        # 2. Compute std = sqrt(2.0 / (in_features + out_features))
        # 3. Initialize with torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute matrix multiplication with transposed weight:
        # return torch.matmul(x, self.weight.t())


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # 1. Allocate uninitialized parameter of shape (num_embeddings, embedding_dim)
        # 2. Initialize with torch.nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Return indexed embedding vectors:
        # return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        # 1. Store norm_eps
        # 2. Allocate self.weight initialized to torch.ones(d_model, ...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Record in_dtype = x.dtype
        # 2. If in_dtype in (float16, bfloat16), x_calc = x.to(torch.float32), else x_calc = x
        # 3. variance = torch.mean(x_calc ** 2, dim=-1, keepdim=True)
        # 4. rms = torch.rsqrt(variance + self.norm_eps)
        # 5. norm_x = x_calc * rms
        # 6. return (self.weight * norm_x).to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    # return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        # 1. self.w_gate = Linear(d_model, d_ff, device=device, dtype=dtype)
        # 2. self.w_up   = Linear(d_model, d_ff, device=device, dtype=dtype)
        # 3. self.w_down = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. gate = silu(self.w_gate(x))
        # 2. up = self.w_up(x)
        # 3. return self.w_down(gate * up)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t())


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_calc = x.to(torch.float32) if in_dtype in (torch.float16, torch.bfloat16) else x
        variance = torch.mean(x_calc ** 2, dim=-1, keepdim=True)
        r_rms = torch.rsqrt(variance + self.norm_eps)
        normalized = x_calc * r_rms
        weight = self.weight.to(x_calc.dtype)
        return (weight * normalized).to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    # raise NotImplementedError("TODO: implement silu function")
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.gate = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.up = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.down = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(silu(self.gate(x)) * self.up(x))
