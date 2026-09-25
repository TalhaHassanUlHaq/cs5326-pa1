"""Adapter boundary between the PA1 tests and student-written code.

Students may use any package layout, module names, or class names. Complete each
adapter by importing your implementation, constructing it with the supplied
arguments, injecting the supplied weights where applicable, and returning the
requested value. Keep the mathematics in your implementation, not in this file.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_load_token_array(path: str | Path) -> np.memmap:
    from src.data import load_token_array
    return load_token_array(path)


def run_get_batch(
    dataset: npt.NDArray[np.uint16],
    batch_size: int,
    sequence_length: int,
    device: str | torch.device,
    generator: torch.Generator,
) -> tuple[Int[Tensor, "batch sequence"], Int[Tensor, "batch sequence"]]:
    from src.data import get_batch
    return get_batch(dataset, batch_size, sequence_length, device, generator)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, "... d_in"],
) -> Float[Tensor, "... d_out"]:
    from src.layers import Linear
    mod = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    mod.weight = torch.nn.Parameter(weights)
    return mod(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, "vocab d_model"],
    token_ids: Int[Tensor, "..."],
) -> Float[Tensor, "... d_model"]:
    from src.layers import Embedding
    mod = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    mod.weight = torch.nn.Parameter(weights)
    return mod(token_ids)


def run_rmsnorm(
    d_model: int,
    norm_eps: float,
    weights: Float[Tensor, "d_model"],
    in_features: Float[Tensor, "... d_model"],
) -> Float[Tensor, "... d_model"]:
    from src.layers import RMSNorm
    mod = RMSNorm(d_model, norm_eps=norm_eps, device=weights.device, dtype=weights.dtype)
    mod.weight = torch.nn.Parameter(weights)
    return mod(in_features)


def run_silu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    from src.layers import silu
    return silu(in_features)


def run_swiglu(
    d_model: int,
    d_ff: int,
    gate_weight: Float[Tensor, "d_ff d_model"],
    down_weight: Float[Tensor, "d_model d_ff"],
    up_weight: Float[Tensor, "d_ff d_model"],
    in_features: Float[Tensor, "... d_model"],
) -> Float[Tensor, "... d_model"]:
    from src.layers import SwiGLU
    mod = SwiGLU(d_model, d_ff, device=gate_weight.device, dtype=gate_weight.dtype)
    mod.gate.weight = torch.nn.Parameter(gate_weight)
    mod.down.weight = torch.nn.Parameter(down_weight)
    mod.up.weight = torch.nn.Parameter(up_weight)
    return mod(in_features)


def run_rope(
    head_dim: int,
    rope_theta: float,
    context_length: int,
    in_query_or_key: Float[Tensor, "... sequence head_dim"],
    token_positions: Int[Tensor, "... sequence"],
) -> Float[Tensor, "... sequence head_dim"]:
    from src.rope import RotaryPositionalEmbedding
    rope = RotaryPositionalEmbedding(rope_theta, head_dim, context_length, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)


def run_softmax(
    in_features: Float[Tensor, "..."], dim: int
) -> Float[Tensor, "..."]:
    from src.attention import softmax
    return softmax(in_features, dim)


def run_scaled_dot_product_attention(
    queries: Float[Tensor, "... queries d_k"],
    keys: Float[Tensor, "... keys d_k"],
    values: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    from src.attention import scaled_dot_product_attention
    return scaled_dot_product_attention(queries, keys, values, mask)


def run_grouped_query_self_attention(
    d_model: int,
    n_q_heads: int,
    n_kv_heads: int,
    context_length: int,
    rope_theta: float,
    q_proj_weight: Float[Tensor, "h_q_times_d_h d_model"],
    k_proj_weight: Float[Tensor, "h_kv_times_d_h d_model"],
    v_proj_weight: Float[Tensor, "h_kv_times_d_h d_model"],
    output_proj_weight: Float[Tensor, "d_model h_q_times_d_h"],
    in_features: Float[Tensor, "batch sequence d_model"],
    token_positions: Int[Tensor, "... sequence"] | None = None,
) -> Float[Tensor, "batch sequence d_model"]:
    from src.attention import CausalGroupedQueryAttention
    mod = CausalGroupedQueryAttention(
        d_model, n_q_heads, n_kv_heads, context_length, rope_theta,
        device=in_features.device, dtype=in_features.dtype
    )
    mod.q_proj.weight = torch.nn.Parameter(q_proj_weight)
    mod.k_proj.weight = torch.nn.Parameter(k_proj_weight)
    mod.v_proj.weight = torch.nn.Parameter(v_proj_weight)
    mod.out_proj.weight = torch.nn.Parameter(output_proj_weight)
    return mod(in_features, token_positions=token_positions)


def run_transformer_block(
    d_model: int,
    n_q_heads: int,
    n_kv_heads: int,
    d_ff: int,
    context_length: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, "batch sequence d_model"],
    token_positions: Int[Tensor, "... sequence"] | None = None,
    norm_eps: float = 1e-5,
) -> Float[Tensor, "batch sequence d_model"]:
    from src.model import TransformerBlock
    block = TransformerBlock(
        d_model, n_q_heads, n_kv_heads, d_ff, context_length, rope_theta, norm_eps=norm_eps,
        device=in_features.device, dtype=in_features.dtype
    )
    block.load_state_dict(weights)
    return block(in_features, token_positions=token_positions)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    n_q_heads: int,
    n_kv_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    token_ids: Int[Tensor, "batch sequence"],
    token_positions: Int[Tensor, "... sequence"] | None = None,
    norm_eps: float = 1e-5,
) -> Float[Tensor, "batch sequence vocab"]:
    from src.model import TransformerLM
    model = TransformerLM(
        vocab_size, context_length, d_model, num_layers, n_q_heads, n_kv_heads, d_ff, rope_theta,
        norm_eps=norm_eps, device=token_ids.device
    )
    model.load_state_dict(weights)
    return model(token_ids, token_positions=token_positions)


def get_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    n_q_heads: int,
    n_kv_heads: int,
    d_ff: int,
    rope_theta: float,
    *,
    norm_eps: float = 1e-5,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    from src.model import TransformerLM
    dev = torch.device(device) if isinstance(device, str) else device
    return TransformerLM(
        vocab_size, context_length, d_model, num_layers, n_q_heads, n_kv_heads, d_ff, rope_theta,
        norm_eps=norm_eps, device=dev, dtype=dtype
    )


def run_cross_entropy(
    logits: Float[Tensor, "... vocab"], targets: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    from src.optim import cross_entropy
    return cross_entropy(logits, targets)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    from src.optim import AdamW
    return AdamW


def run_get_lr_cosine_schedule(
    step: int,
    learning_rate_max: float,
    learning_rate_min: float,
    warmup_steps: int,
    cosine_steps: int,
) -> float:
    from src.optim import get_lr_cosine_schedule
    return get_lr_cosine_schedule(
        step, learning_rate_max, learning_rate_min, warmup_steps, cosine_steps
    )


def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> float:
    from src.optim import gradient_clipping
    return gradient_clipping(parameters, max_l2_norm)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    out: str | Path | BinaryIO,
) -> None:
    from src.checkpoint import save_checkpoint
    save_checkpoint(model, optimizer, next_step, train_generator, val_generator, out)


def run_load_checkpoint(
    src: str | Path | BinaryIO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
) -> int:
    from src.checkpoint import load_checkpoint
    return load_checkpoint(src, model, optimizer, train_generator, val_generator)
