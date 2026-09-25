"""Autoregressive Text Generation with Temperature and Nucleus (Top-p) Sampling.

===============================================================================
WHAT NEEDS TO BE IMPLEMENTED:
===============================================================================
generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:

- Inputs:
    * model: TransformerLM instance with attribute context_length.
    * prompt_ids: Non-empty, 1D torch.long tensor on the model's device.
    * max_new_tokens: Maximum number of additional tokens to generate (int >= 0).
    * context_length: Context window size (must equal model.context_length).
    * temperature: Sampling temperature (float > 0). Lower = more peaky/greedy; higher = more uniform.
    * top_p: Cumulative probability threshold for nucleus sampling (0 < top_p <= 1).
    * eot_token_id: Token ID for end-of-text marker (stop immediately if generated).
    * generator: Optional torch.Generator for reproducible sampling.

- Validation:
    * Require max_new_tokens >= 0.
    * Require context_length > 0 and context_length == model.context_length.
    * Require temperature > 0.
    * Require 0.0 < top_p <= 1.0.
    * If max_new_tokens == 0: return prompt_ids unchanged.

- Execution:
    * Save previous mode: was_training = model.training.
    * Set model to evaluation mode: model.eval().
    * Perform generation under torch.inference_mode():
      - Repeatedly crop sequence to its last context_length tokens.
      - Add batch dimension: sequence[-context_length:].unsqueeze(0).
      - Forward pass: logits = model(input_window)[:, -1, :].
      - Temperature scaling: logits / temperature.
      - Stable softmax to obtain probability distribution q.
      - Nucleus (top-p) truncation:
          1. Sort probabilities in descending order: sorted_probs, sorted_indices = torch.sort(q, descending=True).
          2. Compute cumulative sum: cum_probs = torch.cumsum(sorted_probs, dim=-1).
          3. Mask tokens where cum_probs - sorted_probs > top_p (ensures at least one token is kept).
          4. Zero out masked probabilities and renormalize.
          5. Sample rank using torch.multinomial(sorted_probs, 1, generator=generator).
          6. Map back: token_id = sorted_indices.gather(-1, sample_rank).
      - Append new token to generated sequence.
      - If eot_token_id is not None and token_id == eot_token_id: break.
    * Restore model training mode: model.train(was_training).
    * Return complete sequence (prompt_ids followed by generated completion).

===============================================================================
PSEUDOCODE:
===============================================================================

def generate(model, prompt_ids, max_new_tokens, context_length, *, temperature=1.0, top_p=1.0, eot_token_id=None, generator=None):
    # if max_new_tokens < 0: raise ValueError(...)
    # if context_length != getattr(model, "context_length", None): raise ValueError(...)
    # if temperature <= 0 or not (0 < top_p <= 1.0): raise ValueError(...)
    # if max_new_tokens == 0: return prompt_ids
    # was_training = model.training
    # model.eval()
    # out = prompt_ids.clone()
    # with torch.inference_mode():
    #     for _ in range(max_new_tokens):
    #         window = out[-context_length:].unsqueeze(0)
    #         logits = model(window)[0, -1, :] / temperature
    #         probs = softmax(logits, dim=-1)
    #         if top_p < 1.0:
    #             sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    #             cum_probs = torch.cumsum(sorted_probs, dim=-1)
    #             mask = (cum_probs - sorted_probs) >= top_p
    #             sorted_probs[mask] = 0.0
    #             sorted_probs /= sorted_probs.sum()
    #             idx = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
    #             next_token = sorted_indices[idx]
    #         else:
    #             next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    #         out = torch.cat([out, next_token])
    #         if eot_token_id is not None and next_token.item() == eot_token_id:
    #             break
    # model.train(was_training)
    # return out
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import softmax


def generate(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if getattr(model, "context_length", None) != context_length:
        raise ValueError("context_length must match model's context_length attribute")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if prompt_ids.numel() == 0:
        raise ValueError("prompt_ids must not be empty")

    if max_new_tokens == 0:
        return prompt_ids

    was_training = model.training
    model.eval()
    out = prompt_ids.clone()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            window = out[-context_length:].unsqueeze(0)
            logits = model(window)[0, -1, :] / temperature
            probs = softmax(logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                # Keep shortest prefix where cum_probs >= top_p
                # That means mask elements where cum_probs - sorted_probs >= top_p
                mask = (cum_probs - sorted_probs) >= top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
                sampled_idx = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
                next_token = sorted_indices[sampled_idx]
            else:
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)

            out = torch.cat([out, next_token])
            if eot_token_id is not None and next_token.item() == eot_token_id:
                break

    model.train(was_training)
    return out