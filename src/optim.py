"""Loss and Optimization: Cross-Entropy, AdamW Optimizer, Cosine Schedule, and Gradient Clipping.

===============================================================================
WHAT NEEDS TO BE IMPLEMENTED:
===============================================================================
1. cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
   - logits: (..., vocab_size), targets: (...)
   - Return scalar mean cross-entropy over all batch and sequence positions.
   - Numerically stable log-sum-exp with max shift:
       logsumexp(z) = z_max + log(sum(exp(z - z_max)))
       loss = logsumexp(z) - z[target]
   - RESTRICTION: Do NOT use `torch.softmax`, `torch.log_softmax`,
     `torch.nn.functional.softmax`, `torch.nn.functional.log_softmax`,
     or `torch.nn.functional.cross_entropy`.
     You MAY use `torch.logsumexp`.

2. AdamW(torch.optim.Optimizer):
   - Subclass `torch.optim.Optimizer`.
   - Constructor defaults:
       lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
   - Constraints:
       lr >= 0.0, eps >= 0.0, weight_decay >= 0.0, 0.0 <= beta1 < 1.0, 0.0 <= beta2 < 1.0.
   - Validate both constructor defaults and effective group overrides.
   - step(self, closure=None):
     * If closure is not None, evaluate under torch.enable_grad() and save return value.
     * For each parameter with a dense gradient:
       - Reject sparse gradients (p.grad.is_sparse).
       - Lazily initialize state: step=0, exp_avg=zeros_like(p), exp_avg_sq=zeros_like(p).
       - Increment step t_p only when gradient is present.
       - Update biased moments:
           m_t = beta1 * m_{t-1} + (1 - beta1) * g
           v_t = beta2 * v_{t-1} + (1 - beta2) * (g * g)
       - Compute bias correction factors:
           m_hat = m_t / (1 - beta1 ** t_p)
           v_hat = v_t / (1 - beta2 ** t_p)
       - Apply decoupled weight decay:
           p = p * (1 - lr * weight_decay)
       - Apply adaptive step:
           p = p - lr * m_hat / (sqrt(v_hat) + eps)
     * Autograd graph must NOT be tracked during updates (use @torch.no_grad()).
   - RESTRICTION: Do NOT use `torch.optim.Adam` or `torch.optim.AdamW`.

3. get_lr_cosine_schedule(step, learning_rate_max, learning_rate_min, warmup_steps, cosine_steps) -> float:
   - Validate inputs:
       0 <= step (integer), 0 <= lr_min <= lr_max, 0 <= warmup_steps < cosine_steps.
   - Linear warmup (s < sw):
       lr = (s / sw) * lr_max  (if sw == 0, lr = lr_max)
   - Cosine decay (sw <= s <= sc):
       lr = lr_min + 0.5 * (1 + cos(pi * (s - sw) / (sc - sw))) * (lr_max - lr_min)
   - Floor (s > sc):
       lr = lr_min
   - Return Python float.

4. gradient_clipping(parameters, max_l2_norm: float) -> float:
   - iterable of parameters.
   - Validate max_l2_norm > 0.
   - Compute global L2 norm over all gradients that are not None:
       norm = sqrt(sum(||grad||_2^2 for grad in params))
   - If norm > max_l2_norm:
       scale = max_l2_norm / (norm + 1e-6)
       Multiply each gradient in-place by scale (p.grad.detach().mul_(scale)).
   - Return original norm before clipping as float (0.0 if no gradients).
   - RESTRICTION: Do NOT use `torch.nn.utils.clip_grad_norm_`.

===============================================================================
PSEUDOCODE:
===============================================================================

def cross_entropy(logits, targets):
    # 1. z_max = torch.max(logits, dim=-1, keepdim=True).values
    # 2. lse = z_max.squeeze(-1) + torch.log(torch.sum(torch.exp(logits - z_max), dim=-1))
    # 3. target_logits = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)
    # 4. loss = lse - target_logits
    # 5. return torch.mean(loss)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        # Validate defaults...
        # defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # super().__init__(params, defaults)
        # Validate param groups...

    @torch.no_grad()
    def step(self, closure=None):
        # loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()
        # for group in self.param_groups:
        #     lr, beta1, beta2, eps, wd = group["lr"], group["betas"][0], group["betas"][1], group["eps"], group["weight_decay"]
        #     for p in group["params"]:
        #         if p.grad is None: continue
        #         if p.grad.is_sparse: raise RuntimeError("Sparse gradients not supported")
        #         state = self.state[p]
        #         if len(state) == 0:
        #             state["step"] = 0
        #             state["exp_avg"] = torch.zeros_like(p)
        #             state["exp_avg_sq"] = torch.zeros_like(p)
        #         state["step"] += 1
        #         step = state["step"]
        #         grad = p.grad
        #         exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        #         exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        #         exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        #         m_hat = exp_avg / (1 - beta1 ** step)
        #         v_hat = exp_avg_sq / (1 - beta2 ** step)
        #         p.mul_(1 - lr * wd)
        #         p.addcdiv_(m_hat, torch.sqrt(v_hat) + eps, value=-lr)
        # return loss
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
from torch.optim import Optimizer


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = torch.gather(logits, -1, targets.to(torch.long).unsqueeze(-1)).squeeze(-1)
    loss = lse - target_logits
    return torch.mean(loss)


def _validate_hyperparameters(lr: float, betas: tuple[float, float], eps: float, weight_decay: float) -> None:
    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if eps < 0.0:
        raise ValueError(f"Invalid epsilon: {eps}")
    if weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay: {weight_decay}")
    beta1, beta2 = betas
    if not (0.0 <= beta1 < 1.0):
        raise ValueError(f"Invalid beta1: {beta1}")
    if not (0.0 <= beta2 < 1.0):
        raise ValueError(f"Invalid beta2: {beta2}")


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        _validate_hyperparameters(lr, betas, eps, weight_decay)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            _validate_hyperparameters(
                group["lr"], group["betas"], group["eps"], group["weight_decay"]
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            _validate_hyperparameters(lr, (beta1, beta2), eps, wd)

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                grad = p.grad
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                m_hat = exp_avg / (1.0 - beta1 ** t)
                v_hat = exp_avg_sq / (1.0 - beta2 ** t)

                p.mul_(1.0 - lr * wd)
                p.addcdiv_(m_hat, torch.sqrt(v_hat) + eps, value=-lr)

        return loss


def get_lr_cosine_schedule(
    step: int,
    learning_rate_max: float,
    learning_rate_min: float,
    warmup_steps: int,
    cosine_steps: int,
) -> float:
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
        raise ValueError(f"Invalid step: {step}")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, (int, np.integer)) or warmup_steps < 0:
        raise ValueError(f"Invalid warmup_steps: {warmup_steps}")
    if isinstance(cosine_steps, bool) or not isinstance(cosine_steps, (int, np.integer)) or cosine_steps <= 0:
        raise ValueError(f"Invalid cosine_steps: {cosine_steps}")
    if not (0.0 <= learning_rate_min <= learning_rate_max):
        raise ValueError(f"Invalid learning rates: min {learning_rate_min}, max {learning_rate_max}")
    if warmup_steps >= cosine_steps:
        raise ValueError(f"Invalid warmup / cosine steps: {warmup_steps}, {cosine_steps}")

    if step < warmup_steps:
        if warmup_steps == 0:
            return float(learning_rate_max)
        return float((step / warmup_steps) * learning_rate_max)

    if step > cosine_steps:
        return float(learning_rate_min)

    progress = (step - warmup_steps) / (cosine_steps - warmup_steps)
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(learning_rate_min + decay * (learning_rate_max - learning_rate_min))


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> float:
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")

    params = list(parameters)
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0

    total_norm_sq = sum(g.detach().float().square().sum().item() for g in grads)
    total_norm = math.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for g in grads:
            g.detach().mul_(scale)

    return float(total_norm) 