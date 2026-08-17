# tinylm/training/optimizer.py
# AdamW optimizer construction with weight decay parameter groups.
# The split between decay/no-decay is critical for training quality.

from __future__ import annotations
import torch
import torch.nn as nn

from tinylm.config import Config


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.AdamW:
    """
    Build AdamW with two parameter groups:
      - decay group:    weight matrices (ndim >= 2), weight_decay = cfg value
      - no_decay group: biases, LayerNorm params, 1D tensors, weight_decay = 0

    Args:
        model: GPT model instance
        cfg:   Full Config (reads cfg.train.*)

    Returns:
        Configured AdamW optimizer
    """
    # Separate parameters into two groups
    decay_params    = []
    no_decay_params = []

    # named_parameters() yields (name, tensor) for every parameter in the model
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # Skip frozen parameters (none in our case, but be explicit)
            continue

        if param.dim() >= 2:
            # Weight matrices: apply weight decay
            decay_params.append(param)
        else:
            # Biases, LayerNorm scale/bias, 1D embeddings: no decay
            no_decay_params.append(param)

    n_decay    = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"  Optimizer param groups:")
    print(f"    decay (λ={cfg.train.weight_decay}):    {len(decay_params):>3} tensors, {n_decay:>10,} params")
    print(f"    no_decay (λ=0.0): {len(no_decay_params):>3} tensors, {n_no_decay:>10,} params")

    # Sanity check: total should match model parameter count
    total_decay = n_decay + n_no_decay
    model_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_decay == model_total, (
        f"Parameter count mismatch: {total_decay} in groups vs {model_total} in model. "
        f"A parameter is missing from both groups."
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg.train.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr    = cfg.train.learning_rate,   # overridden each step by scheduler
        betas = (cfg.train.beta1, cfg.train.beta2),
        eps   = 1e-8,
    )

    return optimizer


def optimizer_summary(optimizer: torch.optim.AdamW) -> None:
    """Print optimizer state for inspection."""
    print(f"Optimizer: AdamW")
    for i, group in enumerate(optimizer.param_groups):
        n = sum(p.numel() for p in group["params"])
        print(f"  Group {i}: {n:,} params | "
              f"lr={group['lr']:.2e} | wd={group['weight_decay']}")