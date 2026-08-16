# tinylm/training/scheduler.py
# Learning rate schedule: linear warmup → cosine decay → floor.
# Implemented as a pure function (no PyTorch scheduler object).
#
# Why not torch.optim.lr_scheduler.CosineAnnealingLR?
# The PyTorch scheduler doesn't support custom warmup natively.
# Our pure function is 8 lines and completely transparent.

from __future__ import annotations
import math
import torch

from tinylm.config import TrainConfig


def get_lr(step: int, cfg: TrainConfig) -> float:
    """
    Compute learning rate at a given training step.

    Three regimes:
        [0,         warmup_steps)  → linear warmup
        [warmup_steps, max_steps]  → cosine decay from max_lr to min_lr
        (max_steps, ∞)             → constant min_lr

    Args:
        step: Current training step (0-indexed)
        cfg:  TrainConfig with learning_rate, min_lr, warmup_steps, max_steps

    Returns:
        Learning rate (float) to apply at this step
    """
    max_lr       = cfg.learning_rate
    min_lr       = cfg.min_lr
    warmup_steps = cfg.warmup_steps
    max_steps    = cfg.max_steps

    # ── Regime 1: linear warmup ───────────────────────────────────────
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
        # +1 so step 0 gets lr = max_lr/warmup_steps, not 0
        # (a true zero LR on step 0 wastes the first batch entirely)

    # ── Regime 3: past schedule — hold at min_lr ─────────────────────
    if step > max_steps:
        return min_lr

    # ── Regime 2: cosine decay ────────────────────────────────────────
    # progress goes from 0.0 (at warmup_steps) to 1.0 (at max_steps)
    progress = (step - warmup_steps) / (max_steps - warmup_steps)

    # cos(0) = 1   → lr = max_lr  (start of cosine phase)
    # cos(π) = -1  → lr = min_lr  (end of cosine phase)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine_factor * (max_lr - min_lr)


def set_lr(optimizer: "torch.optim.Optimizer", lr: float) -> None:
    """Update all optimizer parameter groups to the given learning rate."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr