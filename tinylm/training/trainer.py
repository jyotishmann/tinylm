# tinylm/training/trainer.py
# Trainer: encapsulates the full training loop.
# Built incrementally across PRs 022–026 — each PR adds one feature.
#
# PR 022: class skeleton, bare train_step (no mixed precision, no grad accum)
# PR 023: +autocast(bfloat16) in train_step
# PR 024: +gradient accumulation in train loop
# PR 025: +gradient clipping before optimizer.step()
# PR 026: +save_checkpoint() + load_checkpoint()

from __future__ import annotations
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tinylm.config import Config
from tinylm.model import GPT
from tinylm.training.scheduler import get_lr, set_lr


class Trainer:
    """
    Training orchestrator for TinyLM.

    Responsibilities:
      - Training loop (forward → backward → clip → step → schedule)
      - Validation loop (loss + perplexity on held-out data)
      - Checkpointing (save/load model + optimizer state)
      - Logging (console + CSV)

    Args:
        model:        GPT model instance (already on device)
        optimizer:    AdamW configured with param groups
        train_loader: DataLoader for training split
        val_loader:   DataLoader for validation split
        cfg:          Full Config
    """

    def __init__(
        self,
        model:        GPT,
        optimizer:    torch.optim.AdamW,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          Config,
    ) -> None:
        self.model        = model
        self.optimizer    = optimizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg

        # Training state — persisted in checkpoints
        self.step           = 0
        self.best_val_loss  = float("inf")
        self.train_losses:  list[float] = []
        self.val_losses:    list[float] = []

        # Device (must match wherever model lives)
        self.device = next(model.parameters()).device

        # Checkpoint directory
        self.ckpt_dir = Path(cfg.train.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Infinite iterator over training batches
        # (training loop controls steps, not epochs)
        self._train_iter = self._infinite_loader(train_loader)

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _infinite_loader(loader: DataLoader) -> Iterator:
        """Cycles through a DataLoader indefinitely."""
        while True:
            yield from loader

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the next training batch and move to device."""
        x, y = next(self._train_iter)
        return x.to(self.device), y.to(self.device)

    # ── PR 022: Basic training step ───────────────────────────────────

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        One forward + backward pass with mixed precision (bfloat16).

        torch.autocast wraps only the forward pass — the backward pass
        computes gradients in float32 automatically.

        Args:
            x: Input token IDs, shape (B, T)
            y: Target token IDs, shape (B, T)

        Returns:
            Raw loss value (float32 scalar) for this micro-batch
        """
        self.model.train()

        # ── Mixed precision forward pass ──────────────────────────────
        # autocast context:
        #   - MatMuls, attention ops → bfloat16  (fast, low memory)
        #   - Softmax, LayerNorm, loss → float32  (precision-sensitive)
        # device_type must match the model's device ('cuda' or 'cpu')
        # On CPU, bfloat16 is supported from PyTorch 1.10+
        device_type = "cuda" if self.device.type == "cuda" else "cpu"

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            enabled=(device_type == "cuda")):
            _, loss = self.model(x, y)
            # Scale loss before backward so accumulated gradients equal
            # the mean over grad_accum_steps batches (not the sum)
            scaled_loss = loss / self.cfg.train.grad_accum_steps

        # ── Backward pass (runs in float32 automatically) ─────────────
        # autograd casts gradients back to float32 when autocast is off
        scaled_loss.backward()

        return loss.item()  # Unscaled, float32, for logging