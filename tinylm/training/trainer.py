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


    def train(self) -> None:
        """
        Full training loop: max_steps steps with gradient accumulation.

        For each step:
          1. Update LR via cosine schedule
          2. Accumulate gradients over grad_accum_steps micro-batches
          3. Clip gradient norm
          4. Optimizer step
          5. Log, validate, checkpoint at appropriate intervals

        PR 024 adds: gradient accumulation loop
        PR 025 adds: gradient clipping (inside this method)
        PR 026 adds: save_checkpoint() calls (inside this method)
        PR 027 adds: validate() calls + logging (inside this method)
        """
        cfg = self.cfg.train
        print(f"\n{'='*60}")
        print(f"  Training TinyLM (EMG-01)")
        print(f"  Steps:          {cfg.max_steps:,}")
        print(f"  Batch size:     {cfg.batch_size} × {cfg.grad_accum_steps} "
              f"= {cfg.effective_batch_size} effective")
        print(f"  Device:         {self.device}")
        print(f"{'='*60}\n")

        t0 = time.perf_counter()

        while self.step < cfg.max_steps:

            # ── 1. Update learning rate ───────────────────────────────
            lr = get_lr(self.step, cfg)
            set_lr(self.optimizer, lr)

            # ── 2. Gradient accumulation loop ─────────────────────────
            # Zero gradients BEFORE the accumulation loop (not inside)
            self.optimizer.zero_grad(set_to_none=True)

            accum_loss = 0.0  # running sum for logging
            for micro_step in range(cfg.grad_accum_steps):
                x, y = self._get_batch()
                micro_loss = self.train_step(x, y)  # backward() called inside
                accum_loss += micro_loss

            # Average loss over accumulation steps (for logging)
            avg_loss = accum_loss / cfg.grad_accum_steps
            self.train_losses.append(avg_loss)

            # ── 3. Gradient clipping — PR 025 adds this ───────────────
            # Clips after all grad_accum_steps micro-batches are done
            # Returns the pre-clip global norm (useful for monitoring)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm = cfg.grad_clip,   # 1.0 from config
            )
            # grad_norm > 1.0 means clipping fired on this step
            # Monitor via: if grad_norm > cfg.grad_clip: log_clip_event()

            # ── 4. Optimizer step ─────────────────────────────────────
            self.optimizer.step()

            self.step += 1

            # ── 5. Logging ────────────────────────────────────────────
            if self.step % cfg.log_interval == 0:
                elapsed = time.perf_counter() - t0
                tokens_per_sec = (
                    cfg.log_interval
                    * cfg.effective_batch_size
                    * self.cfg.model.context_length
                ) / elapsed
                clipped = "🔴" if grad_norm > cfg.grad_clip else "  "
                print(
                    f"step {self.step:>5} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"‖g‖ {grad_norm:.2f}{clipped} | "
                    f"{tokens_per_sec:.0f} tok/s"
                )
                t0 = time.perf_counter()

            # ── 6. Validate and checkpoint — PRs 026, 027 add this ────
            # (placeholders — filled in by those PRs)    