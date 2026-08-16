# tinylm/model/gpt.py
# The GPT model: assembles all components and implements forward().
# Also: weight initialisation, weight tying, and generate().

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinylm.config import ModelConfig
from tinylm.model.block import TransformerBlock


class GPT(nn.Module):
    """
    Decoder-only GPT-style language model.

    Implements: Embeddings → [TransformerBlock × n_layer] → LayerNorm → LM Head

    Weight tying: lm_head.weight is shared with transformer.wte.weight.
    Weight init: GPT-2 scheme (0.02 std, scaled residual projections).

    Args:
        cfg: ModelConfig — all hyperparameters

    Usage:
        model = GPT(cfg)
        logits, loss = model(idx, targets)
        generated = model.generate(prompt_ids, max_new_tokens=200)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte   = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe   = nn.Embedding(cfg.context_length, cfg.n_embd),
            drop  = nn.Dropout(cfg.dropout),
            h     = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)]),
            ln_f  = nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))

        # LM head: projects hidden states to vocabulary logits
        # bias=False is standard — biases in this projection are rarely useful
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # ── Weight tying ──────────────────────────────────────────────
        # lm_head.weight points to the same storage as wte.weight.
        # Saves vocab_size × n_embd = 5000 × 384 = 1.92M parameters.
        # Must be done BEFORE _init_weights — they'll both be initialised
        # through the same tensor.
        self.transformer.wte.weight = self.lm_head.weight

        # ── Weight initialisation ─────────────────────────────────────
        self.apply(self._init_weights)

        # Scaled initialisation for residual projections (GPT-2 scheme)
        # Keeps residual stream variance constant regardless of depth
        std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=std)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_params_no_emb = n_params - self.transformer.wte.weight.numel()
        print(
            f"GPT initialised | "
            f"{n_params/1e6:.2f}M total params | "
            f"{n_params_no_emb/1e6:.2f}M non-embedding params"
        )

    def _init_weights(self, module: nn.Module) -> None:
        """
        GPT-2 weight initialisation scheme.

        Linear and Embedding weights: Normal(0, 0.02)
        Biases (if present): zeros

        Residual projection weights (c_proj) receive a further 1/sqrt(2*n_layer)
        scaling applied AFTER this in __init__ to control residual stream variance.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx:              torch.Tensor,          # (B, T) integer token IDs
        targets:          Optional[torch.Tensor] = None,  # (B, T) target token IDs
        return_attn_layer: Optional[int] = None,  # if set, return attn weights from this layer
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            idx:               Token ID tensor, shape (B, T)
            targets:           Target token IDs for loss computation (B, T).
                               If None, only logits are returned (inference mode).
            return_attn_layer: If an int, also return attention weights from
                               block[return_attn_layer]. Used by /api/attention.

        Returns:
            logits: Shape (B, T, vocab_size) — pre-softmax vocabulary distribution
            loss:   Scalar cross-entropy loss if targets provided, else None
            attn_weights (optional): (B, n_head, T, T) if return_attn_layer set
        """
        B, T = idx.shape
        assert T <= self.cfg.context_length, (
            f"Input length {T} exceeds context_length {self.cfg.context_length}"
        )

        # ── Embedding ─────────────────────────────────────────────────
        positions = torch.arange(T, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(positions)
        )

        # ── Transformer blocks ────────────────────────────────────────
        captured_weights = None
        for i, block in enumerate(self.transformer.h):
            if return_attn_layer is not None and i == return_attn_layer:
                x, captured_weights = block(x, return_weights=True)
            else:
                x = block(x)

        # ── Final LayerNorm ────────────────────────────────────────────
        # Normalises the residual stream after the last block
        # Pre-LN means block outputs are not normalised — ln_f does it
        x = self.transformer.ln_f(x)

        # ── LM Head ──────────────────────────────────────────────────
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # ── Loss ──────────────────────────────────────────────────────
        loss = None
        if targets is not None:
            # Reshape for cross_entropy: (B, T, V) → (B*T, V), (B, T) → (B*T,)
            # ignore_index=0: PAD tokens don't contribute to loss
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=0,
            )

        if return_attn_layer is not None:
            return logits, loss, captured_weights
        return logits, loss

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "GPT":
        """Convenience constructor. Identical to GPT(cfg) but explicit."""
        return cls(cfg)