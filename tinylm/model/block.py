# tinylm/model/block.py
# One transformer block: Pre-LN attention + Pre-LN FFN, both with residuals.
# This is the repeating unit — our model stacks 6 of these.

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from tinylm.config import ModelConfig
from tinylm.model.attention import MultiHeadAttention
from tinylm.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    One decoder-only transformer block with Pre-LayerNorm.

    Structure (Pre-LN):
        x = x + Attention(LayerNorm(x))    ← self-attention sub-layer
        x = x + FFN(LayerNorm(x))          ← feed-forward sub-layer

    The '+' is the residual connection — allows gradient flow through
    N blocks without vanishing, and lets later blocks refine rather
    than replace earlier representations.

    Args:
        cfg: ModelConfig
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        # LayerNorm before attention (Pre-LN position)
        self.ln1  = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)

        # Multi-head causal self-attention
        self.attn = MultiHeadAttention(cfg)

        # LayerNorm before FFN (Pre-LN position)
        self.ln2  = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)

        # Position-wise feed-forward network
        self.ffn  = FeedForward(cfg)

    def forward(
        self,
        x:              torch.Tensor,          # (B, T, C)
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:              Input tensor (B, T, n_embd)
            return_weights: If True, return (output, attn_weights).
                            Passed through from MultiHeadAttention.

        Returns:
            output: (B, T, n_embd) — same shape as input
            attn_weights (optional): (B, n_head, T, T)
        """
        # ── Attention sub-layer (Pre-LN) ─────────────────────────────
        # Normalise → attend → add residual
        if return_weights:
            attn_out, attn_weights = self.attn(self.ln1(x), return_weights=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.ln1(x))

        # ── Feed-forward sub-layer (Pre-LN) ──────────────────────────
        # Normalise → transform → add residual
        x = x + self.ffn(self.ln2(x))

        if return_weights:
            return x, attn_weights
        return x