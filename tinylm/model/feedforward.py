# tinylm/model/feedforward.py
# Position-wise feed-forward network: the second sub-layer of each block.
# Processes each token independently — no cross-position interaction.

from __future__ import annotations
import torch
import torch.nn as nn

from tinylm.config import ModelConfig


class FeedForward(nn.Module):
    """
    Position-wise two-layer MLP with GELU activation.

    Applied independently to each position after multi-head attention.
    The 4× hidden expansion (n_embd → 4*n_embd → n_embd) gives the model
    capacity to transform representations without mixing positions again.

    Architecture:
        Linear(n_embd → 4*n_embd)  +  GELU  +  Linear(4*n_embd → n_embd)  +  Dropout

    Args:
        cfg: ModelConfig with n_embd, dropout, bias
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # Expansion: projects to 4× the embedding dimension
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias),

            # GELU activation: smooth, no dead neurons, empirically superior to ReLU
            # PyTorch's GELU uses the exact formula by default (not the tanh approximation)
            nn.GELU(),

            # Projection: compress back to n_embd
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias),

            # Dropout: applied AFTER the projection (residual dropout)
            # Not inside the expansion (would waste compute on dropped neurons)
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, n_embd)

        Returns:
            Output tensor of shape (B, T, n_embd) — same shape as input.
            Each position (b, t, :) processed independently.
        """
        return self.net(x)