# tinylm/model/embeddings.py
# Input layer: token IDs → continuous vector representations.
# The only component where integer indices enter the model.
# Everything after this operates on float tensors.

from __future__ import annotations
import torch
import torch.nn as nn

from tinylm.config import ModelConfig


class Embeddings(nn.Module):
    """
    Combined token + positional embedding layer.

    Converts a batch of token ID sequences (B, T) into a batch of
    continuous vector sequences (B, T, n_embd) by summing:
      - a learned token embedding (semantic meaning)
      - a learned positional embedding (sequence position)

    Followed by dropout for regularisation.

    Args:
        cfg: ModelConfig with vocab_size, context_length, n_embd, dropout
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        # Token embedding table: vocab_size × n_embd
        # Each token ID maps to a learned vector
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # Positional embedding table: context_length × n_embd
        # Each position (0..T-1) maps to a learned vector
        self.wpe = nn.Embedding(cfg.context_length, cfg.n_embd)

        self.drop = nn.Dropout(cfg.dropout)
        self.context_length = cfg.context_length

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: Token ID tensor of shape (B, T)

        Returns:
            Embedding tensor of shape (B, T, n_embd)

        Raises:
            AssertionError: if T > context_length
        """
        B, T = idx.shape
        assert T <= self.context_length, (
            f"Sequence length {T} exceeds context_length {self.context_length}. "
            f"Truncate your input before calling forward()."
        )

        # Build position indices [0, 1, 2, ..., T-1] on the same device as idx
        # Using arange ensures this works on CPU, CUDA, and MPS without manual transfer
        positions = torch.arange(T, device=idx.device)  # (T,)

        # Token embedding: each ID → its row in wte   (B, T, n_embd)
        tok_emb = self.wte(idx)

        # Positional embedding: position → its row in wpe   (T, n_embd)
        # PyTorch broadcasts (T, n_embd) to (B, T, n_embd) automatically
        pos_emb = self.wpe(positions)

        # Sum and apply dropout
        # Dropout here regularises the combined representation,
        # not the token or position embeddings individually
        return self.drop(tok_emb + pos_emb)