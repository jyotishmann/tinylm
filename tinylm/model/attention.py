# tinylm/model/attention.py
# Attention implementations — pure function first, module second.
# The pure function (this PR) can be unit-tested in complete isolation.

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinylm.config import ModelConfig


def scaled_dot_product_attention(
    q:        torch.Tensor,         # (B, H, T, d_k)
    k:        torch.Tensor,         # (B, H, T, d_k)
    v:        torch.Tensor,         # (B, H, T, d_v)
    mask:     torch.Tensor,         # (1, 1, T, T) causal mask, 0 = blocked
    dropout:  float     = 0.0,
    training: bool      = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    This is a pure function — no learnable parameters, no state.
    The learnable projections (W_Q, W_K, W_V, W_O) live in MultiHeadAttention.

    Args:
        q:        Query matrix,  shape (B, H, T, d_k)
        k:        Key matrix,    shape (B, H, T, d_k)
        v:        Value matrix,  shape (B, H, T, d_v)
        mask:     Causal mask.   1 = attend, 0 = block. Shape (1, 1, T, T).
        dropout:  Dropout probability for attention weights.
        training: If False, dropout is skipped (evaluation mode).

    Returns:
        output:       Attended values, shape (B, H, T, d_v)
        attn_weights: Post-softmax attention weights, (B, H, T, T).
                      Returned for visualisation — not used in the loss.

    Complexity: O(B * H * T^2 * d_k) — the T^2 term is why attention
                is expensive for long sequences.
    """
    d_k = q.size(-1)  # Head dimension — e.g. 64 for our config

    # ── Step 1: Raw attention scores ─────────────────────────────────
    # (B, H, T, d_k) @ (B, H, d_k, T) → (B, H, T, T)
    # Each query dot-producted with every key
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # ── Step 2: Causal mask ──────────────────────────────────────────
    # Where mask==0 (future positions), set score to -inf
    # After softmax: -inf → 0 (no weight placed on future tokens)
    scores = scores.masked_fill(mask[:, :, :q.size(2), :k.size(2)] == 0,
                                float("-inf"))

    # ── Step 3: Softmax ──────────────────────────────────────────────
    # Normalise along key dimension (last axis)
    # dim=-1: each query's distribution over all keys sums to 1
    # dim=-2 would be wrong: would normalise each key over all queries
    attn_weights = F.softmax(scores, dim=-1)

    # Handle the case where an entire row is -inf (e.g. padding row)
    # Replace NaN (0/0 from softmax) with 0
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # ── Step 4: Attention dropout ────────────────────────────────────
    # Randomly zero some weights — prevents over-reliance on single positions
    # Applied BEFORE the value lookup, not after
    if dropout > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout)

    # ── Step 5: Weighted sum of values ───────────────────────────────
    # (B, H, T, T) @ (B, H, T, d_v) → (B, H, T, d_v)
    output = torch.matmul(attn_weights, v)

    return output, attn_weights


# ── PyTorch 2.0 Note ─────────────────────────────────────────────────
# torch.nn.functional.scaled_dot_product_attention() does the same thing
# but with Flash Attention kernels when available (2-4× faster, O(1) memory).
# In production, swap this function for the PyTorch built-in:
#
#   output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
#
# We implement ours manually because (a) this is a from-scratch project,
# and (b) we need to return attn_weights for the visualisation endpoint.
# Flash Attention does not expose intermediate attention weights.

# Append to tinylm/model/attention.py

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal mask.

    Projects input into Q, K, V using a single fused linear layer,
    splits into n_head heads, applies SDPA in parallel, concatenates,
    and projects back to n_embd.

    Also applies:
      - Attention dropout (inside SDPA)
      - Residual dropout (after output projection)

    The causal mask is pre-computed once and stored as a buffer.
    It does not move to GPU manually — register_buffer handles it.

    Args:
        cfg: ModelConfig with n_embd, n_head, context_length, dropout, bias
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        assert cfg.n_embd % cfg.n_head == 0, (
            f"n_embd ({cfg.n_embd}) must be divisible by n_head ({cfg.n_head})"
        )

        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.d_head  = cfg.n_embd // cfg.n_head  # e.g. 384 // 6 = 64
        self.dropout = cfg.dropout

        # Fused QKV projection: one matrix for all three
        # Output dim = 3 * n_embd so we can split into Q, K, V after
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)

        # Output projection: reassembles concatenated heads → n_embd
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        # Dropout layers
        self.attn_drop  = nn.Dropout(cfg.dropout)  # on attention weights
        self.resid_drop = nn.Dropout(cfg.dropout)  # on output projection

        # Causal mask: lower triangular — position i can only attend to j ≤ i
        # Shape (1, 1, T, T) so it broadcasts over batch and head dimensions
        # register_buffer: not a parameter (no gradient), but moves with the model
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(cfg.context_length, cfg.context_length)
            ).view(1, 1, cfg.context_length, cfg.context_length),
        )

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, C)
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:              Input tensor of shape (B, T, n_embd)
            return_weights: If True, return (output, attn_weights).
                            Used by the attention visualisation API.

        Returns:
            output: Shape (B, T, n_embd)
            attn_weights (optional): Shape (B, n_head, T, T)
        """
        B, T, C = x.shape

        # ── 1. Fused QKV projection ───────────────────────────────────
        # (B, T, C) → (B, T, 3C), then split into 3 × (B, T, C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # ── 2. Reshape into heads ─────────────────────────────────────
        # (B, T, C) → (B, T, H, d_head) → (B, H, T, d_head)
        # The transpose brings the head dimension before the sequence dimension
        # so SDPA sees independent (T, d_head) matrices per head
        def split_heads(t):
            return t.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        # Now each is (B, H, T, d_head)

        # ── 3. Scaled dot-product attention ───────────────────────────
        out, attn_weights = scaled_dot_product_attention(
            q, k, v,
            mask     = self.causal_mask,
            dropout  = self.dropout,
            training = self.training,
        )
        # out: (B, H, T, d_head), attn_weights: (B, H, T, T)

        # ── 4. Reassemble heads ───────────────────────────────────────
        # (B, H, T, d_head) → (B, T, H, d_head) → (B, T, C)
        # .contiguous() required before .view() because .transpose() creates
        # a non-contiguous view — view() requires contiguous memory layout
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # ── 5. Output projection + residual dropout ───────────────────
        out = self.resid_drop(self.c_proj(out))

        if return_weights:
            return out, attn_weights
        return out