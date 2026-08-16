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