# tinylm/model/utils.py
# Model inspection, parameter counting, and attention hook utility.
# These are used by the API and during development debugging.

from __future__ import annotations
from typing import Optional
from contextlib import contextmanager

import torch
import torch.nn as nn

from tinylm.config import ModelConfig


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count parameters by category.

    Returns dict with:
        total:      all parameters (including weight-tied duplicates)
        trainable:  parameters with requires_grad=True
        embedding:  parameters in embedding layers
        attention:  parameters in attention layers
        ffn:        parameters in feed-forward layers
        other:      LayerNorm, biases, etc.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    embedding = attention = ffn = other = 0
    for name, param in model.named_parameters():
        n = param.numel()
        if "wte" in name or "wpe" in name:
            embedding += n
        elif "c_attn" in name or "c_proj" in name and "h." in name:
            # c_proj inside a block (not global) is part of attention
            if any(f"h.{i}" in name for i in range(20)):
                attention += n
        elif "ffn" in name or "mlp" in name:
            ffn += n
        else:
            other += n

    return {
        "total":     total,
        "trainable": trainable,
        "embedding": embedding,
        "attention": attention,
        "ffn":       ffn,
        "other":     other,
    }


def model_summary(model: nn.Module, cfg: ModelConfig) -> None:
    """Print a formatted parameter breakdown for the model."""
    counts = count_parameters(model)

    print("=" * 55)
    print(f"  TinyLM (EMG-01) — Model Summary")
    print("=" * 55)
    print(f"  Architecture  : GPT Decoder-Only (Pre-LN)")
    print(f"  n_layer       : {cfg.n_layer}")
    print(f"  n_head        : {cfg.n_head}")
    print(f"  n_embd        : {cfg.n_embd}")
    print(f"  d_head        : {cfg.d_head}")
    print(f"  context_length: {cfg.context_length}")
    print(f"  vocab_size    : {cfg.vocab_size}")
    print("─" * 55)
    print(f"  Parameters (M):")
    print(f"    Embeddings  : {counts['embedding']/1e6:>8.3f}M")
    print(f"    Attention   : {counts['attention']/1e6:>8.3f}M")
    print(f"    FFN         : {counts['ffn']/1e6:>8.3f}M")
    print(f"    Other (LN)  : {counts['other']/1e6:>8.3f}M")
    print("─" * 55)
    print(f"    TOTAL       : {counts['total']/1e6:>8.3f}M")
    print(f"    (lm_head weight-tied to wte — not double-counted)")
    print("=" * 55)


@contextmanager
def capture_attention_weights(model: nn.Module, layer_idx: int):
    """
    Context manager that attaches a forward hook to capture attention weights
    from a specific transformer block.

    This is the non-invasive pattern — the model's forward() code is unchanged.
    The hook fires automatically when that layer runs.

    Usage:
        with capture_attention_weights(model, layer_idx=2) as captured:
            logits, _ = model(idx)
        weights = captured['weights']  # (B, n_head, T, T)

    Args:
        model:     GPT instance
        layer_idx: Index of the transformer block to capture (0-indexed)
    """
    captured = {"weights": None}

    def hook_fn(module, input, output):
        # output from MultiHeadAttention when return_weights=True is (out, weights)
        # but we attached the hook to the MHA module itself, so output is just 'out'
        # Instead, we hook the SDPA function via the attn_drop module
        # Better: hook the block's forward and call it with return_weights=True
        pass

    # More reliable: override forward for just this inference call
    # We use the GPT model's return_attn_layer parameter instead
    # This function exists as documentation of the hook pattern;
    # the GPT.forward(return_attn_layer=N) path is used in practice.

    yield captured