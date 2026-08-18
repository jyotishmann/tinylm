# tinylm/evaluation/attention.py
# Attention weight extraction for the /api/attention visualisation endpoint.
#
# Uses GPT.forward(return_attn_layer=N) — no monkey-patching or hooks required.
# Returns data formatted for the AttentionResponse Pydantic schema (CELLS-06).

from __future__ import annotations

import numpy as np
import torch

from tinylm.model.gpt import GPT
from tinylm.tokenizer import BPETokenizer
from tinylm.tokenizer.vocab import END_OF_WORD, SPECIAL_TOKENS


def extract_attention_weights(
    model:     GPT,
    tokenizer: BPETokenizer,
    text:      str,
    layer:     int = 0,
) -> dict:
    """
    Extract attention weights from a specific transformer layer.

    Args:
        model:     Trained GPT instance
        tokenizer: Trained BPETokenizer
        text:      Input text (will be tokenized)
        layer:     Which transformer block to extract from (0-indexed)

    Returns:
        Dict with keys:
            weights: list[list[list[float]]]  shape [n_heads, T, T]
                     Each [i][j] = attention weight from position i to position j
            tokens:  list[str]  — human-readable token labels for axis ticks
            layer:   int        — which layer was extracted
            n_heads: int        — number of attention heads

    Raises:
        ValueError: if layer >= model.cfg.n_layer
        ValueError: if text encodes to > context_length tokens
    """
    n_layer = model.cfg.n_layer
    if layer >= n_layer:
        raise ValueError(
            f"Layer {layer} is out of range. "
            f"Model has {n_layer} layers (0–{n_layer-1})."
        )

    device = next(model.parameters()).device
    model.eval()

    # Encode text — no BOS for attention viz (cleaner token labels)
    ids = tokenizer.encode(text, add_bos=False)

    T = len(ids)
    if T == 0:
        raise ValueError("Input text encodes to zero tokens.")
    if T > model.cfg.context_length:
        raise ValueError(
            f"Input encodes to {T} tokens, exceeding context_length "
            f"{model.cfg.context_length}. Use shorter text."
        )

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        _, _, attn_weights = model(idx, return_attn_layer=layer)
    # attn_weights: (1, n_heads, T, T)

    # Convert to Python-native types for JSON serialisation
    # Squeeze batch dim, move to CPU, convert to numpy then list
    weights_np = attn_weights.squeeze(0).cpu().float().numpy()
    # Shape: (n_heads, T, T)

    # Build human-readable token labels for heatmap axis ticks
    tokens = _decode_tokens_for_display(ids, tokenizer)

    return {
        "weights": weights_np.tolist(),    # [n_heads, T, T] nested list
        "tokens":  tokens,                 # [T] human-readable labels
        "layer":   layer,
        "n_heads": model.cfg.n_head,
    }


def _decode_tokens_for_display(
    ids:       list[int],
    tokenizer: BPETokenizer,
) -> list[str]:
    """
    Convert token IDs to display-friendly strings for heatmap axis labels.

    Strips END_OF_WORD markers and replaces special tokens with readable names.
    Truncates long tokens to max 6 characters to fit heatmap cells.
    """
    special_display = {v: k for k, v in SPECIAL_TOKENS.items()}

    labels = []
    for token_id in ids:
        if token_id in special_display:
            labels.append(special_display[token_id])
            continue

        raw = tokenizer.id_to_token.get(token_id, "<?>")
        # Remove end-of-word marker for display
        clean = raw.replace(END_OF_WORD, "")
        # Replace whitespace-only labels (from leading-space tokens)
        if not clean.strip():
            clean = "·"  # middle dot for visual clarity
        # Truncate
        if len(clean) > 6:
            clean = clean[:5] + "…"
        labels.append(clean)

    return labels


def attention_summary(weights_dict: dict) -> None:
    """
    Print a human-readable summary of an attention weight dict.
    Useful for inspecting extraction results in the REPL.
    """
    w    = np.array(weights_dict["weights"])  # (n_heads, T, T)
    toks = weights_dict["tokens"]
    T    = len(toks)

    print(f"Layer {weights_dict['layer']} attention | "
          f"{weights_dict['n_heads']} heads | {T} tokens")
    print(f"Tokens: {toks}")

    for head_idx in range(w.shape[0]):
        head_w = w[head_idx]  # (T, T)
        # Find each token's top attended-to position
        top_attended = head_w.argmax(axis=1)
        print(f"\n  Head {head_idx}:")
        for i, (tok, top_j) in enumerate(zip(toks, top_attended)):
            print(f"    '{tok}' → attends most to '{toks[top_j]}' "
                  f"(weight={head_w[i, top_j]:.3f})")