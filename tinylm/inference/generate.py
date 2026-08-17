# tinylm/inference/generate.py
# Text-in, text-out generation pipeline.
# This is the layer the API and CLI call — they never touch model.generate() directly.

from __future__ import annotations
from typing import Iterator, Optional
import time

import torch
import torch.nn.functional as F

from tinylm.model.gpt import GPT
from tinylm.tokenizer import BPETokenizer
from tinylm.tokenizer.vocab import END_OF_WORD, SPECIAL_TOKENS


def generate_text(
    model:          GPT,
    tokenizer:      BPETokenizer,
    prompt:         str,
    max_new_tokens: int   = 200,
    temperature:    float = 0.8,
    top_k:          int   = 50,
    top_p:          float = 0.9,
    include_prompt: bool  = False,
    seed:           Optional[int] = None,
) -> str:
    """
    Full text-in, text-out generation.

    Encodes the prompt, runs model.generate(), decodes only the new tokens.

    Args:
        model:          Trained GPT instance (eval mode preferred)
        tokenizer:      Trained BPETokenizer
        prompt:         Input text string
        max_new_tokens: Maximum tokens to generate beyond the prompt
        temperature:    Sampling temperature (0=greedy, ~0.8=default, >1=random)
        top_k:          Top-k filtering (0 = disabled)
        top_p:          Nucleus sampling (1.0 = disabled)
        include_prompt: If True, prepend the prompt to the returned string
        seed:           Optional random seed for reproducibility

    Returns:
        Generated text string (prompt excluded unless include_prompt=True)
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    model.eval()

    # Encode prompt — BOS prepended so model has a clean start signal
    prompt_ids    = tokenizer.encode(prompt, add_bos=True)
    n_prompt_toks = len(prompt_ids)

    # Build input tensor: (1, T_prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate — model.generate() returns full sequence (prompt + new tokens)
    with torch.no_grad():
        full_ids = model.generate(
            idx,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
            top_p          = top_p,
            eos_id         = SPECIAL_TOKENS["<EOS>"],
        )

    # Slice out only the generated tokens (strip prompt + BOS)
    new_ids  = full_ids[0, n_prompt_toks:].tolist()
    new_text = tokenizer.decode(new_ids)

    if include_prompt:
        return prompt + new_text
    return new_text


def generate_batch(
    model:          GPT,
    tokenizer:      BPETokenizer,
    prompts:        list[str],
    max_new_tokens: int   = 200,
    temperature:    float = 0.8,
    top_k:          int   = 50,
    top_p:          float = 0.9,
) -> list[str]:
    """
    Generate text for multiple prompts sequentially.
    (True batched generation with padding is left as a future extension.)

    Returns:
        List of generated text strings, one per prompt
    """
    return [
        generate_text(model, tokenizer, p, max_new_tokens, temperature, top_k, top_p)
        for p in prompts
    ]


@torch.no_grad()
def stream_generate(
    model:          GPT,
    tokenizer:      BPETokenizer,
    prompt:         str,
    max_new_tokens: int   = 200,
    temperature:    float = 0.8,
    top_k:          int   = 50,
    top_p:          float = 0.9,
) -> Iterator[tuple[str, int, int]]:
    """
    Streaming text generation — yields one token at a time.

    Implements its own autoregressive loop (cannot use model.generate()
    because we need to yield between steps).

    Yields:
        (token_text, token_id, position) tuples
        token_text: decoded string for this single token
                    (</w> replaced with space, no strip — preserves word boundaries)
        token_id:   integer ID of the generated token
        position:   0-indexed position in the generated sequence

    Usage (in async FastAPI WebSocket):
        async for token_text, token_id, pos in stream_generate(model, tok, prompt):
            await ws.send_json({"type": "token", "token": token_text, ...})
    """
    device = next(model.parameters()).device
    model.eval()

    # Encode prompt with BOS
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    idx        = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    eos_id = SPECIAL_TOKENS["<EOS>"]

    for position in range(max_new_tokens):
        # Sliding window: never exceed context_length
        idx_cond = idx[:, -model.cfg.context_length:]

        # Forward pass — we only need the last position's logits
        logits, _ = model(idx_cond)
        logits     = logits[:, -1, :]         # (1, vocab_size)

        # ── Sampling ─────────────────────────────────────────────────
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)  # (1, 1)
        else:
            logits = logits / temperature

            # Top-k
            if top_k > 0:
                k       = min(top_k, logits.size(-1))
                kth_val = torch.topk(logits, k).values[:, -1].unsqueeze(-1)
                logits  = logits.masked_fill(logits < kth_val, float("-inf"))

            probs = F.softmax(logits, dim=-1)

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_p, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_p, dim=-1)
                remove = (cumsum - sorted_p) > top_p
                sorted_p[remove] = 0.0
                sorted_p /= sorted_p.sum(dim=-1, keepdim=True)
                probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_p)

            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        next_token_id = next_id.item()

        # Decode this single token without calling tokenizer.decode()
        # (which strips trailing spaces — breaking word boundary spacing)
        raw_token = tokenizer.id_to_token.get(next_token_id, "<UNK>")
        # Replace </w> with space (word-boundary marker → space)
        token_text = raw_token.replace(END_OF_WORD, " ")

        yield token_text, next_token_id, position

        # Append to sequence for next step
        idx = torch.cat([idx, next_id], dim=1)

        # Stop on EOS
        if next_token_id == eos_id:
            break