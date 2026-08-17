# tinylm/evaluation/metrics.py
# Language model evaluation metrics.
# All metrics operate on token-level averages (not sequence-level).

from __future__ import annotations
import math
from typing import Optional

import torch
from torch.utils.data import DataLoader

from tinylm.model.gpt import GPT
from tinylm.tokenizer import BPETokenizer
from tinylm.tokenizer.vocab import SPECIAL_TOKENS


def compute_perplexity(
    model:       GPT,
    data_loader: DataLoader,
    device:      torch.device,
    max_batches: Optional[int] = None,
    verbose:     bool = True,
) -> tuple[float, float]:
    """
    Compute token-level perplexity on a dataset.

    Token-level: each token contributes equally regardless of sequence length.
    This is the correct way to report perplexity per the NLP literature.

    Args:
        model:       Trained GPT model
        data_loader: DataLoader for the evaluation dataset
        device:      Device to run inference on
        max_batches: Cap evaluation at this many batches (None = full dataset)
        verbose:     Print progress

    Returns:
        (perplexity, mean_loss) — perplexity = exp(mean_loss)

    Note on numerical stability:
        We cap the exponent at 20 before calling exp() to avoid overflow
        (exp(20) ≈ 485M — already larger than any useful vocab size).
    """
    model.eval()

    total_loss   = 0.0   # sum of (loss_per_batch × n_tokens_per_batch)
    total_tokens = 0     # total non-PAD tokens seen

    pad_id = SPECIAL_TOKENS["<PAD>"]

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x, y = x.to(device), y.to(device)

            # Mixed precision for consistency with training
            device_type = "cuda" if device.type == "cuda" else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=(device_type == "cuda")):
                _, loss = model(x, y)

            # Count non-PAD tokens in this batch (targets with id != pad_id)
            n_tokens = (y != pad_id).sum().item()

            # Accumulate the SUM of per-token losses (loss * n_tokens reverses mean)
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

            if verbose and (batch_idx + 1) % 20 == 0:
                running_ppl = math.exp(min(total_loss / total_tokens, 20))
                print(f"  Batch {batch_idx+1}: running perplexity = {running_ppl:.2f}")

    if total_tokens == 0:
        raise ValueError("No tokens evaluated — check DataLoader is not empty")

    # Token-level mean loss
    mean_loss  = total_loss / total_tokens

    # Numerical stability cap: perplexity > exp(20) ≈ 485M is meaningless
    perplexity = math.exp(min(mean_loss, 20.0))

    model.train()  # Restore training mode
    return perplexity, mean_loss


def bits_per_character(
    model:       GPT,
    tokenizer:   BPETokenizer,
    text:        str,
    device:      torch.device,
) -> float:
    """
    Compute bits-per-character (BPC) — an alternative LM metric.

    BPC = mean_loss / log(2)

    BPC is model-independent: it doesn't change with vocabulary size,
    making it useful for comparing models with different tokenizers.

    For Lovecraft corpus: good models achieve ~1.3–1.8 BPC.
    """
    model.eval()

    ids     = tokenizer.encode(text, add_bos=True)
    idx     = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([ids[1:]],  dtype=torch.long, device=device)

    with torch.no_grad():
        _, loss = model(idx, targets)

    # Convert from nats to bits
    return loss.item() / math.log(2)