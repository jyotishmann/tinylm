# tinylm/tokenizer/tokenizer.py
# BPETokenizer: trains BPE, encodes text → IDs, decodes IDs → text.
# This is the file you demo at interviews.

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from tinylm.tokenizer.vocab import (
    SPECIAL_TOKENS, SPECIAL_IDS, END_OF_WORD,
    build_char_vocab, pretokenize,
)
from tinylm.tokenizer.bpe import (
    build_word_freqs, get_stats, merge_vocab,
)


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer trained from scratch on a text corpus.

    Attributes:
        merges:      Ordered list of (a, b) merge rules (training order).
        token_to_id: Map from token string to integer ID.
        id_to_token: Inverse map.
        _merge_rank: Cache — {pair: rank} for fast encode (built lazily).

    Usage:
        tok = BPETokenizer()
        tok.train(open("corpus.txt").read(), vocab_size=5000)
        ids = tok.encode("The eldritch horror")
        text = tok.decode(ids)
        tok.save("tokenizer.json")

        tok2 = BPETokenizer.load("tokenizer.json")
    """

    def __init__(self) -> None:
        # Merge rules in training order — ORDER MATTERS during encode()
        self.merges:      list[tuple[str, str]] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        # Lazy cache — built on first encode() call, invalidated on train()
        self._merge_rank: Optional[dict[tuple[str, str], int]] = None

    # ── Properties ───────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return SPECIAL_TOKENS["<PAD>"]

    @property
    def unk_id(self) -> int:
        return SPECIAL_TOKENS["<UNK>"]

    @property
    def bos_id(self) -> int:
        return SPECIAL_TOKENS["<BOS>"]

    @property
    def eos_id(self) -> int:
        return SPECIAL_TOKENS["<EOS>"]

    def train(
        self,
        text:       str,
        vocab_size: int  = 5000,
        verbose:    bool = True,
    ) -> None:
        """
        Train BPE on a text corpus.

        Runs (vocab_size - initial_vocab_size) merge iterations.
        After training, self.merges and self.token_to_id are populated.

        Args:
            text:       Raw corpus string (after preprocessing).
            vocab_size: Target vocabulary size (including special tokens).
            verbose:    Show tqdm progress bar.
        """
        # ── Step 1: Initialise vocabulary from corpus characters ─────
        self.token_to_id, self.id_to_token = build_char_vocab(text)
        self._merge_rank = None  # Invalidate any cached merge rank

        # ── Step 2: Build word-frequency table ────────────────────────
        if verbose:
            print("Building word frequency table...")
        word_freqs = build_word_freqs(text)

        if verbose:
            n_unique_words = len(word_freqs)
            n_total_tokens = sum(
                len(w) * f for w, f in word_freqs.items()
            )
            print(f"  Unique word types : {n_unique_words:,}")
            print(f"  Total symbol tokens: {n_total_tokens:,}")
            print(f"  Initial vocab size : {self.vocab_size:,}")
            print(f"  Target vocab size  : {vocab_size:,}")
            print(f"  Merges needed      : {vocab_size - self.vocab_size:,}")

        # ── Step 3: BPE merge loop ────────────────────────────────────
        n_merges = vocab_size - self.vocab_size
        if n_merges <= 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be greater than "
                f"initial vocab size ({self.vocab_size}). "
                f"Increase vocab_size or use a larger corpus."
            )

        iterator = range(n_merges)
        if verbose:
            iterator = tqdm(iterator, desc="BPE training", unit="merge")

        for _ in iterator:
            # Count all adjacent symbol pairs
            stats = get_stats(word_freqs)
            if not stats:
                if verbose:
                    print("No more pairs to merge — stopping early.")
                break

            # Select the most frequent pair
            best_pair = max(stats, key=stats.get)
            best_freq = stats[best_pair]

            # Apply the merge across all words
            word_freqs = merge_vocab(best_pair, word_freqs)

            # Register the new merged token in the vocabulary
            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.token_to_id)
            self.token_to_id[new_token] = new_id
            self.id_to_token[new_id]    = new_token

            # Record the merge rule (ORDER MATTERS — do not sort later)
            self.merges.append(best_pair)

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({
                    "pair": f"{best_pair[0]}+{best_pair[1]}",
                    "freq": best_freq,
                    "vocab": self.vocab_size,
                })

        if verbose:
            print(f"\n✓ Training complete.")
            print(f"  Final vocab size: {self.vocab_size:,}")
            print(f"  Merge rules learned: {len(self.merges):,}")