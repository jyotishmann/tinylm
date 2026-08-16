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

    def _build_merge_rank(self) -> None:
        """
        Build (or rebuild) the merge-rank lookup cache.

        Maps each merge pair to its index in self.merges.
        Lower rank = learned earlier = applied first during encoding.

        Called lazily on first encode() call.
        Must be invalidated (set to None) whenever self.merges changes.
        """
        self._merge_rank = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        

    def _encode_word(self, word: str) -> list[int]:
        """
        Encode a single pre-token (word) to a list of token IDs.

        Algorithm:
          1. Convert word to character list + END_OF_WORD marker
          2. Scan for the lowest-rank applicable merge pair
          3. Apply that merge
          4. Repeat until no applicable pairs remain
          5. Map each resulting symbol to its vocabulary ID

        Args:
            word: A single pre-token string (e.g. "eldritch")

        Returns:
            List of integer token IDs

        Handles OOV characters: if a character is not in the vocabulary
        (shouldn't happen with BPE but we guard anyway), substitute UNK.
        """
        if self._merge_rank is None:
            self._build_merge_rank()

        # Initialise as character sequence + end-of-word
        symbols: list[str] = list(word) + [END_OF_WORD]

        # Iteratively apply the lowest-rank merge that applies
        while len(symbols) > 1:
            # Find the adjacent pair with the lowest merge rank
            best_idx  = -1
            best_rank = float("inf")

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_rank.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx  = i

            if best_rank == float("inf"):
                break  # No applicable merge rules remain

            # Apply the merge at best_idx
            merged = symbols[best_idx] + symbols[best_idx + 1]
            symbols = (
                symbols[:best_idx]
                + [merged]
                + symbols[best_idx + 2:]
            )

        # Convert symbols to IDs, using UNK for any unseen token
        return [
            self.token_to_id.get(sym, self.unk_id)
            for sym in symbols
        ]

    def encode(
        self,
        text:    str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        Encode a text string to a list of integer token IDs.

        Steps:
          1. Pre-tokenise text into words + punctuation
          2. Encode each pre-token via _encode_word()
          3. Flatten the results into a single ID sequence
          4. Optionally prepend BOS and/or append EOS

        Args:
            text:    Input string to encode.
            add_bos: Prepend BOS token ID (for generation prompts).
            add_eos: Append EOS token ID (for training sequences).

        Returns:
            List of integer token IDs.

        Example:
            tok.encode("The eldritch horror")
            → [412, 88, 1023]  (IDs vary with vocabulary)

            tok.encode("The", add_bos=True, add_eos=True)
            → [2, 412, 3]  (BOS=2, "The"=412, EOS=3)
        """
        ids: list[int] = []

        if add_bos:
            ids.append(self.bos_id)

        for word in pretokenize(text):
            ids.extend(self._encode_word(word))

        if add_eos:
            ids.append(self.eos_id)

        return ids


    def decode(
        self,
        ids:            list[int],
        skip_special:   bool = True,
    ) -> str:
        """
        Decode a list of token IDs back to a text string.

        Args:
            ids:          List of integer token IDs.
            skip_special: If True, skip BOS/EOS/PAD tokens (default).
                          Set False to see them as strings (for debugging).

        Returns:
            Decoded text string.

        Example:
            tok.decode([412, 88, 1023])
            → "The eldritch horror"

        Note on END_OF_WORD:
            Token strings contain '</w>' at word boundaries.
            "the</w> eldritch</w> horror</w>" → "the eldritch horror"
            We replace '</w>' with ' ' then strip the trailing space.
        """
        # Filter special token IDs if requested
        if skip_special:
            special_id_set = set(SPECIAL_TOKENS.values())
            ids = [i for i in ids if i not in special_id_set]

        # ID → token string, falling back to <UNK> for unknown IDs
        tokens = [
            self.id_to_token.get(i, "<UNK>")
            for i in ids
        ]

        # Concatenate all tokens and restore word boundaries
        text = "".join(tokens)
        text = text.replace(END_OF_WORD, " ")
        return text.strip()

    def encode_batch(
        self,
        texts:   list[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        """Encode a list of strings. Returns list of ID lists (variable length)."""
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def token_count(self, text: str) -> int:
        """How many tokens does this text encode to? Useful for length checks."""
        return len(self.encode(text))


    def save(self, path: str | Path) -> None:
        """
        Serialise the trained tokenizer to a JSON file.

        The saved file contains everything needed to reconstruct the
        tokenizer exactly: merge rules (in training order) and the
        full vocabulary mapping.

        Args:
            path: File path to save to (e.g. "checkpoints/tokenizer.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version":     "1.0",
            # merges is a list of [a, b] pairs (JSON doesn't have tuples)
            "merges":      [list(pair) for pair in self.merges],
            "token_to_id": self.token_to_id,
            "metadata": {
                "vocab_size":    self.vocab_size,
                "n_merges":      len(self.merges),
                "end_of_word":   END_OF_WORD,
                "special_tokens": SPECIAL_TOKENS,
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✓ Tokenizer saved to {path}")
        print(f"  Vocabulary size : {self.vocab_size:,}")
        print(f"  Merge rules     : {len(self.merges):,}")
        print(f"  File size       : {path.stat().st_size / 1024:.1f} KB")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """
        Load a tokenizer from a saved JSON file.

        Args:
            path: Path to a tokenizer.json file saved by save()

        Returns:
            A fully initialised BPETokenizer ready for encode/decode

        Raises:
            FileNotFoundError: if path does not exist
            ValueError: if the JSON format is unrecognised
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {path}\n"
                "Run: python -m tinylm.tokenizer.train"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("version") != "1.0":
            raise ValueError(
                f"Unrecognised tokenizer version: {data.get('version')}. "
                f"Expected '1.0'."
            )

        tok = cls()
        tok.merges      = [tuple(pair) for pair in data["merges"]]
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(k): v for k, v in
                          {v: k for k, v in tok.token_to_id.items()}.items()}
        # Build integer keys properly from the inverse dict
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        tok._merge_rank = None  # Will be built lazily on first encode()

        return tok