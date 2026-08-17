# tinylm/tokenizer/vocab.py
# Special token definitions and initial character vocabulary builder.
# IDs 0-3 are reserved for special tokens and must never change.

from __future__ import annotations

# ── Special token strings and their fixed IDs ────────────────────────
SPECIAL_TOKENS: dict[str, int] = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}

# Inverse mapping — used during decode to identify special IDs
SPECIAL_IDS: dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

# Marks the end of a word in the BPE representation.
# "eldritch" → ('e','l','d','r','i','t','c','h', END_OF_WORD)
# This is what lets decode() reconstruct word boundaries.
END_OF_WORD: str = "</w>"


def build_char_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build the initial character-level vocabulary from a corpus string.

    Returns:
        token_to_id: maps each character (and special tokens) to an integer ID
        id_to_token: the inverse mapping

    The special tokens occupy IDs 0-3.
    Characters are assigned IDs starting at 4, sorted for determinism.
    The END_OF_WORD marker is included as a vocabulary token.

    Example output (partial):
        token_to_id = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
                       '!': 4, '"': 5, ..., 'z': 78, '</w>': 79}
    """
    # Collect all unique characters in the corpus
    unique_chars = sorted(set(text))

    token_to_id: dict[str, int] = dict(SPECIAL_TOKENS)  # copy, don't mutate
    next_id = len(token_to_id)  # = 4

    # Add END_OF_WORD first (after special tokens) for a clean layout
    token_to_id[END_OF_WORD] = next_id
    next_id += 1

    # Add every character, skipping any that are already in special tokens
    for ch in unique_chars:
        if ch not in token_to_id:
            token_to_id[ch] = next_id
            next_id += 1

    id_to_token = {v: k for k, v in token_to_id.items()}

    return token_to_id, id_to_token


def vocab_summary(token_to_id: dict[str, int]) -> None:
    """Print a human-readable summary of the vocabulary."""
    n_special = len(SPECIAL_TOKENS)
    n_chars = sum(1 for t in token_to_id if len(t) == 1 or t == END_OF_WORD)
    n_bpe = len(token_to_id) - n_special - n_chars

    print(f"Vocabulary size : {len(token_to_id):>6,}")
    print(f"  Special tokens: {n_special:>6,}  {list(SPECIAL_TOKENS.keys())}")
    print(f"  Char tokens   : {n_chars:>6,}")
    print(f"  BPE tokens    : {n_bpe:>6,}")

import re

# Pre-tokenisation pattern.
# Matches: sequences of word characters OR single non-whitespace characters.
# "Hello, world!" → ["Hello", ",", "world", "!"]
# "non-Euclidean" → ["non", "-", "Euclidean"]
_PRETOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def pretokenize(text: str) -> list[str]:
    """
    Split text into pre-tokens (words and punctuation).

    This runs before BPE. It defines what counts as a "word boundary."
    Whitespace is consumed and not returned as a token — spaces are
    implicitly encoded by the END_OF_WORD marker on the preceding token.

    Examples:
        "eldritch horror!"  → ["eldritch", "horror", "!"]
        "non-Euclidean"     → ["non", "-", "Euclidean"]
        "Ph'nglui mglw'nafh"→ ["Ph", "'", "nglui", "mglw", "'", "nafh"]
    """
    return _PRETOK_RE.findall(text)