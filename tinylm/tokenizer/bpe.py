# tinylm/tokenizer/bpe.py
# Core BPE algorithmic functions — stateless, pure, testable in isolation.
# These are the functions an interviewer will ask you to whiteboard.

from __future__ import annotations
from collections import defaultdict

from tinylm.tokenizer.vocab import END_OF_WORD, pretokenize


def build_word_freqs(text: str) -> dict[tuple[str, ...], int]:
    """
    Build word frequency table from raw corpus text.

    Each unique pre-token becomes a tuple of its characters + END_OF_WORD.
    The value is the frequency (how many times that word appears).

    This table is the input to the entire BPE training loop.
    It is built once and then mutated in-place by merge_vocab().

    Args:
        text: Raw corpus string (after preprocessing)

    Returns:
        Dict mapping character-tuple → frequency

    Example:
        text = "the great old ones"
        → {
            ('t','h','e','</w>'): 1,
            ('g','r','e','a','t','</w>'): 1,
            ('o','l','d','</w>'): 1,
            ('o','n','e','s','</w>'): 1,
          }
    """
    word_freqs: dict[tuple[str, ...], int] = defaultdict(int)

    for word in pretokenize(text):
        # Each character becomes a separate symbol; </w> marks word end
        chars = list(word) + [END_OF_WORD]
        word_freqs[tuple(chars)] += 1

    return dict(word_freqs)

def get_stats(
    word_freqs: dict[tuple[str, ...], int]
) -> dict[tuple[str, str], int]:
    """
    Count the frequency of every adjacent symbol pair in the vocabulary.

    Weighted by word frequency — a pair in a word that appears 10,000
    times counts as 10,000, not 1.

    This is called at every BPE training step to find the best merge.
    It is the innermost loop of the algorithm — keep it tight.

    Args:
        word_freqs: Current word frequency table (mutated across training steps)

    Returns:
        Dict mapping (symbol_a, symbol_b) → weighted frequency

    Complexity: O(sum of all word lengths × unique word count)
                For our corpus: ~30K unique words × avg 6 symbols = ~180K ops/step
    """
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for word_tuple, freq in word_freqs.items():
        # Slide a window of size 2 over the symbol sequence
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pair_counts[pair] += freq

    return dict(pair_counts)

def merge_vocab(
    best_pair: tuple[str, str],
    word_freqs: dict[tuple[str, ...], int],
) -> dict[tuple[str, ...], int]:
    """
    Apply a merge rule to the entire word-frequency table.

    Every occurrence of (best_pair[0], best_pair[1]) as adjacent symbols
    is replaced by the concatenated string best_pair[0] + best_pair[1].

    Args:
        best_pair: The (a, b) pair to merge, e.g. ('e', 'l')
        word_freqs: Current vocabulary to update

    Returns:
        New word frequency table with the merge applied

    Example:
        best_pair = ('e', 'l')
        input  = {('e','l','d','r','i','t','c','h','</w>'): 412}
        output = {('el','d','r','i','t','c','h','</w>'): 412}
    """
    new_vocab: dict[tuple[str, ...], int] = {}
    a, b = best_pair
    merged = a + b  # e.g. 'e' + 'l' = 'el'

    for word_tuple, freq in word_freqs.items():
        # Scan through symbols and apply the merge wherever the pair appears
        new_symbols: list[str] = []
        i = 0
        while i < len(word_tuple):
            if (
                i < len(word_tuple) - 1
                and word_tuple[i] == a
                and word_tuple[i + 1] == b
            ):
                new_symbols.append(merged)
                i += 2  # skip both elements of the pair
            else:
                new_symbols.append(word_tuple[i])
                i += 1

        new_vocab[tuple(new_symbols)] = freq

    return new_vocab