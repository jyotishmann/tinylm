# tinylm/tokenizer/train.py
# Training entrypoint for the BPE tokenizer.
# Usage: python -m tinylm.tokenizer.train
#        python -m tinylm.tokenizer.train --config configs/default.yaml
#        python -m tinylm.tokenizer.train --vocab-size 3000

from __future__ import annotations
import argparse
import time
from pathlib import Path

from tinylm.config import load_config
from tinylm.tokenizer.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BPE tokenizer on preprocessed Lovecraft corpus"
    )
    p.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)"
    )
    p.add_argument(
        "--corpus", type=str, default=None,
        help="Override corpus path from config"
    )
    p.add_argument(
        "--vocab-size", type=int, default=None,
        help="Override vocab_size from config"
    )
    p.add_argument(
        "--output", type=str, default="checkpoints/tokenizer.json",
        help="Output path for trained tokenizer (default: checkpoints/tokenizer.json)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────
    cfg = load_config(args.config)
    corpus_path = Path(args.corpus or "data/processed/corpus.txt")
    vocab_size  = args.vocab_size or cfg.model.vocab_size
    output_path = Path(args.output)

    print("=" * 60)
    print("  TinyLM — BPE Tokenizer Training")
    print("=" * 60)
    print(f"  Corpus     : {corpus_path}")
    print(f"  Vocab size : {vocab_size:,}")
    print(f"  Output     : {output_path}")
    print("=" * 60)

    # ── Load corpus ──────────────────────────────────────────────────
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}\n"
            "Run: bash data/build_corpus.sh"
        )
    corpus = corpus_path.read_text(encoding="utf-8")
    print(f"\n✓ Corpus loaded: {len(corpus):,} characters "
          f"({corpus_path.stat().st_size / 1024 / 1024:.2f} MB)\n")

    # ── Train ────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=vocab_size, verbose=True)

    elapsed = time.perf_counter() - t0
    print(f"\n✓ Training complete in {elapsed:.1f}s "
          f"({elapsed / 60:.1f} min)")

    # ── Save ─────────────────────────────────────────────────────────
    tok.save(output_path)

    # ── Smoke test ───────────────────────────────────────────────────
    print("\nSmoke test:")
    test_phrases = [
        "The eldritch horror",
        "cyclopean architecture",
        "non-Euclidean geometry",
        "Ph'nglui mglw'nafh Cthulhu",
    ]
    tok2 = BPETokenizer.load(output_path)
    for phrase in test_phrases:
        ids    = tok2.encode(phrase)
        tokens = [tok2.id_to_token[i] for i in ids]
        decoded = tok2.decode(ids)
        ok = decoded == phrase
        print(f"  {'✓' if ok else '✗'} '{phrase}'")
        print(f"    tokens : {tokens}")
        print(f"    ids    : {ids}")

    print("\n" + "=" * 60)
    print(f"  Tokenizer ready: {output_path}")
    print(f"  Load with: BPETokenizer.load('{output_path}')")
    print("=" * 60)


if __name__ == "__main__":
    main()