# data/verify.py
# Validates the processed corpus before tokenizer training.
# Prints a human-readable report and exits 1 on critical failures.
#
# Usage: python data/verify.py

import sys
from pathlib import Path

PROCESSED_FILE = Path("data/processed/corpus.txt")

# Lovecraft's most distinctive vocabulary — if these are absent,
# the wrong corpus was downloaded.
EXPECTED_WORDS = [
    "Cthulhu", "eldritch", "cyclopean", "Miskatonic",
    "Necronomicon", "blasphemous", "Poe", "Usher",
    "horror", "darkness", "strange",
]

MIN_SIZE_BYTES = 3_000_000  # 3MB minimum — expect ~5-7MB


def verify_corpus() -> None:
    print("=" * 60)
    print("Corpus Verification Report")
    print("=" * 60)

    failures = []

    # ── Check file exists ────────────────────────────────────────────
    if not PROCESSED_FILE.exists():
        print("✗ CRITICAL: corpus.txt not found")
        print("  Run: python data/download.py && python data/preprocess.py")
        sys.exit(1)

    text = PROCESSED_FILE.read_text(encoding="utf-8")
    size_bytes = PROCESSED_FILE.stat().st_size

    # ── Size check ───────────────────────────────────────────────────
    size_mb = size_bytes / 1024 / 1024
    status = "✓" if size_bytes >= MIN_SIZE_BYTES else "✗"
    if size_bytes < MIN_SIZE_BYTES:
        failures.append(f"Corpus too small ({size_mb:.2f} MB < 0.5 MB)")
    print(f"{status} Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    print(f"  Characters: {len(text):,}")
    print(f"  Lines: {text.count(chr(10)):,}")

    # ── Null byte check ──────────────────────────────────────────────
    null_count = text.count("\x00")
    status = "✓" if null_count == 0 else "✗"
    if null_count > 0:
        failures.append(f"Found {null_count} null bytes — corrupt file")
    print(f"{status} Null bytes: {null_count}")

    # ── Character set sanity ─────────────────────────────────────────
    non_ascii = sum(1 for c in text if ord(c) > 127)
    non_ascii_pct = non_ascii / len(text) * 100
    status = "✓" if non_ascii_pct < 2.0 else "⚠"
    print(f"{status} Non-ASCII chars: {non_ascii:,} ({non_ascii_pct:.2f}%)")
    if non_ascii_pct >= 5.0:
        failures.append(f"High non-ASCII ratio ({non_ascii_pct:.1f}%) — check encoding")

    # ── Vocabulary spot-check ────────────────────────────────────────
    print("\nDistinctive vocabulary check:")
    missing = []
    for word in EXPECTED_WORDS:
        count = text.count(word)
        status = "✓" if count > 0 else "✗"
        print(f"  {status} '{word}': {count} occurrences")
        if count == 0:
            missing.append(word)
    if missing:
        failures.append(f"Missing expected words: {missing}")

    # ── Train/val split preview ──────────────────────────────────────
    split_idx = int(len(text) * 0.9)
    train_chars = split_idx
    val_chars   = len(text) - split_idx
    print(f"\nTrain/val split (90/10):")
    print(f"  Train: {train_chars:,} chars ({train_chars/1024/1024:.2f} MB)")
    print(f"  Val:   {val_chars:,} chars ({val_chars/1024/1024:.2f} MB)")

    # ── Final verdict ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures:
        print(f"✗ FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print("✓ All checks passed — corpus ready for tokenizer training")
        print("  Next step: python -m tinylm.tokenizer.train")


if __name__ == "__main__":
    verify_corpus()