# data/preprocess.py
# Cleans the raw concatenated corpus and saves to data/processed/corpus.txt
#
# Usage: python data/preprocess.py
# Run AFTER download.py.

import re
from pathlib import Path

RAW_FILE       = Path("data/raw/lovecraft_raw.txt")
PROCESSED_FILE = Path("data/processed/corpus.txt")


def clean_text(text: str) -> str:
    """
    Clean raw Gutenberg text for language model training.

    Order of operations matters here:
    1. Normalise line endings first (so all subsequent regex work on \n only)
    2. Remove in-text annotations that are artefacts of OCR/conversion
    3. Normalise whitespace (collapse excess blank lines)
    4. Strip leading/trailing whitespace
    """

    # 1. Normalise line endings (Windows → Unix)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Remove Gutenberg in-text annotations
    #    [Illustration: caption], [pg 34], [Footnote: ...], etc.
    text = re.sub(r"\[Illustration[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[pg\s*\d+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Footnote[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Transcriber[^\]]*\]", "", text, flags=re.IGNORECASE)

    # 3. Remove our story-separator lines (=== TITLE ===)
    #    Replace with two newlines (paragraph break) so structure is preserved
    text = re.sub(r"={3,}[^\n]*={3,}", "\n\n", text)

    # 4. Collapse runs of 3+ blank lines into exactly 2 blank lines.
    #    Two blank lines = paragraph / section break signal for the model.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Remove trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def preprocess_corpus() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw corpus not found at {RAW_FILE}. "
            "Run: python data/download.py"
        )

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {RAW_FILE} ...")
    raw = RAW_FILE.read_text(encoding="utf-8")
    print(f"  Raw size: {len(raw):,} characters")

    cleaned = clean_text(raw)

    PROCESSED_FILE.write_text(cleaned, encoding="utf-8")

    print(f"  Cleaned size: {len(cleaned):,} characters")
    print(f"  Reduction: {(1 - len(cleaned)/len(raw))*100:.1f}%")
    print(f"  Lines: {cleaned.count(chr(10)):,}")
    print(f"  Saved to: {PROCESSED_FILE}")
    print(f"  Size on disk: {PROCESSED_FILE.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    preprocess_corpus()