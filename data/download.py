# data/download.py
# Downloads Lovecraft texts from Project Gutenberg and concatenates
# into a single clean corpus file at data/raw/lovecraft_raw.txt
#
# Usage: python data/download.py
# Idempotent: re-running skips already-downloaded files.

import time
import requests
from pathlib import Path

# Local import — works because we ran pip install -e .
from lovecraft_urls import LOVECRAFT_TEXTS, OMNIBUS_FALLBACK

RAW_DIR    = Path("data/raw")
OUTPUT_FILE = RAW_DIR / "lovecraft_raw.txt"

# Gutenberg asks for a User-Agent that identifies the project
HEADERS = {
    "User-Agent": "TinyLM-Research/1.0 (educational transformer training; "
                  "github.com/<YOUR_USERNAME>/tinylm)"
}


def fetch_text(url: str, retries: int = 3, delay: float = 2.0) -> str | None:
    """
    Fetch plain text from a URL with retries.
    Returns the decoded string, or None on failure.
    """
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            # Gutenberg files are UTF-8 with BOM on some; handle both
            return resp.content.decode("utf-8-sig", errors="replace")
        except requests.RequestException as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # exponential backoff
    return None


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg header and footer from a raw text file.

    Gutenberg files have this structure:
        [header — title, credits, legal notice]
        *** START OF THE PROJECT GUTENBERG EBOOK [TITLE] ***
        [actual content]
        *** END OF THE PROJECT GUTENBERG EBOOK [TITLE] ***
        [footer — contact info, license]

    We extract only the content between the START and END markers.
    """
    START_MARKERS = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    ]
    END_MARKERS = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    ]

    start_idx = None
    for marker in START_MARKERS:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            # Move past the marker line itself
            start_idx = text.find("\n", idx) + 1
            break

    end_idx = None
    for marker in END_MARKERS:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            end_idx = idx
            break

    if start_idx is None or end_idx is None:
        # Marker not found — return full text with a warning
        print("    ⚠ Could not find Gutenberg markers; using full text")
        return text.strip()

    return text[start_idx:end_idx].strip()


def download_corpus() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_text_parts: list[str] = []
    successful = 0

    for entry in LOVECRAFT_TEXTS:
        title = entry["title"]
        url   = entry["url"]
        cache_file = RAW_DIR / f"pg{entry['pg_id']}.txt"

        print(f"\n{'─'*60}")
        print(f"  {title}")

        # Use cached file if available (idempotent behaviour)
        if cache_file.exists():
            print(f"  ✓ Using cached: {cache_file}")
            raw = cache_file.read_text(encoding="utf-8")
        else:
            print(f"  ↓ Downloading from {url}")
            raw = fetch_text(url)
            if raw is None:
                print(f"  ✗ Failed — skipping")
                continue
            cache_file.write_text(raw, encoding="utf-8")
            print(f"  ✓ Saved to {cache_file} ({len(raw):,} chars)")
            time.sleep(1.5)  # Be polite to Gutenberg servers

        clean = strip_gutenberg_boilerplate(raw)

        # Wrap each story with a clear delimiter so the tokenizer
        # doesn't see story boundaries as mid-narrative transitions
        story_block = f"\n\n{'='*60}\n{title.upper()}\n{'='*60}\n\n{clean}"
        all_text_parts.append(story_block)
        successful += 1

    if not all_text_parts:
        # All individual downloads failed — try the omnibus fallback
        print("\n⚠ All individual downloads failed. Trying omnibus fallback...")
        raw = fetch_text(OMNIBUS_FALLBACK["url"])
        if raw:
            all_text_parts.append(strip_gutenberg_boilerplate(raw))
            successful = 1

    if not all_text_parts:
        raise RuntimeError(
            "Could not download any Lovecraft texts. "
            "Check your internet connection and verify the URLs in "
            "data/lovecraft_urls.py against https://www.gutenberg.org"
        )

    corpus = "\n\n".join(all_text_parts)
    OUTPUT_FILE.write_text(corpus, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"✓ Downloaded {successful}/{len(LOVECRAFT_TEXTS)} texts")
    print(f"✓ Corpus saved to {OUTPUT_FILE}")
    print(f"✓ Total characters: {len(corpus):,}")
    print(f"✓ Approx. size: {len(corpus.encode('utf-8')) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    download_corpus()