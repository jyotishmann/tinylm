# Run as: python -c "exec(open('test_roundtrip.py').read())"
# or paste into a Python REPL after running train() below

# ── First, train on a tiny corpus to test quickly ────────────────────
from tinylm.tokenizer.tokenizer import BPETokenizer

tiny_corpus = """
The most merciful thing in the world, I think, is the inability
of the human mind to correlate all its contents. We live on a placid
island of ignorance in the midst of black seas of infinity, and it
was not meant that we should voyage far. The sciences, each straining
in its own direction, have hitherto harmed us little; but someday
the piecing together of dissociated knowledge will open up such
terrifying vistas of reality, and of our frightful position therein,
that we shall either go mad from the revelation or flee from the
deadly light into the peace and safety of a new dark age.
"""  # From The Call of Cthulhu — public domain

tok = BPETokenizer()
tok.train(tiny_corpus, vocab_size=200, verbose=False)
print(f"Vocab size: {tok.vocab_size}")

# ── Round-trip tests ────────────────────────────────────────────────
test_cases = [
    "the human mind",
    "terrifying vistas of reality",
    "black seas of infinity",
    "We live on a placid island",
]

print("\nRound-trip tests:")
all_passed = True
for text in test_cases:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    ok = decoded == text
    all_passed = all_passed and ok
    status = "✓" if ok else "✗"
    print(f"  {status} '{text}'")
    if not ok:
        print(f"    encoded: {ids}")
        print(f"    decoded: '{decoded}'")

# ── Peek at tokenisation ─────────────────────────────────────────────
print("\nTokenisation examples:")
for text in ["terrifying", "dissociated knowledge"]:
    ids = tok.encode(text)
    tokens = [tok.id_to_token[i] for i in ids]
    print(f"  '{text}' → {tokens}")

print(f"\n{'All round-trips passed ✓' if all_passed else 'FAILURES detected ✗'}")