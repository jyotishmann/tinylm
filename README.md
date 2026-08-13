# TinyLM

> A GPT-style transformer trained from scratch on H.P. Lovecraft's complete works.  
> Built to demonstrate deep understanding of every component — no HuggingFace, no API calls.

**Status:** 🚧 Under construction — follow the build log in [`CELLS-*.md`](./CELLS-01-foundation.md)

## What this is

- Custom BPE tokenizer (from scratch, ~5k vocab)
- Decoder-only transformer (~12.7M parameters)
- Trained on ~2MB of public-domain Lovecraft prose
- Served via FastAPI with WebSocket streaming
- React frontend with live attention visualisation

## Stack

`PyTorch` · `FastAPI` · `React + Vite` · `TailwindCSS`

## Quick start

_Full instructions in [CELLS-08-integration.md](./CELLS-08-integration.md) once training is complete._

## Architecture

See [`MASTER.md`](./MASTER.md) for the complete design document.

---

*Training corpus: The complete works of H.P. Lovecraft — public domain (died 1937)*