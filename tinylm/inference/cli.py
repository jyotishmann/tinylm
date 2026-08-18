# tinylm/inference/cli.py
# Interactive CLI for text generation.
# Usage:
#   python -m tinylm.inference.cli --prompt "The ancient one"
#   python -m tinylm.inference.cli --interactive

from __future__ import annotations
import argparse
import time

import torch

from tinylm.config import load_config
from tinylm.tokenizer import BPETokenizer
from tinylm.model import GPT
from tinylm.inference.generate import generate_text


BANNER = """
╔══════════════════════════════════════════════════╗
║          TinyLM (EMG-01) — Inference CLI         ║
║   GPT trained from scratch on H.P. Lovecraft     ║
╚══════════════════════════════════════════════════╝
  Commands: 'quit' or Ctrl-C to exit
            'temp=0.5' to change temperature mid-session
            'tokens=200' to change max output length
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TinyLM text generation CLI"
    )
    p.add_argument("--checkpoint",   default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer",    default="checkpoints/tokenizer.json")
    p.add_argument("--config",       default="configs/default.yaml")
    p.add_argument("--prompt",       default=None,
                   help="Single-shot prompt (exits after generation)")
    p.add_argument("--interactive",  action="store_true",
                   help="REPL mode — enter prompts interactively")
    p.add_argument("--max-tokens",   type=int,   default=150)
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--top-k",        type=int,   default=50)
    p.add_argument("--top-p",        type=float, default=0.9)
    p.add_argument("--seed",         type=int,   default=None)
    return p.parse_args()


def load_model(args, cfg) -> tuple[GPT, torch.device]:
    """Load model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.checkpoint} on {device} ...")

    ckpt   = torch.load(args.checkpoint, map_location=device)
    model  = GPT(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    step     = ckpt.get("step", "?")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.1f}M params | step {step} | val_loss {val_loss:.4f}")
    return model, device


def run_generation(
    model, tok, prompt, args, temperature=None, max_tokens=None
) -> str:
    """Run generation and print with timing."""
    temp       = temperature or args.temperature
    max_tok    = max_tokens  or args.max_tokens
    device     = next(model.parameters()).device

    t0  = time.perf_counter()
    out = generate_text(
        model          = model,
        tokenizer      = tok,
        prompt         = prompt,
        max_new_tokens = max_tok,
        temperature    = temp,
        top_k          = args.top_k,
        top_p          = args.top_p,
        seed           = args.seed,
    )
    elapsed = time.perf_counter() - t0

    n_tokens    = len(tok.encode(out))
    tokens_sec  = n_tokens / elapsed if elapsed > 0 else 0

    return out, elapsed, tokens_sec


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    tok  = BPETokenizer.load(args.tokenizer)
    model, device = load_model(args, cfg)

    # ── Single-shot mode ──────────────────────────────────────────────
    if args.prompt and not args.interactive:
        print(f"\nPrompt: {args.prompt}\n")
        out, elapsed, tok_s = run_generation(model, tok, args.prompt, args)
        print(f"Generated:\n{out}")
        print(f"\n[{len(tok.encode(out))} tokens | {elapsed:.1f}s | {tok_s:.0f} tok/s]")
        return

    # ── Interactive REPL mode ─────────────────────────────────────────
    print(BANNER)
    temperature = args.temperature
    max_tokens  = args.max_tokens

    print(f"  temperature={temperature} | top_k={args.top_k} | "
          f"top_p={args.top_p} | max_tokens={max_tokens}\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        # Handle setting commands
        if prompt.startswith("temp="):
            try:
                temperature = float(prompt.split("=")[1])
                print(f"  temperature → {temperature}")
            except ValueError:
                print("  Usage: temp=0.8")
            continue
        if prompt.startswith("tokens="):
            try:
                max_tokens = int(prompt.split("=")[1])
                print(f"  max_tokens → {max_tokens}")
            except ValueError:
                print("  Usage: tokens=200")
            continue

        print()
        out, elapsed, tok_s = run_generation(
            model, tok, prompt, args,
            temperature=temperature, max_tokens=max_tokens,
        )
        print(out)
        print(f"\n  [{len(tok.encode(out))} tokens | {elapsed:.1f}s | "
              f"{tok_s:.0f} tok/s]\n")


if __name__ == "__main__":
    main()