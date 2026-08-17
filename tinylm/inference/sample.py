# tinylm/inference/sample.py
# Sampling parameter exploration — generates text across a grid of settings.
#
# Usage: python -m tinylm.inference.sample
#        python -m tinylm.inference.sample --prompt "The horror" --seed 42

from __future__ import annotations
import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from tinylm.config import load_config
from tinylm.tokenizer import BPETokenizer
from tinylm.model import GPT
from tinylm.inference.generate import generate_text


@dataclass
class SamplingConfig:
    label:       str
    temperature: float
    top_k:       int
    top_p:       float
    description: str


# Curated grid of sampling configurations
SAMPLING_GRID: list[SamplingConfig] = [
    SamplingConfig(
        label       = "greedy",
        temperature = 0.0,
        top_k       = 0,
        top_p       = 1.0,
        description = "Deterministic argmax — always picks the single most likely token",
    ),
    SamplingConfig(
        label       = "conservative",
        temperature = 0.5,
        top_k       = 50,
        top_p       = 0.9,
        description = "Low temperature — coherent, slightly repetitive",
    ),
    SamplingConfig(
        label       = "default",
        temperature = 0.8,
        top_k       = 50,
        top_p       = 0.9,
        description = "Recommended default — good balance of coherence and variety",
    ),
    SamplingConfig(
        label       = "creative",
        temperature = 1.1,
        top_k       = 100,
        top_p       = 0.95,
        description = "High temperature — more surprising word choices",
    ),
    SamplingConfig(
        label       = "chaotic",
        temperature = 1.5,
        top_k       = 0,
        top_p       = 1.0,
        description = "No filtering — full vocabulary distribution at high temperature",
    ),
]


def run_comparison(
    model:          GPT,
    tokenizer:      BPETokenizer,
    prompt:         str,
    max_new_tokens: int           = 100,
    seed:           Optional[int] = None,
    configs:        Optional[list[SamplingConfig]] = None,
) -> dict[str, str]:
    """
    Generate text for each sampling configuration and return results.

    Args:
        model:          Trained GPT
        tokenizer:      Trained BPETokenizer
        prompt:         Prompt to complete
        max_new_tokens: Tokens to generate per config
        seed:           Base random seed (each config uses seed + i for comparability)
        configs:        Override the default SAMPLING_GRID

    Returns:
        Dict mapping config label → generated text
    """
    configs  = configs or SAMPLING_GRID
    results  = {}

    for i, cfg in enumerate(configs):
        config_seed = (seed + i) if seed is not None else None
        text = generate_text(
            model          = model,
            tokenizer      = tokenizer,
            prompt         = prompt,
            max_new_tokens = max_new_tokens,
            temperature    = cfg.temperature,
            top_k          = cfg.top_k,
            top_p          = cfg.top_p,
            seed           = config_seed,
        )
        results[cfg.label] = text

    return results


def print_comparison(
    prompt:  str,
    results: dict[str, str],
    configs: list[SamplingConfig],
    width:   int = 72,
) -> None:
    """Pretty-print the sampling comparison results."""
    print("\n" + "═" * width)
    print(f"  Prompt: \"{prompt}\"")
    print("═" * width)

    config_map = {c.label: c for c in configs}

    for label, text in results.items():
        cfg = config_map.get(label)
        print(f"\n── {label.upper()}")
        if cfg:
            print(f"   temp={cfg.temperature} | top_k={cfg.top_k} | top_p={cfg.top_p}")
            print(f"   {cfg.description}")
        print()
        # Wrap text at width for readable terminal output
        wrapped = textwrap.fill(text.strip(), width=width - 3,
                                initial_indent="   ", subsequent_indent="   ")
        print(wrapped)

    print("\n" + "═" * width)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare sampling parameters on a single prompt"
    )
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer",  default="checkpoints/tokenizer.json")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--prompt",
                   default="The ancient city lay beneath the waves, its cyclopean")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg       = load_config(args.config)
    tok       = BPETokenizer.load(args.tokenizer)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint} ...")
    ckpt      = torch.load(args.checkpoint, map_location=device)
    model     = GPT(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    step      = ckpt.get("step", "?")
    val_loss  = ckpt.get("best_val_loss", float("nan"))
    print(f"  Step {step} | best_val_loss {val_loss:.4f}")

    results   = run_comparison(
        model          = model,
        tokenizer      = tok,
        prompt         = args.prompt,
        max_new_tokens = args.max_tokens,
        seed           = args.seed,
    )

    print_comparison(args.prompt, results, SAMPLING_GRID)


if __name__ == "__main__":
    main()