# tinylm/evaluation/evaluate.py
# Full evaluation suite entrypoint.
# Usage: python -m tinylm.evaluation.evaluate

from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import torch

from tinylm.config import load_config
from tinylm.tokenizer import BPETokenizer
from tinylm.model import GPT, model_summary
from tinylm.training.dataset import build_dataloaders
from tinylm.evaluation.metrics import compute_perplexity, bits_per_character
from tinylm.evaluation.plot import plot_training_curves
from tinylm.inference.generate import generate_text
from tinylm.inference.sample import SAMPLING_GRID


EVAL_PROMPTS = [
    "The ancient city lay beneath the waves, its cyclopean",
    "I have looked upon all that the universe has to hold of horror,",
    "The most merciful thing in the world, I think, is the inability",
    "Ph'nglui mglw'nafh Cthulhu R'lyeh wgah'nagl",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained TinyLM model")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer",  default="checkpoints/tokenizer.json")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--log",        default="logs/train_log.csv")
    p.add_argument("--out-dir",    default="logs")
    p.add_argument("--max-eval-batches", type=int, default=100)
    p.add_argument("--max-tokens",       type=int, default=80)
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    cfg     = load_config(args.config)
    tok     = BPETokenizer.load(args.tokenizer)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt    = torch.load(args.checkpoint, map_location=device)
    model   = GPT(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    step     = ckpt.get("step", "?")
    val_loss_ckpt = ckpt.get("best_val_loss", float("nan"))

    print(f"\n{'═'*60}")
    print(f"  TinyLM (EMG-01) — Evaluation Suite")
    print(f"  Checkpoint: step {step} | val_loss {val_loss_ckpt:.4f}")
    print(f"{'═'*60}\n")

    model_summary(model, cfg.model)

    # ── 1. Perplexity ─────────────────────────────────────────────────
    print("\n[1/4] Computing validation perplexity ...")
    _, val_loader = build_dataloaders(cfg, tok)
    ppl, mean_loss = compute_perplexity(
        model, val_loader, device,
        max_batches=args.max_eval_batches, verbose=False,
    )
    print(f"  Val perplexity: {ppl:.2f}")
    print(f"  Mean loss:      {mean_loss:.4f}")

    # BPC on a standard passage
    passage = "The most merciful thing in the world, I think, is the inability of the human mind to correlate all its contents."
    bpc     = bits_per_character(model, tok, passage, device)
    print(f"  Bits/character: {bpc:.3f}")

    # ── 2. Sample generation ──────────────────────────────────────────
    print("\n[2/4] Generating samples ...")
    samples: dict[str, dict[str, str]] = {}
    for prompt in EVAL_PROMPTS:
        samples[prompt] = {}
        for scfg in SAMPLING_GRID[:3]:  # greedy, conservative, default
            out = generate_text(
                model, tok, prompt,
                max_new_tokens=args.max_tokens,
                temperature=scfg.temperature,
                top_k=scfg.top_k,
                top_p=scfg.top_p,
                seed=42,
            )
            samples[prompt][scfg.label] = out

    # ── 3. Training curves ────────────────────────────────────────────
    print("\n[3/4] Plotting training curves ...")
    curves_path = out_dir / "training_curves.png"
    try:
        plot_training_curves(
            log_path=args.log,
            out_path=str(curves_path),
        )
    except FileNotFoundError:
        print(f"  ⚠ Log file not found: {args.log} — skipping plots")
        curves_path = None

    # ── 4. Write markdown report ──────────────────────────────────────
    print("\n[4/4] Writing evaluation report ...")
    report_path = out_dir / "eval_report.md"

    lines = [
        "# TinyLM (EMG-01) — Evaluation Report\n",
        f"**Checkpoint step:** {step}  ",
        f"**Checkpoint val loss:** {val_loss_ckpt:.4f}  \n",
        "## Model Architecture\n",
        f"- Parameters: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M",
        f"- Layers: {cfg.model.n_layer}  |  Heads: {cfg.model.n_head}  "
        f"|  d_embd: {cfg.model.n_embd}",
        f"- Context length: {cfg.model.context_length}  "
        f"|  Vocab size: {cfg.model.vocab_size}\n",
        "## Evaluation Metrics\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Val Perplexity | {ppl:.2f} |",
        f"| Mean Token Loss | {mean_loss:.4f} |",
        f"| Bits per Character | {bpc:.3f} |",
        "",
    ]

    if curves_path and curves_path.exists():
        lines += [
            "## Training Curves\n",
            f"![Training Curves](training_curves.png)\n",
        ]

    lines += ["## Generated Samples\n"]
    for prompt, prompt_samples in samples.items():
        lines.append(f"### Prompt: *\"{prompt}...\"*\n")
        for label, text in prompt_samples.items():
            cfg_obj = next(c for c in SAMPLING_GRID if c.label == label)
            lines.append(
                f"**{label}** "
                f"(temp={cfg_obj.temperature}, "
                f"top_k={cfg_obj.top_k}, "
                f"top_p={cfg_obj.top_p})\n"
            )
            lines.append(f"> {text.strip()}\n")

    lines += [
        "---",
        f"*Generated by `python -m tinylm.evaluation.evaluate` "
        f"at step {step}*",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ Report saved: {report_path}")

    print(f"\n{'═'*60}")
    print(f"  Evaluation complete!")
    print(f"  Perplexity:  {ppl:.2f}")
    print(f"  BPC:         {bpc:.3f}")
    print(f"  Report:      {report_path}")
    if curves_path:
        print(f"  Curves:      {curves_path}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()