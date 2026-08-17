# tinylm/training/train.py
# Training entrypoint: python -m tinylm.training.train
# Full pipeline: load config → tokenizer → data → model → train

from __future__ import annotations
import argparse
from pathlib import Path

import torch

from tinylm.config import load_config
from tinylm.tokenizer import BPETokenizer
from tinylm.model import GPT, model_summary
from tinylm.training.dataset import build_dataloaders
from tinylm.training.optimizer import build_optimizer
from tinylm.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TinyLM from scratch")
    p.add_argument("--config",   default="configs/default.yaml",
                   help="YAML config path")
    p.add_argument("--tokenizer", default="checkpoints/tokenizer.json",
                   help="Path to trained tokenizer")
    p.add_argument("--resume",   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--device",   default=None,
                   help="Override device (cuda/cpu/mps)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Config ───────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("⚠ No GPU detected — training on CPU will be very slow")
        print("  Use Google Colab (Runtime → T4 GPU) for full training")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Tokenizer ────────────────────────────────────────────────────
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tok_path}\n"
            "Run: python -m tinylm.tokenizer.train"
        )
    tok = BPETokenizer.load(tok_path)
    print(f"Tokenizer: {tok.vocab_size:,} tokens loaded from {tok_path}")

    # Sync vocab_size from tokenizer → config (avoids mismatch)
    cfg.model.vocab_size = tok.vocab_size

    # ── DataLoaders ──────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg, tok)

    # ── Model ────────────────────────────────────────────────────────
    model = GPT(cfg.model).to(device)
    model_summary(model, cfg.model)

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = Trainer(model, optimizer, train_loader, val_loader, cfg)

    # ── Resume ───────────────────────────────────────────────────────
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Train ─────────────────────────────────────────────────────────
    trainer.train()


if __name__ == "__main__":
    main()