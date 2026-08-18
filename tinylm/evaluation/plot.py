# tinylm/evaluation/plot.py
# Training curve visualisation from CSV log.
# Produces a 4-panel figure saved to logs/training_curves.png.

from __future__ import annotations
from pathlib import Path
from typing import Optional

import math
import numpy as np


def smooth(values: list[float], window: int = 10) -> list[float]:
    """
    Simple moving-average smoothing for noisy training curves.
    Pads with edge values to preserve list length.
    """
    if window <= 1 or len(values) < window:
        return values
    arr    = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").tolist()


def plot_training_curves(
    log_path:  str | Path = "logs/train_log.csv",
    out_path:  str | Path = "logs/training_curves.png",
    smoothing: int = 20,
) -> None:
    """
    Load training log CSV and generate a 4-panel visualisation.

    Panel layout:
        [Loss Curve]   [Perplexity]
        [LR Schedule]  [Gradient Norm]

    Args:
        log_path:  Path to CSV produced by Trainer (logs/train_log.csv)
        out_path:  Where to save the PNG
        smoothing: Moving-average window for noisy metrics (0 = disabled)

    Raises:
        FileNotFoundError: if log_path does not exist
        ImportError:       if matplotlib is not installed
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend for server/Colab environments
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. "
            "Install with: pip install matplotlib"
        )

    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(
            f"Training log not found: {log_path}\n"
            "Has training started? Check logs/ directory."
        )

    # ── Load CSV ──────────────────────────────────────────────────────
    data: dict[str, list] = {
        "step": [], "lr": [], "train_loss": [],
        "val_loss": [], "perplexity": [], "grad_norm": [],
    }
    with open(log_path) as f:
        header = next(f).strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < len(header):
                continue
            for key, val in zip(header, parts):
                if key in data:
                    data[key].append(float(val))

    if not data["step"]:
        raise ValueError(f"Log file is empty or has no data rows: {log_path}")

    steps  = data["step"]
    n_steps = steps[-1] if steps else 0

    # ── Extract val-only rows (train_loss is logged every log_interval,
    #    val_loss only every eval_interval) ─────────────────────────────
    val_steps  = [s for s, v in zip(steps, data["val_loss"]) if v > 0]
    val_losses = [v for v in data["val_loss"] if v > 0]
    val_ppls   = [math.exp(min(v, 20)) for v in val_losses]

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"TinyLM (EMG-01) — Training Curves  "
        f"[{int(n_steps):,} steps, best val_loss={min(val_losses or [0]):.4f}]",
        fontsize=13, fontweight="bold",
    )

    train_smooth = smooth(data["train_loss"], smoothing)

    # ── Panel 1: Loss ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(steps, data["train_loss"], alpha=0.25, color="#4C9BE8", label="train (raw)")
    ax.plot(steps, train_smooth,       color="#4C9BE8",  linewidth=2, label="train (smooth)")
    if val_steps:
        ax.plot(val_steps, val_losses, "o-", color="#E85454", linewidth=2,
                markersize=3, label="val")
    ax.set(xlabel="Step", ylabel="Cross-Entropy Loss", title="Loss Curve")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # ── Panel 2: Perplexity ────────────────────────────────────────────
    ax = axes[0, 1]
    if val_steps:
        ax.plot(val_steps, val_ppls, "o-", color="#E8A23A", linewidth=2, markersize=3)
        ax.axhline(y=val_ppls[-1], linestyle="--", color="grey", alpha=0.5,
                   label=f"final: {val_ppls[-1]:.1f}")
        ax.legend(fontsize=9)
    ax.set(xlabel="Step", ylabel="Perplexity", title="Validation Perplexity")
    ax.set_ylim(bottom=0)

    # ── Panel 3: Learning Rate ────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(steps, data["lr"], color="#6DBE6D", linewidth=1.5)
    ax.set(xlabel="Step", ylabel="Learning Rate", title="LR Schedule (warmup + cosine)")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # ── Panel 4: Gradient Norm ────────────────────────────────────────
    ax = axes[1, 1]
    gnorms       = data["grad_norm"]
    gnorm_smooth = smooth(gnorms, smoothing)
    clip_norm    = 1.0  # default — could be read from config
    ax.plot(steps, gnorms,       alpha=0.25, color="#9B59B6", label="grad norm (raw)")
    ax.plot(steps, gnorm_smooth, color="#9B59B6", linewidth=2, label="grad norm (smooth)")
    ax.axhline(y=clip_norm, linestyle="--", color="red", alpha=0.6,
               label=f"clip threshold ({clip_norm})")
    clipped_frac = sum(1 for g in gnorms if g > clip_norm) / max(len(gnorms), 1)
    ax.set_title(f"Gradient Norm  (clipped {clipped_frac*100:.1f}% of steps)")
    ax.set(xlabel="Step", ylabel="Global L2 Norm")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Training curves saved to {out_path}")
    print(f"  Steps logged:    {len(steps)}")
    print(f"  Val checkpoints: {len(val_steps)}")
    if val_losses:
        print(f"  Best val loss:   {min(val_losses):.4f}")
        print(f"  Final val PPL:   {val_ppls[-1]:.2f}")