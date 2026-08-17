# tinylm/config.py
# Typed configuration system — load from YAML, access via dot notation.
# Usage: cfg = load_config("configs/default.yaml")
#        print(cfg.model.n_embd)  # 384

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    """Architecture hyperparameters. See MASTER.md §4.9 for parameter budget."""
    vocab_size:     int   = 5000
    context_length: int   = 256
    n_layer:        int   = 6
    n_head:         int   = 6
    n_embd:         int   = 384
    dropout:        float = 0.1
    bias:           bool  = False

    def __post_init__(self):
        # Validate that n_embd is divisible by n_head at construction time.
        # Catching this here prevents a cryptic reshape error deep inside the model.
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}). "
            f"Got d_head = {self.n_embd / self.n_head:.1f} (non-integer)."
        )

    @property
    def d_head(self) -> int:
        """Dimension of each attention head's Q/K/V vectors."""
        return self.n_embd // self.n_head

    @property
    def n_params_estimate(self) -> int:
        """Rough parameter count estimate (excludes biases, LN params)."""
        # Embeddings
        tok_emb = self.vocab_size * self.n_embd
        pos_emb = self.context_length * self.n_embd
        # Per-layer: attention (4 matrices) + FFN (2 matrices)
        attn    = 4 * self.n_embd * self.n_embd * self.n_layer
        ffn     = 2 * self.n_embd * 4 * self.n_embd * self.n_layer
        # LM head is weight-tied — no extra params
        return tok_emb + pos_emb + attn + ffn


@dataclass
class TrainConfig:
    """Training pipeline configuration. See MASTER.md §5 for full rationale."""
    # Data
    data_path:           str   = "data/processed/corpus.bin"
    train_split:         float = 0.9

    # Batching
    batch_size:          int   = 64
    grad_accum_steps:    int   = 8

    # Schedule
    max_steps:           int   = 5000
    warmup_steps:        int   = 100

    # Optimiser
    learning_rate:       float = 3e-4
    min_lr:              float = 3e-5
    weight_decay:        float = 0.1
    beta1:               float = 0.9
    beta2:               float = 0.95
    grad_clip:           float = 1.0

    # Evaluation & checkpointing
    eval_interval:       int   = 250
    eval_steps:          int   = 50
    checkpoint_interval: int   = 500
    checkpoint_dir:      str   = "checkpoints"

    # Hardware
    device:              str   = "cuda"
    dtype:               str   = "bfloat16"

    # Logging
    log_interval:        int   = 10
    log_dir:             str   = "logs"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps


@dataclass
class Config:
    """Root config — holds model and training sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _overlay(dataclass_obj, updates: dict) -> None:
    """Apply a flat dict of updates onto a dataclass instance in-place."""
    for key, value in updates.items():
        if not hasattr(dataclass_obj, key):
            raise ValueError(
                f"Unknown config key '{key}' for {type(dataclass_obj).__name__}. "
                f"Valid keys: {list(asdict(dataclass_obj).keys())}"
            )
        setattr(dataclass_obj, key, value)


def load_config(*yaml_paths: str | Path) -> Config:
    """
    Load config from one or more YAML files, applied left-to-right.
    Later files override earlier ones (useful for colab_overrides.yaml).

    Example:
        cfg = load_config("configs/default.yaml", "configs/colab.yaml")
    """
    cfg = Config()
    for path in yaml_paths:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if "model" in raw:
            _overlay(cfg.model, raw["model"])
        if "train" in raw:
            _overlay(cfg.train, raw["train"])
    return cfg