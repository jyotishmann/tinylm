# tinylm/training/dataset.py
# Dataset and DataLoader construction for language model training.
#
# Two responsibilities:
#   1. tokenize_and_save(): one-time preprocessing (corpus.txt → corpus.bin)
#   2. LanguageModelDataset + build_dataloaders(): training-time data access

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tinylm.config import Config
from tinylm.tokenizer import BPETokenizer


class LanguageModelDataset(Dataset):
    """
    Sliding-window dataset for next-token prediction.

    Reads a flat 1D array of token IDs. Each sample is a pair:
        x = data[i     : i + context_length]   (input)
        y = data[i + 1 : i + context_length + 1]  (targets, shifted by 1)

    Args:
        data:           1D LongTensor of token IDs
        context_length: Number of tokens per sample
    """

    def __init__(self, data: torch.Tensor, context_length: int) -> None:
        assert data.dtype == torch.long, f"Expected long, got {data.dtype}"
        self.data = data
        self.ctx  = context_length

    def __len__(self) -> int:
        # Number of complete context_length windows in the data
        # -1 because we need one extra token for the target at the last position
        return len(self.data) - self.ctx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x: positions idx .. idx+ctx-1   (B, T) input
        # y: positions idx+1 .. idx+ctx   (B, T) target — shifted right by 1
        x = self.data[idx     : idx + self.ctx]
        y = self.data[idx + 1 : idx + self.ctx + 1]
        return x, y


def tokenize_and_save(
    corpus_path:    str | Path,
    output_dir:     str | Path,
    tokenizer:      BPETokenizer,
    train_split:    float = 0.9,
    force:          bool  = False,
) -> tuple[Path, Path]:
    """
    Tokenize the corpus once and save train/val splits as binary files.

    Output files:
        output_dir/train.bin  — numpy int32 array of training token IDs
        output_dir/val.bin    — numpy int32 array of validation token IDs

    Args:
        corpus_path: Path to data/processed/corpus.txt
        output_dir:  Directory to write .bin files (data/processed/)
        tokenizer:   Trained BPETokenizer
        train_split: Fraction of corpus for training (default 0.9)
        force:       Re-tokenize even if .bin files already exist

    Returns:
        (train_bin_path, val_bin_path)
    """
    corpus_path = Path(corpus_path)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.bin"
    val_path   = output_dir / "val.bin"

    if train_path.exists() and val_path.exists() and not force:
        print(f"✓ Binary token files already exist — skipping tokenization")
        print(f"  (pass force=True to re-tokenize)")
        return train_path, val_path

    print(f"Tokenizing corpus: {corpus_path} ...")
    corpus = corpus_path.read_text(encoding="utf-8")

    # Tokenize the entire corpus (add EOS so model learns to stop)
    token_ids = tokenizer.encode(corpus, add_eos=False)
    print(f"  Total tokens: {len(token_ids):,}")
    print(f"  Vocabulary coverage: {len(set(token_ids)):,} unique IDs used")

    # Split: first 90% train, last 10% val
    # NOT shuffled — temporal order is preserved
    split_idx   = int(len(token_ids) * train_split)
    train_ids   = np.array(token_ids[:split_idx],   dtype=np.int32)
    val_ids     = np.array(token_ids[split_idx:],   dtype=np.int32)

    # Save as raw binary arrays (fast to load, no pickle)
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"  Train: {len(train_ids):>10,} tokens → {train_path}")
    print(f"  Val:   {len(val_ids):>10,} tokens → {val_path}")
    print(f"  Files: {train_path.stat().st_size/1024:.0f}KB, "
          f"{val_path.stat().st_size/1024:.0f}KB")

    return train_path, val_path


def load_token_ids(bin_path: str | Path) -> torch.Tensor:
    """Load a .bin token file into a LongTensor."""
    arr = np.fromfile(bin_path, dtype=np.int32)
    return torch.from_numpy(arr.astype(np.int64))


def build_dataloaders(
    cfg:         Config,
    tokenizer:   BPETokenizer,
    corpus_path: str | Path = "data/processed/corpus.txt",
    data_dir:    str | Path = "data/processed",
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Tokenizes corpus if .bin files don't exist, then wraps in DataLoaders
    configured for efficient GPU training.

    Returns:
        (train_loader, val_loader)
    """
    # Tokenize once and save (idempotent)
    train_bin, val_bin = tokenize_and_save(
        corpus_path  = corpus_path,
        output_dir   = data_dir,
        tokenizer    = tokenizer,
        train_split  = cfg.train.train_split,
    )

    # Load token ID tensors
    train_ids = load_token_ids(train_bin)
    val_ids   = load_token_ids(val_bin)

    ctx = cfg.model.context_length

    # Wrap in Dataset
    train_dataset = LanguageModelDataset(train_ids, ctx)
    val_dataset   = LanguageModelDataset(val_ids,   ctx)

    # Detect if CUDA is available for pin_memory
    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.train.batch_size,
        shuffle     = True,           # Shuffle for training (not evaluation)
        num_workers = 2 if use_cuda else 0,  # Parallel loading on GPU machine
        pin_memory  = use_cuda,       # Page-locked memory → faster GPU transfer
        drop_last   = True,           # Consistent batch size (avoids edge-case shapes)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.train.batch_size,
        shuffle     = False,          # Val always in order (reproducible)
        num_workers = 2 if use_cuda else 0,
        pin_memory  = use_cuda,
        drop_last   = False,
    )

    print(f"  Train dataset: {len(train_dataset):,} samples")
    print(f"  Val dataset:   {len(val_dataset):,} samples")
    print(f"  Batch size:    {cfg.train.batch_size}")
    print(f"  Steps/epoch:   {len(train_loader):,}")

    return train_loader, val_loader