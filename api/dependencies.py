# api/dependencies.py
# FastAPI dependency functions for model, tokenizer, and device.
# Route handlers declare these via Depends() — they never access app.state directly.

from __future__ import annotations
import torch
from fastapi import Request

from tinylm.model.gpt import GPT
from tinylm.tokenizer import BPETokenizer
from tinylm.config import Config

from api.middleware import ModelNotLoadedError


def get_model(request: Request) -> GPT:
    """
    Dependency: retrieve the loaded GPT model from app.state.

    Raises ModelNotLoadedError (→ 503) if the model hasn't loaded yet.
    This can happen if a request arrives during the startup window.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise ModelNotLoadedError(
            "Model is not loaded yet. "
            "The server may still be initialising — retry in a few seconds."
        )
    return model


def get_tokenizer(request: Request) -> BPETokenizer:
    """Dependency: retrieve the tokenizer from app.state."""
    tokenizer = getattr(request.app.state, "tokenizer", None)
    if tokenizer is None:
        raise ModelNotLoadedError("Tokenizer is not loaded yet.")
    return tokenizer


def get_device(request: Request) -> torch.device:
    """Dependency: retrieve the device (cuda/cpu/mps) from app.state."""
    return getattr(request.app.state, "device", torch.device("cpu"))


def get_cfg(request: Request) -> Config:
    """Dependency: retrieve the Config from app.state."""
    return request.app.state.cfg


def get_checkpoint_meta(request: Request) -> dict:
    """
    Dependency: retrieve checkpoint metadata (step, val_loss) if stored.
    Used by the model/info endpoint to include training provenance.
    """
    return getattr(request.app.state, "checkpoint_meta", {})