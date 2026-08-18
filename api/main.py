# api/main.py
# FastAPI application factory.
# Model and tokenizer are loaded once here and shared via app.state.
#
# Start with: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()  # reads .env if present


# ── Configuration from environment ────────────────────────────────────
# These paths can be overridden via .env or environment variables
MODEL_CHECKPOINT = os.getenv("MODEL_PATH",      "checkpoints/best_model.pt")
TOKENIZER_PATH   = os.getenv("TOKENIZER_PATH",  "checkpoints/tokenizer.json")
CONFIG_PATH      = os.getenv("CONFIG_PATH",      "configs/default.yaml")


def _load_resources() -> tuple:
    """
    Load model and tokenizer from disk.
    Called once at startup — NOT called per-request.

    Returns:
        (model, tokenizer, device, cfg)
    """
    from tinylm.config import load_config
    from tinylm.tokenizer import BPETokenizer
    from tinylm.model import GPT

    cfg_path  = Path(CONFIG_PATH)
    tok_path  = Path(TOKENIZER_PATH)
    ckpt_path = Path(MODEL_CHECKPOINT)

    for path, name in [(cfg_path, "Config"), (tok_path, "Tokenizer"),
                       (ckpt_path, "Checkpoint")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found: {path}\n"
                f"Set the correct path via environment variable or .env"
            )

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[startup] Device: {device}")

    # Load tokenizer (~0.2s)
    print(f"[startup] Loading tokenizer from {tok_path} ...")
    tokenizer = BPETokenizer.load(tok_path)
    print(f"[startup] Tokenizer ready: {tokenizer.vocab_size:,} tokens")

    # Load config and sync vocab_size
    cfg = load_config(cfg_path)
    cfg.model.vocab_size = tokenizer.vocab_size

    # Load model weights (~2-5s)
    print(f"[startup] Loading model from {ckpt_path} ...")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = GPT(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    step     = ckpt.get("step",          "unknown")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[startup] Model ready: {n_params:.1f}M params | "
          f"step {step} | val_loss {val_loss:.4f}")

    return model, tokenizer, device, cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context — runs startup code before yield,
    shutdown code after yield.

    app.state stores references shared across all requests:
        app.state.model      — GPT instance (eval mode)
        app.state.tokenizer  — BPETokenizer instance
        app.state.device     — torch.device
        app.state.cfg        — Config instance
    """
    # ── STARTUP ───────────────────────────────────────────────────────
    model, tokenizer, device, cfg = _load_resources()
    app.state.model     = model
    app.state.tokenizer = tokenizer
    app.state.device    = device
    app.state.cfg       = cfg
    print("[startup] API ready ✓")

    yield  # ← server handles requests here

    # ── SHUTDOWN ──────────────────────────────────────────────────────
    print("[shutdown] Releasing model ...")
    del app.state.model
    del app.state.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[shutdown] Done.")


def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI app.

    Separate from module-level instantiation so tests can call
    create_app() with different configurations.
    """
    from api.middleware import add_cors, add_error_handlers, health_router

    app = FastAPI(
        title       = "TinyLM API",
        description = (
            "GPT-style transformer trained from scratch on H.P. Lovecraft. "
            "Every attention head, every BPE merge, every gradient update — "
            "built by hand. No HuggingFace. No API wrappers."
        ),
        version     = "1.0.0",
        lifespan    = lifespan,
    )

    # Middleware (CORS, error handlers)
    add_cors(app)
    add_error_handlers(app)

    # Health check router (no prefix — /health not /api/health)
    app.include_router(health_router)

    # API routers (imported lazily to avoid circular imports)
    from api.routes.generate  import router as gen_router
    from api.routes.model     import router as model_router
    from api.routes.tokenize  import router as tok_router
    from api.routes.attention import router as attn_router
    from api.routes.ws        import router as ws_router

    app.include_router(gen_router,   prefix="/api", tags=["generation"])
    app.include_router(model_router, prefix="/api", tags=["model"])
    app.include_router(tok_router,   prefix="/api", tags=["tokenize"])
    app.include_router(attn_router,  prefix="/api", tags=["attention"])
    app.include_router(ws_router,             tags=["streaming"])

    return app


# Module-level app instance — what uvicorn imports
app = create_app()