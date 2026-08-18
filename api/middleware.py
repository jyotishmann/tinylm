# api/middleware.py
# CORS configuration, custom exception handlers, and health endpoint.
# Imported by create_app() in main.py.

from __future__ import annotations
import os
from typing import Callable

from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ── CORS ──────────────────────────────────────────────────────────────

# Read allowed origins from environment (comma-separated)
# Default: local Vite dev server
_FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
ALLOWED_ORIGINS = [
    url.strip()
    for url in _FRONTEND_URL.split(",")
    if url.strip()
]
# Always include both localhost variants
for _local in ("http://localhost:5173", "http://127.0.0.1:5173"):
    if _local not in ALLOWED_ORIGINS:
        ALLOWED_ORIGINS.append(_local)


def add_cors(app: FastAPI) -> None:
    """
    Add CORSMiddleware to allow cross-origin requests from the React frontend.

    allow_credentials=False: we have no auth cookies (simplifies CORS).
    allow_methods=["GET", "POST"]: our API uses only these two HTTP methods.
    allow_headers=["Content-Type"]: the only header our requests use.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ALLOWED_ORIGINS,
        allow_credentials = False,
        allow_methods     = ["GET", "POST"],
        allow_headers     = ["Content-Type"],
    )


# ── Custom exceptions ──────────────────────────────────────────────────

class ModelNotLoadedError(Exception):
    """Raised when a route handler is called before the model is in app.state."""


class GenerationError(Exception):
    """Raised when text generation fails unexpectedly."""


# ── Error handlers ─────────────────────────────────────────────────────

def add_error_handlers(app: FastAPI) -> None:
    """
    Register custom exception handlers.

    All error responses follow a consistent schema:
        {"detail": {"code": "ERROR_CODE", "message": "Human-readable message"}}

    This matches what the frontend's Axios interceptor expects (CELLS-07).
    """

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
        return JSONResponse(
            status_code = 503,
            content     = {
                "detail": {
                    "code":    "MODEL_NOT_LOADED",
                    "message": str(exc) or "Model is not loaded. Try again in a moment.",
                }
            },
        )

    @app.exception_handler(GenerationError)
    async def generation_error_handler(request: Request, exc: GenerationError):
        return JSONResponse(
            status_code = 500,
            content     = {
                "detail": {
                    "code":    "GENERATION_ERROR",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code = 422,
            content     = {
                "detail": {
                    "code":    "INVALID_INPUT",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        # Catch-all: don't leak internal details to client
        return JSONResponse(
            status_code = 500,
            content     = {
                "detail": {
                    "code":    "INTERNAL_ERROR",
                    "message": "An unexpected error occurred.",
                }
            },
        )


# ── Health endpoint ────────────────────────────────────────────────────

health_router = APIRouter()


@health_router.get("/health", tags=["health"])
async def health_check(request: Request):
    """
    Health check endpoint for Docker/k8s liveness probes.

    Returns 200 if the server is running, with model_loaded flag
    so you can distinguish 'server up but model still loading' from
    'server up and ready to serve requests'.
    """
    model_loaded = (
        hasattr(request.app.state, "model")
        and request.app.state.model is not None
    )
    return {
        "status":       "ok",
        "model_loaded": model_loaded,
        "version":      "1.0.0",
    }