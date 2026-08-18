# api/routes/model.py
# GET /api/model/info — model architecture metadata endpoint.

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.schemas import ModelInfoResponse
from api.dependencies import get_model
from tinylm.model.utils import count_parameters

router = APIRouter()


@router.get(
    "/model/info",
    response_model = ModelInfoResponse,
    summary        = "Get model architecture metadata",
    description    = (
        "Returns static information about the loaded model: parameter count, "
        "layer/head/embedding dimensions, vocabulary size, and training provenance. "
        "Called once on frontend load by useModelInfo() hook."
    ),
)
async def model_info(
    request: Request,
    model   = Depends(get_model),
) -> ModelInfoResponse:
    """
    GET /api/model/info

    No request body — pure read of app.state.
    Always fast (<1ms) — just reads already-computed values.
    """
    cfg    = model.cfg
    counts = count_parameters(model)

    # Checkpoint metadata stored at startup (step, val_loss)
    ckpt_meta = getattr(request.app.state, "checkpoint_meta", {})

    return ModelInfoResponse(
        n_params        = counts["total"],
        n_layers        = cfg.n_layer,
        n_heads         = cfg.n_head,
        n_embd          = cfg.n_embd,
        d_head          = cfg.d_head,
        vocab_size      = cfg.vocab_size,
        context_length  = cfg.context_length,
        architecture    = "GPT-Decoder",
        checkpoint_step = ckpt_meta.get("step"),
        best_val_loss   = ckpt_meta.get("best_val_loss"),
    )