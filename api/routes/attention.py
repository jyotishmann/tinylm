# api/routes/attention.py
# POST /api/attention — extract and return attention weights from a layer.

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.schemas import AttentionRequest, AttentionResponse
from api.dependencies import get_model, get_tokenizer
from tinylm.evaluation.attention import extract_attention_weights

router = APIRouter()


@router.post(
    "/attention",
    response_model = AttentionResponse,
    summary        = "Extract attention weights",
    description    = (
        "Runs a forward pass and returns attention weight matrices from "
        "the specified transformer layer. Used by the AttentionHeatmap "
        "component for visualisation. Returns [n_heads, T, T] weight arrays "
        "where weights[head][i][j] = attention from position i to position j."
    ),
)
async def attention(
    body:      AttentionRequest,
    model     = Depends(get_model),
    tokenizer = Depends(get_tokenizer),
) -> AttentionResponse:
    """
    POST /api/attention

    Slower than /api/tokenize (requires a forward pass through the model)
    but much faster than /api/generate (no autoregressive loop).
    Typical latency: 50–200ms depending on sequence length.
    """
    try:
        result = extract_attention_weights(
            model     = model,
            tokenizer = tokenizer,
            text      = body.text,
            layer     = body.layer,
        )
    except ValueError as exc:
        # Layer out of range, empty text, etc.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AttentionResponse(
        weights = result["weights"],
        tokens  = result["tokens"],
        layer   = result["layer"],
        n_heads = result["n_heads"],
    )