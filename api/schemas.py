# api/schemas.py
# All Pydantic request and response models for the TinyLM API.
# This file is the single source of truth for the API contract.
# Every field here corresponds to a field documented in MASTER.md §6.3 and §8.

from __future__ import annotations
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════
# POST /api/generate
# ══════════════════════════════════════════════════════════════════════

class GenerateRequest(BaseModel):
    """Text generation request.

    Constraints enforce sane generation parameters and prevent
    accidentally expensive or nonsensical inference calls.
    """
    prompt:         str            = Field(...,   min_length=1, max_length=1000,
                                          description="Input text to complete")
    max_new_tokens: int            = Field(200,   ge=1,  le=500,
                                          description="Maximum tokens to generate")
    temperature:    float          = Field(0.8,   ge=0.01, le=2.0,
                                          description="Sampling temperature (0=greedy)")
    top_k:          int            = Field(50,    ge=0,  le=1000,
                                          description="Top-k filtering (0=disabled)")
    top_p:          float          = Field(0.9,   ge=0.0, le=1.0,
                                          description="Nucleus sampling threshold")
    seed:           Optional[int]  = Field(None,
                                          description="Random seed for reproducibility")

    model_config = {"json_schema_extra": {
        "example": {
            "prompt":         "The ancient ruins lay beneath the Pacific",
            "max_new_tokens": 150,
            "temperature":    0.8,
            "top_k":          50,
            "top_p":          0.9,
            "seed":           None,
        }
    }}


class GenerateResponse(BaseModel):
    """Text generation response."""
    generated_text:    str   = Field(..., description="Generated text (prompt excluded)")
    prompt_tokens:     int   = Field(..., description="Number of tokens in the prompt")
    tokens_generated:  int   = Field(..., description="Number of new tokens generated")
    generation_time_ms: float = Field(..., description="Wall-clock generation time")
    tokens_per_second: float  = Field(..., description="Generation throughput")


# ══════════════════════════════════════════════════════════════════════
# GET /api/model/info
# ══════════════════════════════════════════════════════════════════════

class ModelInfoResponse(BaseModel):
    """Model architecture metadata."""
    n_params:       int  = Field(..., description="Total parameter count")
    n_layers:       int  = Field(..., description="Number of transformer blocks")
    n_heads:        int  = Field(..., description="Attention heads per block")
    n_embd:         int  = Field(..., description="Embedding dimension")
    d_head:         int  = Field(..., description="Per-head dimension (n_embd/n_heads)")
    vocab_size:     int  = Field(..., description="Vocabulary size")
    context_length: int  = Field(..., description="Maximum sequence length")
    architecture:   str  = Field(..., description="Architecture family")
    checkpoint_step: Optional[int]   = Field(None, description="Training step of loaded checkpoint")
    best_val_loss:   Optional[float] = Field(None, description="Best validation loss at checkpoint")


# ══════════════════════════════════════════════════════════════════════
# POST /api/tokenize
# ══════════════════════════════════════════════════════════════════════

class TokenizeRequest(BaseModel):
    """Tokenisation request."""
    text: str = Field(..., min_length=1, max_length=5000,
                     description="Text to tokenize")


class TokenInfo(BaseModel):
    """A single token with its ID and string representation."""
    id:   int = Field(..., description="Token integer ID")
    text: str = Field(..., description="Token string (may include </w> marker)")


class TokenizeResponse(BaseModel):
    """Tokenisation response."""
    tokens: list[TokenInfo] = Field(..., description="List of token info objects")
    count:  int             = Field(..., description="Total token count")


# ══════════════════════════════════════════════════════════════════════
# POST /api/attention
# ══════════════════════════════════════════════════════════════════════

class AttentionRequest(BaseModel):
    """Attention weight extraction request."""
    text:  str = Field(..., min_length=1, max_length=512,
                      description="Text to extract attention from")
    layer: int = Field(0,  ge=0,
                      description="Transformer block index (0-indexed)")


class AttentionResponse(BaseModel):
    """
    Attention weight response.

    weights: [n_heads][T][T] — attention[head][query_pos][key_pos]
    tokens:  [T] — human-readable token strings for heatmap axis labels
    """
    weights: list[list[list[float]]] = Field(
        ..., description="Attention weights [n_heads, T, T]"
    )
    tokens:  list[str] = Field(..., description="Token display strings")
    layer:   int       = Field(..., description="Layer index extracted from")
    n_heads: int       = Field(..., description="Number of attention heads")


# ══════════════════════════════════════════════════════════════════════
# WS /ws/generate — WebSocket frame types
# ══════════════════════════════════════════════════════════════════════

class WSGenerateRequest(BaseModel):
    """
    WebSocket generation request — sent as first JSON message after connect.
    """
    prompt:         str   = Field(...,  min_length=1, max_length=1000)
    max_new_tokens: int   = Field(200,  ge=1, le=500)
    temperature:    float = Field(0.8,  ge=0.01, le=2.0)
    top_k:          int   = Field(50,   ge=0, le=1000)
    top_p:          float = Field(0.9,  ge=0.0, le=1.0)


class WSTokenFrame(BaseModel):
    """
    WebSocket token frame — one per generated token.
    Frontend appends `token` to the streaming display buffer.
    """
    type:      Literal["token"] = "token"
    token:     str  = Field(..., description="Decoded token text (</w> → space)")
    token_id:  int  = Field(..., description="Integer token ID")
    position:  int  = Field(..., description="0-indexed position in generated sequence")


class WSDoneFrame(BaseModel):
    """
    WebSocket done frame — sent once after all tokens are generated.
    Frontend uses this to hide the loading indicator.
    """
    type:               Literal["done"] = "done"
    total_tokens:       int   = Field(..., description="Total tokens generated")
    generation_time_ms: float = Field(..., description="Total generation time")


class WSErrorFrame(BaseModel):
    """
    WebSocket error frame — sent if an exception occurs mid-stream.
    Frontend shows this as an error toast.
    """
    type:    Literal["error"] = "error"
    message: str = Field(..., description="Error description")
    code:    int = Field(..., description="HTTP-equivalent error code")