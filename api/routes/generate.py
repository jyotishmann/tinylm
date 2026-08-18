# api/routes/generate.py
# POST /api/generate — synchronous text generation endpoint.

from __future__ import annotations
import time

from fastapi import APIRouter, Depends

from api.schemas import GenerateRequest, GenerateResponse
from api.dependencies import get_model, get_tokenizer
from api.middleware import GenerationError
from tinylm.inference.generate import generate_text

router = APIRouter()


@router.post(
    "/generate",
    response_model = GenerateResponse,
    summary        = "Generate text from a prompt",
    description    = (
        "Runs autoregressive text generation using the trained GPT model. "
        "Returns the full generated text (prompt excluded) with timing stats. "
        "For streaming token-by-token generation, use the WebSocket endpoint."
    ),
)
async def generate(
    body:      GenerateRequest,
    model     = Depends(get_model),
    tokenizer = Depends(get_tokenizer),
) -> GenerateResponse:
    """
    POST /api/generate

    Production note: PyTorch inference is synchronous and CPU/GPU-bound.
    For >1 concurrent user, move inference to a thread pool:
        import asyncio
        text = await asyncio.get_event_loop().run_in_executor(
            None, generate_text, model, tokenizer, body.prompt, ...
        )
    For this demo (single user), async def + synchronous call is acceptable.
    """
    t0 = time.perf_counter()

    # Count prompt tokens for the response metadata
    prompt_ids     = tokenizer.encode(body.prompt)
    n_prompt_toks  = len(prompt_ids)

    try:
        generated = generate_text(
            model          = model,
            tokenizer      = tokenizer,
            prompt         = body.prompt,
            max_new_tokens = body.max_new_tokens,
            temperature    = body.temperature,
            top_k          = body.top_k,
            top_p          = body.top_p,
            seed           = body.seed,
        )
    except Exception as exc:
        raise GenerationError(f"Generation failed: {exc}") from exc

    elapsed_ms   = (time.perf_counter() - t0) * 1000
    n_gen_tokens = len(tokenizer.encode(generated))
    tok_per_sec  = n_gen_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0

    return GenerateResponse(
        generated_text     = generated,
        prompt_tokens      = n_prompt_toks,
        tokens_generated   = n_gen_tokens,
        generation_time_ms = round(elapsed_ms, 2),
        tokens_per_second  = round(tok_per_sec, 1),
    )