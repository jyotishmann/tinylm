# api/routes/ws.py
# WS /ws/generate — streaming token-by-token generation via WebSocket.
#
# Protocol (documented in MASTER.md §8.2):
#   Client  → Server: connect, then send WSGenerateRequest as JSON
#   Server  → Client: stream WSTokenFrame JSON messages (one per token)
#   Server  → Client: send WSDoneFrame JSON when complete
#   Server  → Client: send WSErrorFrame JSON on exception
#   Either side can close the connection at any time.

from __future__ import annotations
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from api.schemas import (
    WSGenerateRequest, WSTokenFrame, WSDoneFrame, WSErrorFrame,
)
from api.middleware import ModelNotLoadedError
from tinylm.inference.generate import stream_generate

router = APIRouter()


@router.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket) -> None:
    """
    WS /ws/generate

    WebSocket lifecycle:
        1. Client connects
        2. Server accepts
        3. Client sends WSGenerateRequest as JSON string
        4. Server validates, starts streaming
        5. Server sends WSTokenFrame for each generated token
        6. Server sends WSDoneFrame when max_new_tokens reached or EOS
        7. Connection remains open (client closes when ready)

    Disconnect handling:
        If client disconnects mid-stream (browser reload, stop button),
        WebSocketDisconnect is raised on the next send_json() call.
        We catch it and exit cleanly — no zombie generation on the server.
    """
    await websocket.accept()

    # Access app.state directly (Depends() is less clean for WebSocket)
    model     = getattr(websocket.app.state, "model",     None)
    tokenizer = getattr(websocket.app.state, "tokenizer", None)

    if model is None or tokenizer is None:
        await websocket.send_json(
            WSErrorFrame(message="Model not loaded", code=503).model_dump()
        )
        await websocket.close(code=1011)  # 1011 = internal error
        return

    t0       = time.perf_counter()
    n_tokens = 0

    try:
        # ── Receive and validate request ──────────────────────────────
        raw_data = await websocket.receive_json()
        try:
            request = WSGenerateRequest(**raw_data)
        except ValidationError as exc:
            await websocket.send_json(
                WSErrorFrame(
                    message = f"Invalid request: {exc.errors()[0]['msg']}",
                    code    = 422,
                ).model_dump()
            )
            return

        # ── Stream tokens ─────────────────────────────────────────────
        # stream_generate() is a synchronous generator.
        # Iterating it here blocks the event loop per token (~10-50ms each).
        # For production: asyncio.to_thread() or an async generator wrapper.
        for token_text, token_id, position in stream_generate(
            model          = model,
            tokenizer      = tokenizer,
            prompt         = request.prompt,
            max_new_tokens = request.max_new_tokens,
            temperature    = request.temperature,
            top_k          = request.top_k,
            top_p          = request.top_p,
        ):
            frame = WSTokenFrame(
                token    = token_text,
                token_id = token_id,
                position = position,
            )
            # This raises WebSocketDisconnect if the client has disconnected
            await websocket.send_json(frame.model_dump())
            n_tokens += 1

        # ── Send done frame ───────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        done_frame = WSDoneFrame(
            total_tokens       = n_tokens,
            generation_time_ms = round(elapsed_ms, 2),
        )
        await websocket.send_json(done_frame.model_dump())

    except WebSocketDisconnect:
        # Client closed the connection — clean exit, no error logging needed
        # The generation loop is interrupted naturally here
        pass

    except Exception as exc:
        # Unexpected exception — try to notify the client before closing
        try:
            error_frame = WSErrorFrame(message=str(exc), code=500)
            await websocket.send_json(error_frame.model_dump())
        except Exception:
            pass  # WebSocket may already be closed