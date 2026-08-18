# api/routes/tokenize.py
# POST /api/tokenize — tokenise text and return token IDs + strings.

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas import TokenizeRequest, TokenizeResponse, TokenInfo
from api.dependencies import get_tokenizer
from tinylm.tokenizer.vocab import END_OF_WORD

router = APIRouter()


@router.post(
    "/tokenize",
    response_model = TokenizeResponse,
    summary        = "Tokenise text",
    description    = (
        "Converts input text to BPE token IDs and their string representations. "
        "Used by the TokenViewer component with 300ms debounce. "
        "The 'text' field of each token includes the </w> end-of-word marker "
        "so the frontend can reconstruct word boundaries."
    ),
)
async def tokenize(
    body:      TokenizeRequest,
    tokenizer = Depends(get_tokenizer),
) -> TokenizeResponse:
    """
    POST /api/tokenize

    Note: encode() without add_bos=True so BOS doesn't appear in the
    token visualisation (it's a structural token, not part of the text).
    """
    ids    = tokenizer.encode(body.text, add_bos=False)

    tokens = [
        TokenInfo(
            id   = token_id,
            # Return the raw token string including </w>
            # Frontend can display it as-is or strip </w> for cleaner labels
            text = tokenizer.id_to_token.get(token_id, "<UNK>"),
        )
        for token_id in ids
    ]

    return TokenizeResponse(tokens=tokens, count=len(tokens))