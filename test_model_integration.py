# Full integration test — run this before tagging v0.3.0
# Verifies every component path from token IDs to loss

import math
import torch
from tinylm.config import load_config
from tinylm.model import GPT, model_summary, count_parameters
from tinylm.tokenizer import BPETokenizer

cfg = load_config("configs/default.yaml")
tok = BPETokenizer.load("checkpoints/tokenizer.json")

print("=" * 60)
print("  CELLS-03 Integration Test")
print("=" * 60)

# ── 1. Instantiation ─────────────────────────────────────────────────
model = GPT(cfg.model)
model.eval()
model_summary(model, cfg.model)

# ── 2. Forward pass with loss ────────────────────────────────────────
B, T = 8, 128
idx     = torch.randint(4, cfg.model.vocab_size, (B, T))  # avoid special tokens
targets = torch.randint(4, cfg.model.vocab_size, (B, T))

with torch.no_grad():
    logits, loss = model(idx, targets)

assert logits.shape == (B, T, cfg.model.vocab_size)
print(f"\n✓ Forward pass: logits {logits.shape}")

expected_loss = math.log(cfg.model.vocab_size)
actual_loss   = loss.item()
deviation     = abs(actual_loss - expected_loss) / expected_loss
print(f"✓ Loss at init: {actual_loss:.4f}  (expected ≈ {expected_loss:.4f}, "
      f"deviation {deviation*100:.1f}%)")
assert deviation < 0.20, (  # allow 20% deviation
    f"Loss too far from expected. Init went wrong. "
    f"Got {actual_loss:.3f}, expected {expected_loss:.3f}"
)

# ── 3. Real tokenizer roundtrip ──────────────────────────────────────
prompt    = "The ancient city of R'lyeh lay sunken beneath the Pacific"
token_ids = tok.encode(prompt, add_bos=True)
idx_real  = torch.tensor([token_ids])  # (1, T_prompt)

with torch.no_grad():
    logits_real, _ = model(idx_real)

T_prompt = len(token_ids)
assert logits_real.shape == (1, T_prompt, cfg.model.vocab_size)
print(f"✓ Real text forward: '{prompt[:40]}...' → logits {logits_real.shape}")

# ── 4. Generation ────────────────────────────────────────────────────
with torch.no_grad():
    generated = model.generate(idx_real, max_new_tokens=30, temperature=0.8)

generated_text = tok.decode(generated[0].tolist())
print(f"✓ Generation (untrained): '{generated_text[:80]}...'")

# ── 5. Attention weights ─────────────────────────────────────────────
with torch.no_grad():
    _, _, attn_w = model(idx_real, return_attn_layer=0)
assert attn_w.shape == (1, cfg.model.n_head, T_prompt, T_prompt)
print(f"✓ Attention weights (layer 0): {attn_w.shape}")

# ── 6. Weight tying ──────────────────────────────────────────────────
assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
print("✓ Weight tying: lm_head.weight IS wte.weight")

# ── 7. No gradient in no_grad context ──────────────────────────────
with torch.no_grad():
    out, _ = model(idx[:1, :16])
assert not out.requires_grad, "no_grad context should disable gradient tracking"
print("✓ No gradient in no_grad context")

print("\n" + "=" * 60)
print("  All integration tests passed ✓")
print("  Model is ready for training (CELLS-04)")
print("=" * 60)