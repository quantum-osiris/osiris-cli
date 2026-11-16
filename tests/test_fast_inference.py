"""
Tests for OSIRIS v5.0 C-SIMD and Fast KV-Cached Inference Engine
"""

import numpy as np
import pytest

from osiris.nclm.autograd import no_grad
from osiris.nclm.fast_inference import FastKVCache, HAVE_C_SIMD, decode_step_kv, fast_generate, prefill_kv
from osiris.nclm.transformer import SovereignConfig, SovereignTransformerV2


@pytest.mark.parametrize(
    "cfg_kwargs",
    [
        dict(dim=128, n_layers=2, n_heads=2, max_seq_len=64, pilot_wave=False, golden_scale=False, positional="standard"),
        dict(dim=128, n_layers=2, n_heads=2, max_seq_len=64),
        dict(dim=128, n_layers=2, n_heads=2, max_seq_len=64, torsion_lock=True, phase_conjugate=True, positional="phi"),
        dict(dim=64, n_layers=2, n_heads=2, max_seq_len=64, torsion_lock=True, phase_conjugate=True, fractal_embedding=True),
    ],
)
def test_fast_inference_numerical_equivalence_across_architectures(cfg_kwargs):
    np.random.seed(0)
    config = SovereignConfig(**cfg_kwargs)
    model = SovereignTransformerV2(config)
    model.eval()

    prompt = "User: hello OSIRIS\nOSIRIS:"
    prompt_ids = list(prompt.encode("utf-8"))

    with no_grad():
        ref_logits = model.forward(np.array([prompt_ids], dtype=np.int64)).data[0, -1, :]

    kv = FastKVCache(len(model.blocks), config.max_seq_len, config.n_heads, config.dim // config.n_heads)
    prefill_logits = prefill_kv(model, prompt_ids, kv)
    assert np.max(np.abs(ref_logits - prefill_logits)) < 1e-4

    next_tok = int(np.argmax(ref_logits))
    with no_grad():
        ref_step1 = model.forward(np.array([prompt_ids + [next_tok]], dtype=np.int64)).data[0, -1, :]
    step1_logits = decode_step_kv(model, next_tok, len(prompt_ids), kv)
    assert np.max(np.abs(ref_step1 - step1_logits)) < 1e-4


def test_fast_generate_matches_forward_across_sliding_context_window():
    np.random.seed(7)
    # CRSM architecture with small max_seq_len=32 so prompt (40 bytes) + 20 new bytes crosses window boundary
    config = SovereignConfig(dim=64, n_layers=2, n_heads=2, ff_dim=128, max_seq_len=32, torsion_lock=True, phase_conjugate=True)
    model = SovereignTransformerV2(config)

    prompt = "organism Bell { gene g() on a, b { } }  "  # 40 bytes (> max_seq_len=32)
    fast_out = fast_generate(model, prompt, max_new_tokens=20, temperature=0.0, top_k=0)

    # Reference greedy loop via model.forward()
    ref_ids = list(prompt.encode("utf-8"))
    model.eval()
    with no_grad():
        for _ in range(20):
            ctx = ref_ids[-config.max_seq_len :]
            logits = model.forward(np.array([ctx], dtype=np.int64)).data[0, -1, :]
            ref_ids.append(int(np.argmax(logits)))
    model.train()
    ref_out = bytes(ref_ids).decode("utf-8", errors="replace")
    assert len(ref_ids) == len(prompt.encode("utf-8")) + 20
    assert fast_out == ref_out


def test_fast_generate_produces_text():
    config = SovereignConfig(dim=128, n_layers=2, n_heads=2, max_seq_len=256)
    model = SovereignTransformerV2(config)

    prompt = "Test prompt"
    out = fast_generate(model, prompt, max_new_tokens=16, temperature=0.0)
    assert out.startswith(prompt)
    assert len(out) > len(prompt)


def test_c_simd_availability():
    if HAVE_C_SIMD:
        from osiris.nclm import _c_simd
        assert hasattr(_c_simd, "fast_layernorm")
        assert hasattr(_c_simd, "fast_gelu")
        assert hasattr(_c_simd, "fast_kv_attention_step")
        assert hasattr(_c_simd, "fast_cross_entropy")
