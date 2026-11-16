"""
Fast C-SIMD and KV-Cached Inference Engine for OSIRIS Living Core
================================================================
Provides O(1) per-token autoregressive generation for step-invariant blocks
and vectorized autograd-free evaluation for sequence-coupled 11D-CRSM blocks
(TorsionLockedAttention, PhaseConjugateCorrector, FractalAntennaEmbedding),
preserving float32 numerical equivalence with `model.forward()` across all
architectures and across sliding context windows (`len >= max_seq_len`).
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional
import numpy as np

try:
    from . import _c_simd
    HAVE_C_SIMD = True
except (ImportError, OSError):
    _c_simd = None
    HAVE_C_SIMD = False

from .autograd import no_grad

_SQRT_2_OVER_PI = 0.7978845608028654
_PHI_INV = 1.0 / 1.618033988749895


class FastKVCache:
    """Pre-allocated key-value and token-window cache across transformer blocks."""

    def __init__(self, n_layers: int, max_seq_len: int, n_heads: int, head_dim: int):
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.k_cache: List[Optional[np.ndarray]] = [None] * n_layers
        self.v_cache: List[Optional[np.ndarray]] = [None] * n_layers
        self.token_ids: List[int] = []
        self.seq_len = 0

    def reset(self) -> None:
        self.k_cache = [None] * self.n_layers
        self.v_cache = [None] * self.n_layers
        self.token_ids = []
        self.seq_len = 0


def _layernorm(h: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    if HAVE_C_SIMD:
        return _c_simd.fast_layernorm(np.ascontiguousarray(h, dtype=np.float32), gamma, beta, 1e-5)
    mean = np.mean(h, axis=-1, keepdims=True)
    var = np.var(h, axis=-1, keepdims=True)
    return (h - mean) / np.sqrt(var + 1e-5) * gamma + beta


def _linear_forward(layer, x: np.ndarray) -> np.ndarray:
    """Fast linear projection: x @ W^T + b without autograd graph."""
    out = x @ layer.weight.data.T
    if getattr(layer, "bias_param", None) is not None:
        out = out + layer.bias_param.data
    return out


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * (x * x * x))))


def _embed_tokens(model, token_ids: List[int]) -> np.ndarray:
    """Vectorized embedding supporting both standard Embedding and FractalAntennaEmbedding."""
    if getattr(model, "_use_fractal", False):
        emb = model.tok_emb
        x_coding = emb.byte_emb.data[token_ids, :]  # (T, D)
        harmonics = (x_coding @ emb.resonance_in.data.T) * emb.omega[np.newaxis, :].astype(np.float32)
        x_resonant = np.tanh(harmonics) @ emb.resonance_out.data.T
        x_vacuum = (x_coding @ emb.vacuum_proj.data.T) * emb.xi_coupling.data
        return x_coding + x_resonant + x_vacuum
    return model.tok_emb.weight.data[token_ids, :]


def _is_sequence_coupled(model) -> bool:
    """True when block computations depend on full current window length T (CRSM or multi-layer pilot-wave)."""
    if getattr(model, "_use_sovereign", False) or getattr(model, "_use_fractal", False):
        return True
    return any(getattr(b.attn, "pilot_wave", False) for b in model.blocks) and len(model.blocks) > 1


def _forward_window_no_grad(model, window_ids: List[int], kv_cache: Optional[FastKVCache] = None) -> np.ndarray:
    """Exact vectorized no-autograd forward pass returning final-position logits (V,)."""
    max_len = model.config.max_seq_len
    ids = window_ids[-max_len:]
    P = len(ids)
    x = _embed_tokens(model, ids)
    pe = model.pos_enc.data[:P, :]
    h = (x + pe)[np.newaxis, :, :]  # (1, P, D)

    B = 1
    H = model.config.n_heads
    HD = model.config.dim // H
    scale = 1.0 / math.sqrt(HD)
    mask = model._causal_mask[:, :, :P, :P]
    eye = None

    for i, block in enumerate(model.blocks):
        norm_h = _layernorm(h, block.ln1.gamma.data, block.ln1.beta.data)
        q = _linear_forward(block.attn.q_proj, norm_h).reshape(B, P, H, HD).transpose(0, 2, 1, 3)
        k = _linear_forward(block.attn.k_proj, norm_h).reshape(B, P, H, HD).transpose(0, 2, 1, 3)
        v = _linear_forward(block.attn.v_proj, norm_h).reshape(B, P, H, HD).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            kv_cache.k_cache[i] = np.ascontiguousarray(k, dtype=np.float32)
            kv_cache.v_cache[i] = np.ascontiguousarray(v, dtype=np.float32)

        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        # TorsionLockedAttention (SovereignBlock)
        if hasattr(block.attn, "tlock_alpha"):
            if eye is None:
                eye = np.eye(P, dtype=np.float32)[np.newaxis, np.newaxis, :, :]
            alpha = block.attn.tlock_alpha.data.reshape(1, H, 1, 1)
            beta = block.attn.tlock_beta.data.reshape(1, H, 1, 1)
            scores = scores * alpha + (scores * eye) * beta
            pw = block.attn._pilot_wave_factor(P)
            scores = scores * pw
        elif getattr(block.attn, "pilot_wave", False):
            pw = block.attn._pilot_wave_factor(P)
            scores = scores * pw[np.newaxis, np.newaxis, :, :]

        scores = scores + mask
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, P, -1)
        out = _linear_forward(block.attn.out_proj, out)
        h = h + out

        # PhaseConjugateCorrector (SovereignBlock)
        pc = getattr(block, "phase_corrector", None)
        if pc is not None:
            var = np.var(h, axis=-1, keepdims=True)
            mean_var = float(np.mean(var))
            gamma = float(np.clip(mean_var / (mean_var + 1.0), 0.0, 1.0))
            if gamma > pc.gamma_critical:
                h_conj = h.copy()
                h_conj[:, :, 1::2] *= -1.0
                h_healed = _linear_forward(pc.theta_proj, h_conj)
                gate = (np.tanh(_linear_forward(pc.conj_gate, h)) + 1.0) * 0.5
                h = gate * h_healed + (1.0 - gate) * h

        norm_h = _layernorm(h, block.ln2.gamma.data, block.ln2.beta.data)
        ffn_out = _linear_forward(block.ffn.fc2, _gelu(_linear_forward(block.ffn.fc1, norm_h)))
        if hasattr(block, "phase_corrector") or getattr(block.ffn, "golden_scale", False):
            ffn_out = ffn_out * _PHI_INV
        h = h + ffn_out

    # Only project final token onto vocabulary (P-fold reduction in LM-head FLOPs)
    last_h = _layernorm(h[:, -1:, :], model.ln_f.gamma.data, model.ln_f.beta.data)
    logits = _linear_forward(model.lm_head, last_h)
    if kv_cache is not None:
        kv_cache.token_ids = list(ids)
        kv_cache.seq_len = P
    return logits[0, 0, :]


def prefill_kv(model, prompt_ids: List[int], kv_cache: FastKVCache) -> np.ndarray:
    """Process prompt in one vectorized forward pass and populate KV cache."""
    return _forward_window_no_grad(model, prompt_ids, kv_cache=kv_cache)


def decode_step_kv(model, token_id: int, pos: int, kv_cache: FastKVCache) -> np.ndarray:
    """Decode next single token, using O(1) KV cache when step-invariant or sliding window when needed."""
    kv_cache.token_ids.append(token_id)
    max_len = model.config.max_seq_len

    # Sequence-coupled CRSM/pilot-wave models or full context windows re-evaluate the exact bounded window
    if _is_sequence_coupled(model) or pos >= max_len or kv_cache.k_cache[0] is None:
        return _forward_window_no_grad(model, kv_cache.token_ids[-max_len:], kv_cache=kv_cache)

    w_emb = model.tok_emb.weight.data
    x = w_emb[token_id : token_id + 1, :]
    pe = model.pos_enc.data[pos : pos + 1, :]
    h = (x + pe)[np.newaxis, :, :]  # (1, 1, D)

    B = 1
    H = model.config.n_heads
    HD = model.config.dim // H
    scale = 1.0 / math.sqrt(HD)

    for i, block in enumerate(model.blocks):
        norm_h = _layernorm(h, block.ln1.gamma.data, block.ln1.beta.data)

        q = _linear_forward(block.attn.q_proj, norm_h).reshape(B, 1, H, HD).transpose(0, 2, 1, 3)
        k_new = _linear_forward(block.attn.k_proj, norm_h).reshape(B, 1, H, HD).transpose(0, 2, 1, 3)
        v_new = _linear_forward(block.attn.v_proj, norm_h).reshape(B, 1, H, HD).transpose(0, 2, 1, 3)

        k = np.ascontiguousarray(np.concatenate([kv_cache.k_cache[i], k_new], axis=2), dtype=np.float32)
        v = np.ascontiguousarray(np.concatenate([kv_cache.v_cache[i], v_new], axis=2), dtype=np.float32)
        kv_cache.k_cache[i] = k
        kv_cache.v_cache[i] = v

        if HAVE_C_SIMD and not getattr(block.attn, "pilot_wave", False):
            q_2d = np.ascontiguousarray(q[0, :, 0, :], dtype=np.float32)
            k_3d = np.ascontiguousarray(k[0, :, :, :], dtype=np.float32)
            v_3d = np.ascontiguousarray(v[0, :, :, :], dtype=np.float32)
            out_heads = _c_simd.fast_kv_attention_step(q_2d, k_3d, v_3d, float(scale))
            out = out_heads.reshape(B, 1, -1)
        else:
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale
            if getattr(block.attn, "pilot_wave", False):
                cur_len = pos + 1
                pw = block.attn._pilot_wave_factor(cur_len)
                scores = scores * pw[-1:, :][np.newaxis, np.newaxis, :, :]
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, 1, -1)

        out = _linear_forward(block.attn.out_proj, out)
        h = h + out

        norm_h = _layernorm(h, block.ln2.gamma.data, block.ln2.beta.data)
        ffn_out = _linear_forward(block.ffn.fc2, _gelu(_linear_forward(block.ffn.fc1, norm_h)))
        if getattr(block.ffn, "golden_scale", False):
            ffn_out = ffn_out * _PHI_INV
        h = h + ffn_out

    norm_h = _layernorm(h, model.ln_f.gamma.data, model.ln_f.beta.data)
    logits = _linear_forward(model.lm_head, norm_h)
    kv_cache.seq_len = pos + 1
    return logits[0, 0, :]


def fast_generate(
    model,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 40,
    callback: Optional[Callable[[str], None]] = None,
) -> str:
    """High-speed autoregressive generation matching `model.forward()` across all NCLM architectures."""
    prompt_ids = list(prompt.encode("utf-8", errors="replace"))
    if not prompt_ids:
        prompt_ids = [0]

    n_layers = len(model.blocks)
    head_dim = model.config.dim // model.config.n_heads
    max_len = model.config.max_seq_len
    kv_cache = FastKVCache(n_layers, max_len, model.config.n_heads, head_dim)

    model.eval()
    with no_grad():
        logits = prefill_kv(model, prompt_ids[-max_len:], kv_cache)

        output_ids = list(prompt_ids)
        cur_pos = min(len(prompt_ids), max_len)

        for step in range(max_new_tokens):
            next_logits = logits.copy()
            if temperature > 0:
                next_logits = next_logits / temperature
                if 0 < top_k < model.config.vocab_size:
                    indices_to_remove = np.argsort(next_logits)[:-top_k]
                    next_logits[indices_to_remove] = -1e9
                probs = np.exp(next_logits - np.max(next_logits))
                probs = probs / probs.sum()
                next_id = int(np.random.choice(model.config.vocab_size, p=probs))
            else:
                next_id = int(np.argmax(next_logits))

            output_ids.append(next_id)
            if callback:
                callback(bytes([next_id]).decode("utf-8", errors="replace"))

            current_str = bytes(output_ids[len(prompt_ids) :]).decode("utf-8", errors="replace")
            if "\nUser:" in current_str or "\nuser:" in current_str:
                break

            if step + 1 < max_new_tokens:
                logits = decode_step_kv(model, next_id, cur_pos, kv_cache)
                cur_pos = min(cur_pos + 1, max_len)

    model.train()
    return bytes(output_ids).decode("utf-8", errors="replace")
