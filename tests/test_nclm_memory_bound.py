"""Regression tests ensuring native NCLM autograd & training do not leak computation graphs."""

from __future__ import annotations

import gc
import numpy as np

from osiris.nclm.autograd import Tensor, clip_grad_norm, cross_entropy_loss, no_grad
from osiris.nclm.optimizer import AdamW
from osiris.nclm.transformer import SovereignConfig, SovereignTransformerV2


def test_no_grad_builds_zero_graph_edges() -> None:
    cfg = SovereignConfig(
        vocab_size=256, dim=64, n_layers=2, n_heads=4, ff_dim=128, max_seq_len=32, dropout=0.0,
        torsion_lock=True, phase_conjugate=True, fractal_embedding=True,
    )
    model = SovereignTransformerV2(cfg)
    x = np.arange(32, dtype=np.int64).reshape(1, 32)
    y = ((np.arange(32) + 1) % 256).astype(np.int64).reshape(1, 32)

    with no_grad():
        logits = model.forward(x)
        loss = cross_entropy_loss(logits, y)

    assert logits.requires_grad is False
    assert logits._prev == []
    assert logits._backward_fn is None
    assert loss.requires_grad is False
    assert loss._prev == []
    assert loss._backward_fn is None


def test_cross_entropy_backward_releases_all_intermediate_tensors() -> None:
    np.random.seed(7)
    cfg = SovereignConfig(
        vocab_size=256, dim=64, n_layers=2, n_heads=4, ff_dim=128, max_seq_len=32, dropout=0.0,
        torsion_lock=True, phase_conjugate=True, fractal_embedding=True,
    )
    model = SovereignTransformerV2(cfg)
    opt = AdamW(model.parameters(), lr=1e-3)
    param_ids = {id(p) for p in model.parameters()} | {id(model.pos_enc)}

    x = np.random.randint(0, 256, size=(4, 32), dtype=np.int64)
    y = np.random.randint(0, 256, size=(4, 32), dtype=np.int64)

    gc.collect()
    gc.disable()
    try:
        for _ in range(5):
            opt.zero_grad()
            logits = model.forward(x)
            loss = cross_entropy_loss(logits, y)
            loss.backward()
            clip_grad_norm(model.parameters(), 1.0)
            opt.step()
            assert loss._prev == []
            assert logits._prev == []
            assert logits.grad is None
            del logits, loss

        # With cyclic GC disabled, ZERO non-parameter Tensor objects remain alive in memory!
        live_nonparam_tensors = [
            obj for obj in gc.get_objects()
            if isinstance(obj, Tensor) and id(obj) not in param_ids
        ]
        assert live_nonparam_tensors == []
    finally:
        gc.enable()
