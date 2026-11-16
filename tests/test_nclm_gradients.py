"""The NCLM's backward pass matches its forward pass, for both organism architectures.

Each parameter tensor is moved along its own gradient direction; the measured slope of the loss must
equal the gradient's norm. A whole-tensor move is robust to float32 noise, unlike single entries."""
import random

import numpy as np
import pytest

import osiris_termux_console as otc
from osiris.nclm import cross_entropy_loss
from osiris.nclm.autograd import no_grad

# Declared but never used in the forward pass (PhaseConjugateCorrector.zero_point): one tensor per block.
UNUSED = {"phase_conj.zero_point"}


def _loss(model, x, y):
    with no_grad():
        return float(np.float64(cross_entropy_loss(model.forward(x), y).data))


@pytest.mark.parametrize("arch", ["crsm", "standard"])
def test_every_parameter_gradient_matches_the_measured_slope(arch):
    np.random.seed(0)
    random.seed(0)
    model, opt = otc._organism_build(arch)
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 256, size=(2, 48)), rng.integers(0, 256, size=(2, 48))
    opt.zero_grad()
    cross_entropy_loss(model.forward(x), y).backward()
    bad = []
    for i, p in enumerate(model.parameters()):
        name = getattr(p, "name", "")
        if p.grad is None:
            if name not in UNUSED:
                bad.append((i, name, "no gradient"))
            continue
        g = np.asarray(p.grad, dtype=np.float64)
        norm = float(np.linalg.norm(g))
        if norm < 1e-9:
            continue
        d, old, best = (g / norm).astype(p.data.dtype), p.data.copy(), None
        for eps in (3e-2, 1e-2, 3e-3):
            p.data[...] = old + eps * d
            up = _loss(model, x, y)
            p.data[...] = old - eps * d
            down = _loss(model, x, y)
            p.data[...] = old
            ratio = (up - down) / (2 * eps) / norm
            best = ratio if best is None or abs(ratio - 1) < abs(best - 1) else best
        if abs(best - 1) > 0.05:
            bad.append((i, name, round(best, 3)))
    assert bad == []


@pytest.mark.parametrize("arch", ["crsm", "standard"])
def test_a_later_byte_never_changes_an_earlier_prediction(arch):
    np.random.seed(0)
    model, _ = otc._organism_build(arch)
    x = np.random.default_rng(1).integers(0, 256, size=(1, 40))
    changed = x.copy()
    changed[0, 30] = (changed[0, 30] + 7) % 256
    with no_grad():
        a, b = model.forward(x).data, model.forward(changed).data
    assert np.abs(a[0, :30] - b[0, :30]).max() == 0.0 and np.abs(a[0, 30:] - b[0, 30:]).max() > 0
