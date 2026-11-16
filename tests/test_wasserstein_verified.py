"""
Unit & Mathematical Property Tests for Verified Optimal Transport (`wasserstein_verified`)
==========================================================================================

Covers Phase 2 Section 2 (Correction A — NB-11):
  - Proof that historical independent-sampling expectation C(p,p) = 0.5 != 0 = W_2^2(p,p)
    on p = q = 0.5*delta_0 + 0.5*delta_1 with unit cost
  - Identity of indiscernibles: W_p(p, p) = 0 and W_p(p, q) > 0 for p != q
  - Symmetry: W_p(p, q) = W_p(q, p)
  - Triangle inequality: W_p(p, r) <= W_p(p, q) + W_p(q, r)
  - Closed-form analytic distributions (Dirac delta shift, Bernoulli 2-atom W_1/W_2,
    product Bernoulli on {0,1}^n with Hamming distance)
  - Certified Sinkhorn primal/dual bounds bracketing exact LP distance
  - Sliced Wasserstein consistency and signed/complex input rejection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from osiris_cli.metrics import (
    debiased_empirical_wasserstein,
    defective_independent_cost,
    exact_wasserstein_1d,
    exact_wasserstein_lp,
    hamming_cost_matrix,
    sinkhorn_wasserstein_bounded,
    sliced_wasserstein_hypercube,
    validate_probability_measure,
)


def test_nb11_defective_formula_vs_true_wasserstein_counterexample():
    """
    Phase 2 Section 2 Correction A benchmark check:
    For p = q = 0.5*delta_0 + 0.5*delta_1 with unit cost between outcomes 0 and 1:
      - Defective formula C(p, p) = sum_{x,y} c(x,y) p(x) p(y) = 0.5
      - True W_2^2(p, p) = 0.0 and W_2(p, p) = 0.0
    """
    p = np.array([0.5, 0.5], dtype=np.float64)
    unit_cost = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    c_pp = defective_independent_cost(p, p, unit_cost)
    ot_pp = exact_wasserstein_lp(p, p, unit_cost, order=2.0)

    assert math.isclose(c_pp, 0.5, abs_tol=1e-15)
    assert math.isclose(ot_pp.transport_cost, 0.0, abs_tol=1e-13)
    assert math.isclose(ot_pp.distance, 0.0, abs_tol=1e-13)


@pytest.mark.parametrize("order", [1.0, 2.0])
def test_metric_axioms_identity_symmetry_triangle_on_hamming_cube(order: float):
    rng = np.random.default_rng(2026)
    n_bits = 4
    states = list(range(1 << n_bits))
    d_mat = hamming_cost_matrix(states, states, normalize_by_n=n_bits)

    p = rng.dirichlet(np.ones(len(states)))
    q = rng.dirichlet(np.ones(len(states)))
    r = rng.dirichlet(np.ones(len(states)))

    # Identity
    assert exact_wasserstein_lp(p, p, d_mat, order=order).distance < 1e-12
    assert exact_wasserstein_lp(q, q, d_mat, order=order).distance < 1e-12

    # Strict positivity for distinct distributions
    w_pq = exact_wasserstein_lp(p, q, d_mat, order=order).distance
    w_qp = exact_wasserstein_lp(q, p, d_mat, order=order).distance
    assert w_pq > 0.0

    # Symmetry
    assert math.isclose(w_pq, w_qp, abs_tol=1e-12)

    # Triangle inequality
    w_pr = exact_wasserstein_lp(p, r, d_mat, order=order).distance
    w_qr = exact_wasserstein_lp(q, r, d_mat, order=order).distance
    assert w_pr <= w_pq + w_qr + 1e-11


def test_analytic_closed_form_distributions():
    # 1. Two-point Bernoulli: P = (1-a) d_0 + a d_L, Q = (1-b) d_0 + b d_L on {0, L}
    L = 3.5
    a, b = 0.2, 0.85
    p = np.array([1.0 - a, a])
    q = np.array([1.0 - b, b])
    d_mat = np.array([[0.0, L], [L, 0.0]])

    w1_lp = exact_wasserstein_lp(p, q, d_mat, order=1.0).distance
    w2_lp = exact_wasserstein_lp(p, q, d_mat, order=2.0).distance
    w1_1d = exact_wasserstein_1d([0.0, L], [0.0, L], p, q, order=1.0).distance
    w2_1d = exact_wasserstein_1d([0.0, L], [0.0, L], p, q, order=2.0).distance

    assert math.isclose(w1_lp, abs(a - b) * L, abs_tol=1e-12)
    assert math.isclose(w1_1d, abs(a - b) * L, abs_tol=1e-12)
    assert math.isclose(w2_lp, math.sqrt(abs(a - b)) * L, abs_tol=1e-12)
    assert math.isclose(w2_1d, math.sqrt(abs(a - b)) * L, abs_tol=1e-12)

    # 2. Product Bernoulli(a)^n vs Bernoulli(b)^n on {0,1}^n with unnormalized Hamming metric:
    # Exact W_1 equals n * |a - b| (achieved by coordinatewise monotone coupling).
    n = 5
    states = list(range(1 << n))
    d_ham = hamming_cost_matrix(states, states)
    weights_a = np.array([
        (a ** bin(s).count("1")) * ((1.0 - a) ** (n - bin(s).count("1"))) for s in states
    ])
    weights_b = np.array([
        (b ** bin(s).count("1")) * ((1.0 - b) ** (n - bin(s).count("1"))) for s in states
    ])
    w1_prod = exact_wasserstein_lp(weights_a, weights_b, d_ham, order=1.0).distance
    assert math.isclose(w1_prod, n * abs(a - b), abs_tol=1e-11)


def test_sinkhorn_certified_bounds_bracket_exact_lp():
    rng = np.random.default_rng(77)
    n = 4
    states = list(range(1 << n))
    d_mat = hamming_cost_matrix(states, states, normalize_by_n=n)
    p = rng.dirichlet(np.ones(len(states)))
    q = rng.dirichlet(np.ones(len(states)))

    exact = exact_wasserstein_lp(p, q, d_mat, order=2.0)
    sh = sinkhorn_wasserstein_bounded(p, q, d_mat, order=2.0, epsilon=0.005, max_iter=2000)

    assert sh.distance_lower_bound <= exact.distance + 1e-10
    assert exact.distance <= sh.distance_upper_bound + 1e-10
    assert abs(sh.midpoint_estimate - exact.distance) <= sh.declared_max_error + 1e-10
    assert sh.declared_max_error < 0.05


def test_sliced_wasserstein_and_measure_validation():
    c1 = {"0000": 500, "1111": 500}
    c2 = {"0000": 200, "1111": 800}
    sw_id = sliced_wasserstein_hypercube(c1, c1, order=2.0, n_projections=128, seed=1)
    sw_diff = sliced_wasserstein_hypercube(c1, c2, order=2.0, n_projections=128, seed=1)
    assert sw_id["sliced_distance"] < 1e-12
    assert sw_diff["sliced_distance"] > 0.2
    assert sw_diff["mc_95ci_halfwidth"] > 0.0

    # Signed or complex arrays must be rejected
    with pytest.raises(ValueError, match="negative"):
        validate_probability_measure(np.array([1.2, -0.2]))
    with pytest.raises(ValueError, match="complex"):
        validate_probability_measure(np.array([0.5 + 0.1j, 0.5 - 0.1j]))
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_probability_measure(np.array([0.2, 0.2]))


def test_wide_hamming_above_64_qubits_and_split_sample_debiasing():
    # 128-qubit bitstrings must not overflow uint64
    s_zero = "0" * 128
    s_half = "1" * 64 + "0" * 64
    s_ones = "1" * 128
    c128 = hamming_cost_matrix([s_zero, s_half], [s_zero, s_ones], normalize_by_n=128)
    np.testing.assert_allclose(c128, [[0.0, 1.0], [0.5, 0.5]], atol=1e-15)

    # Split-sample empirical OT debiasing removes O(S^{-1/Q}) self-distance inflation on 27Q draws
    rng = np.random.default_rng(2026)
    draws_p = rng.integers(0, 1 << 27, size=256)
    draws_q = rng.integers(0, 1 << 27, size=256)
    cnt_p: dict[str, int] = {}
    cnt_q: dict[str, int] = {}
    for x in draws_p:
        bs = format(int(x), "027b")
        cnt_p[bs] = cnt_p.get(bs, 0) + 1
    for x in draws_q:
        bs = format(int(x), "027b")
        cnt_q[bs] = cnt_q.get(bs, 0) + 1

    deb = debiased_empirical_wasserstein(cnt_p, cnt_q, order=2.0, seed=42)
    assert deb["raw_plug_in_distance"] > 0.20
    assert deb["debiased_distance"] < 0.05
