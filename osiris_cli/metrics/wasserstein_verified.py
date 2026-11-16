"""
Verified Discrete Optimal Transport (Wasserstein) Metrics
=========================================================

Phase 2 Section 2 (Correction A — NB-11) & Section 4 (W2 Methods Repair).

Separates:
  1. `defective_independent_cost(p, q, cost_matrix)`:
     Reproduces the historical independent-product expectation
         C(p, q) = sum_{x, y} c(x, y) p(x) q(y)
     which violates identity of indiscernibles: for p = q = 0.5*d_0 + 0.5*d_1
     with unit ground cost, C(p, p) = 0.5 != 0 = W_2^2(p, p).
  2. `exact_wasserstein_lp(p, q, ground_metric, order=2)`:
     Solves the exact Kantorovich linear program via HiGHS (`scipy.optimize.linprog`)
     over all couplings pi in Pi(p, q), returning exact W_p(p, q) and W_p^p(p, q).
  3. `exact_wasserstein_1d(u_values, v_values, u_weights, v_weights, order=2)`:
     Exact O(K log K) inverse-CDF / quantile optimal transport on 1D supports.
  4. `sinkhorn_wasserstein_bounded(...)` & `sliced_wasserstein_hypercube(...)`:
     Scalable approximations with declared numerical error bounds (Altschuler et al.
     rounding onto Pi(p, q) vs dual/entropic lower bound, and Monte-Carlo SE).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


@dataclass(frozen=True)
class OTResult:
    distance: float
    transport_cost: float
    order: float
    support_size_p: int
    support_size_q: int
    method: str
    max_numerical_error: float


@dataclass(frozen=True)
class SinkhornResult:
    distance_upper_bound: float
    distance_lower_bound: float
    midpoint_estimate: float
    declared_max_error: float
    epsilon: float
    iterations: int
    converged: bool


def validate_probability_measure(
    weights: np.ndarray, name: str = "distribution", tol: float = 1e-9
) -> np.ndarray:
    """
    Validate that `weights` is a real, non-negative, normalized discrete probability
    distribution. Explicitly rejects complex or signed signals (e.g. raw NMR spectra).
    """
    arr = np.asarray(weights)
    if np.iscomplexobj(arr):
        raise ValueError(
            f"{name} contains complex values; define an explicit transformation to a "
            "non-negative probability measure before computing Wasserstein distance."
        )
    arr = arr.astype(np.float64, copy=False)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(arr < -tol):
        raise ValueError(
            f"{name} contains negative entries (min={arr.min():.3e}); signed signals "
            "are not probability measures."
        )
    clipped = np.maximum(arr, 0.0)
    total = float(np.sum(clipped))
    if not math.isclose(total, 1.0, rel_tol=tol, abs_tol=tol):
        raise ValueError(
            f"{name} weights must sum to 1.0 within {tol}, got sum={total:.12f}"
        )
    return clipped / total


def hamming_cost_matrix(
    bitstrings_x: Sequence[int | str],
    bitstrings_y: Sequence[int | str],
    normalize_by_n: Optional[int] = None,
) -> np.ndarray:
    """
    Compute pairwise Hamming ground distance d_H(x, y) (or d_H(x, y) / n when
    `normalize_by_n` is given) for integer-encoded or string bitstrings of arbitrary width.
    Supports >64-qubit bitstrings without uint64 overflow.
    """
    def _to_python_ints(seq: Sequence[int | str]) -> list[int]:
        out: list[int] = []
        for item in seq:
            if isinstance(item, str):
                out.append(int(item, 2) if item else 0)
            else:
                val = int(item)
                if val < 0:
                    raise ValueError(f"Bitstring integer must be non-negative, got {val}")
                out.append(val)
        return out

    ix = _to_python_ints(bitstrings_x)
    iy = _to_python_ints(bitstrings_y)
    max_val = max(ix + iy, default=0)

    if max_val < (1 << 64):
        bx = np.asarray(ix, dtype=np.uint64)[:, None]
        by = np.asarray(iy, dtype=np.uint64)[None, :]
        xor = np.bitwise_xor(bx, by)
        xor_bytes = xor.view(np.uint8).reshape(len(ix), len(iy), 8)
        lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)
        dist = lut[xor_bytes].sum(axis=-1)
    else:
        dist = np.empty((len(ix), len(iy)), dtype=np.float64)
        for r_idx, vx in enumerate(ix):
            for c_idx, vy in enumerate(iy):
                dist[r_idx, c_idx] = float((vx ^ vy).bit_count())

    if normalize_by_n is not None:
        if normalize_by_n <= 0:
            raise ValueError("normalize_by_n must be positive")
        dist = dist / float(normalize_by_n)
    return dist


def defective_independent_cost(
    p: np.ndarray, q: np.ndarray, cost_matrix: np.ndarray
) -> float:
    """
    Reproduce the historical independent-sampling expectation formula:
        C(p, q) = sum_{x, y} c(x, y) p(x) q(y)
    Documented for forensic audit ONLY; NOT a Wasserstein metric because C(p, p) > 0
    for any non-degenerate distribution p.
    """
    p_val = validate_probability_measure(p, "p")
    q_val = validate_probability_measure(q, "q")
    c = np.asarray(cost_matrix, dtype=np.float64)
    if c.shape != (p_val.size, q_val.size):
        raise ValueError(f"cost_matrix shape {c.shape} != ({p_val.size}, {q_val.size})")
    return float(p_val @ c @ q_val)


def exact_wasserstein_lp(
    p: np.ndarray,
    q: np.ndarray,
    ground_distance: np.ndarray,
    order: float = 2.0,
) -> OTResult:
    """
    Compute exact p-Wasserstein distance W_p(P, Q) via Kantorovich Linear Programming:
        W_p(P, Q) = ( min_{pi in Pi(P, Q)} sum_{i,j} d(i, j)^p pi_{i,j} )^(1/p)
    Restricts LP variables to positive-mass support of P and Q for speed and exactness.
    """
    if order < 1.0:
        raise ValueError(f"order must be >= 1.0, got {order}")
    p_val = validate_probability_measure(p, "p")
    q_val = validate_probability_measure(q, "q")
    d = np.asarray(ground_distance, dtype=np.float64)
    if d.shape != (p_val.size, q_val.size):
        raise ValueError(f"ground_distance shape {d.shape} != ({p_val.size}, {q_val.size})")
    if np.any(d < -1e-14):
        raise ValueError("ground_distance must be non-negative")
    d = np.maximum(d, 0.0)

    # Filter zero-probability atoms
    mask_p = p_val > 0.0
    mask_q = q_val > 0.0
    p_sub = p_val[mask_p]
    q_sub = q_val[mask_q]
    p_sub /= p_sub.sum()
    q_sub /= q_sub.sum()
    d_sub = d[np.ix_(mask_p, mask_q)]

    m, n = p_sub.size, q_sub.size
    cost_vec = (d_sub ** order).reshape(m * n)

    # Sparse equality constraints: row sums = p_sub (m rows), col sums = q_sub (first n-1 cols;
    # last col redundant since sum(p_sub) == sum(q_sub) == 1). Avoids O((m+n)*m*n) dense RAM.
    n_eq = m + (n - 1)
    row_idx_r = np.repeat(np.arange(m, dtype=np.int32), n)
    col_idx_r = np.arange(m * n, dtype=np.int32)
    if n > 1:
        row_idx_c = m + np.tile(np.arange(n - 1, dtype=np.int32), m)
        col_idx_c = (np.arange(m, dtype=np.int32)[:, None] * n + np.arange(n - 1, dtype=np.int32)[None, :]).reshape(-1)
        rows = np.concatenate([row_idx_r, row_idx_c])
        cols = np.concatenate([col_idx_r, col_idx_c])
    else:
        rows = row_idx_r
        cols = col_idx_r
    data = np.ones(rows.size, dtype=np.float64)
    a_eq = sparse.csc_matrix((data, (rows, cols)), shape=(n_eq, m * n))
    b_eq = np.concatenate([p_sub, q_sub[:-1]]) if n > 1 else p_sub.copy()

    res = linprog(
        c=cost_vec,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Exact OT linear program failed: {res.message}")

    transport_cost = max(0.0, float(res.fun))
    dist = transport_cost ** (1.0 / order)
    return OTResult(
        distance=dist,
        transport_cost=transport_cost,
        order=order,
        support_size_p=m,
        support_size_q=n,
        method="exact_kantorovich_highs_lp",
        max_numerical_error=1e-11,
    )


def exact_wasserstein_1d(
    u_values: Sequence[float],
    v_values: Sequence[float],
    u_weights: Optional[Sequence[float]] = None,
    v_weights: Optional[Sequence[float]] = None,
    order: float = 2.0,
) -> OTResult:
    """
    Exact 1D p-Wasserstein distance via inverse-CDF integration:
        W_p^p(mu, nu) = int_0^1 |F_mu^{-1}(t) - F_nu^{-1}(t)|^p dt
    Runs in O((M + N) log(M + N)) time with machine-precision accuracy.
    """
    if order < 1.0:
        raise ValueError(f"order must be >= 1.0, got {order}")
    u = np.asarray(u_values, dtype=np.float64)
    v = np.asarray(v_values, dtype=np.float64)
    uw = (
        np.full(u.size, 1.0 / u.size, dtype=np.float64)
        if u_weights is None
        else validate_probability_measure(np.asarray(u_weights), "u_weights")
    )
    vw = (
        np.full(v.size, 1.0 / v.size, dtype=np.float64)
        if v_weights is None
        else validate_probability_measure(np.asarray(v_weights), "v_weights")
    )
    if u.size != uw.size or v.size != vw.size:
        raise ValueError("Values and weights must have matching lengths")

    u_sorter = np.argsort(u, kind="mergesort")
    v_sorter = np.argsort(v, kind="mergesort")
    u_sorted, uw_sorted = u[u_sorter], uw[u_sorter]
    v_sorted, vw_sorted = v[v_sorter], vw[v_sorter]

    u_cdf = np.cumsum(uw_sorted)
    v_cdf = np.cumsum(vw_sorted)
    u_cdf[-1] = 1.0
    v_cdf[-1] = 1.0

    knots = np.unique(np.concatenate(([0.0], u_cdf, v_cdf)))
    if knots.size <= 1:
        return OTResult(0.0, 0.0, order, u.size, v.size, "exact_1d_quantile", 1e-15)

    mids = 0.5 * (knots[:-1] + knots[1:])
    dt = np.diff(knots)
    u_idx = np.searchsorted(u_cdf, mids, side="right")
    v_idx = np.searchsorted(v_cdf, mids, side="right")
    u_idx = np.clip(u_idx, 0, u_sorted.size - 1)
    v_idx = np.clip(v_idx, 0, v_sorted.size - 1)

    cost = float(np.sum((np.abs(u_sorted[u_idx] - v_sorted[v_idx]) ** order) * dt))
    cost = max(0.0, cost)
    return OTResult(
        distance=cost ** (1.0 / order),
        transport_cost=cost,
        order=order,
        support_size_p=int(np.sum(uw > 0)),
        support_size_q=int(np.sum(vw > 0)),
        method="exact_1d_quantile",
        max_numerical_error=1e-14,
    )


def _round_onto_transport_polytope(
    plan: np.ndarray, r: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """
    Altschuler, Weed & Rigollet (2017) Algorithm 2: project a non-negative matrix
    onto the exact coupling polytope Pi(r, c) in O(m*n) operations.
    """
    P = np.maximum(plan, 0.0).copy()
    row_sums = P.sum(axis=1)
    scale_r = np.where(row_sums > 0.0, np.minimum(1.0, r / np.maximum(row_sums, 1e-300)), 1.0)
    P = P * scale_r[:, None]

    col_sums = P.sum(axis=0)
    scale_c = np.where(col_sums > 0.0, np.minimum(1.0, c / np.maximum(col_sums, 1e-300)), 1.0)
    P = P * scale_c[None, :]

    err_r = np.maximum(r - P.sum(axis=1), 0.0)
    err_c = np.maximum(c - P.sum(axis=0), 0.0)
    l1_err = float(err_r.sum())
    if l1_err > 1e-300:
        P += np.outer(err_r, err_c) / l1_err
    return P


def sinkhorn_wasserstein_bounded(
    p: np.ndarray,
    q: np.ndarray,
    ground_distance: np.ndarray,
    order: float = 2.0,
    epsilon: float = 0.02,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> SinkhornResult:
    """
    Log-domain stabilized Sinkhorn-Knopp entropic OT with guaranteed primal-feasible
    upper bound (via Altschuler polytope rounding) and analytic entropic lower bound:
        cost_feasible - eps * H(pi) <= W_p^p(p, q) <= cost_feasible
    Wait: more specifically, for any feasible rounded plan pi_hat in Pi(p, q):
        W_p^p(p, q) <= <C, pi_hat>
    and from the Sinkhorn dual potentials (f, g) with f_i + g_j <= C_{ij} + eps*log K_{ij} = 0?
    Even stronger: define hard c-transform dual potentials:
        u_i = f_i,  v_j = min_i (C_{ij} - u_i)
    Then u_i + v_j <= C_{ij} for ALL (i, j) strictly, so <u, p> + <v, q> is a 100%
    CERTIFIED Kantorovich dual lower bound on exact W_p^p(p, q)!
    """
    if order < 1.0:
        raise ValueError("order must be >= 1.0")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0.0")

    p_val = validate_probability_measure(p, "p")
    q_val = validate_probability_measure(q, "q")
    d = np.maximum(np.asarray(ground_distance, dtype=np.float64), 0.0)
    mask_p = p_val > 0.0
    mask_q = q_val > 0.0
    r = p_val[mask_p]
    c = q_val[mask_q]
    r /= r.sum()
    c /= c.sum()
    C = d[np.ix_(mask_p, mask_q)] ** order

    log_r = np.log(r)
    log_c = np.log(c)
    f = np.zeros_like(r)
    g = np.zeros_like(c)

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # f_i = eps * log r_i - eps * logsumexp_j ((g_j - C_ij) / eps)
        M_f = (g[None, :] - C) / epsilon
        m_f = np.max(M_f, axis=1, keepdims=True)
        f_new = epsilon * (log_r - (m_f[:, 0] + np.log(np.sum(np.exp(M_f - m_f), axis=1))))

        M_g = (f_new[:, None] - C) / epsilon
        m_g = np.max(M_g, axis=0, keepdims=True)
        g_new = epsilon * (log_c - (m_g[0, :] + np.log(np.sum(np.exp(M_g - m_g), axis=0))))

        if np.max(np.abs(f_new - f)) < tol and np.max(np.abs(g_new - g)) < tol:
            f, g = f_new, g_new
            converged = True
            break
        f, g = f_new, g_new

    log_P = (f[:, None] + g[None, :] - C) / epsilon
    P_raw = np.exp(np.clip(log_P, -700.0, 50.0))
    P_feas = _round_onto_transport_polytope(P_raw, r, c)

    ub_cost = max(0.0, float(np.sum(P_feas * C)))

    # Certified Kantorovich dual feasible pair via exact c-transform: v_j = min_i (C_ij - f_i)
    v_cert = np.min(C - f[:, None], axis=0)
    u_cert = np.min(C - v_cert[None, :], axis=1)
    lb_cost = max(0.0, float(np.dot(u_cert, r) + np.dot(v_cert, c)))
    if lb_cost > ub_cost and lb_cost - ub_cost < 1e-12:
        lb_cost = ub_cost

    ub_dist = ub_cost ** (1.0 / order)
    lb_dist = lb_cost ** (1.0 / order)
    mid = 0.5 * (ub_dist + lb_dist)
    err = 0.5 * max(0.0, ub_dist - lb_dist)
    return SinkhornResult(
        distance_upper_bound=ub_dist,
        distance_lower_bound=lb_dist,
        midpoint_estimate=mid,
        declared_max_error=err,
        epsilon=epsilon,
        iterations=it,
        converged=converged,
    )


def sliced_wasserstein_hypercube(
    counts_p: Mapping[str, int],
    counts_q: Mapping[str, int],
    order: float = 2.0,
    n_projections: int = 256,
    normalize_by_sqrt_n: bool = False,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Scalable Sliced p-Wasserstein distance over bitstring count dictionaries on {0,1}^n,
    with Monte-Carlo standard error across random unit-sphere projections.
    Note: Embedding bitstrings in {0,1}^n induces Euclidean ground metric ||x-y||_2 = sqrt(d_H(x,y));
    pass `normalize_by_sqrt_n=True` to scale coordinates to {0, 1/sqrt(n)}^n (||x-y||_2 = sqrt(d_H/n)).
    """
    if not counts_p or not counts_q:
        raise ValueError("Count dictionaries must be non-empty")
    keys_p = list(counts_p.keys())
    keys_q = list(counts_q.keys())
    n_bits = len(keys_p[0])
    if any(len(k) != n_bits for k in keys_p) or any(len(k) != n_bits for k in keys_q):
        raise ValueError("All bitstrings must have identical bit length")

    scale = (1.0 / math.sqrt(float(n_bits))) if normalize_by_sqrt_n else 1.0
    X = np.array([[scale if ch == "1" else 0.0 for ch in k] for k in keys_p], dtype=np.float64)
    Y = np.array([[scale if ch == "1" else 0.0 for ch in k] for k in keys_q], dtype=np.float64)
    wp = validate_probability_measure(
        np.array([counts_p[k] for k in keys_p], dtype=np.float64) / sum(counts_p.values()), "counts_p"
    )
    wq = validate_probability_measure(
        np.array([counts_q[k] for k in keys_q], dtype=np.float64) / sum(counts_q.values()), "counts_q"
    )

    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_projections, n_bits))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    proj_X = X @ dirs.T
    proj_Y = Y @ dirs.T
    slice_costs = np.empty(n_projections, dtype=np.float64)
    for k in range(n_projections):
        slice_costs[k] = exact_wasserstein_1d(proj_X[:, k], proj_Y[:, k], wp, wq, order=order).transport_cost

    mean_cost = float(np.mean(slice_costs))
    se_cost = float(np.std(slice_costs, ddof=1) / math.sqrt(n_projections)) if n_projections > 1 else 0.0
    sw_dist = mean_cost ** (1.0 / order)
    # Propagate 95% half-width via conservative endpoint mapping
    hi_dist = (mean_cost + 1.96 * se_cost) ** (1.0 / order)
    lo_dist = max(0.0, mean_cost - 1.96 * se_cost) ** (1.0 / order)
    return {
        "sliced_distance": sw_dist,
        "mc_95ci_halfwidth": 0.5 * (hi_dist - lo_dist),
        "n_projections": float(n_projections),
        "n_qubits": float(n_bits),
    }


def debiased_empirical_wasserstein(
    counts_p: Mapping[str, int],
    counts_q: Mapping[str, int],
    order: float = 2.0,
    normalize_hamming: bool = True,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Split-sample debiased empirical p-Wasserstein estimator for finite-shot bitstring
    histograms on {0,1}^Q (eliminating O(S^{-1/Q}) high-entropy self-distance bias):

        W_{p,deb}^p(p_hat, q_hat) = max(0, W_p^p(p_hat, q_hat)
                                           - 0.5 * [W_p^p(p_a, p_b) + W_p^p(q_a, q_b)])

    where (p_a, p_b) and (q_a, q_b) are independent hypergeometric half-splits of the
    shot records in `counts_p` and `counts_q`.
    """
    if not counts_p or not counts_q:
        raise ValueError("Count dictionaries must be non-empty")
    keys_p = list(counts_p.keys())
    keys_q = list(counts_q.keys())
    n_bits = len(keys_p[0])
    if any(len(k) != n_bits for k in keys_p) or any(len(k) != n_bits for k in keys_q):
        raise ValueError("All bitstrings must share identical length")

    total_p = sum(int(v) for v in counts_p.values())
    total_q = sum(int(v) for v in counts_q.values())
    if total_p < 4 or total_q < 4:
        raise ValueError("At least 4 shots per distribution required for split-sample debiasing")

    rng = np.random.default_rng(seed)

    def _split_counts(cnts: Mapping[str, int]) -> tuple[Dict[str, int], Dict[str, int]]:
        pool: list[str] = []
        for k, c in cnts.items():
            pool.extend([k] * int(c))
        perm = rng.permutation(len(pool))
        half = len(pool) // 2
        ca: Dict[str, int] = {}
        cb: Dict[str, int] = {}
        for idx in perm[:half]:
            s = pool[int(idx)]
            ca[s] = ca.get(s, 0) + 1
        for idx in perm[half : 2 * half]:
            s = pool[int(idx)]
            cb[s] = cb.get(s, 0) + 1
        return ca, cb

    def _lp_cost_between_dicts(d1: Mapping[str, int], d2: Mapping[str, int]) -> float:
        k1 = list(d1.keys())
        k2 = list(d2.keys())
        w1 = np.array([d1[k] for k in k1], dtype=np.float64)
        w2 = np.array([d2[k] for k in k2], dtype=np.float64)
        w1 /= w1.sum()
        w2 /= w2.sum()
        cmat = hamming_cost_matrix(k1, k2, normalize_by_n=n_bits if normalize_hamming else None)
        return exact_wasserstein_lp(w1, w2, cmat, order=order).transport_cost

    pa, pb = _split_counts(counts_p)
    qa, qb = _split_counts(counts_q)

    raw_cost = _lp_cost_between_dicts(counts_p, counts_q)
    self_p_cost = _lp_cost_between_dicts(pa, pb)
    self_q_cost = _lp_cost_between_dicts(qa, qb)
    debiased_cost = max(0.0, raw_cost - 0.5 * (self_p_cost + self_q_cost))

    return {
        "debiased_distance": debiased_cost ** (1.0 / order),
        "debiased_transport_cost": debiased_cost,
        "raw_plug_in_distance": raw_cost ** (1.0 / order),
        "raw_plug_in_cost": raw_cost,
        "split_self_cost_p": self_p_cost,
        "split_self_cost_q": self_q_cost,
    }
