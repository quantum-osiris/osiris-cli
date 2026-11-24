"""
osiris_qscore — scoring for decoded IBM Quantum counts.

Conventions
-----------
Qiskit order everywhere: the rightmost character is qubit/clbit 0, for both
bitstrings and basis labels. "XYZZ" means qubit 0 and 1 measured in Z,
qubit 2 in Y, qubit 3 in X. Outcome bit 0 -> eigenvalue +1, bit 1 -> -1, in
every basis (X measured via H, Y via Sdg then H, before a Z measurement).

Finite-shot entropy ceilings
----------------------------
For an ``n``-bit register sampled ``N`` times, plug-in Shannon entropy cannot
exceed ``min(n, log2(N))`` bits. Comparing ``H`` against ``n`` when ``N < 2**n``
confuses finite sample size with physical entropy suppression (see claims
``NEGENTROPY_136`` and ``ENTROPY_SUPPRESSION_Z29``). ``entropy_ceiling`` and
``score_entropy`` enforce the finite-shot bound and flag ``SATURATED_ENTROPY``
and ``ALL_ZERO`` / ``UNMEASURED_REGISTER`` registers prior to downstream analysis.

Population is not fidelity
--------------------------
ghz_population() is P(0...0) + P(1...1) from computational-basis counts. It is
an upper bound on GHZ fidelity, not the fidelity: a classical 50/50 mixture of
|0...0> and |1...1> has population 1.0 and fidelity 0.5. Fidelity also needs
the coherence between |0...0> and |1...1>, which only X/Y settings measure.

ghz_fidelity() computes the fidelity exactly from Pauli measurement settings
using the GHZ stabilizer expansion, and refuses to return a number when the
settings needed for any term are missing.

Uncertainties are exact standard errors of the linear estimator (no MLE, no
bootstrap): settings are independent, and within a setting all terms are
evaluated on the same shots, so correlations are accounted for.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

Counts = Mapping[str, int]

SATURATION_RATIO_THRESHOLD = 0.98
SATURATED_ENTROPY = "SATURATED_ENTROPY"
ALL_ZERO = "ALL_ZERO"
UNMEASURED_REGISTER = "UNMEASURED_REGISTER"


@dataclass(frozen=True)
class Estimate:
    value: float
    stderr: float
    shots: int

    def sigma_above(self, threshold: float) -> float:
        """How many standard errors the value sits above threshold."""
        if self.stderr == 0:
            return math.inf if self.value > threshold else (0.0 if self.value == threshold else -math.inf)
        return (self.value - threshold) / self.stderr

    def __str__(self) -> str:
        return f"{self.value:.4f} ± {self.stderr:.4f} (N={self.shots})"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _width(counts: Counts) -> int:
    widths = {len(k) for k in counts}
    if len(widths) != 1:
        raise ValueError(f"bitstrings have mixed widths {sorted(widths)}")
    w = widths.pop()
    for k in counts:
        if set(k) - {"0", "1"}:
            raise ValueError(f"not a bitstring: {k!r}")
    return w


def _shots(counts: Counts) -> int:
    n = sum(counts.values())
    if n <= 0:
        raise ValueError("no shots in counts")
    return n


def _bits_matrix(counts: Counts) -> tuple[np.ndarray, np.ndarray]:
    """(K, n) boolean matrix in string positions, and (K,) weights."""
    keys = list(counts)
    bits = np.array([[c == "1" for c in k] for k in keys], dtype=bool)
    weights = np.array([counts[k] for k in keys], dtype=float)
    return bits, weights


def _pos(qubit: int, n: int) -> int:
    """String position of a qubit index in Qiskit order."""
    if not 0 <= qubit < n:
        raise ValueError(f"qubit {qubit} out of range for {n} bits")
    return n - 1 - qubit


# --------------------------------------------------------------------------
# Finite-shot Shannon entropy & circuit-distribution scoring
# --------------------------------------------------------------------------

def entropy_ceiling(num_bits: int, shots: int) -> float:
    """Maximum observable plug-in Shannon entropy (bits) for ``num_bits`` and ``shots``."""
    if num_bits < 0 or shots < 0:
        raise ValueError(f"num_bits ({num_bits}) and shots ({shots}) must be non-negative")
    if num_bits == 0 or shots <= 1:
        return 0.0
    return min(float(num_bits), float(math.log2(shots)))


def shannon_entropy(counts: Counts) -> float:
    """Plug-in Shannon entropy ``-sum p_x log2(p_x)`` in bits."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0))


def miller_madow_entropy(counts: Counts, num_bits: int | None = None) -> float:
    """First-order Miller-Madow bias-corrected Shannon entropy (clipped at ``num_bits``)."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = shannon_entropy(counts)
    nonzero_bins = sum(1 for c in counts.values() if c > 0)
    corrected = h + (nonzero_bins - 1) / (2.0 * total * math.log(2.0))
    if num_bits is not None and num_bits >= 0:
        return float(min(float(num_bits), corrected))
    return float(corrected)


def entropy_plug_in_stderr(counts: Counts) -> float:
    """Delta-method plug-in standard error of ``shannon_entropy`` (in bits)."""
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    h = -sum(p * math.log2(p) for p in probs)
    second_moment = sum(p * (math.log2(p) ** 2) for p in probs)
    var_h = max(0.0, second_moment - h * h) / total
    return float(math.sqrt(var_h))


def distribution_tvd(counts: Counts, reference_probs: Mapping[str, float]) -> float:
    """Total variation distance between empirical ``counts`` and ``reference_probs``."""
    total = _shots(counts)
    keys = set(counts) | set(reference_probs)
    return float(0.5 * sum(abs(counts.get(k, 0) / total - reference_probs.get(k, 0.0)) for k in keys))


def distribution_kl(counts: Counts, reference_probs: Mapping[str, float], smoothing: float = 1e-12) -> float:
    """KL divergence ``D_KL(P_emp || P_ref)`` in bits with floor ``smoothing`` on ``P_ref``."""
    total = _shots(counts)
    kl = 0.0
    for k, c in counts.items():
        if c <= 0:
            continue
        p = c / total
        q = max(float(reference_probs.get(k, 0.0)), smoothing)
        kl += p * math.log2(p / q)
    return float(max(0.0, kl))


@dataclass(frozen=True)
class EntropyScore:
    num_bits: int
    shots: int
    unique_outcomes: int
    shannon_entropy: float
    stderr: float
    miller_madow_entropy: float
    ceiling: float
    ceiling_ratio: float
    flags: tuple[str, ...]


def score_entropy(
    counts: Counts,
    num_bits: int | None = None,
    saturation_threshold: float = SATURATION_RATIO_THRESHOLD,
) -> EntropyScore:
    """Compute finite-shot-normalized Shannon entropy and register health flags."""
    nonzero = {k: int(v) for k, v in counts.items() if v > 0}
    if not nonzero:
        w = int(num_bits or 0)
        return EntropyScore(
            num_bits=w, shots=0, unique_outcomes=0, shannon_entropy=0.0,
            stderr=0.0, miller_madow_entropy=0.0, ceiling=0.0, ceiling_ratio=0.0,
            flags=(UNMEASURED_REGISTER,),
        )
    inferred_w = _width(nonzero)
    w = inferred_w if num_bits is None else int(num_bits)
    if w != inferred_w:
        raise ValueError(f"num_bits={w} does not match bitstring width {inferred_w}")
    total = sum(nonzero.values())
    ceil_h = entropy_ceiling(w, total)
    h = shannon_entropy(nonzero)
    se = entropy_plug_in_stderr(nonzero)
    mm = miller_madow_entropy(nonzero, num_bits=w)
    ratio = (h / ceil_h) if ceil_h > 0 else 0.0

    flags: list[str] = []
    if len(nonzero) == 1 and set(next(iter(nonzero))) == {"0"}:
        flags.extend([ALL_ZERO, UNMEASURED_REGISTER])
    elif ceil_h > 0 and ratio >= saturation_threshold:
        flags.append(SATURATED_ENTROPY)

    return EntropyScore(
        num_bits=w,
        shots=total,
        unique_outcomes=len(nonzero),
        shannon_entropy=h,
        stderr=se,
        miller_madow_entropy=mm,
        ceiling=ceil_h,
        ceiling_ratio=ratio,
        flags=tuple(flags),
    )


# --------------------------------------------------------------------------
# Computational-basis quantities
# --------------------------------------------------------------------------

def population(counts: Counts, outcomes: Iterable[str]) -> Estimate:
    """Fraction of shots landing in `outcomes`, over the true shot count."""
    n = _width(counts)
    outcomes = list(outcomes)
    for o in outcomes:
        if len(o) != n:
            raise ValueError(f"outcome {o!r} has width {len(o)}, counts have width {n}")
    N = _shots(counts)
    p = sum(counts.get(o, 0) for o in set(outcomes)) / N
    return Estimate(p, math.sqrt(p * (1 - p) / N), N)


def ghz_population(counts: Counts) -> Estimate:
    """P(0...0) + P(1...1). An upper bound on GHZ fidelity, not the fidelity."""
    n = _width(counts)
    return population(counts, ["0" * n, "1" * n])


def pauli_expectation(counts: Counts, qubits: Iterable[int]) -> Estimate:
    """<P> for a Pauli acting on `qubits`, from counts taken in P's eigenbasis.

    The caller is responsible for the counts coming from a setting that
    measured each of `qubits` in the right basis.
    """
    n = _width(counts)
    pos = [_pos(q, n) for q in qubits]
    bits, w = _bits_matrix(counts)
    eig = 1 - 2 * (bits[:, pos].sum(axis=1) % 2)
    N = w.sum()
    mean = float((w * eig).sum() / N)
    var = max(0.0, 1 - mean**2)
    return Estimate(mean, math.sqrt(var / N), int(N))


# --------------------------------------------------------------------------
# GHZ fidelity from Pauli settings
# --------------------------------------------------------------------------

def ghz_stabilizer_terms(n: int) -> list[tuple[str, int]]:
    """|GHZ_n><GHZ_n| = 2^-n * sum(coeff * P) over the returned (label, coeff).

    Z-type terms: Z on every even-size subset (identity included), coeff +1.
    X/Y-type terms: X on every qubit except Y on an even-size subset T,
    coeff (-1)^(|T|/2). Labels are in Qiskit order.
    """
    if n < 2:
        raise ValueError("GHZ needs at least 2 qubits")
    terms = []
    for k in range(0, n + 1, 2):
        for subset in itertools.combinations(range(n), k):
            terms.append(("".join("Z" if i in subset else "I" for i in range(n)), 1))
    for k in range(0, n + 1, 2):
        for subset in itertools.combinations(range(n), k):
            terms.append(("".join("Y" if i in subset else "X" for i in range(n)), (-1) ** (k // 2)))
    return terms


@dataclass(frozen=True)
class GHZFidelity:
    fidelity: Estimate
    population: float      # P(0...0) + P(1...1) implied by the Z-type terms
    coherence: float       # 2 Re rho_{0...0,1...1} implied by the X/Y-type terms
    settings_used: int
    n_qubits: int

    @property
    def witness_sigma(self) -> float:
        """Standard errors above 0.5. F > 0.5 witnesses genuine multipartite entanglement."""
        return self.fidelity.sigma_above(0.5)

    def __str__(self) -> str:
        return (f"GHZ-{self.n_qubits} fidelity {self.fidelity}  "
                f"[population {self.population:.4f}, coherence {self.coherence:.4f}, "
                f"{self.settings_used} settings, {self.witness_sigma:.2f}σ above 0.5]")


def ghz_fidelity(settings: Mapping[str, Counts]) -> GHZFidelity:
    """Linear-inversion GHZ fidelity from Pauli measurement settings.

    settings: {basis_label: counts}, e.g. {"ZZZZ": {...}, "XXXX": {...}, ...},
    labels in Qiskit order, characters from "XYZ". Full tomography (3^n settings)
    works; so does the minimal set (all-Z plus the X/Y settings with an even
    number of Y). Each Pauli term is averaged over every setting that measures it.

    Raises ValueError if any stabilizer term is not covered by a setting.
    """
    if not settings:
        raise ValueError("no settings supplied")
    labels = list(settings)
    n = len(labels[0])
    for lbl in labels:
        if len(lbl) != n or set(lbl) - set("XYZ"):
            raise ValueError(f"bad basis label {lbl!r}: need {n} characters from X, Y, Z")
        if _width(settings[lbl]) != n:
            raise ValueError(f"counts for {lbl!r} are not {n} bits wide")

    dim = 2**n
    terms = [(lbl, c) for lbl, c in ghz_stabilizer_terms(n) if set(lbl) != {"I"}]

    def covers(term: str, setting: str) -> bool:
        return all(p == "I" or p == s for p, s in zip(term, setting))

    compat = {t: [s for s in labels if covers(t, s)] for t, _ in terms}
    missing = [t for t, ss in compat.items() if not ss]
    if missing:
        n_xy = sum(1 for t in missing if "X" in t or "Y" in t)
        raise ValueError(
            f"cannot compute GHZ fidelity: {len(missing)} of {len(terms)} stabilizer terms are not "
            f"measured by any setting ({n_xy} of them X/Y terms), e.g. {missing[:4]}. "
            "Computational-basis data alone gives the population, not the fidelity."
        )

    # Per-setting, per-shot score g_s(b); F = 1/dim + sum_s mean_s(g_s).
    plan: dict[str, list[tuple[list[int], float, bool]]] = defaultdict(list)
    for t, c in terms:
        share = c / (dim * len(compat[t]))
        support = [i for i, p in enumerate(t) if p != "I"]
        for s in compat[t]:
            plan[s].append((support, share, "Z" in t))

    F, var, zpart, shots_used = 1.0 / dim, 0.0, 1.0 / dim, 0
    for s, items in plan.items():
        bits, w = _bits_matrix(settings[s])
        N = w.sum()
        g = np.zeros(len(w))
        gz = np.zeros(len(w))
        for support, share, is_z in items:
            eig = 1 - 2 * (bits[:, support].sum(axis=1) % 2)
            g += share * eig
            if is_z:
                gz += share * eig
        mean = float((w * g).sum() / N)
        F += mean
        zpart += float((w * gz).sum() / N)
        if N > 1:
            var += float((w * (g - mean) ** 2).sum() / (N - 1)) / N
        shots_used += int(N)

    xypart = F - zpart
    return GHZFidelity(
        fidelity=Estimate(F, math.sqrt(var), shots_used),
        population=2 * zpart,
        coherence=2 * xypart,
        settings_used=len(plan),
        n_qubits=n,
    )


def tomography_labels(n: int) -> list[str]:
    """All 3^n Pauli settings in itertools.product("XYZ") order.

    This is only a convenience. Which PUB holds which setting depends on how
    the circuits were generated: confirm the mapping from the generating script
    or the circuit metadata before scoring. A wrong mapping gives a wrong number
    without any error.
    """
    return ["".join(p) for p in itertools.product("XYZ", repeat=n)]
