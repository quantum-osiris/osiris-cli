"""Three candidate samplers for the bivariate-bicycle (BB) family, plus exact GF(2) algebra.

Mathematics (Bravyi et al. 2024, "High-threshold and low-overhead fault-tolerant quantum memory", the BB
construction; derivation re-checked numerically in tests, not assumed): with A, B commuting circulant blocks of
the group algebra R = F2[x,y]/(x^l - 1, y^m - 1), HX = [A | B], HZ = [B^T | A^T], n = 2*l*m and

    k = n - rank(HX) - rank(HZ) = 2*(l*m - rank([A | B]))  = 2 * dim(ker A^T  intersect  ker B^T).

So k > 0 exactly when A^T and B^T have a common nonzero kernel vector, i.e. when A and B share a common
zero-divisor direction. A trinomial A has an odd number of terms, so A(1,1) = 1 and the all-ones vector is never
in a kernel; a nonzero kernel needs a common root of A and B at (alpha, beta) with alpha^l = beta^m = 1 in some
extension field. For two independently uniform trinomials that is rare, which is why the uniform sampler gives
k = 0 most of the time.

Samplers (family, n range, weight and constraints identical; only the proposal distribution differs):
  A  uniform      : the original sampler (qldpc_sweep.sample_specs).
  B  rank-aware   : same uniform proposals, accepted only if the exact k > 0 (rejection on exact rank).
  C  structure    : per (l,m), precompute the trinomials whose own kernel is non-zero (necessary for k>0), propose
                    A and B only from that set, and accept only if the exact k > 0.
Both B and C verify k exactly by GF(2) elimination; no code is accepted on a heuristic.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, Iterable, List, Sequence, Tuple

from osiris_cli import qldpc_sweep as sw

Term = Tuple[int, int]


def _row_bits(ell: int, m: int, terms: Sequence[Term], i: int, j: int) -> int:
    v = 0
    for a, b in terms:
        v |= 1 << (((i + a) % ell) * m + (j + b) % m)
    return v


def circulant_rows(ell: int, m: int, terms: Sequence[Term]) -> List[int]:
    return [_row_bits(ell, m, terms, i, j) for i in range(ell) for j in range(m)]


def gf2_rank(rows: Iterable[int]) -> int:
    """Exact rank over GF(2) of bit-packed rows (Gaussian elimination)."""
    piv: Dict[int, int] = {}
    r = 0
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h]
            else:
                piv[h] = v
                r += 1
                break
    return r


def k_of(ell: int, m: int, A: Sequence[Term], B: Sequence[Term]) -> int:
    """k = 2*(l*m - rank([A|B])), exact."""
    N = ell * m
    ra, rb = circulant_rows(ell, m, A), circulant_rows(ell, m, B)
    return 2 * (N - gf2_rank((a | (b << N)) for a, b in zip(ra, rb)))


def kernel_dim(ell: int, m: int, terms: Sequence[Term]) -> int:
    return ell * m - gf2_rank(circulant_rows(ell, m, terms))


def zero_divisor_trinomials(ell: int, m: int) -> List[Tuple[Term, Term, Term]]:
    """All trinomials containing (0,0), up to translation, with non-zero kernel (exact)."""
    cells = [(i, j) for i in range(ell) for j in range(m) if (i, j) != (0, 0)]
    out = []
    for p, q in itertools.combinations(cells, 2):
        t = ((0, 0), p, q)
        if kernel_dim(ell, m, t) > 0:
            out.append(t)
    return out


class UniformSampler:
    name = "A_uniform"

    def __init__(self, seed: int, n_range=(60, 126)):
        self.rng = random.Random(seed)
        self.lms = [(a, b) for a in range(2, 64) for b in range(a, 64) if n_range[0] <= 2 * a * b <= n_range[1]]
        self.proposals = 0

    def _propose(self):
        ell, m = self.rng.choice(self.lms)
        cells = [(i, j) for i in range(ell) for j in range(m)]
        self.proposals += 1
        return ell, m, self.rng.sample(cells, 3), self.rng.sample(cells, 3)

    def next(self):
        return self._propose()


class RankAwareSampler(UniformSampler):
    name = "B_rank_aware"

    def next(self):
        while True:
            ell, m, A, B = self._propose()
            if k_of(ell, m, A, B) > 0:
                return ell, m, A, B


class StructureSampler(UniformSampler):
    name = "C_structure_guided"
    _cache: Dict[Tuple[int, int], list] = {}

    def __init__(self, seed: int, n_range=(60, 126)):
        super().__init__(seed, n_range)
        self.usable = [lm for lm in self.lms if self._zd(*lm)]

    @classmethod
    def _zd(cls, ell, m):
        if (ell, m) not in cls._cache:
            cls._cache[(ell, m)] = zero_divisor_trinomials(ell, m)
        return cls._cache[(ell, m)]

    def next(self):
        while True:
            ell, m = self.rng.choice(self.usable)
            zd = self._zd(ell, m)
            self.proposals += 1
            sh = lambda t: [((a + self.rng.randrange(ell)) % ell, (b + self.rng.randrange(m)) % m) for a, b in t]  # noqa: E731
            A, B = sh(self.rng.choice(zd)), sh(self.rng.choice(zd))
            if k_of(ell, m, A, B) > 0:
                return ell, m, A, B


SAMPLERS = (UniformSampler, RankAwareSampler, StructureSampler)


def canonical(ell, m, A, B):
    return sw.canonical_key(ell, m, A, B)
