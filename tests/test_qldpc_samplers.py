import os
import random
import pytest
from osiris_cli import qldpc_samplers as qs

REPO = "/home/enki/github-backup/qldpc-challenge"


def test_gf2_rank_basic():
    assert qs.gf2_rank([0b11, 0b01, 0b10]) == 2 and qs.gf2_rank([0, 0]) == 0


def test_trinomials_never_have_all_ones_in_kernel_and_k_is_even():
    rng = random.Random(5)
    for _ in range(40):
        ell, m = rng.choice([(3, 4), (4, 6), (6, 6), (2, 9)])
        cells = [(i, j) for i in range(ell) for j in range(m)]
        A, B = rng.sample(cells, 3), rng.sample(cells, 3)
        assert qs.k_of(ell, m, A, B) % 2 == 0
        ones = (1 << (ell * m)) - 1
        # A*ones = (#terms mod 2) * ones = ones, so ones is not in ker A
        assert all(bin(r & ones).count("1") % 2 == 1 for r in qs.circulant_rows(ell, m, A))


@pytest.mark.skipif(not os.path.isdir(REPO + "/verify"), reason="challenge repo absent")
def test_k_matches_repos_compute_k():
    import sys
    sys.dont_write_bytecode = True
    sys.path[:0] = [REPO + "/research/kit", REPO + "/verify"]
    import bb
    import css
    rng = random.Random(9)
    for _ in range(15):
        ell, m = rng.choice([(3, 5), (4, 6), (6, 6), (3, 10)])
        cells = [(i, j) for i in range(ell) for j in range(m)]
        A, B = rng.sample(cells, 3), rng.sample(cells, 3)
        HX, HZ = bb.build_bb(ell, m, A, B)
        assert css.compute_k(HX, HZ) == qs.k_of(ell, m, A, B)


def test_samplers_deterministic_and_b_c_always_k_positive():
    for cls in (qs.UniformSampler, qs.RankAwareSampler, qs.StructureSampler):
        a = [cls(7).next() for _ in range(1)]
        assert a == [cls(7).next() for _ in range(1)]
    for cls in (qs.RankAwareSampler, qs.StructureSampler):
        s = cls(3)
        for _ in range(5):
            ell, m, A, B = s.next()
            assert qs.k_of(ell, m, A, B) > 0 and 60 <= 2 * ell * m <= 126
