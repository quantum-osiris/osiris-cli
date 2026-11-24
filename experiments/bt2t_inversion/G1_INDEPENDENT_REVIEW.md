# G1 Independent Review Sign-Off — `OSIRIS-BT2T-001` (`bt2t-enc-1`)

- **Queue Item:** `phase1/verification_queue.md` Check #2 (`BT2T-2`, Gate `G1 -> G2` & Hardware Pre-Gate `(a)`)
- **Original Author Session:** Claude Code (`2026-11-22`, commit `e179cfb1` file digest)
- **Independent Reviewer:** Antigravity Scientific Rigor Auditor (`google-deepmind/antigravity`, separate agent family & clean-room verification harness)
- **Review Timestamp (UTC):** `2026-11-24T01:14:06Z`
- **Audited Source Digest (`experiments/bt2t_inversion/bt2t.py`):** `e179cfb1cd431615a75b2cf30a389d5ba1d406c289b394919b23e5c594117dac`
- **Frozen Encoding Digest (`bt2t-enc-1`):** `3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5`
- **Verdict:** **PASS — GATE G1 SIGNED OFF INDEPENDENTLY**

---

## 1. Clean-Room Algebraic Rederivation (`SU(2)` Matrix Oracle)

Without calling `bt2t.qmul`, `bt2t.multiplication_table`, or `bt2t.conjugate`, we independently constructed the 24 elements of the binary tetrahedral group $2T \subset SU(2)$ directly from the prose encoding specification (`bt2t.py` lines 14–23) via the faithful $2\times 2$ complex matrix representation
$$
\mathbf{1} \mapsto I_2,\quad
\mathbf{i} \mapsto -i\sigma_x,\quad
\mathbf{j} \mapsto -i\sigma_y,\quad
\mathbf{k} \mapsto -i\sigma_z,
$$
where $(-i\sigma_x)(-i\sigma_y) = -i\sigma_z = \mathbf{k}$ and $(-i\sigma_x)^2 = (-i\sigma_y)^2 = (-i\sigma_z)^2 = -I_2$.

1. **Unitarity & Determinant Check (`0..23`):** All 24 matrices satisfy $U_x U_x^\dagger = I_2$ ($\|\cdot\|_\infty < 10^{-14}$) and $\det(U_x) = +1$ ($|\det(U_x)-1| < 10^{-14}$), confirming all 24 encoded elements are distinct unit Hurwitz quaternions in $SU(2)$.
2. **Adjoint Inversion Check (`0..23`) & Unused Extension (`24..31`):** Computing $U_x^{-1} = U_x^\dagger$ directly in $SU(2)$ yields a unique index $\pi_{\text{SU(2)}}(x) \in \{0,\dots,23\}$ for every $x \in \{0,\dots,23\}$. Extending by the identity on unused codewords $x \in \{24,\dots,31\}$ reproduces `bt2t.inversion_permutation()` on **32 / 32** computational basis states (`su2_adjoint_matches_bt2t_perm_all_32 = True`), with fixed-point set $\{0, 1\} \cup \{24,\dots,31\}$ (10 fixed points and 11 disjoint transpositions).

---

## 2. Independent Boolean Derivation of `inversion_circuit_structured`

Let $|x\rangle = |b_4 b_3 b_2 b_1 b_0\rangle$ in Qiskit little-endian order ($x = \sum_{q=0}^4 b_q 2^q$):

1. **Sector Partition by $(b_4, b_3)$:**
   - $(b_4, b_3) = (0, 0)$ ($x \in \{0,\dots,7\}$): $Q_8 = \{\pm 1, \pm i, \pm j, \pm k\}$, with $b_4 \oplus b_3 = 0$.
   - $(b_4, b_3) \in \{(0, 1), (1, 0)\}$ ($x \in \{8,\dots,23\}$): Notice $x - 8 = 16 b_4 + 8 b_3 + 4 b_2 + 2 b_1 + b_0 - 8$.
     - When $(b_4, b_3) = (0, 1)$ ($x \in 8..15$), $n = x - 8 \in 0..7$ has bit 3 equal to $0$ ($s_a = +1$).
     - When $(b_4, b_3) = (1, 0)$ ($x \in 16..23$), $n = x - 8 \in 8..15$ has bit 3 equal to $1$ ($s_a = -1$).
     - In **both** cases, $(b_2, b_1, b_0)$ encode the imaginary sign bits $(s_b, s_c, s_d)$, and $b_4 \oplus b_3 = 1$!
   - $(b_4, b_3) = (1, 1)$ ($x \in \{24,\dots,31\}$): Unused codewords, with $b_4 \oplus b_3 = 0$.

2. **Hurwitz Imaginary Sign Flip (`CX(3->t), CX(4->t)` for `t in {0, 1, 2}`):**
   - Applies $b_t \leftarrow b_t \oplus b_3 \oplus b_4$ for $t \in \{0, 1, 2\}$.
   - On $x \in 8..23$ ($b_3 \oplus b_4 = 1$): preserves $(b_4, b_3)$ ($s_a$ unchanged) and complements $(b_2, b_1, b_0)$ ($(s_b, s_c, s_d) \mapsto (-s_b, -s_c, -s_d)$), implementing exact quaternion conjugation $(a, b, c, d)/2 \mapsto (a, -b, -c, -d)/2$.
   - On $Q_8$ ($0..7$) and unused codewords ($24..31$): $b_3 \oplus b_4 = 0$, so the 6 CNOTs act as the exact identity.

3. **$Q_8$ Inversion (`CCX` + 4-control `MCX` with $X$ Conjugations):**
   - Subsequent gates flip $b_0$ iff $(\neg b_3 \land \neg b_4) \oplus (\neg b_1 \land \neg b_2 \land \neg b_3 \land \neg b_4) = (\neg b_3 \land \neg b_4) \land (b_1 \lor b_2)$.
   - On $Q_8$ ($b_3 = b_4 = 0$): for $x \in \{0, 1\}$ ($\pm 1$, where $b_1 = b_2 = 0$), both controls fire and cancel ($b_0$ unchanged); for $x \in \{2,\dots,7\}$ ($\pm i, \pm j, \pm k$, where $b_1 \lor b_2 = 1$), only the `CCX` fires, flipping $b_0$ ($2 \leftrightarrow 3$, $4 \leftrightarrow 5$, $6 \leftrightarrow 7$).
   - On Hurwitz ($8..23$) and unused ($24..31$): $\neg b_3 \land \neg b_4 = 0$, so neither `CCX` nor `MCX` fires.

4. **Exact $32 \times 32$ Complex Unitary Matrix Verification (Including Phases):**
   - Both `inversion_circuit()` and `inversion_circuit_structured()` consist purely of computational-basis permutation gates (`X`, `CX`, `CCX`, `MCX`) with zero relative phase drift:
     - $\|\text{Operator}(\text{generic}) - P_{32}\|_\infty = 3.475 \times 10^{-14} < 10^{-9}$
     - $\|\text{Operator}(\text{structured}) - P_{32}\|_\infty = 3.553 \times 10^{-15} < 10^{-9}$
   - On `HEAVY_HEX_5Q` (`optimization_level=3, seed_transpiler=20261122`), `Operator.from_circuit(t)` (accounting for SABRE initial and final index layouts) matches $P_{32}$ up to global phase at `72 CZ` (`structured`) vs `723 CZ` (`generic`), and all 32 basis states map deterministically (`100.0%`, error `< 10^{-11}`) when prepared on `initial_index_layout` and decoded via `final_index_layout`.

---

## 3. Sign-Off Decision

All algebraic, phase, unused-codeword, and layout-aware compilation checks pass without defect. Gate **G1** (`Check #2` in `phase1/verification_queue.md`) is formally signed off.
