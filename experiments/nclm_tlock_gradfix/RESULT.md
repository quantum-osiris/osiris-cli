# T-Lock backward-pass correction (osiris-cli be8f506d, parent 968b7516)

**Finding.** Backward-pass correction supported by numerical checks on the tested cases; forward outputs
unchanged on those cases. No improvement in training performance is claimed or shown.

**Change.** `osiris/nclm/sovereign_mechanics.py`, `TorsionLockedAttention.__call__`: `diag(R)` was copied out of
`scores.data` (a constant in the backward pass); it is now `scores * I` with a constant identity mask.

**Gradient check** (`validate.py`, docstring gives the definition and acceptance rule; `results/*.json` hold every
tensor at all six perturbation sizes; `results/SUMMARY.txt` is the table). float32, dropout 0, three inputs per
architecture, no independent autodiff reference.
- crsm, pre-fix: 41, 37 and 39 of 80 tensors fail (no perturbation size within 2 %); best-case error up to 11.6-15.7 %.
- crsm, fixed: 0 of 80 fail; every judged tensor is within 0.23 % at its best perturbation size. No single size
  suits all tensors: a few large-norm projection weights agree within 2 % at only one or two sizes (curvature at
  large steps, float32 noise at small ones).
- standard: 0 of 56 fail in both revisions, identical numbers.
- Zero gradients: none met the 1e-10 threshold. No gradient at all: the four `phase_conj.zero_point` tensors
  (declared, never used in the forward pass) in both revisions. That is a separate open issue; not changed here.

**Forward.** Logits (float32) and loss for the 3 inputs x 2 architectures are `np.array_equal` between the two
revisions: 12 of 12 arrays. Tolerance 0; observed maximum difference 0. Those cases only.

**Training** (`training/`, exploratory, not the registered NCLM-ARCH-1 protocol; `TABLE.txt`). crsm, batch 8,
window 128, 1,200 steps, AdamW 3e-4 with the console's schedule, gradient clip 1.0, seeds 11-13, same corpus
and hash-of-path split for both arms. Best = final in every run. Mean change after - before: +0.020 bits/byte
(worse); one of three seeds better. Not recorded: training loss, gradient norms, non-finite checks; no
checkpoints were saved. A first attempt (2,000 steps, six processes at once) was killed by the kernel for
memory and produced no results; it is a failed run, not rerun at that size.
