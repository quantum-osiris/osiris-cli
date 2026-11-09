# Frozen protocol: BB sampler comparison (written and committed BEFORE the benchmark is run)

Date 2026-11-09. Baseline: archived campaign osiris-bb-w6-unrestricted-n60-126-20261109 (unchanged; 400 sampled,
376 with k=0, 24 with k>0, 3 gate acceptances, 0 board advances).

## Root cause (derived, then checked)
For BB codes k = n - rank(HX) - rank(HZ) = 2*(l*m - rank[A|B]) = 2*dim(ker A^T ∩ ker B^T). Verified: this formula
reproduces the archived k for 400/400 candidates and matches the challenge repo's css.compute_k on 60/60 sampled.
k>0 needs A and B to share a kernel direction. Measured on the archive: A zero-divisor 99/400, B zero-divisor 123/400,
both 55/400, k>0 24/400. So the k=0 rate (94%) is the product of "both must be zero-divisors" (13.75%) and "their
kernels must intersect" (24 of 55), not a construction bug. Trinomials have odd weight so A(1,1)=1: the trivial
all-ones direction never helps.

## Samplers (qldpc_samplers.py). Family, 2lm in [60,126], l<=m, 3-term A and B, weight 6: unchanged.
A uniform (original distribution); B rank-aware (uniform proposals, exact-k>0 rejection); C structure-guided
(A,B proposed only from trinomials with non-zero kernel, exact-k>0 acceptance). Every accepted code has k verified
by exact GF(2) elimination; none is accepted on a heuristic. Samplers do not use the archive and have no tuned
parameters (nothing is fit, so there is no tuning data to leak).

## Design
5 replicates per sampler, independent seeds: A 1101-1105, B 1201-1205, C 1301-1305 (none used before; the earlier
campaign used sampler seed 101 and screening seeds 101/211/307). Each replicate emits 400 candidates (matched
OUTPUT budget; proposals and CPU are measured, not matched). Per replicate surrogate-screen (200 trials, seeds 101,
211, 307; min taken) the first 40 k>0 candidates in generation order. Sequential, one process, RAM guard 1.5 GB.
Structure sampler's one-time zero-divisor tables are timed separately and reported both separately and amortised.

## Measures (descriptive; no significance claim with n=5)
valid-construction rate (CSS commutation + weight 6 + n in range, checked on the repo's build_bb for a random 20 per
replicate); fraction k>0 among proposals; distribution of k; unique canonical structures per replicate and
overlap with the 400 archived structures; proposals per accepted candidate; process CPU seconds per unique k>0
candidate; surrogate min-upper-bound distribution on the screened subset; within-replicate duplicate rate.

## Decision rule (fixed now)
A sampler is called MORE EFFICIENT at producing usable (k>0, unique) candidates only if its worst replicate beats A's
best replicate on CPU seconds per unique k>0 candidate. It is called PRODUCTIVE for the gate only if it yields k>0
candidates with min surrogate bound >= 6 outside the archived structures. Neither is evidence of better codes:
any quantum-code claim requires the official validator, which this stage does not run.
Held-out requirement: seeds above were never used for development; if the code or protocol changes after the run,
a new versioned protocol and fresh seeds are required.
