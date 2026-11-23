# Pre-Registration: `OSIRIS-XTOP-006B-FEZ` — Confirmatory Intra-Block 4-Twirl Landsman–Figgatt Scrambling-vs-Decoherence Discriminator on `ibm_fez`

- **Protocol ID:** `OSIRIS-XTOP-006B-FEZ` (Confirmatory `H2` closure following `OSIRIS-XTOP-006-FEZ` `dc1tqaanj2vs73dfrt40`)
- **Target Processor:** `ibm_fez` (`[99, 115, 95, 93, 94, 92, 79]`, constant `28 CZ`, `126 SX` skeleton)
- **Root Cause Resolved from `XTOP-006-FEZ`:**
  In `dc1tqaanj2vs73dfrt40`, `DECOH_CTRL` rotated a single Pauli-$Z$ twirl `(1,1,0), (1,0,1), (0,1,1), (1,1,1)` per block; although the 4-twirl pooled mean hit `0.6572` (`+19.94 pp` below `0.8566` coherent, matching the `2/3 = 0.6667` classical ceiling), deterministic twirl-to-twirl differences (`1/3, 1/3, 1, 1` noiseless) inflated un-stratified block SE (`0.132`).
- **Stratified Intra-Block 4-Twirl Design (`65,536` shots, `~18–19s` QPU):**
  Every block $b \in \{0, 1, 2, 3\}$ executes **all 4 Pauli-$Z$ twirls** (`48` `DECOH_4TWIRL` PUBs/block) alongside `12` `COHERENT` PUBs/block and `4` A/A + readout sentinels (`64` PUBs/block $\times$ `4` blocks = `256` PUBs $\times$ `256` shots = `65,536` shots), eliminating twirl-choice variance between blocks identically (`noiseless block mean = 0.666667` at $\theta=\pi/2$ and `0.642657` at $\theta=3\pi/8$ in every block).

## Preregistered Confirmatory Hypothesis (`H2_CONFIRMATORY`)

- **Pass Criterion:** Paired across the $N=4$ randomized blocks (`df = 3`, Holm–Bonferroni family-wise corrected across $\theta \in \{3\pi/8, \pi/2\}$):
  1. $\Delta F(\pi/2) = F_{\text{COHERENT}}(\pi/2 \mid \Phi^+) - F_{\text{DECOH\_4TWIRL}}(\pi/2 \mid \Phi^+) > +0.15$ (`+15 pp`) with $p_{\text{Holm}} < 0.01$, AND
  2. $\Delta F(3\pi/8) = F_{\text{COHERENT}}(3\pi/8 \mid \Phi^+) - F_{\text{DECOH\_4TWIRL}}(3\pi/8 \mid \Phi^+) > +0.10$ (`+10 pp`) with $p_{\text{Holm}} < 0.01$.
- **Kill Rule:** Refuted if either contrast fails its preregistered effect-size floor or $p_{\text{Holm}} \ge 0.01$.
