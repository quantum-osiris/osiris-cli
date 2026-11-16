# Quantinuum Startup Partner Program — Approved Submission Package (`Track A`)

- **Track Classification:** Commercial & Defense Product R&D (`Agile Defense Systems LLC` — strictly firewalled from DOE OLCF QCUP Open-Science `Track B`)
- **Company Name:** Agile Defense Systems LLC
- **Principal Contact / Title:** Devin Phillip Davis — Founder / Principal Investigator
- **Areas of Interest (Form Checkboxes):** `Software`, `AI`, `Optimization`, `Physics` *(optional: `Cryptography`)*
- **Supporting Verified Evidence DOIs / Records:**
  - `10.5281/zenodo.24287505` (`OSIRIS-HP-004` `ibm_marrakesh` Job `dbti1fbsbsbs73cb2vu0` & `OSIRIS-HP-004B` `ibm_fez` Job `dbtj61bnlcms73ajc380`, `585,728` raw shots)
  - `10.5281/zenodo.24293756` (`OSIRIS-QEC-001` `ibm_fez` Job `dbtl2vb2kf8s73c9a0d0`, `393,216` raw shots; total `978,944` pre-registered QPU shots)
  - `10.5281/zenodo.22862567` (`dnalang-core` OpenQASM 3 compiler & conformance suite, `20,000` random programs, `< 0.015%` failure rate)
  - `10.5281/zenodo.22870279` (40-qubit planar routing vs. chain rerun: `39` to `166` two-qubit gate inflation, `3.3x` fidelity ratio)

---

## Application Prompt Response (Copy-Paste Ready)

**Prompt:** *"Describe how access to the world’s most powerful quantum computing resources would accelerate or enable your product development process."*

Agile Defense Systems LLC develops **DNA-Lang** and **OSIRIS**: a verifiable quantum-classical co-design compiler, hardware-in-the-loop evolutionary circuit discovery engine, and cryptographically governed R&D workbench for high-assurance scientific, enterprise, and defense applications. DNA-Lang compiles modular, parameterized circuit families (`genes`/`genomes`) to conformance-tested OpenQASM 3 (20,000 random conformance programs, < 0.015% failure rate) while recording every evolutionary lineage step, pre-registered hypothesis, circuit digest, and raw shot histogram in an append-only RFC 8785 / Ed25519 evidence ledger—ensuring AI-assisted and evolutionary quantum workflows remain reproducible, falsifiable, and immune to post-hoc selection bias.

Across three prospectively pre-registered superconducting hardware campaigns totaling 978,944 raw unmitigated shots (`10.5281/zenodo.24287505` and `10.5281/zenodo.24293756`), we (i) certified Schmidt number >= 3 across two independent 156-qubit processors (raw joint fidelities 0.633423 and 0.730713 vs. the 0.5000 ceiling) and established a Terhal–Vollbrecht fidelity-witness lower bound on entanglement of formation above one ebit on `ibm_fez` (1.147 ebits; 1.124 ebits at the 95% lower confidence limit), and (ii) demonstrated monotonic multi-round sub-threshold error suppression across distances d in {3, 5, 7} and cycles R in {1, 3, 5, 7} with 7.22x space-time Blossom V MWPM error reduction (9.74% to 1.35% at d = 7, R = 7). Crucially, these runs also quantified two planar transmon bottlenecks that limit our commercial roadmap: 4x SWAP-routing gate inflation (39 to 166 two-qubit gates, 3.3x support loss) on non-planar graphs, and 3.344 us/round mid-circuit readout/reset spectator dephasing (0.1265–0.1284 phase-flip detector defect floor).

Access to Quantinuum’s H-Series/Helios QCCD processors, TKET, and Nexus would directly accelerate three product milestones:
1. **Native Nexus & TKET Provenance Capsules:** Targeting TKET passes and Nexus workflows from DNA-Lang’s OpenQASM 3 IR, providing tamper-evident pre-registration, number-provenance firewalls, and independent raw-count verification for enterprise and defense Nexus users.
2. **Routing-Free Evolutionary & qLDPC Co-Design:** Using QCCD all-to-all connectivity and isolated mid-circuit measurement/reset zones to evolve compact state-preparation circuits and co-compiled Steane `[[7,1,3]]`, Tesseract `[[16,6,4]]`, and Bivariate-Bicycle `[[72,12,6]]` qLDPC error-detection/correction gadgets decoded via Belief Propagation + Ordered Statistics Decoding (`BP+OSD`), free of planar SWAP and spectator-readout confounding.
3. **High-Dimensional Entanglement & Debiased Transport Benchmarks on Helios:** Scaling joint-stabilizer entanglement-dimensionality witnesses and split-sample-debiased Kantorovich/Sinkhorn distributional stability diagnostics into the 50–98 physical qubit and multi-logical-qubit regime.
