# Aeterna Time-Dependent Controls — Forensic Audit & Methodological Repair

**Status:** `Implemented and tested` (`phase2/aeterna_w2/aeterna_controls_repaired.py`, `tests/test_phase2_aeterna_w2.py`; GHz continuous-wave regime classified as `Simulated` under OpenPlan gate-level constraints)  
**Audited Historical Sources:** `legacy-mirrors/ENKI-420/aeterna-porta-v2.git` (`HEAD` `50d22675`):
- `aeterna_porta_v2_ibm_nighthawk_circuit.py` (`sha256: bc1c4cdce676ab23883b724147cd21387c2b2c8e2390fb346a17916f40184eb2`)
- `deploy_aeterna_porta_v2_IGNITION.py` (`sha256: 5700a3fbf7bc94b3348be9151675fad0deaa9c48721fb8d1836664641161f530`)
- `deploy_aeterna_porta_v2_SWEEP.py` (`sha256: eb1174ba55a620e9a4100298899e822c625b8153bf520b572914b1a3fb2be1a8`)

---

## 1. Audit of Historical Time-Dependent & Control Defects

1. **1.0 GHz Integer-Nanosecond Aliasing (`aeterna_porta_v2_ibm_nighthawk_circuit.py:72–84`):**
   - `drive_frequency = 2 * np.pi * 1.0e9`, `phase = drive_frequency * step * 1e-9 = 2 * pi * step` for `step in range(10)`.
   - Every sampled angle `0.5 * sin(phase)` had magnitude $\le 2.573 \times 10^{-15}\text{ rad}$.
   - Under `transpile(..., optimization_level=3)`, **all 100 Floquet `rz` gates compiled to 0 instructions** (`audit_historical_aeterna_bug()`).
2. **Hardware Nyquist & `dt` Alignment Failure:**
   - Even without integer-phase coincidence, IBM Heron r2 enforces a discrete instruction timing grid `dt = 4.0 ns` ($f_{\text{sample,max}} = 250\text{ MHz}$, Nyquist limit $f_{\text{Nyq}} = 125\text{ MHz}$). A `1.0 GHz` drive cannot be stroboscopically sampled via gate-level `delay` + discrete gates without severe aliasing.
   - Additionally, original circuits contained **zero `qc.delay(...)` instructions**, so loop index `step` corresponded to zero scheduled idle duration, and consecutive `rz` gates commuted and fused into a single virtual-Z frame update.
3. **Zeno Cycle Floor-Truncation (`int(f * 1e-6)`):**
   - Original v2.0 (`1.0 MHz * 1 µs`) and proposed v2.1 (`1.25 MHz * 1 µs`) both evaluated `int(...) == 1` cycle, rendering the v2.1 parameter change a no-op.
4. **Weak-Measurement Operator Mismatch & Inert Pre-Readout `RZ`:**
   - `qc.cry(0.1, data, anc)` implements $|0\rangle\langle 0|\otimes I + |1\rangle\langle 1|\otimes R_Y(0.1)$, not $\exp(-i\alpha Z_d \otimes Y_a)$.
   - Feed-forward `qc.rz(deg2rad(THETA_LOCK), target)` immediately before Z-basis `measure_all()` is a pure phase on computational basis states and cannot change readout probabilities.
5. **Qubit Budget Overflow, Saturating Metrics & Fabricated Submission Path:**
   - `TOTAL_QUBITS = 60 + 60 + 10 = 130` exceeded the 120-qubit target, and `QuantumCircuit(130, 130)` + `measure_all()` allocated 260 classical bits.
   - `deploy_aeterna_porta_v2_IGNITION.py` imported no quantum SDK, slept 5.5 s, generated `ign_v2.1_<timestamp>`, and wrote hardcoded `"phi_estimation": 0.782` (`QUARANTINED` as a fabricated-output script).
   - `deploy_aeterna_porta_v2_SWEEP.py` normalized empirical Shannon entropy by $\log_2(\text{unique shots})$, guaranteeing $\hat{\Phi} = 1.000 > 0.7734$ on pure 120-qubit noise.

---

## 2. Repaired Implementation & Explicit Simulation-Only Boundary

Implemented in `phase2/aeterna_w2/aeterna_controls_repaired.py`:
- **`FloquetScheduleSpec.validate()`:** Enforces exact integer `dt` alignment (`sample_interval_s = k * 4.0 ns`), strict Nyquist anti-aliasing accounting for finite 2-`SX` `RX(theta)` synthesis duration ($T_{\text{step}} = \Delta t_{\text{delay}} + 2 T_{\text{SX}}$, e.g. $16\text{ ns} + 48\text{ ns} = 64\text{ ns} \implies f_{\text{Nyq}} = 7.81\text{ MHz}$ on `ibm_fez` and $16\text{ ns} + 72\text{ ns} = 88\text{ ns} \implies f_{\text{Nyq}} = 5.68\text{ MHz}$ on `ibm_marrakesh`), midpoint sample timing $t_k = (k + \frac{1}{2})T_{\text{step}}$, and non-zero RMS modulation check.
- **Transpilation Survival:** Uses non-commuting transverse `RX(theta_k)` (or `SX-RZ-SXdg`) interleaved with explicit `qc.delay(dt_ticks, q, unit="dt")`. Verified via `verify_floquet_survives_transpile(..., optimization_level=3)` that all scheduled `delay` and `sx` modulation layers survive Qiskit level-3 optimization.
- **Calibrated Stroboscopic Zeno Coupling & Mid-Circuit Latency Bound:** Uses `RY(2*alpha)` on reset $|0\rangle_{\text{anc}}$ followed by a single native `CZ(data, anc)` to realize exact $\exp(-i\alpha Z_d Y_a)| \psi\rangle|0\rangle_a$, with rounded integer cycle counts (`compute_zeno_cycle_count`) and explicit `total_qubits <= backend_qubit_limit` validation. Records calibrated `measure + reset` latency (`3,144 ns` on `ibm_fez`, capping reused-ancilla Zeno to $f_{\max} \approx 318\text{ kHz}$; `5,304 ns` on `ibm_marrakesh`, $f_{\max} \approx 188.5\text{ kHz}$, requiring fresh ancilla registers without mid-circuit reset for $\ge 1\text{ MHz}$ monitoring).
- **Explicit Hardware Limitation (`Simulated` Regime):** Sub-nanosecond `1.0 GHz` continuous Floquet modulation requires custom AWG/Pulse IQ envelopes not exposed on IBM Quantum OpenPlan dynamic circuits (`rz, sx, x, cz, measure, reset, delay`). Any `> 125 MHz` (or `> 7.81 MHz` 2-`SX` stroboscopic) Floquet investigation is classified `Simulated` (continuous-time Lindblad/Floquet solver) unless pulse-level hardware access is available.
