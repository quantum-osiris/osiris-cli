"""
Repaired Time-Dependent Floquet & Stroboscopic Zeno Controls (Aeterna Audit)
===========================================================================

Phase 2 Section 4 (Aeterna Methods Repair).

Forensic defects identified in `aeterna_porta_v2_ibm_nighthawk_circuit.py` and
`deploy_aeterna_porta_v2_IGNITION.py` / `SWEEP.py`:
  1. Integer-nanosecond aliasing defect:
       phase_k = (2*pi * 1.0e9 Hz) * (k * 1.0e-9 s) = 2*pi*k
     giving |0.5 * sin(phase_k)| <= 2.573e-15 rad for all k=0..9; every Floquet
     rotation vanished in floating-point arithmetic and was stripped by transpilation.
  2. Nyquist violation vs hardware `dt`:
     On IBM Heron r2 (`dt = 4.0 ns`, Nyquist limit f_Nyq = 1 / (2*dt) = 125 MHz),
     a 1.0 GHz modulation CANNOT be synthesized via gate-level `delay` + discrete
     rotations without aliasing; sub-`dt` 1 GHz modulation requires analog microwave
     IQ waveform control (simulation-only when OpenPlan lacks custom Pulse gates).
  3. Commuting virtual-Z fusion defect:
     Even at non-aliased phases, consecutive `RZ(theta_k)` gates separated only by
     `delay` commute and fuse into a single net Z rotation under `transpile(...,
     optimization_level >= 1)`, erasing multi-step Floquet modulation unless
     non-commuting transverse operations (`RX` or `SX-RZ-SXdg`) are scheduled.
  4. Zeno cycle floor-truncation defect:
     `int(1.25e6 * 1e-6) == 1` silently truncated the v2.1 Zeno upgrade to 1 cycle.
  5. Inert post-measurement / pre-readout `RZ`:
     `RZ(theta_lock)` immediately preceding Z-basis `measure_all()` changes zero
     outcome probabilities.
  6. Hardware width & fabricated-submission quarantine:
     130-qubit circuit exceeded 120-qubit target; `IGNITION.py` slept and wrote
     hardcoded `phi_estimation: 0.782` without importing Qiskit or submitting jobs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile

HERON_R2_DT_S: float = 4.0e-9  # 4.0 ns hardware timing resolution on IBM Heron r2
HERON_R2_FEZ_SX_NS: float = 24.0  # 6 dt on ibm_fez
HERON_R2_MARRAKESH_SX_NS: float = 36.0  # 9 dt on ibm_marrakesh
HERON_R2_FEZ_MEASURE_RESET_NS: float = 3144.0  # 1,560 ns measure + 1,584 ns reset on ibm_fez
HERON_R2_MARRAKESH_MEASURE_RESET_NS: float = 5304.0  # 2,584 ns measure + 2,720 ns reset on ibm_marrakesh
HERON_R2_MEASURE_RESET_NS: float = HERON_R2_FEZ_MEASURE_RESET_NS


@dataclass(frozen=True)
class FloquetScheduleSpec:
    drive_frequency_hz: float
    drive_amplitude_rad: float
    n_steps: int
    sample_interval_s: float
    dt_s: float = HERON_R2_DT_S
    gate_synthesis_s: float = 48.0e-9  # 2x SX duration (48 ns on ibm_fez; 72 ns on ibm_marrakesh)
    phase_offset_rad: float = 0.0
    transverse_axis: str = "rx"  # 'rx' ensures non-commuting transverse Floquet modulation

    @property
    def physical_step_s(self) -> float:
        return self.sample_interval_s + max(0.0, self.gate_synthesis_s)

    def validate(self) -> Dict[str, float | int | bool | str]:
        if self.drive_frequency_hz <= 0.0:
            raise ValueError("drive_frequency_hz must be positive")
        if self.n_steps < 2:
            raise ValueError("n_steps must be >= 2 for time-dependent modulation")
        if self.sample_interval_s <= 0.0 or self.dt_s <= 0.0:
            raise ValueError("sample_interval_s and dt_s must be positive")

        # 1. Hardware dt alignment check on both idle delay and gate synthesis
        dt_ticks_float = self.sample_interval_s / self.dt_s
        dt_ticks = int(round(dt_ticks_float))
        if dt_ticks < 1 or not math.isclose(dt_ticks * self.dt_s, self.sample_interval_s, rel_tol=1e-9, abs_tol=1e-15):
            raise ValueError(
                f"sample_interval_s ({self.sample_interval_s:.3e} s) is not a positive integer "
                f"multiple of hardware dt ({self.dt_s:.3e} s)"
            )

        # 2. Strict Nyquist anti-aliasing check accounting for finite 2-SX RX(theta) synthesis duration!
        eff_step_s = self.physical_step_s
        nyquist_hz = 0.5 / eff_step_s
        if self.drive_frequency_hz >= nyquist_hz * (1.0 - 1e-12):
            raise ValueError(
                f"Nyquist aliasing violation: drive_frequency_hz={self.drive_frequency_hz:.3e} Hz "
                f">= Nyquist limit {nyquist_hz:.3e} Hz for physical_step_s={eff_step_s:.3e} s "
                f"(delay={self.sample_interval_s:.3e} s + gate={self.gate_synthesis_s:.3e} s). "
                "Use continuous-time simulation for GHz drives or lower f_drive < 1/(2*T_step)."
            )

        angles = self.step_angles()
        if float(np.max(np.abs(angles))) < 1e-9:
            raise ValueError("All sampled Floquet rotation angles vanished (< 1e-9 rad)")

        return {
            "dt_ticks_per_step": dt_ticks,
            "physical_step_ns": round(eff_step_s * 1e9, 3),
            "nyquist_limit_hz": nyquist_hz,
            "oversampling_ratio": (1.0 / eff_step_s) / self.drive_frequency_hz,
            "max_abs_angle_rad": float(np.max(np.abs(angles))),
            "rms_angle_rad": float(np.sqrt(np.mean(angles ** 2))),
            "hardware_gate_level_feasible": True,
        }

    def step_times_s(self) -> np.ndarray:
        """Midpoint-sampled physical times t_k = (k + 0.5) * T_step (avoids t=0 endpoint degeneracy)."""
        return (np.arange(self.n_steps, dtype=np.float64) + 0.5) * self.physical_step_s

    def step_angles(self) -> np.ndarray:
        omega = 2.0 * math.pi * self.drive_frequency_hz
        phases = omega * self.step_times_s() + self.phase_offset_rad
        return self.drive_amplitude_rad * np.sin(phases)


def audit_historical_aeterna_bug() -> Dict[str, object]:
    """Quantify the exact historical 1 GHz @ 1 ns aliasing + Zeno truncation defects."""
    drive_amp = 0.5
    drive_omega = 2.0 * math.pi * 1.0e9
    steps = np.arange(10, dtype=np.float64)
    phases = drive_omega * steps * 1.0e-9
    angles = drive_amp * np.sin(phases)

    qc = QuantumCircuit(10)
    for ang in angles:
        for q in range(10):
            qc.rz(float(ang), q)
    qc_t = transpile(qc, basis_gates=["rz", "sx", "x", "cz"], optimization_level=3, seed_transpiler=42)

    return {
        "max_abs_floquet_angle_rad": float(np.max(np.abs(angles))),
        "pre_transpile_rz_count": int(qc.count_ops().get("rz", 0)),
        "post_transpile_opt3_gate_count": int(sum(qc_t.count_ops().values())),
        "zeno_v2_0_cycles": int(1_000_000.0 * 1e-6),
        "zeno_v2_1_cycles_truncated": int(1.25e6 * 1e-6),
        "floquet_survived_transpilation": int(sum(qc_t.count_ops().values())) > 0,
    }


def compute_zeno_cycle_count(window_s: float, zeno_frequency_hz: float, max_cycles: int = 100) -> int:
    """
    Compute exact integer Zeno monitoring cycles without silent `int(...)` floor truncation.
    Requires `window_s * zeno_frequency_hz >= 1` and rounds to nearest integer cycle,
    raising ValueError if two distinct configurations collide unintentionally or exceed cap.
    """
    if window_s <= 0.0 or zeno_frequency_hz <= 0.0:
        raise ValueError("window_s and zeno_frequency_hz must be positive")
    raw = window_s * zeno_frequency_hz
    cycles = int(round(raw))
    if cycles < 1:
        raise ValueError(
            f"Requested window_s={window_s:.3e} s * zeno_frequency_hz={zeno_frequency_hz:.3e} Hz "
            f"= {raw:.3f} < 1 cycle"
        )
    if cycles > max_cycles:
        raise ValueError(f"Computed Zeno cycles ({cycles}) exceeds max_cycles={max_cycles}")
    return cycles


def build_repaired_aeterna_stroboscopic_circuit(
    *,
    n_left: int = 4,
    n_right: int = 4,
    n_anc: int = 2,
    backend_qubit_limit: int = 156,
    theta_prep_rad: float = 0.65,
    zeno_window_s: float = 4.0e-6,
    zeno_frequency_hz: float = 1.0e6,
    weak_zy_coupling_rad: float = 0.10,
    floquet_spec: Optional[FloquetScheduleSpec] = None,
    include_mid_circuit_zeno: bool = True,
) -> Tuple[QuantumCircuit, Dict[str, object]]:
    """
    Build a mathematically well-defined, hardware-`dt`-aligned stroboscopic test circuit:
      - Enforces `n_left + n_right + n_anc <= backend_qubit_limit` (no 130Q-on-120Q overflow)
      - Single classical register layout (no duplicate 260-bit classical allocation)
      - Exact Z_data (x) Y_ancilla weak coupling U = exp(-i * alpha * Z_d (x) Y_a) via
        H_z-controlled decomposition: `CX(d, a)` / `RY` rather than uncalibrated `CRY`:
          exp(-i * alpha * Z_d Y_a) = |0><0|_d (x) RY(2*alpha)_a + |1><1|_d (x) RY(-2*alpha)_a
                                    = RY(2*alpha)_a · CX(d, a) · RY(-2*alpha)_a · CX(d, a) ?
        Let's verify: CX Y CX = -Y, so:
          RY(alpha)_a · CX(d, a) · RY(-alpha)_a · CX(d, a):
            if d=0: RY(alpha) RY(-alpha) = I -> wait!
            For exp(-i * alpha * Z_d Y_a):
              d=0 (Z=+1) -> exp(-i * alpha * Y) = RY(2*alpha)
              d=1 (Z=-1) -> exp(+i * alpha * Y) = RY(-2*alpha)
            Compare `CX(d, a) · RY(-2*alpha)_a · CX(d, a)`:
              if d=0: RY(-2*alpha)
              if d=1: X RY(-2*alpha) X = RY(+2*alpha)!
            Even simpler: `RY(2*alpha)_a` when d=0 and `RY(-2*alpha)_a` when d=1 is achieved by
            `X(d); CRY(4*alpha)?` No: `CX(d, a)` flips the sign of `RY`:
            If ancilla starts in |0>_a (immediately after reset!), then:
              d=0: |0>_d |0>_a -> |0>_d (cos(alpha)|0> + sin(alpha)|1>)_a
              d=1: |1>_d |0>_a -> |1>_d (cos(alpha)|0> - sin(alpha)|1>)_a
            Apply `RY(2*alpha)` on `a`, THEN `CZ(d, a)`!
              d=0: |0>_d (cos(alpha)|0> + sin(alpha)|1>)_a
              d=1: |1>_d (cos(alpha)|0> - sin(alpha)|1>)_a !!
            Only ONE native two-qubit `CZ` gate per weak measurement on |0>_a!!
      - Non-commuting `RX(theta_k)` + `delay(dt_ticks, unit='dt')` Floquet steps that
        provably survive Qiskit `optimization_level=3` transpilation when delays are bound.
      - Removes dead pre-readout `RZ` gates unless followed by a transverse measurement basis.
    """
    total_qubits = n_left + n_right + n_anc
    if n_left <= 0 or n_right <= 0 or n_anc < 0:
        raise ValueError("Invalid partition sizes")
    if total_qubits > backend_qubit_limit:
        raise ValueError(
            f"Circuit requires {total_qubits} qubits, exceeding backend limit {backend_qubit_limit}"
        )

    if floquet_spec is None:
        # Default hardware-realizable 4.0 MHz Floquet modulation with 16 ns delay + 48 ns 2-SX synthesis
        # (T_step = 64 ns = 16 dt on ibm_fez; f_sample = 15.625 MHz, f_Nyquist = 7.8125 MHz > 4.0 MHz!)
        floquet_spec = FloquetScheduleSpec(
            drive_frequency_hz=4.0e6,
            drive_amplitude_rad=0.35,
            n_steps=10,
            sample_interval_s=16.0e-9,
            dt_s=HERON_R2_DT_S,
            gate_synthesis_s=2.0 * HERON_R2_FEZ_SX_NS * 1.0e-9,
        )
    floquet_meta = floquet_spec.validate()
    dt_ticks = int(floquet_meta["dt_ticks_per_step"])

    zeno_cycles = compute_zeno_cycle_count(zeno_window_s, zeno_frequency_hz) if include_mid_circuit_zeno and n_anc > 0 else 0
    zeno_interval_ticks = (
        max(1, int(round((zeno_window_s / zeno_cycles) / floquet_spec.dt_s)))
        if zeno_cycles > 0
        else 0
    )

    n_data = n_left + n_right
    n_clbits = n_data + (zeno_cycles * min(n_left, n_anc))
    qc = QuantumCircuit(total_qubits, n_clbits, name="aeterna_repaired_stroboscopic")

    # Stage 1: Entangled pair preparation
    n_pairs = min(n_left, n_right)
    for i in range(n_pairs):
        ql = i
        qr = n_left + i
        qc.h(ql)
        qc.ry(theta_prep_rad, ql)
        qc.cx(ql, qr)
        qc.ry(0.5 * theta_prep_rad, qr)

    # Stage 2: Calibrated stroboscopic Zeno weak monitoring on |0>_anc via RY(2*alpha) + CZ
    anc_start = n_data
    cl_cursor = n_data
    monitored = min(n_left, n_anc)
    for _ in range(zeno_cycles):
        for i in range(monitored):
            qd = i
            qa = anc_start + i
            qc.ry(2.0 * weak_zy_coupling_rad, qa)
            qc.cz(qd, qa)
            qc.measure(qa, cl_cursor)
            qc.reset(qa)
            cl_cursor += 1
        for qd in range(n_data):
            qc.delay(zeno_interval_ticks, qd, unit="dt")

    # Stage 3: Non-aliased, dt-aligned transverse Floquet modulation
    throat_qubits = list(range(max(0, n_left - min(2, n_left)), min(n_data, n_left + min(2, n_right))))
    step_angles = floquet_spec.step_angles()
    for theta_k in step_angles:
        for q in throat_qubits:
            if floquet_spec.transverse_axis == "rx":
                qc.rx(float(theta_k), q)
            else:
                qc.sx(q)
                qc.rz(float(theta_k), q)
                qc.sxdg(q)
            qc.delay(dt_ticks, q, unit="dt")

    # Stage 4: Data readout (no dead Z-rotations before Z-basis measurement)
    for q in range(n_data):
        qc.measure(q, q)

    scheduled_floquet_ns = len(step_angles) * (floquet_spec.physical_step_s * 1e9)
    scheduled_zeno_ns = zeno_cycles * (
        zeno_interval_ticks * floquet_spec.dt_s * 1e9 + HERON_R2_FEZ_MEASURE_RESET_NS
    )
    meta = {
        "total_qubits": total_qubits,
        "total_clbits": n_clbits,
        "zeno_cycles": zeno_cycles,
        "reused_ancilla_max_zeno_hz_fez": round(1.0e9 / HERON_R2_FEZ_MEASURE_RESET_NS, 1),
        "reused_ancilla_max_zeno_hz_marrakesh": round(1.0e9 / HERON_R2_MARRAKESH_MEASURE_RESET_NS, 1),
        "floquet": floquet_meta,
        "step_angles_rad": [float(a) for a in step_angles],
        "throat_qubits": throat_qubits,
        "logical_depth": qc.depth(),
        "estimated_duration_ns": scheduled_floquet_ns + scheduled_zeno_ns,
        "hardware_limitations": (
            "Gate-level stroboscopic Floquet control requires f_drive < 1/(2*(Delta_t + T_2SX)) "
            "(<= 7.81 MHz at 16 ns delay + 48 ns 2-SX synthesis on ibm_fez; <= 5.68 MHz at 16 ns + "
            "72 ns on ibm_marrakesh). Reused-ancilla mid-circuit measure+reset takes 3,144 ns on "
            "ibm_fez (f_max = 318 kHz) and 5,304 ns on ibm_marrakesh (f_max = 188.5 kHz); >= 1 MHz "
            "stroboscopic Zeno requires fresh ancilla pools without mid-circuit reset, and 1.0 GHz "
            "driving remains strictly Simulated unless analog AWG Pulse access is available."
        ),
    }
    return qc, meta


def verify_floquet_survives_transpile(
    qc: QuantumCircuit, optimization_level: int = 3
) -> Dict[str, object]:
    """
    Verify that scheduled `delay` instructions prevent the Qiskit optimizer from
    collapsing the multi-step Floquet sequence into a single single-qubit gate.
    """
    compiled = transpile(
        qc,
        basis_gates=["rz", "sx", "x", "cz", "delay", "reset"],
        optimization_level=optimization_level,
        seed_transpiler=42,
    )
    ops = {k: int(v) for k, v in compiled.count_ops().items()}
    delay_count = ops.get("delay", 0)
    sx_count = ops.get("sx", 0)
    return {
        "optimization_level": optimization_level,
        "compiled_depth": compiled.depth(),
        "compiled_ops": ops,
        "delay_count": delay_count,
        "sx_count": sx_count,
        "floquet_modulation_preserved": delay_count >= 10 and sx_count >= 10,
    }
