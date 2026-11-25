#!/usr/bin/env python3
"""Deterministic offline verification & manifest generator for OSIRIS-BT2T-004 (Milestone 4).

Verifies:
  1. Frozen ``bt2t-enc-1`` encoding SHA-256 integrity (untouched Milestone 1, 2 & 3 primitives).
  2. Gauge-fixed 10-qubit coupled 2-plaquette holonomy formulation vs 35-qubit unfixed 7-link ladder:
     * 2x1 ladder with 6 vertices, 7 directed links (24^7 = 4,586,471,424 configurations, 35Q)
       reduced by maximal spanning-tree gauge fixing (5 tree links = +1) to 2 plaquette holonomies
       (W_1, W_B) in 2T x 2T (24^2 = 576 active states in a 10-qubit 1024-D register with 448
       decoupled padded states: 7,962,624x configuration reduction, 3.5x qubit reduction);
     * Exact proof by Burnside's lemma, conjugacy-class centralizer summation, Peter-Weyl
       Clebsch-Gordan multiplicity summation, and explicit orbit enumeration that the residual
       diagonal-conjugation-invariant physical Hilbert space H_phys^{(2P)} has dimension 76.
  3. Coupled 2-plaquette Kogut-Susskind Hamiltonian H_{2P}(g_YM^2) & shared-link electric operator:
     * Exact identity between M_{2T}^dagger (Delta_{2T}^{(1)} (x) I) M_{2T} and the direct
       opposite-orientation shared-link Cayley shift Laplacian;
     * Exact Hermiticity, positive semi-definiteness, mutual commutativity of T_E^{(1)}, T_E^{(2)},
       T_E^{(shared)}, 0.0 leakage into the 448 padded states, global gauge conjugation invariance
       [H_{2P}, Ad_g (x) Ad_g] = 0 (< 1e-14), and 3-link 13,824-state unfixed ladder intertwining.
  4. Exact Wilson-loop area-law vs perimeter-law confinement crossover across
     g_YM^2 in {0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0}:
     * Monotonic decrease of W_{1x1} and W_{1x2}, strictly positive connected correlator
       C_{12} = W_{1x2} - W_{1x1}^2 > 0, and monotonic growth of effective string tension
       sigma_eff(g_YM^2) = -ln(W_{1x2} / W_{1x1}).
  5. Second-order coupled Strang Trotterization O(dt^2) slope (~ 2.000) vs first-order Lie-Trotter
     O(dt^1) slope (~ 1.000) across N in {4, 8, 16, 32, 64, 128}, plus local PROXY Qiskit and
     pytket compilations proving 32 CZ all-to-all / 38 CZ heavy-hex 2xT5 for U_B^{(2P)}(dt),
     >100x CZ compression vs generic 10Q dense synthesis, and >10x CZ reduction vs 35Q unfixed
     7-link Trotterization.

Execution tier: SIMULATED (evidence_class = SIMULATED, provider_job_id = null, qpu_seconds = 0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    import rfc8785 as _rfc8785  # type: ignore[import-not-found]
except ImportError:
    _rfc8785 = None

HERE = Path(__file__).resolve().parent
if __package__:
    from . import bt2t_two_plaquette_ladder as ladder
else:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import bt2t_two_plaquette_ladder as ladder  # type: ignore[no-redef]

SEED = 20261124
DEFAULT_MANIFEST_PATH = HERE / "manifest_bt2t004.json"


def _clean_float(val: float) -> float:
    if val == 0.0:
        return 0.0
    return float(f"{float(val):.12e}")


def _clean_json_tree(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return _clean_float(obj)
    if isinstance(obj, (list, tuple)):
        return [_clean_json_tree(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _clean_json_tree(v) for k, v in obj.items()}
    raise TypeError(f"Unsupported manifest value type: {type(obj)}")


def _jcs_format_float(val: float) -> str:
    """ECMA-262 / RFC 8785 canonical decimal serialization for finite IEEE-754 floats."""
    if not math.isfinite(val):
        raise ValueError(f"Non-finite float not permitted in RFC 8785 JCS: {val}")
    if val == 0.0:
        return "0"
    neg = val < 0.0
    av = -val if neg else val
    s = repr(av).lower()
    if "e" in s:
        mant, exp_s = s.split("e")
        exp = int(exp_s)
    else:
        mant, exp = s, 0
    if "." in mant:
        int_p, frac_p = mant.split(".")
        raw = int_p + frac_p
        stripped = raw.lstrip("0")
        leading_zeros = len(raw) - len(stripped)
        dec_exp = exp + len(int_p) - leading_zeros - 1
    else:
        stripped = mant.lstrip("0")
        dec_exp = exp + len(stripped) - 1
    digits = stripped.rstrip("0") or "0"
    k = len(digits)
    n = dec_exp + 1
    if k <= n <= 21:
        out = digits + ("0" * (n - k))
    elif 0 < n <= 21:
        out = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        out = "0." + ("0" * (-n)) + digits
    else:
        sign_e = "+" if (n - 1) >= 0 else "-"
        mant_out = digits if k == 1 else (digits[0] + "." + digits[1:])
        out = f"{mant_out}e{sign_e}{abs(n - 1)}"
    return ("-" + out) if neg else out


def pure_python_jcs_bytes(payload: Any) -> bytes:
    """Pure-Python RFC 8785 JSON Canonicalization Scheme (JCS) serializer."""

    def _ser(val: Any) -> str:
        if val is None:
            return "null"
        if val is True:
            return "true"
        if val is False:
            return "false"
        if isinstance(val, int):
            return str(val)
        if isinstance(val, float):
            return _jcs_format_float(val)
        if isinstance(val, str):
            return json.dumps(val, ensure_ascii=False)
        if isinstance(val, (list, tuple)):
            return "[" + ",".join(_ser(item) for item in val) + "]"
        if isinstance(val, Mapping):
            items = sorted(val.items(), key=lambda kv: str(kv[0]).encode("utf-16-be"))
            return (
                "{"
                + ",".join(f"{json.dumps(str(k), ensure_ascii=False)}:{_ser(v)}" for k, v in items)
                + "}"
            )
        raise TypeError(f"Unsupported type for RFC 8785 JCS: {type(val)}")

    return _ser(payload).encode("utf-8")


def jcs_canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Canonicalize ``payload`` under RFC 8785 JCS using ``rfc8785`` or pure-Python fallback."""
    if _rfc8785 is not None:
        return bytes(_rfc8785.dumps(dict(payload)))
    return pure_python_jcs_bytes(payload)


def jcs_sha256(payload: Mapping[str, Any]) -> str:
    """Return hex SHA-256 digest of RFC 8785 canonical JSON bytes of ``payload``."""
    return hashlib.sha256(jcs_canonical_bytes(payload)).hexdigest()


def build_manifest_bt2t004() -> Dict[str, object]:
    """Run all deterministic OSIRIS-BT2T-004 checks and return canonical manifest dict."""
    enc_rec = ladder.verify_frozen_encoding()
    reduction_rep = ladder.verify_two_plaquette_gauge_reduction()
    hamiltonian_rep = ladder.verify_coupled_hamiltonian_and_shared_link(g_ym_sq=1.0, dt=0.2)
    confinement_rep = ladder.benchmark_wilson_loop_confinement_crossover(
        couplings=ladder.CONFINEMENT_COUPLING_SWEEP
    )
    trotter_rep = ladder.benchmark_coupled_trotter_scaling(
        t_final=0.5,
        couplings=ladder.TROTTER_COUPLING_REGIMES,
        step_counts=ladder.TROTTER_STEP_COUNTS,
    )
    comp_rep = ladder.benchmark_native_compilations_bt2t004(dt=0.2, g_ym_sq=1.0, seed=SEED)

    all_checks: Dict[str, bool] = {
        "encoding_sha256_verified": enc_rec["sha256"] == ladder.ENCODING_SHA256,
    }
    for prefix, sub_rep in (
        ("reduction", reduction_rep),
        ("hamiltonian", hamiltonian_rep),
        ("confinement", confinement_rep),
        ("trotter", trotter_rep),
        ("compilation", comp_rep),
    ):
        for k, v in sub_rep["checks"].items():  # type: ignore[union-attr]
            all_checks[f"{prefix}_{k}"] = bool(v)

    all_passed = all(all_checks.values())
    body_raw: Dict[str, object] = {
        "experiment_id": "OSIRIS-BT2T-004",
        "milestone": (
            "M4: 10-qubit coupled 2-plaquette 2T subset SU(2) Kogut-Susskind lattice gauge ladder (offline)"
        ),
        "execution_tier": "SIMULATED",
        "evidence_class": "SIMULATED",
        "execution_class": "NO_EXECUTION (exact linear algebra and local PROXY compilation only)",
        "provider_job_id": None,
        "qpu_seconds": 0,
        "preregistration": None,
        "encoding_version": ladder.ENCODING_VERSION,
        "encoding_sha256": ladder.ENCODING_SHA256,
        "seed": SEED,
        "part1_gauge_reduction_and_76d_physical_subspace": {
            k: v for k, v in reduction_rep.items() if k != "checks"
        },
        "part2_coupled_hamiltonian_and_shared_link_electric_operator": {
            k: v for k, v in hamiltonian_rep.items() if k != "checks"
        },
        "part3_wilson_loop_confinement_crossover_and_string_tension": {
            k: v for k, v in confinement_rep.items() if k != "checks"
        },
        "part4_coupled_strang_trotter_convergence_and_dynamics": {
            k: v for k, v in trotter_rep.items() if k != "checks"
        },
        "part5_native_10q_and_35q_circuit_compilations": {
            k: v for k, v in comp_rep.items() if k != "checks"
        },
        "checks": all_checks,
        "all_passed": all_passed,
        "all_checks_passed": all_passed,
        "governance_note": (
            "Offline exact non-abelian 2-plaquette lattice gauge simulation and local PROXY "
            "circuit compilation only; no quantum hardware or cloud emulator was invoked "
            "(0 QPU seconds, provider_job_id=null)."
        ),
    }

    cleaned: Dict[str, object] = _clean_json_tree(body_raw)
    cleaned["rfc8785_canonical_sha256"] = jcs_sha256(cleaned)
    return cleaned


def write_manifest_bt2t004(output_path: Path = DEFAULT_MANIFEST_PATH) -> Dict[str, object]:
    """Generate and write deterministic ``manifest_bt2t004.json`` with LF line endings."""
    manifest = build_manifest_bt2t004()
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output_path.write_bytes(serialized.encode("utf-8"))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()
    manifest = write_manifest_bt2t004(args.out)
    ok = bool(manifest["all_passed"])
    n_checks = len(manifest["checks"])  # type: ignore[arg-type]
    print(f"wrote {args.out}; all_passed={ok} ({n_checks} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
