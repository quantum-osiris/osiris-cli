#!/usr/bin/env python3
"""Deterministic offline verification & manifest generator for OSIRIS-BT2T-003 (Milestone 3).

Verifies:
  1. Frozen ``bt2t-enc-1`` encoding SHA-256 integrity (untouched Milestone 1 & 2 primitives).
  2. Gauge-fixed 5-qubit Kogut-Susskind 2T single-plaquette Hamiltonian H_KS = H_E + H_B:
     * 7 conjugacy classes matching the 7 irreps of 2T;
     * Exact bi-invariance [Delta_{2T}, L_h] = [Delta_{2T}, R_h] = 0 (< 1e-14) across all 24 h in 2T;
     * Exact residual conjugation Gauss's law [H_KS, Ad_h] = 0 (< 1e-14);
     * Exact Peter-Weyl non-abelian QFT diagonalization of H_E into SU(2)-calibrated Casimir sectors;
     * Exact decoupling (0.0 generator, +1 propagator) of all 8 unused codewords (24..31).
  3. Unfixed multi-link non-abelian holonomy accumulation & gauge-invariant isomorphism:
     * Exhaustive 15Q 2-link (13,824 triples) and 25Q 4-link (331,776 configurations) classical
       holonomy accumulation and ancilla uncomputation;
     * Exact 4-vertex Gauss-law invariance on H_phys (dim = 7) and <= 1e-13 dynamical agreement
       between 4-link unfixed evolution and 5-qubit gauge-fixed evolution.
  4. Real-time plaquette dynamics across strong/intermediate/weak couplings g_YM^2 in {0.5, 1.0, 2.0}
     and 1st-order Lie-Trotter O(dt^1) vs 2nd-order Strang O(dt^2) spectral norm error scaling.
  5. Optional local PROXY Qiskit (rz/sx/x/cz on all-to-all & heavy-hex T5/20Q/25Q) and pytket
     (ZZPhase/PhasedX/Rz) compilations comparing 5Q structured synthesis vs 5Q dense UnitaryGate
     and 20Q/25Q unfixed 4-link circuits.

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
    from . import bt2t_plaquette_dynamics as dyn
else:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import bt2t_plaquette_dynamics as dyn  # type: ignore[no-redef]

SEED = 20261124
DEFAULT_MANIFEST_PATH = HERE / "manifest_bt2t003.json"


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
                + ",".join(
                    f"{json.dumps(str(k), ensure_ascii=False)}:{_ser(v)}" for k, v in items
                )
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


def build_manifest_bt2t003() -> Dict[str, object]:
    """Run all deterministic OSIRIS-BT2T-003 checks and return canonical manifest dict."""
    enc_rec = dyn.verify_frozen_encoding()
    gauss_rep = dyn.verify_gauss_law_bi_invariance(g_ym_sq=1.0, dt=0.2)
    iso_rep = dyn.verify_multilink_gauge_isomorphism(
        dt=0.15, g_ym_sq=1.0, n_steps=5, seed=SEED
    )
    regime_rep = dyn.simulate_plaquette_regimes(
        t_final=1.0, n_strang_steps=64, couplings=dyn.COUPLING_REGIMES
    )
    scaling_rep = dyn.benchmark_trotter_error_scaling(
        t_final=1.0,
        couplings=dyn.COUPLING_REGIMES,
        step_counts=(8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    )
    comp_rep = dyn.benchmark_native_compilations(dt=0.2, g_ym_sq=1.0, seed=SEED)

    all_checks: Dict[str, bool] = {
        "encoding_sha256_verified": enc_rec["sha256"] == dyn.ENCODING_SHA256,
    }
    for prefix, sub_rep in (
        ("gauss", gauss_rep),
        ("multilink", iso_rep),
        ("dynamics", regime_rep),
        ("trotter", scaling_rep),
        ("compilation", comp_rep),
    ):
        for k, v in sub_rep["checks"].items():  # type: ignore[union-attr]
            all_checks[f"{prefix}_{k}"] = bool(v)

    body_raw: Dict[str, object] = {
        "experiment_id": "OSIRIS-BT2T-003",
        "milestone": (
            "M3: Kogut-Susskind 2T non-abelian lattice gauge single-plaquette Trotter dynamics (offline)"
        ),
        "execution_tier": "SIMULATED",
        "evidence_class": "SIMULATED",
        "execution_class": "NO_EXECUTION (exact linear algebra and local PROXY compilation only)",
        "provider_job_id": None,
        "qpu_seconds": 0,
        "preregistration": None,
        "encoding_version": dyn.ENCODING_VERSION,
        "encoding_sha256": dyn.ENCODING_SHA256,
        "seed": SEED,
        "part1_gauss_law_and_peter_weyl_hamiltonian": {
            k: v for k, v in gauss_rep.items() if k != "checks"
        },
        "part2_multilink_holonomy_and_gauge_isomorphism": {
            k: v for k, v in iso_rep.items() if k != "checks"
        },
        "part3_plaquette_regime_dynamics": {
            k: v for k, v in regime_rep.items() if k != "checks"
        },
        "part4_trotter_suzuki_error_scaling": {
            k: v for k, v in scaling_rep.items() if k != "checks"
        },
        "part5_native_circuit_compilations": {
            k: v for k, v in comp_rep.items() if k != "checks"
        },
        "checks": all_checks,
        "all_checks_passed": all(all_checks.values()),
        "governance_note": (
            "Offline exact non-abelian lattice gauge simulation and local PROXY circuit "
            "compilation only; no quantum hardware or cloud emulator was invoked "
            "(0 QPU seconds, provider_job_id=null)."
        ),
    }

    cleaned: Dict[str, object] = _clean_json_tree(body_raw)
    cleaned["rfc8785_canonical_sha256"] = jcs_sha256(cleaned)
    return cleaned


def write_manifest_bt2t003(output_path: Path = DEFAULT_MANIFEST_PATH) -> Dict[str, object]:
    """Generate and write deterministic ``manifest_bt2t003.json`` with LF line endings."""
    manifest = build_manifest_bt2t003()
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output_path.write_bytes(serialized.encode("utf-8"))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()
    manifest = write_manifest_bt2t003(args.out)
    ok = bool(manifest["all_checks_passed"])
    n_checks = len(manifest["checks"])  # type: ignore[arg-type]
    print(f"wrote {args.out}; all_checks_passed={ok} ({n_checks} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
