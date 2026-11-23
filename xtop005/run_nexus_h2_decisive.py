#!/usr/bin/env python3
"""OSIRIS-XTOP-005 Hash-Locked Nexus H2-1E / H2-1 Decisive 18-Circuit Runner & Bundle Generator.

Decisive L=3 comparison on Quantinuum H2 all-to-all QCCD:
  * Q0 (BARE, 8 qubits):          9 two-qubit Pauli readout settings x 500 shots
  * Q3 (ENC_ALL_CHK, 18 qubits):  9 two-qubit Pauli readout settings x 500 shots
                                  (all 4 registers in [[4,2,2]], verified prep + flagged weight-4 X/Z checks)

Governance invariants:
  * Default (`--dry-run`, whenever `--submit` / `--live-approved` is absent):
      - Zero network calls.
      - Verifies RFC 8785 JCS canonicalization of `nexus_h2_decisive_manifest.json`.
      - Verifies every SHA-256 entry in `SHA256SUMS.txt`, per-circuit `tket_sha256`, and `compiled_batch_sha256`.
      - Runs Stage-0 noiseless simulation directly on the 18 offline-compiled H2-1 pytket circuits
        (`F = 1.000000`, `0` shots rejected).
      - Emits an `OFFLINE_VERIFIED` readiness receipt (`evidence_class: "OFFLINE_COMPILED"`, `provider_job_id: null`).
  * `--submit --device H2-1E --operator-approval-token <token>`:
      - Fails closed if manifest/circuit SHA-256 hashes mismatch, Stage-0 fails, operator approval token is
        absent/invalid, HQC cap is exceeded, or `qnexus` / Nexus credentials are unavailable.
      - Never marks `HARDWARE_ATTESTED` unless `NexusSdkRunner` confirms a completed Tier 3 QPU receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rfc8785

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osiris_cli.nexus_ledger import NexusAuditLedger  # noqa: E402
from osiris_cli.providers.nexus_sdk import (  # noqa: E402
    NexusSdkRunner,
    NexusSdkUnavailable,
    batch_sha256,
    load_circuits,
    load_qnexus,
    local_hqc,
    tket_sha256,
)
from osiris_cli.providers.quantinuum import resolve_quantinuum_target  # noqa: E402
from osiris_cli.providers.quantinuum_nexus import GovernanceError  # noqa: E402
from osiris_cli.quantinuum_gate import QuantinuumGateError, verify_tier3_token  # noqa: E402
from xtop005 import osiris_xtop_005 as x  # noqa: E402
from xtop005 import quantinuum as xq  # noqa: E402

BUNDLE_DIR = Path(__file__).resolve().parent / "nexus_h2_bundle"
MANIFEST_NAME = "nexus_h2_decisive_manifest.json"
BUNDLE_CIRCUITS_NAME = "circuits_h2_decisive_18_tket.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"

DECISIVE_ARMS: Tuple[str, ...] = ("Q0", "Q3")
DECISIVE_DEPTH: int = 3
DEFAULT_SHOTS: int = 500
H2_ALLOWED_OP_NAMES = frozenset(
    {"PhasedX", "Rz", "ZZMax", "ZZPhase", "Measure", "Reset", "Barrier"}
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jcs_bytes(obj: Any) -> bytes:
    return rfc8785.dumps(obj)


def _detailed_tk_counts(tk_circ: Any) -> Dict[str, int]:
    """Return exact H2-1 compiled gate counts (both total 1Q and physical PhasedX vs virtual Rz)."""
    from pytket import OpType

    n1_total = 0
    n1_phys = 0
    n1_rz = 0
    n2 = 0
    n_meas = 0
    n_reset = 0
    swaps = 0
    for cmd in tk_circ.get_commands():
        op = cmd.op
        while op.type == OpType.Conditional:
            op = op.op
        t = op.type
        op_name = t.name
        if op_name == "SWAP":
            swaps += 1
        if op_name not in H2_ALLOWED_OP_NAMES:
            raise ValueError(f"Non-native H2-1 operation {op_name!r} in {tk_circ.name}")
        if t == OpType.Measure:
            n_meas += 1
        elif t == OpType.Reset:
            n_reset += 1
        elif t == OpType.Barrier or not cmd.qubits:
            continue
        elif len(cmd.qubits) == 1:
            n1_total += 1
            if t == OpType.Rz:
                n1_rz += 1
            else:
                n1_phys += 1
        elif len(cmd.qubits) == 2:
            n2 += 1
    n_qubits = int(tk_circ.n_qubits)
    n_m = n_meas + n_reset + n_qubits
    return {
        "qubits": n_qubits,
        "N_1q": n1_total,
        "N_1q_physical": n1_phys,
        "N_1q_virtual_rz": n1_rz,
        "N_2q": n2,
        "N_meas": n_meas,
        "N_reset": n_reset,
        "N_m": n_m,
        "routing_swaps": swaps,
    }


def _conservative_hqc(counts: Dict[str, int], shots: int) -> float:
    """H-series published HQC formula counting Rz in N_1q (matches xtop005/quantinuum.py)."""
    return 5.0 + shots * (counts["N_1q"] + 10 * counts["N_2q"] + 5 * counts["N_m"]) / 5000.0


def run_stage0_on_compiled_tket(
    compiled_by_arm: Dict[str, List[Tuple[x.Built, Any]]],
    shots: int = 128,
    seed: int = 20261121,
) -> Dict[str, Dict[str, Any]]:
    """Verify exact Stage-0 noiseless recovery (F = 1.000000, 0 rejected shots) on compiled H2-1 circuits."""
    from pytket.extensions.qiskit import tk_to_qiskit
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    frame = x.logical_frame(DECISIVE_DEPTH)
    summary: Dict[str, Dict[str, Any]] = {}
    for qarm, pairs in compiled_by_arm.items():
        qcs = [tk_to_qiskit(tk_c, replace_implicit_swaps=True) for _, tk_c in pairs]
        res0 = sim.run(qcs, shots=shots, seed_simulator=seed).result()
        per0 = {
            "".join(b.setting): x.correlators(b, res0.get_counts(idx))
            for idx, (b, _) in enumerate(pairs)
        }
        f_raw = float(x.fidelity(per0, frame, "raw"))
        f_acc = float(x.fidelity(per0, frame, "acc"))
        total_shots = sum(int(v["raw"]["n"]) for v in per0.values())
        acc_shots = sum(int(v["acc"]["n"]) for v in per0.values())
        rejected = total_shots - acc_shots
        if abs(f_raw - 1.0) > 1e-9 or abs(f_acc - 1.0) > 1e-9 or rejected != 0:
            raise AssertionError(
                f"Stage-0 failed for {qarm}: F_raw={f_raw}, F_acc={f_acc}, rejected={rejected}"
            )
        summary[qarm] = {
            "stage0_F_raw": f"{f_raw:.6f}",
            "stage0_F_acc": f"{f_acc:.6f}",
            "stage0_total_shots": total_shots,
            "stage0_rejected_shots": rejected,
        }
    return summary


def generate_decisive_bundle(
    bundle_dir: Path = BUNDLE_DIR,
    shots_hqc: int = DEFAULT_SHOTS,
    shots_stage1_sim: int = 2000,
    seed: int = 20261121,
) -> Dict[str, Any]:
    """Compile the 18 decisive L=3 circuits offline for H2-1, run Stage-0 & Stage-1, and write JCS bundle."""
    from qiskit_aer import AerSimulator

    bundle_dir = Path(bundle_dir)
    circuits_subdir = bundle_dir / "circuits"
    circuits_subdir.mkdir(parents=True, exist_ok=True)

    noisy_sim = AerSimulator(method="stabilizer", noise_model=xq.h2_noise())
    rng = np.random.default_rng(seed)
    frame = x.logical_frame(DECISIVE_DEPTH)

    compiled_by_arm: Dict[str, List[Tuple[x.Built, Any]]] = {}
    circuit_entries: List[Dict[str, Any]] = []
    bundle_list: List[Dict[str, Any]] = []
    arm_summaries: Dict[str, Dict[str, Any]] = {}
    sha_rows: List[Tuple[str, str]] = []

    for qarm in DECISIVE_ARMS:
        label = xq.ARMS_Q[qarm][0]
        built_qcs = [xq.build_q(qarm, DECISIVE_DEPTH, st) for st in x.SETTINGS]

        # Stage-1 H2-like Pauli noise prediction (matching xtop005/quantinuum.py)
        res_noisy = noisy_sim.run(
            [qc for _, qc in built_qcs],
            shots=shots_stage1_sim,
            seed_simulator=seed + DECISIVE_DEPTH,
        ).result()
        per_noisy = {
            "".join(b.setting): x.correlators(b, res_noisy.get_counts(idx))
            for idx, (b, _) in enumerate(built_qcs)
        }
        which = "acc" if qarm != "Q0" else "raw"
        f_raw = float(x.fidelity(per_noisy, frame, "raw"))
        f_used = float(x.fidelity(per_noisy, frame, which))
        ci_lo, ci_hi = x.bootstrap_ci(per_noisy, frame, which, rng)
        p_keep = sum(v["acc"]["n"] for v in per_noisy.values()) / sum(
            v["raw"]["n"] for v in per_noisy.values()
        )

        arm_pairs: List[Tuple[x.Built, Any]] = []
        arm_counts: List[Dict[str, int]] = []
        arm_hqc_cons = 0.0
        arm_hqc_phys = Decimal("0")

        for b, qc in built_qcs:
            tk_c = xq.tk_compile(qc)
            tk_c.name = qc.name
            arm_pairs.append((b, tk_c))

            tk_dict = tk_c.to_dict()
            c_bytes = _jcs_bytes(tk_dict)
            c_rel = f"circuits/{qc.name}.tket.json"
            c_path = bundle_dir / c_rel
            c_path.write_bytes(c_bytes)
            file_sha = _sha256_bytes(c_bytes)
            tk_sha = tket_sha256(tk_c)
            sha_rows.append((file_sha, c_rel))
            bundle_list.append({"name": qc.name, "tket": tk_dict})

            cnt = _detailed_tk_counts(tk_c)
            arm_counts.append(cnt)
            h_cons = _conservative_hqc(cnt, shots_hqc)
            h_phys = local_hqc(tk_c, shots_hqc)
            arm_hqc_cons += h_cons
            arm_hqc_phys += h_phys

            circuit_entries.append(
                {
                    "name": qc.name,
                    "arm": qarm,
                    "label": label,
                    "depth": DECISIVE_DEPTH,
                    "setting": "".join(b.setting),
                    "qubits": cnt["qubits"],
                    "N_1q": cnt["N_1q"],
                    "N_1q_physical": cnt["N_1q_physical"],
                    "N_1q_virtual_rz": cnt["N_1q_virtual_rz"],
                    "N_2q": cnt["N_2q"],
                    "N_meas": cnt["N_meas"],
                    "N_reset": cnt["N_reset"],
                    "N_m": cnt["N_m"],
                    "routing_swaps": cnt["routing_swaps"],
                    "shots": shots_hqc,
                    "hqc_conservative": f"{h_cons:.4f}",
                    "hqc_physical": str(h_phys),
                    "tket_sha256": tk_sha,
                    "file_relpath": c_rel,
                    "file_sha256": file_sha,
                }
            )

        compiled_by_arm[qarm] = arm_pairs
        arm_summaries[qarm] = {
            "arm": qarm,
            "label": label,
            "depth": DECISIVE_DEPTH,
            "n_circuits": len(arm_pairs),
            "qubits": arm_counts[0]["qubits"],
            "totals_9_circuits": {
                k: int(sum(c_[k] for c_ in arm_counts))
                for k in (
                    "N_1q",
                    "N_1q_physical",
                    "N_1q_virtual_rz",
                    "N_2q",
                    "N_meas",
                    "N_reset",
                    "N_m",
                    "routing_swaps",
                )
            },
            "averages_per_circuit": {
                k: f"{float(np.mean([c_[k] for c_ in arm_counts])):.2f}"
                for k in (
                    "N_1q",
                    "N_1q_physical",
                    "N_1q_virtual_rz",
                    "N_2q",
                    "N_meas",
                    "N_reset",
                    "N_m",
                )
            },
            "hqc_9_settings_500_conservative": f"{arm_hqc_cons:.4f}",
            "hqc_9_settings_500_physical": str(arm_hqc_phys),
            "stage1_predicted": {
                "noise_model": {
                    "p_1q": "0.000030",
                    "p_2q": "0.001000",
                    "p_meas": "0.001000",
                    "p_reset": "0.001000",
                    "shots_sim": shots_stage1_sim,
                    "seed": seed,
                },
                "F_raw": f"{f_raw:.6f}",
                "F_used": f"{f_used:.6f}",
                "F_used_ci99": [f"{ci_lo:.6f}", f"{ci_hi:.6f}"],
                "p_keep": f"{p_keep:.6f}",
            },
        }

    stage0_res = run_stage0_on_compiled_tket(compiled_by_arm, shots=128, seed=seed)
    for qarm, st0 in stage0_res.items():
        arm_summaries[qarm]["stage0_noiseless"] = st0

    bundle_bytes = _jcs_bytes(bundle_list)
    bundle_path = bundle_dir / BUNDLE_CIRCUITS_NAME
    bundle_path.write_bytes(bundle_bytes)
    bundle_sha = _sha256_bytes(bundle_bytes)
    sha_rows.append((bundle_sha, BUNDLE_CIRCUITS_NAME))

    total_cons = sum(float(c_["hqc_conservative"]) for c_ in circuit_entries)
    total_phys = sum((Decimal(c_["hqc_physical"]) for c_ in circuit_entries), Decimal("0"))
    hqc_budget_ceil = int(math.ceil(total_cons))
    comp_batch_sha = batch_sha256([c_["tket_sha256"] for c_ in circuit_entries])

    manifest_obj: Dict[str, Any] = {
        "schema": "osiris.xtop005.nexus_h2_decisive_manifest/1",
        "experiment_id": "OSIRIS-XTOP-005",
        "title": "Decisive L=3 Code-Space Hayden-Preskill Comparison (BARE 8Q vs ENC_ALL_CHK 18Q)",
        "canonicalization": "RFC8785-JCS",
        "evidence_class": "OFFLINE_COMPILED",
        "provider_job_id": None,
        "hardware_executed": False,
        "compiler_backend": "pytket.extensions.quantinuum.QuantinuumAPIOffline",
        "compile_target": "H2-1",
        "staging_emulator_target": "H2-1E",
        "optimisation_level": 2,
        "depth": DECISIVE_DEPTH,
        "shots_per_circuit": shots_hqc,
        "n_circuits": len(circuit_entries),
        "hqc_budget": hqc_budget_ceil,
        "hqc_total_conservative": f"{total_cons:.4f}",
        "hqc_total_physical": str(total_phys),
        "hqc_formula": "5 + shots * (N_1q + 10 * N_2q + 5 * N_m) / 5000",
        "compiled_batch_sha256": comp_batch_sha,
        "bundle_file_relpath": BUNDLE_CIRCUITS_NAME,
        "bundle_file_sha256": bundle_sha,
        "arms": arm_summaries,
        "circuits": circuit_entries,
    }
    manifest_bytes = _jcs_bytes(manifest_obj)
    manifest_path = bundle_dir / MANIFEST_NAME
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = _sha256_bytes(manifest_bytes)
    sha_rows.append((manifest_sha, MANIFEST_NAME))

    sha_lines = [f"{digest}  {rel}" for digest, rel in sorted(sha_rows, key=lambda t: t[1])]
    (bundle_dir / SHA256SUMS_NAME).write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    return manifest_obj


def verify_decisive_bundle_offline(
    bundle_dir: Path = BUNDLE_DIR,
    stage0_shots: int = 128,
    seed: int = 20261121,
) -> Dict[str, Any]:
    """Verify SHA-256 lock, RFC 8785 JCS canonicalization, H2-1 ISA compliance, and Stage-0 F=1.0."""
    from pytket import Circuit

    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_NAME
    sums_path = bundle_dir / SHA256SUMS_NAME
    if not manifest_path.is_file():
        raise GovernanceError(f"Missing decisive manifest: {manifest_path}")
    if not sums_path.is_file():
        raise GovernanceError(f"Missing SHA256SUMS lockfile: {sums_path}")

    raw_manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest_bytes.decode("utf-8"))

    # 1. Verify RFC 8785 JCS byte-level canonicalization
    if raw_manifest_bytes != _jcs_bytes(manifest):
        raise GovernanceError("Manifest is not RFC 8785 JCS-canonicalized (byte mismatch)")
    if manifest.get("evidence_class") != "OFFLINE_COMPILED" or manifest.get("provider_job_id") is not None:
        raise GovernanceError(
            "Offline bundle manifest must declare evidence_class='OFFLINE_COMPILED' and provider_job_id=null"
        )

    # 2. Verify SHA256SUMS.txt
    sums_map: Dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("  ", 1)
        if len(parts) != 2:
            raise GovernanceError(f"Malformed SHA256SUMS entry: {line!r}")
        digest, rel = parts
        target_file = bundle_dir / rel
        if not target_file.is_file():
            raise GovernanceError(f"Missing locked file referenced in SHA256SUMS.txt: {rel}")
        actual = _sha256_bytes(target_file.read_bytes())
        if actual != digest:
            raise GovernanceError(
                f"SHA-256 mismatch for {rel}: expected {digest}, found {actual}"
            )
        sums_map[rel] = actual

    if MANIFEST_NAME not in sums_map or BUNDLE_CIRCUITS_NAME not in sums_map:
        raise GovernanceError("SHA256SUMS.txt must cover both manifest and 18-circuit batch bundle")

    # 3. Verify every circuit file, tket_sha256, ISA gates, and rebuild Stage-0 pairs
    circuits_meta = manifest.get("circuits", [])
    if len(circuits_meta) != 18:
        raise GovernanceError(f"Expected exactly 18 decisive circuits, found {len(circuits_meta)}")

    compiled_by_arm: Dict[str, List[Tuple[x.Built, Any]]] = {"Q0": [], "Q3": []}
    tket_hashes: List[str] = []

    for entry in circuits_meta:
        rel = entry["file_relpath"]
        c_path = bundle_dir / rel
        c_bytes = c_path.read_bytes()
        if _sha256_bytes(c_bytes) != entry["file_sha256"]:
            raise GovernanceError(f"Circuit file SHA-256 mismatch on {rel}")
        if sums_map.get(rel) != entry["file_sha256"]:
            raise GovernanceError(f"Circuit file {rel} missing or mismatched in SHA256SUMS.txt")

        tk_dict = json.loads(c_bytes.decode("utf-8"))
        tk_c = Circuit.from_dict(tk_dict)
        tk_c.name = entry["name"]
        actual_tk_sha = tket_sha256(tk_c)
        if actual_tk_sha != entry["tket_sha256"]:
            raise GovernanceError(
                f"tket_sha256 mismatch for {entry['name']}: expected {entry['tket_sha256']}, got {actual_tk_sha}"
            )
        cnt = _detailed_tk_counts(tk_c)
        if cnt["N_2q"] != entry["N_2q"] or cnt["N_1q"] != entry["N_1q"] or cnt["N_m"] != entry["N_m"]:
            raise GovernanceError(f"Gate count mismatch against manifest for {entry['name']}")

        tket_hashes.append(actual_tk_sha)
        setting = (entry["setting"][0], entry["setting"][1])
        built, _ = xq.build_q(entry["arm"], int(entry["depth"]), setting)
        compiled_by_arm[entry["arm"]].append((built, tk_c))

    actual_batch_sha = batch_sha256(tket_hashes)
    if actual_batch_sha != manifest["compiled_batch_sha256"]:
        raise GovernanceError(
            f"compiled_batch_sha256 mismatch: expected {manifest['compiled_batch_sha256']}, got {actual_batch_sha}"
        )

    # Also verify load_circuits compatibility on the 18-circuit bundle file
    bundle_loaded = load_circuits(bundle_dir / manifest["bundle_file_relpath"])
    if [tket_sha256(c_) for c_ in bundle_loaded] != tket_hashes:
        raise GovernanceError("18-circuit batch file disagrees with individual circuit digests")

    # 4. Stage-0 noiseless check on all 18 compiled circuits
    stage0 = run_stage0_on_compiled_tket(compiled_by_arm, shots=stage0_shots, seed=seed)

    receipt_body = {
        "schema": "osiris.xtop005.nexus_readiness_receipt/1",
        "experiment_id": "OSIRIS-XTOP-005",
        "status": "OFFLINE_VERIFIED",
        "evidence_class": "OFFLINE_COMPILED",
        "hardware_executed": False,
        "provider_job_id": None,
        "compile_target": manifest["compile_target"],
        "staging_emulator_target": manifest["staging_emulator_target"],
        "depth": manifest["depth"],
        "n_circuits": len(circuits_meta),
        "shots_per_circuit": manifest["shots_per_circuit"],
        "hqc_budget": manifest["hqc_budget"],
        "hqc_total_conservative": manifest["hqc_total_conservative"],
        "hqc_total_physical": manifest["hqc_total_physical"],
        "compiled_batch_sha256": actual_batch_sha,
        "manifest_sha256": sums_map[MANIFEST_NAME],
        "stage0": stage0,
    }
    receipt_sha = _sha256_bytes(_jcs_bytes(receipt_body))
    receipt_body["readiness_receipt_sha256"] = receipt_sha
    return receipt_body


def submit_decisive_bundle(
    device: str,
    operator_approval_token: str,
    bundle_dir: Path = BUNDLE_DIR,
    shots: int = DEFAULT_SHOTS,
    max_emulator_hqc: float = 1400.0,
    project_name: str = "OSIRIS",
    osiris_home: Optional[str] = None,
    qnx_override: Any = None,
) -> Dict[str, Any]:
    """Submit the verified 18-circuit decisive bundle through OSIRIS Nexus governance gates.

    Fails closed if:
      1. Offline SHA-256 or Stage-0 verification fails.
      2. `operator_approval_token` is empty or invalid.
      3. Target is invalid or Tier 1 syntax-checker only.
      4. HQC budget exceeds cap (`max_emulator_hqc` for Tier 2, or signed token `max_hqc` for Tier 3).
      5. `qnexus` SDK or Nexus authentication credentials are absent.
    Never emits `HARDWARE_ATTESTED` unless `NexusSdkRunner` independently confirms `COMPLETED` on a Tier 3 QPU.
    """
    readiness = verify_decisive_bundle_offline(bundle_dir=bundle_dir)
    spec = resolve_quantinuum_target(device)
    if spec.tier == 1:
        raise GovernanceError(
            f"--submit requires an emulator (H2-1E/H2-2E) or QPU (H2-1/H2-2) target; got {spec.name}"
        )

    tok = (operator_approval_token or "").strip()
    if not tok:
        raise GovernanceError(
            f"Submission to {spec.name} requires a non-empty --operator-approval-token"
        )

    circuits = load_circuits(Path(bundle_dir) / BUNDLE_CIRCUITS_NAME)
    est_phys = sum((local_hqc(c_, shots) for c_ in circuits), Decimal("0"))
    compiled_batch = readiness["compiled_batch_sha256"]

    if spec.tier == 3:
        try:
            verify_tier3_token(
                tok,
                compiled_batch,
                spec,
                shots,
                est_phys,
                osiris_home=osiris_home,
                compiled_sha256=compiled_batch,
            )
        except QuantinuumGateError as exc:
            raise GovernanceError(str(exc)) from exc
    elif spec.tier == 2:
        if est_phys > Decimal(str(max_emulator_hqc)):
            raise GovernanceError(
                f"Estimated HQC ({est_phys}) exceeds --max-emulator-hqc ({max_emulator_hqc}) for {spec.name}"
            )
        # If the operator supplied a structured Tier-3-style signed token, verify signature; otherwise require
        # explicit credential presence before allowing remote Nexus emulator execution.
        if tok.startswith("osiris-t3"):
            raise GovernanceError(
                f"Tier 3 QPU token supplied for Tier 2 emulator target {spec.name}; pass an emulator approval token"
            )

    qnx = qnx_override if qnx_override is not None else load_qnexus()
    # Fail closed if real qnexus has no configured Nexus auth token / credentials in non-mocked mode
    if qnx_override is None and not (
        os.environ.get("QUANTINUUM_NEXUS_TOKEN", "").strip()
        or (Path.home() / ".config" / "qnexus").exists()
    ):
        raise GovernanceError(
            "Nexus credentials absent: no active qnexus login session or QUANTINUUM_NEXUS_TOKEN found"
        )

    ledger = NexusAuditLedger(osiris_home)
    runner = NexusSdkRunner(
        qnx=qnx,
        ledger=ledger,
        project_name=project_name,
        project_description="OSIRIS-XTOP-005 decisive L=3 H2 submission",
    )
    compile_for = spec.name[:-1] if spec.name.endswith("E") else spec.name
    pre = runner.preflight(
        circuits,
        target=compile_for,
        shots=shots,
        optimisation_level=0,
        job_name="xtop005-l3-decisive-preflight",
        approve_remote=True,
    )
    run_out = runner.run(
        pre,
        target=spec.name,
        shots=shots,
        approval_token=tok if spec.tier == 3 else None,
        max_emulator_hqc=max_emulator_hqc,
        job_name="xtop005-l3-decisive-run",
        approve_remote=True,
    )
    evidence_class = (
        "HARDWARE_ATTESTED" if run_out.get("disposition") == "HARDWARE_ATTESTED" else "SIMULATED"
    )
    return {
        "experiment_id": "OSIRIS-XTOP-005",
        "evidence_class": evidence_class,
        "provider_job_id": run_out.get("job_id"),
        "offline_readiness": readiness,
        "preflight": pre.summary(),
        "execution": run_out,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="OSIRIS-XTOP-005 hash-locked Nexus H2-1E / H2-1 decisive 18-circuit runner"
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=BUNDLE_DIR,
        help="Path to hash-locked nexus_h2_bundle directory",
    )
    parser.add_argument(
        "--generate-bundle",
        action="store_true",
        help="Compile all 18 L=3 circuits offline and write the JCS-canonicalized bundle + SHA256SUMS.txt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify SHA-256 lock + Stage-0 noiseless simulation offline (default unless --submit/--live-approved)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit verified bundle via OSIRIS Nexus governance gate (requires --operator-approval-token)",
    )
    parser.add_argument(
        "--live-approved",
        action="store_true",
        help="Explicit live/remote submission confirmation flag (companion/alias for --submit)",
    )
    parser.add_argument(
        "--device",
        default="H2-1E",
        help="Quantinuum Nexus device target for --submit (default: H2-1E)",
    )
    parser.add_argument(
        "--operator-approval-token",
        default="",
        help="Operator approval token (mandatory for --submit)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help="Shots per circuit (default: 500)",
    )
    parser.add_argument(
        "--max-emulator-hqc",
        type=float,
        default=1400.0,
        help="Hard HQC budget cap for Tier 2 emulator runs (default: 1400.0 for the 1,336 HQC bundle)",
    )
    parser.add_argument(
        "--project",
        default="OSIRIS",
        help="Quantinuum Nexus project name (default: OSIRIS)",
    )
    parser.add_argument(
        "--osiris-home",
        default=None,
        help="Optional OSIRIS_HOME override for ledger/key lookup",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.generate_bundle:
            manifest = generate_decisive_bundle(bundle_dir=args.bundle_dir, shots_hqc=args.shots)
            print(
                json.dumps(
                    {
                        "status": "BUNDLE_GENERATED",
                        "bundle_dir": str(args.bundle_dir),
                        "n_circuits": manifest["n_circuits"],
                        "hqc_budget": manifest["hqc_budget"],
                        "hqc_total_conservative": manifest["hqc_total_conservative"],
                        "hqc_total_physical": manifest["hqc_total_physical"],
                        "compiled_batch_sha256": manifest["compiled_batch_sha256"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        wants_submit = bool(args.submit or args.live_approved)
        if not wants_submit or args.dry_run:
            receipt = verify_decisive_bundle_offline(bundle_dir=args.bundle_dir)
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0

        res = submit_decisive_bundle(
            device=args.device,
            operator_approval_token=args.operator_approval_token,
            bundle_dir=args.bundle_dir,
            shots=args.shots,
            max_emulator_hqc=args.max_emulator_hqc,
            project_name=args.project,
            osiris_home=args.osiris_home,
        )
        print(json.dumps(res, indent=2, sort_keys=True, default=str))
        return 0
    except (GovernanceError, NexusSdkUnavailable, QuantinuumGateError, AssertionError, ValueError) as exc:
        print(f"NEXUS_GOVERNANCE_GATE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
