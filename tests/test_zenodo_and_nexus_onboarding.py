"""Unit tests for `papers/publish_nov23_papers_to_zenodo.py` and `scripts/nexus_onboard_preflight.py`.

Verifies:
  1. Zero-network `--dry-run` staging of all 4 Nov 23, 2026 manuscripts (`SYS-024`, `SHA256SUMS.txt`,
     `reproducibility_capsule.json`, Zenodo Deposition metadata, deterministic `zenodo_staging_manifest_nov23.json`).
  2. Fail-closed `SYS-024` (un-negated refuted claim & quarantined constant) and `SHA256SUMS.txt` tamper detection.
  3. Fail-closed `--execute` guards without `ZENODO_TOKEN` or `--confirm-publish`, plus mock HTTP draft upload.
  4. Zero-network `--offline-dry-run` / `--status` of `nexus_onboard_preflight.py` (`ENG-NEXUS-4`),
     `osiris.quantum_protocol/1` pre-gate validation, and exact HQC formula verification.
  5. Fail-closed `--preflight-live` guards (missing `--operator-approved`, missing credentials, paid QPU target,
     multi-circuit canary) and mocked single-circuit `H2-1SC` / `H2-1E` HQC comparison + secret redaction.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ZENODO_SCRIPT_PATH = REPO_ROOT / "papers" / "publish_nov23_papers_to_zenodo.py"
NEXUS_SCRIPT_PATH = REPO_ROOT / "scripts" / "nexus_onboard_preflight.py"

EXPECTED_NOV23_PAPERS = (
    "xtop006_non_clifford_kraus_fez",
    "wp2_27q_debiased_ot_fez",
    "nonabelian_2bga_and_adv1",
    "heron_r2_calibration_staleness_and_colored_nulls",
    "bt2t001_binary_tetrahedral_fez",
    "bt2t002_mul_and_nonabelian_qft",
    "echo_and_card_c_qec_gates",
    "rep3_echo_fez_and_multidistance_qec",
    "bt2t003_kogut_susskind_plaquette",
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zenodo_mod = _load_module("publish_nov23_papers_to_zenodo", ZENODO_SCRIPT_PATH)
nexus_mod = _load_module("nexus_onboard_preflight", NEXUS_SCRIPT_PATH)


def _forbid_network(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError(f"Unexpected network call invoked: args={args!r}, kwargs={kwargs!r}")


# ---------------------------------------------------------------------------
# 1. Zenodo Nov 23 Dry-Run, SYS-024, SHA256SUMS & Fail-Closed Live Guards
# ---------------------------------------------------------------------------


def test_zenodo_dry_run_all_four_papers_pass_sys024_and_sha256sums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)
    out_a = tmp_path / "zenodo_staging_manifest_nov23.json"
    out_b = tmp_path / "zenodo_staging_manifest_nov23_repeat.json"

    rc_a = zenodo_mod.main(["--dry-run", "--manifest-out", str(out_a)])
    rc_b = zenodo_mod.main(["--manifest-out", str(out_b)])  # default mode must also be --dry-run
    assert rc_a == 0
    assert rc_b == 0
    assert out_a.read_bytes() == out_b.read_bytes()

    manifest = json.loads(out_a.read_text(encoding="utf-8"))
    assert manifest["schema"] == "osiris.zenodo_staging_manifest/1"
    assert manifest["mode"] == "DRY_RUN_STAGED"
    assert manifest["network_calls"] == 0
    assert manifest["publication_date"] == "2026-11-23"
    assert manifest["paper_count"] == len(EXPECTED_NOV23_PAPERS)
    assert tuple(manifest["paper_order"]) == EXPECTED_NOV23_PAPERS

    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    assert manifest["manifest_sha256"] == zenodo_mod.sha256_bytes(
        zenodo_mod.canonical_json_bytes(body)
    )

    for key in EXPECTED_NOV23_PAPERS:
        entry = manifest["papers"][key]
        assert entry["sha256sums_check"]["passed"] is True
        assert entry["sha256sums_check"]["verified_count"] >= 6
        assert entry["sys024_check"]["passed"] is True
        assert entry["sys024_check"]["flagged_claims"] == []
        assert entry["sys024_check"]["quarantined_constants"] == []

        meta = entry["metadata"]
        assert meta["upload_type"] == "publication"
        assert meta["publication_type"] in ("preprint", "article")
        assert meta["publication_date"] == "2026-11-23"
        assert meta["creators"] == [
            {"name": "Davis, Devin Phillip", "affiliation": "Agile Defense Systems LLC"}
        ]
        assert len(meta["title"]) > 20
        assert len(meta["description"]) > 100
        assert len(meta["keywords"]) >= 5
        assert any(
            r.get("identifier") == "https://github.com/quantum-osiris/osiris-cli"
            for r in meta["related_identifiers"]
        )

        files = entry["files"]
        assert len(files) == entry["total_files"] >= 10
        upload_names = [f["upload_name"] for f in files]
        assert len(upload_names) == len(set(upload_names))
        for fobj in files:
            assert len(fobj["sha256"]) == 64
            assert int(fobj["size_bytes"]) > 0


def _build_minimal_paper_fixture(root: Path, paper_key: str = "demo_paper") -> Dict[str, Any]:
    pdir = root / "papers" / paper_key
    pdir.mkdir(parents=True, exist_ok=True)

    gen_tex = "\\newcommand{\\DemoShots}{4,096}\n"
    tex = (
        "\\documentclass{article}\n\\input{generated.tex}\n\\begin{document}\n"
        "A preregistered hardware experiment with \\DemoShots\\ shots measured Bell fidelity.\n"
        "\\end{document}\n"
    )
    (pdir / "generated.tex").write_text(gen_tex, encoding="utf-8")
    (pdir / f"{paper_key}.tex").write_text(tex, encoding="utf-8")

    cap_body = {
        "paper_id": "DEMO-PAPER",
        "schema": "osiris.reproducibility_capsule/1",
    }
    cap_sha = zenodo_mod.sha256_bytes(zenodo_mod.canonical_json_bytes(cap_body))
    capsule = {**cap_body, "rfc8785_canonical_sha256": cap_sha}
    (pdir / "reproducibility_capsule.json").write_text(
        json.dumps(capsule, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sums_lines = []
    for fname in ("generated.tex", f"{paper_key}.tex", "reproducibility_capsule.json"):
        digest = hashlib.sha256((pdir / fname).read_bytes()).hexdigest()
        sums_lines.append(f"{digest}  {fname}")
    (pdir / "SHA256SUMS.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

    return {
        "dir": f"papers/{paper_key}",
        "tex_file": f"{paper_key}.tex",
        "publication_type": "preprint",
        "version": "1.0.0",
        "related": [],
        "files": [
            (f"papers/{paper_key}/{paper_key}.tex", None),
            (f"papers/{paper_key}/generated.tex", None),
            (f"papers/{paper_key}/reproducibility_capsule.json", None),
            (f"papers/{paper_key}/SHA256SUMS.txt", None),
        ],
        "optional_files": [
            (f"papers/{paper_key}/{paper_key}.pdf", None),
        ],
        "meta": {
            "title": "Preregistered benchmark study on superconducting qubits",
            "description": "<p>Clean preregistered benchmark description.</p>",
            "keywords": ["quantum benchmarking", "preregistration"],
        },
    }


def test_zenodo_sys024_and_sha256sums_tamper_guards_fail_closed(tmp_path: Path) -> None:
    cfg = _build_minimal_paper_fixture(tmp_path)
    pdir = tmp_path / "papers" / "demo_paper"

    # Clean fixture stages cleanly (and picks up optional PDF when present)
    (pdir / "demo_paper.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    staged = zenodo_mod.stage_paper_bundle("demo_paper", cfg, root=tmp_path)
    assert staged["pdf_compiled"] is True
    assert any(f["upload_name"] == "demo_paper.pdf" for f in staged["files"])

    # 1. Corrupting a locked file triggers SHA256_MISMATCH
    orig_tex = (pdir / "demo_paper.tex").read_text(encoding="utf-8")
    (pdir / "demo_paper.tex").write_text(orig_tex + "% tampered\n", encoding="utf-8")
    with pytest.raises(zenodo_mod.ZenodoPublishError) as exc_sha:
        zenodo_mod.stage_paper_bundle("demo_paper", cfg, root=tmp_path)
    assert exc_sha.value.code == "SHA256_MISMATCH"

    # 2. Injecting an un-negated refuted claim (even with updated SHA256SUMS) triggers SYS024_FLAGGED_CLAIM
    bad_tex = (
        "\\documentclass{article}\n\\input{generated.tex}\n\\begin{document}\n"
        "OSIRIS operates on a 156-qubit quantum processor during training.\n"
        "\\end{document}\n"
    )
    (pdir / "demo_paper.tex").write_text(bad_tex, encoding="utf-8")
    sums_lines = [
        f"{hashlib.sha256((pdir / fn).read_bytes()).hexdigest()}  {fn}"
        for fn in ("generated.tex", "demo_paper.tex", "reproducibility_capsule.json")
    ]
    (pdir / "SHA256SUMS.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")
    with pytest.raises(zenodo_mod.ZenodoPublishError) as exc_claim:
        zenodo_mod.stage_paper_bundle("demo_paper", cfg, root=tmp_path)
    assert exc_claim.value.code == "SYS024_FLAGGED_CLAIM"

    # 3. Injecting a quarantined SYS-024 constant into metadata triggers SYS024_QUARANTINED_CONSTANT
    (pdir / "demo_paper.tex").write_text(orig_tex, encoding="utf-8")
    sums_lines_clean = [
        f"{hashlib.sha256((pdir / fn).read_bytes()).hexdigest()}  {fn}"
        for fn in ("generated.tex", "demo_paper.tex", "reproducibility_capsule.json")
    ]
    (pdir / "SHA256SUMS.txt").write_text("\n".join(sums_lines_clean) + "\n", encoding="utf-8")
    bad_cfg = {
        **cfg,
        "meta": {
            **cfg["meta"],
            "description": "<p>Resonance angle theta_lock = 51.843 degrees.</p>",
        },
    }
    with pytest.raises(zenodo_mod.ZenodoPublishError) as exc_const:
        zenodo_mod.stage_paper_bundle("demo_paper", bad_cfg, root=tmp_path)
    assert exc_const.value.code == "SYS024_QUARANTINED_CONSTANT"


def test_zenodo_execute_fails_closed_without_token_or_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    monkeypatch.delenv("ZENODO_API_TOKEN", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)

    manifest_out = tmp_path / "manifest.json"
    # Missing ZENODO_TOKEN -> exit code 2, 0 network calls
    rc_no_tok = zenodo_mod.main(
        ["--execute", "--confirm-publish", "--paper", "xtop006", "--manifest-out", str(manifest_out)]
    )
    assert rc_no_tok == 2

    # Present ZENODO_TOKEN, missing --confirm-publish -> exit code 2, 0 network calls
    monkeypatch.setenv("ZENODO_TOKEN", "zenodo-unit-test-secret-token")
    rc_no_confirm = zenodo_mod.main(
        ["--execute", "--paper", "xtop006", "--manifest-out", str(manifest_out)]
    )
    assert rc_no_confirm == 2

    # Mocked HTTP execution with both ZENODO_TOKEN and confirm_publish=True succeeds offline
    recorded_http: List[Tuple[str, str]] = []

    def fake_http(
        method: str,
        url: str,
        token: str,
        data: Optional[Dict[str, Any]],
        raw: Optional[bytes],
        ctype: str,
    ) -> Tuple[int, Dict[str, Any]]:
        assert token == "zenodo-unit-test-secret-token"
        recorded_http.append((method, url))
        if method == "POST" and url.endswith("/depositions"):
            return 201, {
                "id": 998877,
                "links": {
                    "bucket": "https://zenodo.org/api/files/mock-bucket",
                    "html": "https://zenodo.org/deposit/998877",
                },
            }
        if method == "PUT" and url.endswith("/depositions/998877"):
            assert data is not None and data["metadata"]["publication_date"] == "2026-11-23"
            return 200, {"metadata": {"prereserve_doi": {"doi": "10.5281/zenodo.998877"}}}
        if method == "PUT" and "mock-bucket/" in url:
            assert raw is not None and ctype == "application/octet-stream"
            return 201, {"size": len(raw)}
        raise AssertionError(f"Unexpected call {method} {url}")

    exec_out = zenodo_mod.execute_zenodo_upload(
        paper_keys=["xtop006_non_clifford_kraus_fez"],
        manifest_out=manifest_out,
        confirm_publish=True,
        receipt_dir=tmp_path / "receipts",
        env={"ZENODO_TOKEN": "zenodo-unit-test-secret-token"},
        http_call=fake_http,
    )
    assert exec_out["mode"] == "EXECUTED"
    rec = exec_out["receipts"]["xtop006_non_clifford_kraus_fez"]
    assert rec["deposition_id"] == 998877
    assert rec["state"] == "draft"
    assert rec["doi"] == "10.5281/zenodo.998877"
    assert (tmp_path / "receipts" / "zenodo_receipt_xtop006_non_clifford_kraus_fez.json").is_file()
    assert len(recorded_http) == 2 + len(rec["files"])


# ---------------------------------------------------------------------------
# 2. Nexus Onboarding & ENG-NEXUS-4 Single-Circuit Preflight Tests
# ---------------------------------------------------------------------------


class _MockRef:
    def __init__(self, payload: Any = None) -> None:
        self.id = str(uuid.uuid4())
        self.payload = payload

    def download_circuit(self) -> Any:
        return self.payload.copy()


class _MockCompileResult:
    def __init__(self, ref: _MockRef) -> None:
        self._ref = ref

    def get_output(self) -> _MockRef:
        return self._ref


class FakeQnexusForPreflight:
    """Offline stand-in matching `qnexus` 0.51.4 signatures used by `NexusSdkRunner.preflight`."""

    def __init__(self, forced_cost: Optional[float] = None) -> None:
        self.projects = self.circuits = self.jobs = self
        self._jobs: Dict[str, List[_MockCompileResult]] = {}
        self.forced_cost = forced_cost
        self.execute_called = False

    def QuantinuumConfig(self, *, device_name: str, **kw: Any) -> Dict[str, Any]:
        return {"type": "QuantinuumConfig", "device_name": device_name, **kw}

    def get_or_create(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> _MockRef:
        return _MockRef()

    def upload(
        self,
        *,
        circuit: Any,
        project: Any = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> _MockRef:
        return _MockRef(circuit)

    def start_compile_job(
        self,
        *,
        programs: List[_MockRef],
        backend_config: Dict[str, Any],
        name: str,
        project: Any = None,
        optimisation_level: int = 0,
        **kw: Any,
    ) -> _MockRef:
        job = _MockRef()
        self._jobs[job.id] = [_MockCompileResult(_MockRef(r.payload)) for r in programs]
        return job

    def wait_for(self, *, job: _MockRef, timeout: Optional[float] = None, **kw: Any) -> None:
        return None

    def results(self, *, job: _MockRef, **kw: Any) -> List[_MockCompileResult]:
        return self._jobs[job.id]

    def cost(
        self,
        *,
        circuit_ref: _MockRef,
        n_shots: int,
        backend_config: Dict[str, Any],
        syntax_checker: Optional[str] = None,
        project: Any = None,
        timeout: Optional[float] = None,
    ) -> float:
        assert syntax_checker == "H2-1SC"
        assert project is not None
        if self.forced_cost is not None:
            return self.forced_cost
        from osiris_cli.providers.nexus_sdk import local_hqc

        return float(local_hqc(circuit_ref.payload, n_shots))

    def start_execute_job(self, **kw: Any) -> Any:
        self.execute_called = True
        raise AssertionError("ENG-NEXUS-4 preflight must NEVER start a paid execution job")


def test_nexus_offline_dry_run_and_status_verify_hqc_and_pregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)
    empty_home = tmp_path / "clean_home"
    empty_home.mkdir()

    out1 = tmp_path / "receipt_1.json"
    out2 = tmp_path / "receipt_2.json"

    r1 = nexus_mod.run_offline_preflight(
        mode_label="OFFLINE_DRY_RUN",
        wen_bundle_dir=tmp_path / "absent_wen_dir",
        shots=500,
        receipt_out=out1,
        env={},
        home_dir=empty_home,
    )
    r2 = nexus_mod.run_offline_preflight(
        mode_label="OFFLINE_DRY_RUN",
        wen_bundle_dir=tmp_path / "absent_wen_dir",
        shots=500,
        receipt_out=out2,
        env={},
        home_dir=empty_home,
    )
    assert out1.read_bytes() == out2.read_bytes()
    assert r1["receipt_sha256"] == r2["receipt_sha256"]
    assert r1["claim_id"] == "ENG-NEXUS-4"
    assert r1["verification_queue_check"] == 8
    assert r1["network_calls"] == 0
    assert r1["hardware_executed"] is False
    assert r1["paid_qpu_hqc_spent"] == "0.0000"
    assert r1["auth_status"]["auth_configured"] is False
    assert r1["protocol_pregate"]["passed"] is True
    assert r1["protocol_pregate"]["schema"] == "osiris.quantum_protocol_pregate/1"

    canary = r1["primary_canary"]
    assert canary["name"] == "xtop005q_Q0_L3_ZZ"
    assert canary["qubits"] == 8
    assert canary["N_1q_physical"] == 39
    assert canary["N_1q_virtual_rz"] == 4
    assert canary["N_2q"] == 22
    assert canary["N_m"] == 12
    assert canary["routing_swaps"] == 0
    assert canary["local_hqc_physical"] == "36.9000"
    assert canary["local_hqc_conservative"] == "37.3000"
    assert canary["bundle_manifest_verified"] is True
    assert r1["wen_plaquette_canary"]["present"] is False

    # CLI --status entrypoint also succeeds with exit code 0
    status_out = tmp_path / "status_receipt.json"
    assert nexus_mod.main(["--status", "--receipt-out", str(status_out)]) == 0
    st_json = json.loads(status_out.read_text(encoding="utf-8"))
    assert st_json["mode"] == "STATUS"
    if nexus_mod.DEFAULT_WEN_BUNDLE_DIR.is_dir():
        assert st_json["wen_plaquette_canary"]["present"] is True
        assert st_json["wen_plaquette_canary"]["local_hqc_physical"] == "70.2000"
        assert st_json["wen_plaquette_canary"]["hqc_1000_shots_meas_reset_only"] == "115.4000"


def test_nexus_protocol_pregate_blocks_incomplete_manifest(tmp_path: Path) -> None:
    bad_proto = {
        "schema": "osiris.quantum_protocol/1",
        "shots_per_circuit": 500,
        "num_blocks": 3,
        # Missing classical_baseline and colored_noise / drift controls
    }
    with pytest.raises(nexus_mod.NexusPreflightError) as exc_info:
        nexus_mod.run_offline_preflight(
            protocol_manifest=bad_proto,
            receipt_out=tmp_path / "blocked.json",
            env={},
            home_dir=tmp_path,
        )
    assert exc_info.value.code == "BLOCKED_PROTOCOL_PREGATE"


def test_nexus_preflight_live_fail_closed_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)
    clean_home = tmp_path / "no_creds_home"
    clean_home.mkdir()
    fake_qnx = FakeQnexusForPreflight()

    # 1. Paid Tier 3 QPU target is strictly forbidden in onboarding preflight
    with pytest.raises(nexus_mod.NexusPreflightError) as exc_qpu:
        nexus_mod.run_live_canary_preflight(
            target="H2-1",
            operator_approved=True,
            env={"QNEXUS_TOKEN": "tok-12345"},
            home_dir=clean_home,
            qnx_override=fake_qnx,
        )
    assert exc_qpu.value.code == "BLOCKED_PAID_QPU_TARGET"

    # 2. Missing --operator-approved fails closed even when credentials are present
    with pytest.raises(nexus_mod.NexusPreflightError) as exc_app:
        nexus_mod.run_live_canary_preflight(
            target="H2-1SC",
            operator_approved=False,
            env={"QNEXUS_TOKEN": "tok-12345"},
            home_dir=clean_home,
            qnx_override=fake_qnx,
        )
    assert exc_app.value.code == "BLOCKED_OPERATOR_APPROVAL"

    # 3. Missing credentials fails closed even when --operator-approved is True
    with pytest.raises(nexus_mod.NexusPreflightError) as exc_cred:
        nexus_mod.run_live_canary_preflight(
            target="H2-1SC",
            operator_approved=True,
            env={},
            home_dir=clean_home,
            qnx_override=fake_qnx,
        )
    assert exc_cred.value.code == "BLOCKED_MISSING_CREDENTIALS"

    # 4. Multi-circuit bundle passed as canary fails closed (single-circuit canary invariant)
    multi_bundle = REPO_ROOT / "xtop005" / "nexus_h2_bundle" / "circuits_h2_decisive_18_tket.json"
    with pytest.raises(nexus_mod.NexusPreflightError) as exc_multi:
        nexus_mod.run_live_canary_preflight(
            target="H2-1SC",
            operator_approved=True,
            canary_path=multi_bundle,
            env={"QNEXUS_TOKEN": "tok-12345"},
            home_dir=clean_home,
            qnx_override=fake_qnx,
        )
    assert exc_multi.value.code == "MULTI_CIRCUIT_CANARY_FORBIDDEN"

    # CLI --preflight-live without --operator-approved exits 2
    monkeypatch.delenv("QNEXUS_TOKEN", raising=False)
    monkeypatch.delenv("QUANTINUUM_NEXUS_TOKEN", raising=False)
    assert nexus_mod.main(["--preflight-live", "--receipt-out", str(tmp_path / "r.json")]) == 2


def test_nexus_preflight_live_single_canary_compares_hqc_and_redacts_secrets(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pytket")
    fake_home = tmp_path / "op_home"
    (fake_home / ".qnx").mkdir(parents=True)
    (fake_home / ".qnx" / "auth.json").write_text('{"logged_in": true}\n', encoding="utf-8")

    secret_tok = "qnx-top-secret-credential-value-987654"
    env = {"QNEXUS_TOKEN": secret_tok}
    out_ok = tmp_path / "live_supported_receipt.json"
    fake_ok = FakeQnexusForPreflight()

    res_ok = nexus_mod.run_live_canary_preflight(
        target="H2-1E",
        operator_approved=True,
        wen_bundle_dir=tmp_path / "none",
        shots=500,
        receipt_out=out_ok,
        osiris_home=tmp_path / "osiris_home",
        env=env,
        home_dir=fake_home,
        qnx_override=fake_ok,
    )
    assert fake_ok.execute_called is False
    assert res_ok["mode"] == "PREFLIGHT_LIVE"
    assert res_ok["requested_target"] == "H2-1E"
    assert res_ok["emulator_config_verified"] is True
    assert res_ok["canary_circuits_uploaded"] == 1
    assert res_ok["sdk_calls_recorded"].count("circuits.upload") == 1
    assert res_ok["sdk_calls_recorded"].count("circuits.cost") == 1
    assert "start_execute_job" not in res_ok["sdk_calls_recorded"]
    assert res_ok["paid_qpu_hqc_spent"] == "0.0000"
    assert res_ok["hardware_executed"] is False
    assert res_ok["hqc_comparison"]["provider_hqc"] == "36.9000"
    assert res_ok["hqc_comparison"]["canary_local_hqc_physical"] == "36.9000"
    assert res_ok["hqc_comparison"]["exact_match_physical"] is True
    assert res_ok["eng_nexus_4_status"] == "SUPPORTED"

    raw_written = out_ok.read_text(encoding="utf-8")
    assert secret_tok not in raw_written
    assert str(fake_home.resolve()) not in raw_written

    # Divergent provider HQC triggers ADAPTER_HQC_MISMATCH cleanly
    fake_mismatch = FakeQnexusForPreflight(forced_cost=99.5)
    res_bad = nexus_mod.run_live_canary_preflight(
        target="H2-1SC",
        operator_approved=True,
        wen_bundle_dir=tmp_path / "none",
        shots=500,
        receipt_out=tmp_path / "live_mismatch_receipt.json",
        osiris_home=tmp_path / "osiris_home_2",
        env=env,
        home_dir=fake_home,
        qnx_override=fake_mismatch,
    )
    assert res_bad["eng_nexus_4_status"] == "ADAPTER_HQC_MISMATCH"
    assert res_bad["hqc_comparison"]["within_tolerance"] is False


def test_wen_plaquette_bundle_verification_and_tamper_guard(tmp_path: Path) -> None:
    wen_dir = tmp_path / "wen_bundle"
    wen_dir.mkdir()
    tk_json = {
        "bits": [["c", [0]], ["c", [1]]],
        "commands": [
            {"args": [["q", [0]]], "op": {"params": ["0.5", "0.0"], "type": "PhasedX"}},
            {"args": [["q", [0]], ["q", [1]]], "op": {"params": ["0.5"], "type": "ZZPhase"}},
            {
                "args": [["c", [0]], ["q", [1]]],
                "op": {
                    "conditional": {"op": {"params": ["1.0"], "type": "Rz"}, "value": 1, "width": 1},
                    "type": "Conditional",
                },
            },
            {"args": [["q", [0]], ["c", [0]]], "op": {"type": "Measure"}},
            {"args": [["q", [0]]], "op": {"type": "Reset"}},
            {"args": [["q", [1]], ["c", [1]]], "op": {"type": "Measure"}},
        ],
        "created_qubits": [],
        "discarded_qubits": [],
        "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]],
        "name": "wen_plaquette_qec002_x_basis_h2_isa",
        "phase": "0.0",
        "qubits": [["q", [0]], ["q", [1]]],
    }
    isa_path = wen_dir / "wen_plaquette_qec002_x_basis_h2_isa.json"
    isa_bytes = json.dumps(tk_json, sort_keys=True, separators=(",", ":")).encode("utf-8")
    isa_path.write_bytes(isa_bytes)
    isa_sha = hashlib.sha256(isa_bytes).hexdigest()

    # 1 PhasedX + 10*1 ZZPhase + 5*(2 Measure + 1 Reset) = 1 + 10 + 15 = 26 -> @1000 shots: 5 + 26/5 = 10.2
    manifest = {
        "circuits": {
            "x_basis": {
                "file_sha256": isa_sha,
                "hqc_budget": {
                    "n_1q_phasedx": 1,
                    "n_2q_zzphase": 1,
                    "n_m_total": 3,
                    "hqc_by_shots": {"1000": 10.2},
                },
            }
        }
    }
    (wen_dir / "nexus_qec002_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (wen_dir / "SHA256SUMS.txt").write_text(
        f"{isa_sha}  wen_plaquette_qec002_x_basis_h2_isa.json\n",
        encoding="utf-8",
    )

    verified = nexus_mod.verify_wen_plaquette_canary(wen_bundle_dir=wen_dir, shots=500)
    assert verified["present"] is True
    assert verified["manifest_verified"] is True
    # With state prep: N_m = 2 + 2 + 1 = 5 -> 1 + 10 + 25 = 36 units -> 5 + 500*36/5000 = 8.6000
    assert verified["local_hqc_physical"] == "8.6000"
    assert verified["hqc_1000_shots_meas_reset_only"] == "10.2000"

    # Corrupting SHA256SUMS triggers fail-closed WEN_SHA256_MISMATCH
    (wen_dir / "SHA256SUMS.txt").write_text(
        f"{'0' * 64}  wen_plaquette_qec002_x_basis_h2_isa.json\n",
        encoding="utf-8",
    )
    with pytest.raises(nexus_mod.NexusPreflightError) as exc_wen:
        nexus_mod.verify_wen_plaquette_canary(wen_bundle_dir=wen_dir, shots=500)
    assert exc_wen.value.code == "WEN_SHA256_MISMATCH"


def test_committed_staging_manifest_and_offline_receipt_integrity() -> None:
    manifest_path = REPO_ROOT / "papers" / "zenodo_staging_manifest_nov23.json"
    receipt_path = REPO_ROOT / "xtop005" / "eng_nexus_4_preflight_receipt.json"
    assert manifest_path.is_file()
    assert receipt_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    m_body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    assert manifest["manifest_sha256"] == zenodo_mod.sha256_bytes(
        zenodo_mod.canonical_json_bytes(m_body)
    )
    assert set(manifest["papers"].keys()) == set(EXPECTED_NOV23_PAPERS)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    r_body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    assert receipt["receipt_sha256"] == nexus_mod.sha256_bytes(
        nexus_mod.canonical_json_bytes(r_body)
    )
    assert receipt["paid_qpu_hqc_spent"] == "0.0000"
    assert receipt["hardware_executed"] is False

