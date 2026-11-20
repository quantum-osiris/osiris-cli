"""Test-session setup shared by every test file.

The swarm and console call whatever Ollama server OLLAMA_HOST names (default
localhost:11434). A busy local Ollama made tests/test_integration.py hang for
minutes on a developer machine, while CI (no Ollama) passed. Tests therefore run
against an address nothing listens on, so every model call takes the documented
fallback path; set OSIRIS_TEST_LIVE_OLLAMA=1 to test against a real server.
"""
import atexit
import importlib.util
import os
import shutil
import sys
import tempfile

import pytest

if os.environ.get("OSIRIS_TEST_LIVE_OLLAMA") != "1":
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"      # discard port: connection refused at once
    os.environ["OSIRIS_GEMMA_MOCK"] = "1"

# Provide backward compatibility for fake Heron backend if not exported by runtime
try:
    import qiskit_ibm_runtime.fake_provider as _fp
    if not hasattr(_fp, "FakeKingston") and hasattr(_fp, "FakeMarrakesh"):
        _fp.FakeKingston = _fp.FakeMarrakesh
except ImportError:
    pass


# ── the operator's signing key ────────────────────────────────────────────────────────────────
# A CodeAgent built without an explicit enclave signs approvals with the key in
# pqc_enclave.default_enclave_dir(): ~/.osiris/enclave unless OSIRIS_ENCLAVE_DIR names another
# place. Tests did that with the real directory. On a developer's machine they signed with the
# operator's own key and appended to its consumed-grants log; on a machine with no key (CI) the
# first such test failed, a later one answered "yes" to key setup and created a key there, and
# every test after it passed -- so the result depended on what an earlier run had left behind.
# The whole session therefore uses a throwaway enclave with a key already set up.
_TEST_ENCLAVE = tempfile.mkdtemp(prefix="osiris-test-enclave-")
os.environ["OSIRIS_ENCLAVE_DIR"] = _TEST_ENCLAVE
atexit.register(shutil.rmtree, _TEST_ENCLAVE, ignore_errors=True)

_TEST_EXEC_DIR = tempfile.mkdtemp(prefix="osiris-test-exec-")
os.environ.setdefault("OSIRIS_EXEC_JOURNAL", os.path.join(_TEST_EXEC_DIR, "exec_journal.jsonl"))
os.environ.setdefault("OSIRIS_EXEC_ARTIFACTS_DIR", os.path.join(_TEST_EXEC_DIR, "runs"))
atexit.register(shutil.rmtree, _TEST_EXEC_DIR, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _throwaway_operator_key():
    from osiris_cli.pqc_enclave import keys_present, setup_local_keys
    if not keys_present(_TEST_ENCLAVE):
        setup_local_keys(_TEST_ENCLAVE, confirm=lambda prompt: True)
    yield _TEST_ENCLAVE


# ── Python 3.9 ────────────────────────────────────────────────────────────────────────────────
# osiris-cli supports Python 3.9, but dnalang and osiris-governance (pyproject's [research]
# extra) require 3.10+. On 3.9 the tests below therefore cannot run. Each is skipped by name with the package it needs as the reason -- and only when
# Python is older than 3.10 AND that package really is not importable. On 3.10+ nothing here
# applies: a missing package is a collection error or a failure there, never a skip.
PY39_MIN_FOR_RESEARCH = (3, 10)
# Test modules that import the package while being imported.
PY39_MODULES = {
    "test_compiler_strict.py": "dnalang",
    "test_currency_card.py": "dnalang",
    "test_dnalang_loop.py": "dnalang",
    "test_edge_provisioner_x3_port.py": "dnalang",
    "test_edge_provisioning_and_e0_e3.py": "dnalang",
    "test_evolution_bridge.py": "dnalang",
    "test_falsification_engine.py": "dnalang",
    "test_proof_carrying_evolution.py": "dnalang",
    "test_trio_authz.py": "dnalang",
    "test_trio_command.py": "dnalang",
}
# Single tests that import it when they run.
PY39_TESTS = {
    "test_compiler_fallback_characterization.py::test_loop_validation_accepts_a_genome_whose_body_names_a_forbidden_op":
        "dnalang",
    "test_production_synth_and_r0_r4.py::test_r3_sprint_execute_runs_verified_first_passing_and_records_run_artifact":
        "dnalang",   # the run is recorded in the genome ledger, whose hash chain is dnalang's
    "test_workstream_h_corpus.py::test_fixture_08_malformed_dnalang_rejected_without_fake_simulation_or_receipt":
        "osiris_governance",
    "test_workstream_h_corpus.py::test_fixture_10_two_qubit_bell_simulator_real_counts_and_reproducible_seed":
        "osiris_governance",
    "test_workstream_h_corpus.py::test_fixture_11_tampered_verification_receipt_fails_closed": "osiris_governance",
    "test_workstream_h_corpus.py::test_h13_quantum_origin_mermin_m3_pqc_enclave_bridge": "osiris_governance",
}


def _unavailable_on_this_python(package: str) -> bool:
    return sys.version_info < PY39_MIN_FOR_RESEARCH and importlib.util.find_spec(package) is None


def _py39_reason(package: str) -> str:
    return (f"needs {package}, which is not installed here and is unavailable on Python "
            f"{sys.version_info[0]}.{sys.version_info[1]} (it requires 3.10+)")


class _SkippedOnPython39(pytest.Module):
    def collect(self):
        pytest.skip(_py39_reason(PY39_MODULES[self.path.name]), allow_module_level=True)


def pytest_pycollect_makemodule(module_path, parent):
    package = PY39_MODULES.get(module_path.name) if module_path.parent.name == "tests" else None
    if package and _unavailable_on_this_python(package):
        return _SkippedOnPython39.from_parent(parent, path=module_path)
    return None


def pytest_collection_modifyitems(config, items):
    for item in items:
        key = f"{item.path.name}::{getattr(item, 'originalname', None) or item.name}"
        package = PY39_TESTS.get(key)
        if package and _unavailable_on_this_python(package):
            item.add_marker(pytest.mark.skip(reason=_py39_reason(package)))
