"""What the test environment must contain, so that a missing package fails loudly instead of
quietly shrinking the suite."""
import importlib.util
import os
import sys

import pytest

from tests import conftest

TESTS = os.path.dirname(os.path.abspath(__file__))
PY310 = sys.version_info >= conftest.PY39_MIN_FOR_RESEARCH


@pytest.mark.skipif(not PY310, reason="dnalang and osiris-governance require Python >= 3.10")
@pytest.mark.parametrize("package", ["dnalang", "organism_sim", "osiris_governance", "qiskit", "qiskit_aer"])
def test_research_packages_are_installed_on_python_310_and_later(package):
    """On 3.10+ the [research] extra is part of the test environment. Without this, a missing
    osiris_governance turns whole files into pytest.importorskip skips and a missing dnalang
    stops collection, as it did in CI until 2026-11-20."""
    assert importlib.util.find_spec(package) is not None, (
        f"{package} is not installed; set up the test environment with `pip install -e '.[research]'`")


def test_nothing_is_skipped_for_python_39_on_python_310_and_later():
    if PY310:
        for package in set(conftest.PY39_MODULES.values()) | set(conftest.PY39_TESTS.values()):
            assert not conftest._unavailable_on_this_python(package)
    else:
        assert conftest._unavailable_on_this_python("dnalang")


def test_every_test_named_for_python_39_exists():
    """The lists in conftest name real files and real tests, so none outlives what it names."""
    for name in conftest.PY39_MODULES:
        assert os.path.isfile(os.path.join(TESTS, name)), name
    for key in conftest.PY39_TESTS:
        name, test = key.split("::")
        with open(os.path.join(TESTS, name), encoding="utf-8") as f:
            assert f"def {test}(" in f.read(), key


def test_the_default_enclave_is_a_throwaway_one_not_the_operators(tmp_path, monkeypatch):
    """Whatever HOME is, the enclave a test gets by default is the session's temporary one,
    with a key already set up; nothing is created under the home directory."""
    from osiris_cli import pqc_enclave
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    enclave = pqc_enclave.default_enclave_dir()
    assert enclave == conftest._TEST_ENCLAVE
    assert not enclave.startswith(os.path.join(os.path.expanduser("~"), ".osiris"))
    assert pqc_enclave.keys_present()
    gate = pqc_enclave.load_or_onboard(confirm=lambda prompt: False)
    assert gate.key_fingerprint
    assert not (home / ".osiris").exists()
