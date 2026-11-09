import pytest

from osiris_cli import sandbox

pytestmark = pytest.mark.skipif(not sandbox.available(), reason="bwrap/user namespaces unavailable")


def test_selftest_all_contained():
    res = sandbox.selftest()
    assert res and all(res.values()), res


def test_unavailable_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "available", lambda: False)
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.run(["true"], str(tmp_path))


def test_refuses_home_as_workdir():
    import os
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.run(["true"], os.path.expanduser("~"))


def test_trio_sandboxed_run(tmp_path):
    from osiris_cli.orchestration.trio import DeveloperOutput, run_trio
    r = tmp_path / "r"
    r.mkdir()
    (r / "calc.py").write_text("def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n")
    (r / "test_calc.py").write_text("from calc import mean\n\ndef test_m():\n    assert mean([2, 4]) == 3\n")
    fixed = "def mean(xs):\n    return sum(xs) / len(xs)\n"
    res = run_trio(str(r), lambda f, sb: DeveloperOutput({"calc.py": fixed}, "Fixed the divisor."),
                   str(tmp_path / "o"), sandboxed=True)
    assert res.outcome == "PENDING_HUMAN_APPROVAL", res.reasons
