import pytest

from osiris_cli import sandbox

pytestmark = pytest.mark.skipif(not sandbox.available(), reason="bwrap/user namespaces unavailable")


def test_selftest_all_contained():
    res = sandbox.selftest()
    assert res and all(res.values()), res


def test_runs_when_user_already_has_more_tasks_than_procs(tmp_path):
    # Regression (2026-11-09): procs was an absolute RLIMIT_NPROC; with ~225 threads already
    # owned by the user (llama-server), bwrap's namespace clone failed with EAGAIN every time.
    assert sandbox._uid_tasks() > 8
    res = sandbox.run(["/usr/bin/true"], str(tmp_path), limits=sandbox.Limits(procs=8))
    assert res.returncode == 0, res.output


def test_procs_still_bounds_new_processes(tmp_path):
    import sys
    code = ("import subprocess\nn=0\nps=[]\ntry:\n for i in range(100):\n  ps.append(subprocess.Popen(['sleep','5']));n+=1\n"
            "except OSError: pass\nprint('SPAWNED',n)\n[p.kill() for p in ps]")
    res = sandbox.run([sys.executable, "-I", "-c", code], str(tmp_path),
                      limits=sandbox.Limits(wall_s=20, procs=20))
    assert "SPAWNED" in res.output, res.output
    assert int(res.output.split("SPAWNED")[1].split()[0]) < 100


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
