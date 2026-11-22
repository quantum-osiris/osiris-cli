"""Node Alpha audit F-03: timed-out attempts are not calibration evidence; inference honours the horizon."""

import importlib

from osiris_cli import calibration


def _cand(task, mentor, own_ok, hidden_ok, no_output):
    return {"entry": f"{task}-{mentor}", "run": "bench-r1", "task": task, "mentor": mentor, "n": 1,
            "own_ok": own_ok, "hidden_ok": hidden_ok, "no_output": no_output,
            "module_sha256": None if no_output else "ab" * 32}


def test_timed_out_attempts_do_not_earn_a_perfect_brier_score(tmp_path):
    cal_path = tmp_path / "calibration.jsonl"
    cands = [
        _cand("backlog", "ollama/gemma3:1b", False, False, True),    # timed out
        _cand("claims", "ollama/gemma3:1b", False, False, True),     # timed out
        _cand("gaps", "ollama/gemma3:1b", False, False, False),      # produced code, own test failed
        _cand("backlog", "ollama/qwen2.5-coder:7b", True, False, False),
    ]
    assert calibration.sync_from_bench_candidates(cands, path=cal_path) == 2
    obs = calibration.CalibrationLedger(cal_path).observations()
    assert sorted((o.model_id, o.task_id) for o in obs) == [
        ("ollama/gemma3:1b", "gaps"), ("ollama/qwen2.5-coder:7b", "backlog")]
    rep = calibration.compute_calibration(obs, num_bins=10, model_filter="ollama/gemma3:1b")
    assert rep.num_samples == 1  # one real prediction, not three


def test_inference_timeout_is_clamped_to_the_substrate_horizon(monkeypatch):
    import osiris_hardware_mesh
    console = importlib.import_module("osiris_termux_console")
    monkeypatch.delenv("OSIRIS_INFERENCE_TIMEOUT", raising=False)

    class P:
        execution_timeout_s = 300.0

    monkeypatch.setattr(osiris_hardware_mesh, "get_current_profile", lambda: P())
    assert console.inference_timeout_s() == 300.0          # Node Alpha: 300 s, not 420 s
    P.execution_timeout_s = 1800.0
    assert console.inference_timeout_s() == 420.0          # Node Beta: the 420 s ceiling still applies
    monkeypatch.setenv("OSIRIS_INFERENCE_TIMEOUT", "90")
    assert console.inference_timeout_s() == 90.0           # explicit operator override
    monkeypatch.setenv("OSIRIS_INFERENCE_TIMEOUT", "nonsense")
    assert console.inference_timeout_s() == 420.0
