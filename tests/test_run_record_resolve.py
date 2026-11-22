"""'Last run' must ignore execution-kernel artifact directories that share ~/.osiris/runs."""

import pytest

import run_record


def _sprint_run(runs, run_id):
    (runs / run_id).mkdir(parents=True)
    (runs / run_id / "run.json").write_text("{}", encoding="utf-8")


def test_newest_run_skips_kernel_artifact_dirs(tmp_path):
    runs = tmp_path / "runs"
    _sprint_run(runs, "run-20261121T010000Z-aaaaaa")
    _sprint_run(runs, "run-20261122T020000Z-bbbbbb")
    kernel = runs / "run-d3b4bb35bcea4a51b14867e65b8fb83d"   # sorts after every run-2026... id
    kernel.mkdir()
    (kernel / "output.jsonl").write_text("{}\n", encoding="utf-8")
    assert run_record.resolve(None, runs_dir=str(runs)) == "run-20261122T020000Z-bbbbbb"
    assert run_record.resolve("aaaaaa", runs_dir=str(runs)) == "run-20261121T010000Z-aaaaaa"
    with pytest.raises(LookupError):
        run_record.resolve("d3b4bb", runs_dir=str(runs))


def test_only_kernel_dirs_means_no_runs_yet(tmp_path):
    runs = tmp_path / "runs"
    (runs / "run-0123abcd").mkdir(parents=True)
    (runs / "run-0123abcd" / "output.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(LookupError, match="no runs recorded yet"):
        run_record.resolve(None, runs_dir=str(runs))


def test_has_runs_ignores_kernel_dirs(tmp_path):
    runs = tmp_path / "runs"
    (runs / "run-0123abcd").mkdir(parents=True)
    assert run_record.has_runs(str(runs)) is False
    _sprint_run(runs, "run-20261122T020000Z-cccccc")
    assert run_record.has_runs(str(runs)) is True


def test_kernel_dirs_are_not_reported_as_orphaned_sprint_runs(tmp_path):
    runs = tmp_path / "runs"
    (runs / "run-d3b4bb35bcea4a51b14867e65b8fb83d").mkdir(parents=True)    # kernel artifact
    (runs / "run-20261122T020000Z-dddddd").mkdir()                         # sprint run, no run.json
    assert run_record.orphans([], runs_dir=str(runs)) == ["run-20261122T020000Z-dddddd"]
    assert run_record.is_sprint_run_id(run_record.new_run_id())
