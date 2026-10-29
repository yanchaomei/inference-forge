import sys

from forge.processes import run_bounded


def test_external_command_exceeding_whole_run_budget_is_terminated():
    result = run_bounded([sys.executable,"-c","import time; time.sleep(60)"], .1)
    assert result["timed_out"] is True
    assert result["returncode"] != 0
    assert result["wall_seconds"] < 5


def test_successful_command_is_not_reported_as_budget_exhaustion():
    result = run_bounded([sys.executable,"-c","raise SystemExit(0)"], 5)
    assert result["timed_out"] is False
    assert result["returncode"] == 0
