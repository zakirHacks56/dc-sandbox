"""Unit tests for the error handling added to beacon/tick.py's same-run
pipeline: `_run_fixer` must never raise (a launch-level crash / subprocess
failure becomes a 'crash:...' outcome instead of aborting the tick)."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

BEACON = Path(__file__).resolve().parent.parent.parent / "beacon"
FIXER = Path(__file__).resolve().parent.parent / "fixer"


def _tick(monkeypatch):
    sys.path.insert(0, str(BEACON))
    sys.path.insert(0, str(FIXER))
    import beacon_util as util
    import tick
    monkeypatch.setattr(util, "log", lambda *a, **k: None)
    monkeypatch.setattr(util, "env_for_fixer", lambda *a, **k: {})
    monkeypatch.setattr(util, "FIXER", Path("."))
    monkeypatch.setattr(util, "FIXER_SCRIPT", Path("fixer_agent.py"))
    return tick


def test_run_fixer_catches_os_error(monkeypatch):
    tick = _tick(monkeypatch)

    def _boom(cmd, **kw):
        raise OSError("python not found")

    monkeypatch.setattr(tick.subprocess, "run", _boom)
    assert tick._run_fixer(["--issue", "1"]) == "crash:OSError"


def test_run_fixer_catches_generic_subprocess_error(monkeypatch):
    tick = _tick(monkeypatch)

    def _boom(cmd, **kw):
        raise subprocess.SubprocessError("broken pipe")

    monkeypatch.setattr(tick.subprocess, "run", _boom)
    assert tick._run_fixer(["--issue", "1"]) == "crash:SubprocessError"


def test_run_fixer_timeout_still_timeout(monkeypatch):
    tick = _tick(monkeypatch)

    def _boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, timeout=3600)

    monkeypatch.setattr(tick.subprocess, "run", _boom)
    assert tick._run_fixer([]) == "timeout"


def test_run_fixer_nonzero_exit_reported(monkeypatch):
    tick = _tick(monkeypatch)
    monkeypatch.setattr(
        tick.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=2, stdout="boom", stderr="err"),
    )
    assert tick._run_fixer(["--issue", "1"]) == "exit2"


def test_run_fixer_ok_exit(monkeypatch):
    tick = _tick(monkeypatch)
    monkeypatch.setattr(
        tick.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="done", stderr=""),
    )
    assert tick._run_fixer(["--issue", "1"]) == "ok"