"""Tests for fixer/guards.py (W15/W16/W22/W10)."""
import time
from pathlib import Path

import pytest

import guards


@pytest.fixture
def iso_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(guards, "FAILURES_FILE", tmp_path / "failures.jsonl")
    monkeypatch.setattr(guards, "METRICS_FILE", tmp_path / "metrics.jsonl")
    monkeypatch.setattr(guards, "CKPT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(guards, "DATA", tmp_path)
    return tmp_path


def _fresh_usage(monkeypatch, tmp_path):
    usage = tmp_path / "provider_usage.json"
    monkeypatch.setenv("PROVIDER_USAGE_FILE", str(usage))
    usage.write_text('{"providers": {"gemini": {"date": "' + time.strftime('%Y-%m-%d') +
                     '", "tokens": 1234, "calls": 2, "fails": 0, "down": false, '
                     '"cooldown_until": 0}}}', encoding="utf-8")
    return usage


def test_budget_limits(monkeypatch):
    b = guards.IssueBudget(max_tokens=100, max_iterations=3, max_wall_clock_seconds=60)
    assert b.exceeded() is None
    b.tally(tokens=100, iterations=1)
    assert b.exceeded() == "token_budget_exceeded"
    b = guards.IssueBudget(max_iterations=2, max_wall_clock_seconds=60)
    b.tally(iterations=2)
    assert b.exceeded() == "max_iterations_reached"
    b = guards.IssueBudget(max_wall_clock_seconds=0.0001)
    time.sleep(0.02)
    assert b.exceeded() == "time_budget_exceeded"


def test_abandon_writes_failures(iso_tmp):
    guards.abandon({"repo": "a/b", "issue": 7, "language": "python"},
                   "token_budget_exceeded")
    rows = guards.read_jsonl(guards.FAILURES_FILE)
    assert len(rows) == 1
    assert rows[0]["type"] == "abandoned"
    assert rows[0]["issue"] == 7


def test_failure_isolation_captures(iso_tmp):
    @guards.failure_isolation("hunt")
    def boom(repo, issue):
        raise RuntimeError("kaboom")

    result = boom("a/b", 3)
    assert result["isolated"] is True
    rows = guards.read_jsonl(guards.FAILURES_FILE)
    assert rows[0]["stage"] == "hunt"
    assert "kaboom" in rows[0]["error"]


def test_metrics_row(iso_tmp):
    guards.metrics({"repo": "a/b", "issue": 1, "language": "py"}, "draft_pr",
                   tokens_spent=42, iterations=3)
    rows = guards.read_jsonl(guards.METRICS_FILE)
    assert rows[0]["outcome"] == "draft_pr"
    assert rows[0]["tokens_spent"] == 42


def test_checkpoint_roundtrip(iso_tmp):
    path = guards.checkpoint("a/b", 5, {"attempt": 2, "state": "IMPLEMENTING"})
    assert path.exists()
    assert guards.load_checkpoint("a/b", 5)["state"]["attempt"] == 2
    assert guards.load_checkpoint("x/y", 9) is None


def test_total_spent_tokens(iso_tmp, monkeypatch):
    usage = _fresh_usage(monkeypatch, iso_tmp)
    assert guards.total_spent_tokens() == 1234
    usage.unlink()
    assert guards.total_spent_tokens() == 0


def test_digest_failures_window(iso_tmp):
    guards.abandon({"repo": "a/b", "issue": 2}, "max_iterations_reached")
    assert len(guards.digest_failures(days=1)) == 1
    # Far into the past -> excluded.
    row = guards.read_jsonl(guards.FAILURES_FILE)[0]
    import json
    row["ts"] = "2020-01-01T00:00:00+00:00"
    guards.FAILURES_FILE.write_text(
        "\n".join(json.dumps(r) for r in [row]) + "\n", encoding="utf-8")
    assert guards.digest_failures(days=1) == []