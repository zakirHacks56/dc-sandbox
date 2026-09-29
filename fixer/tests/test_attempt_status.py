"""Unit tests for oss_agent_v2 run-once attempt_status lifecycle (new_workflow,
_migrate_workflow_record, set_terminal_record) and the discover_valid_issue
selection guards (_attempt_consumed, _issue_pref_bucket, _repo_activity_ok).
No network -- gh is a fake, WORKFLOWS_DIR is redirected to tmp."""
import json
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest

import oss_agent_v2 as agent


@pytest.fixture
def tmp_wo(tmp_path, monkeypatch):
    wf_root = tmp_path / "workflows"
    wf_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "WORKFLOWS_DIR", wf_root)
    return wf_root


def _write_record(wf_root, repo_full, number, record):
    rec_dir = wf_root / str(repo_full).replace("/", "-")
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / f"issue-{number}.json").write_text(json.dumps(record), encoding="utf-8")


# --- new_workflow / migration ----------------------------------------------

def test_new_workflow_defaults_not_attempted(tmp_wo):
    rec = agent.new_workflow("a/b", 7, "t")
    assert rec["attempt_status"] == "not_attempted"
    assert rec["state"] == agent.WF.TASK_RECEIVED


def test_migrate_legacy_completed_maps_success(tmp_wo):
    rec = agent._migrate_workflow_record({"repo": "a/b", "issue": 1, "state": agent.WF.COMPLETED})
    assert rec["attempt_status"] == "attempted_success"


def test_migrate_legacy_pr_number_maps_success(tmp_wo):
    rec = agent._migrate_workflow_record(
        {"repo": "a/b", "issue": 2, "state": "IMPLEMENTING", "pr_number": 42})
    assert rec["attempt_status"] == "attempted_success"


def test_migrate_legacy_abandoned_maps_failed(tmp_wo):
    rec = agent._migrate_workflow_record({"repo": "a/b", "issue": 3, "state": agent.WF.ABANDONED})
    assert rec["attempt_status"] == "attempted_failed"


def test_migrate_inflight_stays_not_attempted(tmp_wo):
    rec = agent._migrate_workflow_record({"repo": "a/b", "issue": 4, "state": agent.WF.IMPLEMENTING})
    assert rec["attempt_status"] == "not_attempted"


def test_migrate_existing_status_untouched(tmp_wo):
    rec = agent._migrate_workflow_record({"repo": "a/b", "issue": 5, "state": agent.WF.COMPLETED,
                                          "attempt_status": "budget_exhausted"})
    assert rec["attempt_status"] == "budget_exhausted"


# --- set_terminal_record stamping -----------------------------------------

def test_terminal_parked_pr_stamps_success(tmp_wo):
    rec = agent.new_workflow("a/b", 5, "t")
    agent.set_terminal_record(rec, agent.AttemptOutcome.PR_DRAFTED, "pr")
    assert rec["attempt_status"] == "attempted_success"
    assert (tmp_wo / "a-b" / "issue-5.json").exists()


def test_terminal_gate_deferred_stamps_success(tmp_wo):
    rec = agent.new_workflow("a/b", 6, "t")
    agent.set_terminal_record(rec, agent.AttemptOutcome.GATE_DEFERRED)
    assert rec["attempt_status"] == "attempted_success"


def test_terminal_abandoned_failed_stamps(tmp_wo):
    rec = agent.new_workflow("a/b", 7, "t")
    agent.set_terminal_record(rec, agent.AttemptOutcome.ABANDONED_FAILED, "tests failed")
    assert rec["attempt_status"] == "attempted_failed"
    assert rec["state"] == agent.WF.ABANDONED
    assert (tmp_wo / "a-b" / "issue-7.json").exists()


def test_terminal_noop_stays_not_attempted(tmp_wo):
    rec = agent.new_workflow("a/b", 8, "t")
    agent.set_terminal_record(rec, agent.AttemptOutcome.NOOP, "declined")
    assert rec["attempt_status"] == "not_attempted"


def test_terminal_budget_exhausted_wins(tmp_wo):
    rec = agent.new_workflow("a/b", 9, "t")
    rec["state"] = agent.WF.IMPLEMENTING
    rec["attempt_status"] = "budget_exhausted"
    rec["baseline_tested"] = False  # unknown key, tolerated
    agent.set_terminal_record(rec, agent.AttemptOutcome.ABANDONED_FAILED, "ran out")
    assert rec["state"] == agent.WF.ABANDONED
    assert rec["attempt_status"] == "budget_exhausted"


# --- discover_valid_issue selection guards ---------------------------------

def test_issue_pref_bucket_bounds():
    assert agent._issue_pref_bucket(1) == 0
    assert agent._issue_pref_bucket(150) == 0
    assert agent._issue_pref_bucket(151) == 1
    assert agent._issue_pref_bucket(0) == 1


def test_attempt_consumed_statuses(tmp_wo):
    _write_record(tmp_wo, "a/b", 21,
                  {"state": agent.WF.ABANDONED, "attempt_status": "budget_exhausted"})
    _write_record(tmp_wo, "a/b", 22, {"state": agent.WF.IMPLEMENTING})
    assert agent._attempt_consumed("a/b", 20) == ""
    assert agent._attempt_consumed("a/b", 21) == "budget_exhausted"
    assert agent._attempt_consumed("a/b", 22) == "in_flight"


class _FakeRepo:
    def __init__(self, pushed_at=None, archived=False, disabled=False, owner=None):
        self.pushed_at = pushed_at
        self.archived = archived
        self.disabled = disabled
        self.owner = owner or SimpleNamespace(suspended_at=None)


class _FakeGh:
    def __init__(self, repo):
        self.repo = repo
        self.calls = 0

    def get_repo(self, name):
        self.calls += 1
        return self.repo


def test_repo_activity_ok_active_with_daily_cache(tmp_path, monkeypatch):
    repo = _FakeRepo(pushed_at=datetime.now(timezone.utc) - timedelta(days=1))
    gh = _FakeGh(repo)
    monkeypatch.setattr(agent, "_REPO_ACTIVITY_FILE", str(tmp_path / "repo_activity.json"))
    assert agent._repo_activity_ok(gh, "o/r") == (True, "")
    assert agent._repo_activity_ok(gh, "o/r") == (True, "")
    assert gh.calls == 1
    assert agent._repo_activity_ok(gh, "o/r", force=True) == (True, "")
    assert gh.calls == 2


def test_repo_activity_ok_archived(tmp_path, monkeypatch):
    gh = _FakeGh(_FakeRepo(pushed_at=datetime.now(timezone.utc) - timedelta(days=1),
                           archived=True))
    monkeypatch.setattr(agent, "_REPO_ACTIVITY_FILE", str(tmp_path / "repo_activity.json"))
    assert agent._repo_activity_ok(gh, "o/r")[0] is False


def test_repo_activity_ok_stale(tmp_path, monkeypatch):
    gh = _FakeGh(_FakeRepo(pushed_at=datetime.now(timezone.utc) - timedelta(days=400)))
    monkeypatch.setattr(agent, "_REPO_ACTIVITY_FILE", str(tmp_path / "repo_activity.json"))
    ok, reason = agent._repo_activity_ok(gh, "o/r")
    assert ok is False and "no commits" in reason


def test_repo_activity_ok_owner_suspended(tmp_path, monkeypatch):
    gh = _FakeGh(_FakeRepo(pushed_at=datetime.now(timezone.utc) - timedelta(days=1),
                           owner=SimpleNamespace(suspended_at="2026-01-01")))
    monkeypatch.setattr(agent, "_REPO_ACTIVITY_FILE", str(tmp_path / "repo_activity.json"))
    ok, reason = agent._repo_activity_ok(gh, "o/r")
    assert ok is False and "suspended" in reason


def test_repo_activity_ok_fetch_error(tmp_path, monkeypatch):
    class _Boom:
        def get_repo(self, name):
            raise RuntimeError("connection reset")

    monkeypatch.setattr(agent, "_REPO_ACTIVITY_FILE", str(tmp_path / "repo_activity.json"))
    ok, reason = agent._repo_activity_ok(_Boom(), "o/r")
    assert ok is False and reason.startswith("fetch error:")


def test_repo_activity_ok_no_commits(tmp_path, monkeypatch):
    gh = _FakeGh(_FakeRepo(pushed_at=None))
    monkeypatch.setattr(agent, "_REPO_ACTIVITY_FILE", str(tmp_path / "repo_activity.json"))
    assert agent._repo_activity_ok(gh, "o/r") == (False, "no commits")