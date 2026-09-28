"""Unit tests for beacon/tick.py `_reconcile_prs`: GitHub-truth re-filing of
parked lanes whose PR merged/closed upstream. No network -- util.gh_api is
patched per-case."""
import sys
from pathlib import Path

BEACON = Path(__file__).resolve().parent.parent.parent / "beacon"
FIXER = Path(__file__).resolve().parent.parent / "fixer"


def _tick_module():
    sys.path.insert(0, str(BEACON))
    sys.path.insert(0, str(FIXER))
    import beacon_util as util
    import tick
    return util, tick


def _fake_records(tmp_path, records, monkeypatch, tick, util, keep_updated="2026-01-01T00:00:00Z"):
    wf_root = tmp_path / "fixer" / ".agent_data" / "workflows"
    rec_dir = wf_root / "owner-repo"
    rec_dir.mkdir(parents=True)
    for i, overrides in enumerate(records):
        rec = {"repo": "owner/repo", "issue": 100 + i, "pr_number": 1 + i,
               "state": "WAITING_FOR_FEEDBACK", "updated_at": keep_updated}
        rec.update(overrides)
        (rec_dir / f"issue-{100 + i}.json").write_text(
            __import__("json").dumps(rec), encoding="utf-8")
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    return wf_root


def test_reconcile_merged_marks_completed(tmp_path, monkeypatch):
    util, tick = _tick_module()
    _fake_records(tmp_path, [{}], monkeypatch, tick, util)
    monkeypatch.setattr(
        util, "gh_api",
        lambda method, path, payload=None: {"state": "closed", "merged": True} if "/pulls/" in path else None,
    )
    calls = {"thanks": 0}
    monkeypatch.setattr(tick, "_thank_merge",
                        lambda repo, pr: calls.update(thanks=calls["thanks"] + 1))
    board = {"lanes": {}}
    assert tick._reconcile_prs(board, {"owner/repo"}) == 1
    lane = board["lanes"]["owner/repo#100"]
    assert lane["state"] == "COMPLETED"
    assert calls["thanks"] == 1


def test_reconcile_closed_unmerged_abandons(tmp_path, monkeypatch):
    util, tick = _tick_module()
    _fake_records(tmp_path, [{}], monkeypatch, tick, util)
    monkeypatch.setattr(
        util, "gh_api",
        lambda method, path, payload=None: {"state": "closed", "merged": False} if "/pulls/" in path else None,
    )
    monkeypatch.setattr(tick, "_thank_merge", lambda repo, pr: None)
    board = {"lanes": {}}
    assert tick._reconcile_prs(board, {"owner/repo"}) == 1
    assert board["lanes"]["owner/repo#100"]["state"] == "ABANDONED"


def test_reconcile_open_pr_closed_issue_closes_once(tmp_path, monkeypatch):
    util, tick = _tick_module()
    # Two lanes, both with open PRs but closed issues: only ONE courteous-close
    # subprocess per tick (tick is a single unit of work).
    _fake_records(tmp_path, [{}, {"pr_number": 2, "issue": 101}], monkeypatch, tick, util)
    pr = {"state": "open", "merged": False}
    issue_closed = {"state": "closed"}
    monkeypatch.setattr(util, "gh_api", lambda method, path, payload=None:
                        pr if "/pulls/" in path else issue_closed)
    close_calls = []
    monkeypatch.setattr(tick, "_run_fixer",
                        lambda args, extra_env=None: close_calls.append(args) or "sweep: ok")
    board = {"lanes": {}}
    assert tick._reconcile_prs(board, {"owner/repo"}) == 1
    assert len(close_calls) == 1
    state = board["lanes"]["owner/repo#100"]["state"]
    assert state == "ABANDONED"


def test_reconcile_open_pr_open_issue_leaves_lane(tmp_path, monkeypatch):
    util, tick = _tick_module()
    _fake_records(tmp_path, [{}], monkeypatch, tick, util)
    monkeypatch.setattr(util, "gh_api", lambda method, path, payload=None:
                        {"state": "open", "merged": False} if "/pulls/" in path
                        else {"state": "open"})
    close_calls = []
    monkeypatch.setattr(tick, "_run_fixer",
                        lambda args, extra_env=None: close_calls.append(args) or "ok")
    board = {"lanes": {}}
    assert tick._reconcile_prs(board, {"owner/repo"}) == 0
    assert close_calls == []
    assert "owner/repo#100" not in board["lanes"]


def test_reconcile_skips_disabled_repo_and_terminal(tmp_path, monkeypatch):
    util, tick = _tick_module()
    _fake_records(tmp_path, [{"state": "COMPLETED", "pr_number": 3}], monkeypatch, tick, util)
    monkeypatch.setattr(util, "gh_api", lambda method, path, payload=None:
                        {"state": "closed", "merged": True})
    board = {"lanes": {}}
    assert tick._reconcile_prs(board, {"owner/repo"}) == 0  # COMPLETED skipped
    assert tick._reconcile_prs(board, set()) == 0  # repo not enabled