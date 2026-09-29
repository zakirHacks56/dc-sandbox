"""Unit tests for beacon/hunter_stage.py selection logic: issue-number
preference + newest-created scoring, once-daily repo-activity gating, and the
run-once attempt_status skip. No network -- util.gh_api / util.gh_paged are
patched per-case."""
import json
import sys
from pathlib import Path

BEACON = Path(__file__).resolve().parent.parent.parent / "beacon"

sys.path.insert(0, str(BEACON))
import beacon_util as util  # noqa: E402
import hunter_stage as hunter  # noqa: E402

_ACTIVE_META = {
    "stargazers_count": 10,
    "archived": False,
    "disabled": False,
    "pushed_at": "2026-09-01T00:00:00Z",
    "owner": {"suspended_at": None},
    "default_branch": "master",
}


def _issue(number, created, label="good first issue"):
    return {
        "number": number,
        "title": f"issue {number}",
        "body": "small reproducible bug",
        "state": "open",
        "created_at": created,
        "labels": [{"name": label}],
        "locked": False,
    }


class _FakeGH:
    def __init__(self, meta=_ACTIVE_META, issues=None, pulls=None):
        self.meta = meta
        self.issues = issues or []
        self.pulls = pulls if pulls is not None else []
        self.repo_meta_calls = 0
        self.paged_paths = []

    def api(self, method, path, payload=None):
        if path == "/user":
            return {"login": "me"}
        if path == f"/repos/o/r":
            self.repo_meta_calls += 1
            return self.meta
        return None

    def paged(self, path):
        self.paged_paths.append(path)
        if "issues" in path:
            return self.issues
        if "pulls" in path:
            return self.pulls
        return None


def _patch(monkeypatch, gh):
    monkeypatch.setattr(util, "gh_api", gh.api)
    monkeypatch.setattr(util, "gh_paged", gh.paged)


def _conf():
    return {
        "max_stars": 1000,
        "max_file_bytes": 0,
        "max_attempts_per_repo_day": 0,
        "default_labels": ["good first issue"],
        "targets": [{"repo": "o/r", "enabled": True}],
    }


def _write_record(tmp_path, repo_full, number, record):
    safe = str(repo_full).replace("/", "-")
    rec_dir = tmp_path / "fixer" / ".agent_data" / "workflows" / safe
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / f"issue-{number}.json").write_text(json.dumps(record), encoding="utf-8")


# --- _pick_issue scoring -----------------------------------------------------

def test_pick_prefers_low_number_over_newest(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH())
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    issues = [_issue(200, "2026-09-01T00:00:00Z"), _issue(1, "2026-01-01T00:00:00Z")]
    picked = hunter._pick_issue(issues, set(), "o/r")
    assert picked is not None and picked[0] == 1


def test_pick_uses_newest_within_bucket(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH())
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    issues = [_issue(10, "2020-01-01T00:00:00Z"), _issue(8, "2026-09-01T00:00:00Z")]
    picked = hunter._pick_issue(issues, set(), "o/r")
    assert picked is not None and picked[0] == 8


def test_pick_skips_consumed_attempts(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH())
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    _write_record(tmp_path, "o/r", 1, {"state": "ABANDONED",
                                       "attempt_status": "budget_exhausted"})
    issues = [_issue(1, "2026-09-01T00:00:00Z"), _issue(2, "2026-08-01T00:00:00Z")]
    picked = hunter._pick_issue(issues, set(), "o/r")
    assert picked is not None and picked[0] == 2


def test_pick_all_consumed_returns_none(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH())
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    _write_record(tmp_path, "o/r", 1, {"state": "IMPLEMENTING"})
    _write_record(tmp_path, "o/r", 2, {"state": "COMPLETED",
                                       "attempt_status": "attempted_success"})
    assert hunter._pick_issue([_issue(1, "2026-09-01T00:00:00Z"),
                               _issue(2, "2026-09-02T00:00:00Z")],
                              set(), "o/r") is None


# --- _attempt_status ---------------------------------------------------------

def test_attempt_status_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    _write_record(tmp_path, "o/r", 5,
                  {"state": "ABANDONED", "attempt_status": "budget_exhausted"})
    _write_record(tmp_path, "o/r", 6, {"state": "IMPLEMENTING"})
    _write_record(tmp_path, "o/r", 7,
                  {"state": "COMPLETED", "attempt_status": "attempted_success"})
    assert hunter._attempt_status("o/r", 5) == "budget_exhausted"
    assert hunter._attempt_status("o/r", 6) == "in_flight"
    assert hunter._attempt_status("o/r", 7) == "attempted_success"
    assert hunter._attempt_status("o/r", 99) == ""


def test_attempt_status_legacy_terminal_treated_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    flat = tmp_path / "fixer" / ".agent_data" / "workflows" / "o-r_issue9.json"
    flat.parent.mkdir(parents=True, exist_ok=True)
    flat.write_text(json.dumps({"state": "COMPLETED"}), encoding="utf-8")
    assert hunter._attempt_status("o/r", 9) == ""


def test_attempt_status_corrupt_is_consumed(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    rec_dir = tmp_path / "fixer" / ".agent_data" / "workflows" / "o-r"
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / "issue-3.json").write_text("{not json", encoding="utf-8")
    assert hunter._attempt_status("o/r", 3) == "corrupt"


# --- _repo_active ----------------------------------------------------------

def test_repo_active_daily_cache(tmp_path, monkeypatch):
    gh = _FakeGH()
    _patch(monkeypatch, gh)
    board = {}
    assert hunter._repo_active("o/r", board) is True
    assert hunter._repo_active("o/r", board) is True
    assert gh.repo_meta_calls == 1
    assert board["repo_activity"]["results"]["o/r"] is True


def test_repo_inactive_archived(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH(meta={**_ACTIVE_META, "archived": True}))
    assert hunter._repo_active("o/r", {}) is False


def test_repo_inactive_stale(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH(
        meta={**_ACTIVE_META, "pushed_at": "2020-01-01T00:00:00Z"}))
    assert hunter._repo_active("o/r", {}) is False


def test_repo_inactive_owner_suspended(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH(meta={**_ACTIVE_META,
                                      "owner": {"suspended_at": "2026-01-01"}}))
    assert hunter._repo_active("o/r", {}) is False


def test_repo_active_meta_missing_means_inactive(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH(meta=None))
    assert hunter._repo_active("o/r", {}) is False


# --- find_candidate integration ---------------------------------------------

def test_find_candidate_prefers_low_number(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH(
        issues=[_issue(200, "2026-09-01T00:00:00Z"), _issue(1, "2026-01-01T00:00:00Z")]))
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    assert hunter.find_candidate(_conf(), {}) == ("o/r", 1)


def test_find_candidate_reroute_to_newest_when_preferred_consumed(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH(
        issues=[_issue(200, "2026-09-01T00:00:00Z"), _issue(1, "2026-01-01T00:00:00Z")]))
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    _write_record(tmp_path, "o/r", 1,
                  {"state": "ABANDONED", "attempt_status": "budget_exhausted"})
    assert hunter.find_candidate(_conf(), {}) == ("o/r", 200)


def test_find_candidate_none_when_repo_inactive(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH(
        meta={**_ACTIVE_META, "archived": True},
        issues=[_issue(1, "2026-09-01T00:00:00Z")]))
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    assert hunter.find_candidate(_conf(), {}) is None


def test_find_candidate_none_when_all_consumed(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH(
        issues=[_issue(1, "2026-09-01T00:00:00Z"), _issue(2, "2026-09-02T00:00:00Z")]))
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    _write_record(tmp_path, "o/r", 1, {"state": "IMPLEMENTING"})
    _write_record(tmp_path, "o/r", 2,
                  {"state": "COMPLETED", "attempt_status": "attempted_success"})
    assert hunter.find_candidate(_conf(), {}) is None


# --- error handling: a hunt must never crash the tick ------------------------

def test_find_candidate_bad_config_int_never_raises(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH(issues=[_issue(1, "2026-09-01T00:00:00Z")]))
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    conf = _conf()
    conf["max_stars"] = "not-an-int"
    conf["max_file_bytes"] = "definitely-not-an-int"
    assert hunter.find_candidate(conf, {}) == ("o/r", 1)


def test_find_candidate_target_missing_repo_never_raises(tmp_path, monkeypatch):
    _patch(monkeypatch, _FakeGH(issues=[_issue(1, "2026-09-01T00:00:00Z")]))
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    conf = _conf()
    conf["targets"] = [{"enabled": True}, {"repo": "o/r", "enabled": True}]
    assert hunter.find_candidate(conf, {}) == ("o/r", 1)


def test_find_candidate_unexpected_error_returns_none(tmp_path, monkeypatch):
    def _boom(method, path, payload=None):
        raise RuntimeError("vault locked")

    monkeypatch.setattr(util, "gh_api", _boom)
    monkeypatch.setattr(util, "FIXER", tmp_path / "fixer")
    assert hunter.find_candidate(_conf(), {}) is None