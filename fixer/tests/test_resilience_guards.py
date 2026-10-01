from types import SimpleNamespace

import pytest

import oss_agent_v2 as o


def _issue(title="Document the setup flow", body=(
        "Add a section describing how to install dependencies and run the "
        "worker, mirroring the existing usage section.")):
    return SimpleNamespace(title=title, body=body)


# ---------------------------------------------------------------- watchdog tuning

def test_watchdog_defaults_are_short_and_capped():
    # Pick #1: watchdog = one stalled call dies in 40s (was 90), and at most
    # OMNIROUTE_STALL_CAP=2 consecutive stalls before a clean abandon -- NOT a
    # blind "raise everything" that chases a flaky gateway forever.
    assert o.OMNIROUTE_STALL_SECONDS == 40
    assert o.OMNIROUTE_STALL_CAP == 2
    assert issubclass(o.OmniRouteStallExceeded, RuntimeError)


def test_call_model_abandons_cleanly_after_stall_cap(monkeypatch):
    called = {"n": 0}

    def stall(*a, **k):
        called["n"] += 1
        raise TimeoutError("ReadTimeout")  # type name not in _is_timeout_error list

    fake = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=stall)))
    monkeypatch.setattr(o, "ai_client", fake)
    monkeypatch.setattr(o.time, "sleep", lambda s: None)
    with pytest.raises(o.OmniRouteStallExceeded):
        o.call_model("hi", fast=False)
    # cap=2: exactly two combos get their quick probe, then stop -- we do not
    # drain all 4 fallback combos' stall windows by default.
    assert called["n"] == o.OMNIROUTE_STALL_CAP


def test_call_model_abandons_on_router_watchdog_exhaustion(monkeypatch):
    # The REAL stall signature seen in live runs: the router raises
    # AllProvidersExhaustedError whose last error is a watchdog kill. It must
    # count against the stall cap too (and abandon cleanly), otherwise the run
    # chains every fallback combo through the same broken path.
    import llm_router as router
    called = {"n": 0}

    def stall(*a, **k):
        called["n"] += 1
        raise router.AllProvidersExhaustedError(
            "all providers for tier 'primary' failed or were over budget "
            "(last: provider call exceeded watchdog window of 40s)")

    fake = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=stall)))
    monkeypatch.setattr(o, "ai_client", fake)
    monkeypatch.setattr(o.time, "sleep", lambda s: None)
    with pytest.raises(o.OmniRouteStallExceeded):
        o.call_model("hi", fast=False)
    assert called["n"] == o.OMNIROUTE_STALL_CAP


# ------------------------------------------------------- vague -> HARD escalation

def test_escalated_difficulty_normal_issue_stays_as_classified():
    cls = {"difficulty": "easy", "domain": "docs"}
    eff, vague, docs_vague = o.escalated_difficulty(cls, _issue())
    assert eff == "easy"
    assert vague is False
    assert docs_vague is False


def test_escalated_difficulty_vague_non_docs_forces_hard():
    cls = {"difficulty": "easy", "domain": "backend"}
    eff, vague, docs_vague = o.escalated_difficulty(
        cls, _issue(title="fix the thing", body="make it work or whatever"))
    assert eff == "hard"
    assert vague is True
    assert docs_vague is False


def test_escalated_difficulty_vague_docs_keeps_routine_budget():
    # Pick #3 (originally mislabeled "environmental"): a short/ambiguous but
    # docs-only issue must NOT go plan-first HARD (multiplies calls + watchdog
    # exposure for a one-line README edit).
    cls = {"difficulty": "easy", "domain": "docs"}
    eff, vague, docs_vague = o.escalated_difficulty(
        cls, _issue(title="add a readme section", body="just like the english one"))
    assert eff == "easy"
    assert vague is True
    assert docs_vague is True


def test_escalated_difficulty_hard_classification_wins_even_for_docs():
    cls = {"difficulty": "hard", "domain": "docs"}
    eff, _, _ = o.escalated_difficulty(cls, _issue())
    assert eff == "hard"


# ------------------------------------------------------- docs-exemption loophole

def test_promote_envbroken_baseline_code_patch_rejected():
    # Pick #5: baseline collection-broken (0 tests EVER ran) makes "no new
    # failures" trivially true for ANY patch, code included. A code change
    # must NOT sail through on that.
    baseline = "ERROR collecting tests/test_app.py - ModuleNotFoundError"
    ok, why = o._promote_failing_suite(set(), True, ["src/app.py"], baseline)
    assert ok is False
    assert why


def test_promote_envbroken_baseline_docs_patch_accepted():
    baseline = "ERROR collecting tests/test_app.py - ModuleNotFoundError"
    ok, why = o._promote_failing_suite(set(), False, ["README.md"], baseline)
    assert ok is True
    assert "docs-only" in why


def test_promote_envbroken_baseline_code_patch_rejected_even_if_output_looks_clean():
    # The loophole: a code patch + env-broken baseline + an after-output that
    # LOOKS trustworthy ("no tests ran") must still not pass -- only a docs-only
    # patch survives an untestable baseline.
    baseline = "ERROR collecting tests - ImportError: no module named x"
    ok, _ = o._promote_failing_suite(set(), True, ["src/app.py"], baseline)
    assert ok is False
    ok, why = o._promote_failing_suite(set(), True, ["README.md"], baseline)
    assert ok is True


def test_promote_healthy_baseline_trustworthy_no_new_failures_accepted():
    baseline = "3 passed, 1 failed"
    ok, why = o._promote_failing_suite(set(), True, ["src/app.py"], baseline)
    assert ok is True
    assert "pre-existing" in why


def test_promote_new_failures_never_accepted_even_docs():
    ok, _ = o._promote_failing_suite({"tests/test_a.py::test_b"}, True,
                                     ["README.md"], "3 passed")
    assert ok is False


def test_promote_healthy_baseline_untrustworthy_output_code_patch_rejected():
    ok, _ = o._promote_failing_suite(set(), False, ["src/app.py"], "3 passed")
    assert ok is False
    ok, why = o._promote_failing_suite(set(), False, ["README.md"], "3 passed")
    assert ok is True
    assert "docs-only" in why


# ------------------------------------------------- test-in-production guard

def _repo_with_diff(tmp_path, files: dict):
    import subprocess
    repo = tmp_path / "r"
    repo.mkdir()
    run = lambda *a: subprocess.run(a, cwd=repo, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@t.t")
    run("git", "config", "user.name", "t")
    for name, body in files.items():
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "base")
    return repo, run


def test_scan_flags_new_test_def_in_production_file(tmp_path):
    repo, run = _repo_with_diff(tmp_path, {
        "src/config/writer.py": "def save(cfg):\n    return cfg\n",
        "README.md": "# readme\n",
    })
    (repo / "src/config/writer.py").write_text(
        "def save(cfg):\n    return cfg\n\n\ndef test_save_roundtrip():\n"
        "    assert save({}) == {}\n", encoding="utf-8")
    (repo / "README.md").write_text("# readme\n\nnew section\n", encoding="utf-8")
    found = o.scan_for_tests_in_production(repo, ["src/config/writer.py", "README.md"])
    assert len(found) == 1 and "src/config/writer.py" in found[0]


def test_scan_ignores_tests_in_real_test_files(tmp_path):
    repo, run = _repo_with_diff(tmp_path, {"src/config/writer.py": "x = 1\n"})
    (repo / "tests").mkdir()
    (repo / "tests/test_writer.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")
    found = o.scan_for_tests_in_production(repo, ["tests/test_writer.py"])
    assert found == []


def test_scan_ignores_preexisting_test_helper_mention(tmp_path):
    # A production file that already contained a test helper must not be
    # flagged: only lines the PATCH adds count.
    repo, run = _repo_with_diff(tmp_path, {
        "src/config/writer.py": "def test_legacy_helper():\n    return 1\n",
    })
    (repo / "src/config/writer.py").write_text(
        "def test_legacy_helper():\n    return 1\n\n\ndef save(cfg):\n    return cfg\n",
        encoding="utf-8")
    found = o.scan_for_tests_in_production(repo, ["src/config/writer.py"])
    assert found == []