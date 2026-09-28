"""Golden-path test (Step 6): a well-formed unified diff applies on a real
git repo, its test results read honestly, and retries escalate to the
stronger reasoning combo. No network -- the model chain never fires."""
import sys
import subprocess
from pathlib import Path

FIXER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FIXER))


def _git(repo: Path, *args):
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                          text=True)


def test_golden_path_diff_applies(tmp_path):
    import oss_agent_v2 as o
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "a@b.c")
    _git(repo, "config", "user.name", "t")
    (repo / "app.py").write_text("def add(a, b):\n    return a - b\n",
                                 encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-q", "-m", "init")
    diff = ("```diff\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a - b\n"
            "+    return a + b\n"
            "```")
    touched = o.apply_fix(repo, diff, allowed_paths={"app.py"})
    assert touched == ["app.py"], touched
    assert "a + b" in (repo / "app.py").read_text(encoding="utf-8")
    # The tree must stay commit-clean except for our one edited file.
    status = _git(repo, "status", "--porcelain").stdout
    assert "app.py" in status and "tmp_fix.patch" not in status


def test_golden_path_garbage_diff_does_not_corrupt_tree(tmp_path):
    import oss_agent_v2 as o
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "a@b.c")
    _git(repo, "config", "user.name", "t")
    (repo / "keep.py").write_text("KEEP = 1\n", encoding="utf-8")
    (repo / "other.py").write_text("OTHER = 2\n", encoding="utf-8")
    _git(repo, "add", "keep.py", "other.py")
    _git(repo, "commit", "-q", "-m", "init")
    # Garbage with MULTIPLE files in scope: the raw-response fallback (single
    # file) must NOT fire, so neither file gets overwritten -- instead the
    # call rejects loudly and the tree stays untouched.
    junk = "not a patch at all, no format markers"
    try:
        touched = o.apply_fix(repo, junk,
                              allowed_paths={"keep.py", "other.py"},
                              relevant_files=[("keep.py", "KEEP = 1\n"),
                                              ("other.py", "OTHER = 2\n")])
        loaded = touched
    except ValueError:
        loaded = None
    assert loaded is None, "unparseable multi-file output must be rejected"
    assert (repo / "keep.py").read_text(encoding="utf-8") == "KEEP = 1\n"
    assert (repo / "other.py").read_text(encoding="utf-8") == "OTHER = 2\n"
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""

def test_retry_escalates_chain_to_reasoning(tmp_path, monkeypatch):
    import oss_agent_v2 as o
    monkeypatch.setattr(o, "_using_fallback_endpoint", lambda: False)
    # Non-fast normal call: best-coding leads.
    normal = o._model_chain(fast=False, escalate=False)
    assert normal and normal[0] == o.OMNIROUTE_MODEL
    # Retry of a failed fix: strongest reasoning combo leads.
    retry_chain = o._model_chain(fast=False, escalate=True)
    assert retry_chain[0] == o.OMNIROUTE_REASONING_MODEL
    assert retry_chain[0] not in ("", "auto")
    # Cheap fast steps still use the fast chain even on a retry.
    fast_retry = o._model_chain(fast=True, escalate=True)
    assert fast_retry[0] == o.OMNIROUTE_FAST_MODEL


def test_suite_result_guard_keeps_honesty(tmp_path):
    import oss_agent_v2 as o
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_x.py").write_text(
        "def test_true():\n    assert True\n", encoding="utf-8")
    passed, output = o.run_tests(repo, "pytest")
    assert passed, output
    assert o.suite_result_is_trustworthy(output)
    assert o.extract_failing_tests(output) == set()