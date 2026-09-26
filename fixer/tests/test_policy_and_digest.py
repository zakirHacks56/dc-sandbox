"""Tests for fixer/repo_policy.py (W4) and the digest/audit helpers
(W17 / W3)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import repo_policy

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "beacon"))
import digest  # noqa: E402

_audit_spec = importlib.util.spec_from_file_location(
    "audit_tokens_ut", ROOT / "scripts" / "audit_tokens.py")
audit_tokens = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(audit_tokens)


class SimpleObj:
    def __init__(self, text):
        self._text = text

    @property
    def decoded_content(self):
        return self._text.encode("utf-8")


class FakeRepo:
    def __init__(self, archived=False, disabled=False, has_issues=True,
                 default_branch="main", docs=None):
        self.archived = archived
        self.disabled = disabled
        self.has_issues = has_issues
        self.default_branch = default_branch
        self._docs = docs or {}

    def with_doc(self, name, text):
        self._docs[name] = text
        return self

    def get_contents(self, name, ref=None):
        text = self._docs.get(name)
        if text is None:
            raise Exception("not found")
        return SimpleObj(text)


class FakeGh:
    def __init__(self, repos):
        self._repos = repos

    def get_repo(self, name):
        if name not in self._repos:
            raise Exception("no repo")
        return self._repos[name]


def test_clean_repo_is_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(repo_policy, "_SKIP_FILE", tmp_path / "skip.json")
    repo = FakeRepo().with_doc("README.md", "Welcome! PRs welcome.")
    gh = FakeGh({"a/b": repo})
    assert repo_policy.etiquette_ok("a/b", gh) == ""


def test_archived_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(repo_policy, "_SKIP_FILE", tmp_path / "skip.json")
    gh = FakeGh({"a/b": FakeRepo(archived=True)})
    assert repo_policy.etiquette_ok("a/b", gh) == "archived"


def test_banned_docs_refused_and_remembered(tmp_path, monkeypatch):
    monkeypatch.setattr(repo_policy, "_SKIP_FILE", tmp_path / "skip.json")
    repo = FakeRepo().with_doc("CONTRIBUTING.md", "We do not accept no automated PRs.")
    gh = FakeGh({"a/b": repo})
    assert repo_policy.etiquette_ok("a/b", gh)
    assert repo_policy.is_remembered_refused("a/b") is True


def test_skip_list_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(repo_policy, "_SKIP_FILE", tmp_path / "skip.json")
    repo_policy.remember_refused("a/b", "test")
    gh = FakeGh({"a/b": FakeRepo()})
    assert repo_policy.etiquette_ok("a/b", gh) == "already refused us (skip-list)"


def test_filter_ok_repos(tmp_path, monkeypatch):
    monkeypatch.setattr(repo_policy, "_SKIP_FILE", tmp_path / "skip.json")
    gh = FakeGh({
        "nice/repo": FakeRepo().with_doc("README.md", "PRs welcome"),
        "mean/repo": FakeRepo().with_doc("README.md", "no automated PRs, ever"),
    })
    assert repo_policy.filter_ok_repos(["nice/repo", "mean/repo"], gh) == ["nice/repo"]


def test_digest_builds_from_jsonl(tmp_path, monkeypatch):
    failures = tmp_path / "failures.jsonl"
    metrics = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(digest, "FAILURES", failures)
    monkeypatch.setattr(digest, "METRICS", metrics)
    failures.write_text(json.dumps({
        "ts": "2026-09-26T10:00:00+00:00", "type": "failure",
        "stage": "generate_fix", "repo": "a/b",
        "issue": {"repo": "a/b", "issue": 2}, "error": "x"}) + "\n")
    metrics.write_text(json.dumps({
        "ts": "2026-09-26T10:00:00+00:00", "repo": "a/b", "issue": 2,
        "outcome": "failed", "tokens_spent": 30, "iterations": 3}) + "\n")
    text = digest.build_digest(days=1)
    assert "a/b" in text
    assert "failed=1" in text
    assert "tokens spent: 30" in text


def test_audit_tokens_masks_values(tmp_path, monkeypatch):
    env = tmp_path / "manual.env"
    env.write_text('GITHUB_TOKEN="ghp_1234567890abcdef"\nOMNIROUTE_API_KEY=sk_secret\n',
                   encoding="utf-8")
    monkeypatch.setattr(audit_tokens, "ROOT", tmp_path)
    lines = audit_tokens.audit()
    joined = "\n".join(lines)
    assert "ghp_1234567890abcdef" not in joined
    assert "sk_secret" not in joined
    assert "cdef" in joined  # masked tail present, full value never printed