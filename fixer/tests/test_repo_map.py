"""Tests for fixer/repo_map.py (W6)."""
import sys
from pathlib import Path

import pytest

import repo_map

FIXER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FIXER))


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text(
        "import os\n"
        "from pathlib import Path\n\n"
        "class User:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n\n"
        "def validate_token(token):\n"
        "    return bool(token)\n\n"
        "async def refresh_session(session):\n"
        "    return session\n", encoding="utf-8")
    (tmp_path / "src" / "server.js").write_text(
        "const express = require('express');\n"
        "function handle(req, res) { return res; }\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("hi", encoding="utf-8")
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "x.pyc").write_text("x", encoding="utf-8")
    return tmp_path


def test_build_repo_map_python_functions_classes_imports(sample_repo):
    m = repo_map.build_repo_map(sample_repo)
    files = {f.path: f for f in m.files}
    auth = files.get("src/auth.py")
    assert auth is not None
    funcs = dict(auth.functions)
    assert "validate_token" in funcs
    assert any(name.endswith("refresh_session") for name in funcs)
    assert any("User" in name for name, _ in auth.classes)


def test_skips_non_code_and_cache_dirs(sample_repo):
    m = repo_map.build_repo_map(sample_repo)
    paths = [f.path for f in m.files]
    assert "README.md" not in paths
    assert "src/server.js" in paths


def test_js_functions_via_regex(sample_repo):
    m = repo_map.build_repo_map(sample_repo)
    js = next(f for f in m.files if f.path == "src/server.js")
    assert any(name == "handle" for name, _ in js.functions)


def test_render_contains_paths(sample_repo):
    text = repo_map.render_compact(sample_repo, max_chars=4000)
    assert "src/auth.py" in text
    assert "validate_token" in text


def test_repo_context_respects_allowlist(sample_repo):
    ctx = repo_map.RepoContext(sample_repo, allowed_paths={"src/auth.py"})
    assert "validate_token" in ctx.read("src/auth.py")
    assert "ERROR" in ctx.read("src/server.js")
    ctx2 = repo_map.RepoContext(sample_repo, allowed_paths=set())
    assert "validate_token" in ctx2.read("src/auth.py")


def test_repo_context_truncation(sample_repo):
    ctx = repo_map.RepoContext(sample_repo, allowed_paths={"src/auth.py"}, max_chars=40)
    out = ctx.read("src/auth.py")
    assert "more lines omitted" in out