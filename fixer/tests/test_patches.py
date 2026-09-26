"""Tests for fixer/patches.py (W7)."""
import sys
from pathlib import Path

import pytest

import patches

FIXER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FIXER))

PATCH1 = (
    "### FILE: src/app.py\n"
    "<<<<<<< SEARCH\n"
    "def add(a, b):\n"
    "    return a - b\n"
    "=======\n"
    "def add(a, b):\n"
    "    return a + b\n"
    ">>>>>>> REPLACE\n"
)


def test_parse_blocks():
    blocks = patches.parse_blocks(PATCH1)
    assert len(blocks) == 1
    path, search, replace = blocks[0]
    assert path == "src/app.py"
    assert "a - b" in search
    assert "a + b" in replace


def test_apply_search_replace_success():
    text, errors = patches.apply_search_replace(
        "def add(a, b):\n    return a - b\n", [("src/app.py", "a - b", "a + b")])
    assert errors == []
    assert "a + b" in text


def test_apply_search_replace_no_match():
    text, errors = patches.apply_search_replace(
        "def add(a, b):\n    return a - b\n",
        [("src/app.py", "a * b", "a + b")])
    assert errors and "not found" in errors[0]


def test_apply_search_replace_ambiguous():
    _, errors = patches.apply_search_replace(
        "x = 'hint'\ny = 'hint'\n", [("f.py", "hint", "HINT")])
    assert errors and "matched 2 times" in errors[0]


def test_apply_patch_end_to_end(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8")
    result = patches.apply_patch(tmp_path, PATCH1)
    assert not result.errors
    assert "a + b" in result.applied["src/app.py"]
    # No partial writes happened for a failing block (SEARCH side broken).
    bad = PATCH1.replace("return a - b", "return a * b * c")
    result2 = patches.apply_patch(tmp_path, bad)
    assert result2.errors
    assert "src/app.py" not in result2.applied


def test_apply_patch_refuses_outside_allowlist(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    result = patches.apply_patch(tmp_path, PATCH1, allowed_paths={"other.py"})
    assert result.errors.get("src/app.py")


def test_full_file_blocks(tmp_path):
    block = "### FILE: hello.txt\n<<<CONTENT>>>\nline one\nline two\n<<<END>>>\n"
    (tmp_path / "hello.txt").write_text("old\n", encoding="utf-8")
    result = patches.apply_patch(tmp_path, block, allowed_paths={"hello.txt"})
    assert result.applied["hello.txt"] == "line one\nline two\n"


def test_dict_variant():
    files = {"a.py": "x = 1\n"}
    updated, errors = patches.apply_search_replace_dict(
        files, [("a.py", "x = 1", "x = 2")])
    assert errors == {}
    assert updated["a.py"] == "x = 2\n"
    updated2, errors2 = patches.apply_search_replace_dict(
        files, [("missing.py", "x", "y")])
    assert "missing.py" in errors2


def test_unified_diff_parse():
    text = ("```diff\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "+def foo():\n"
            "+    return 1\n"
            "```")
    files = patches.parse_unified_diff(text)
    assert "a.py" in files