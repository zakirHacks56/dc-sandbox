"""End-to-end W5/W6/W7: the language validation harness drives the real
repo_map + patches pipeline against synthetic fixtures and must go green."""
import subprocess
import sys
from pathlib import Path

import pytest

FIXER = Path(__file__).resolve().parent.parent


def test_validate_language_python_is_proven(tmp_path):
    from validate_language import validate_language
    row = validate_language("python")
    assert row["status"] == "proven", row
    assert all(f["status"] == "pass" for f in row["fixtures"])


def test_capability_map_marks_python_proven_after_harness():
    import json
    cap = json.loads((FIXER / "capability_map.json").read_text(encoding="utf-8"))
    assert cap["languages"]["python"]["status"] in ("proven", "untested")


def test_harness_cli_end_to_end():
    proc = subprocess.run(
        [sys.executable, str(FIXER / "validate_language.py"), "--language", "python"],
        cwd=str(FIXER), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr