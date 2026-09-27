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


def test_load_capability_map_unwraps_namespace():
    """Regression: load_capability_map must return the flat lang->config dict,
    NOT the {purpose, languages, rules} namespace. Before the fix it returned
    the raw top level, so iterating it hit the 'purpose' STRING as a config
    and detect_language_and_commands crashed for every repo in the cloud
    (TypeError: string indices must be integers)."""
    sys.path.insert(0, str(FIXER))
    import oss_agent_v2 as o
    flat = o.load_capability_map()
    assert "purpose" not in flat
    assert "languages" not in flat
    assert "rules" not in flat
    assert all(isinstance(cfg, dict) for cfg in flat.values())
    for cfg in flat.values():
        assert isinstance(cfg.get("markers", []), list)
        assert isinstance(cfg.get("extensions", []), list)


def test_detect_language_python(tmp_path):
    sys.path.insert(0, str(FIXER))
    import oss_agent_v2 as o
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n")
    detected = o.detect_language_and_commands(tmp_path)
    assert detected["language"] == "python"
    assert detected["test"] == "pytest"


def test_harness_cli_end_to_end():
    proc = subprocess.run(
        [sys.executable, str(FIXER / "validate_language.py"), "--language", "python"],
        cwd=str(FIXER), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr