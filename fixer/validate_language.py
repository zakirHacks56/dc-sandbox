"""
validate_language.py
Language validation harness (W5).

Proves that the fixer's mechanical pipeline (repo-map -> SEARCH/REPLACE patch
-> run tests) actually produces green results on a language BEFORE that
language is trusted on real issues. Each language has deterministic synthetic
fixtures -- a small repo with a seeded bug + a failing test -- and the harness
runs the exact same code path the agent uses:

  1. build_repo_map() on the fixture           (W6)
  2. apply a SEARCH/REPLACE patch              (W7)
  3. run the language's test command            (W2/W5)
  4. report proven / unproven

Python is always available; other languages are env-guarded (skipped with a
clear status if the toolchain is missing). Run:

    python fixer/validate_language.py --language python
    python fixer/validate_language.py --all
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from repo_map import build_repo_map
from patches import apply_patch

CAPABILITY_FILE = Path(__file__).with_name("capability_map.json")
REPORT_FILE = Path(__file__).resolve().parent.parent / "data" / "validation.json"


FIXTURES = {
    "python": [
        {
            "name": "off-by-one",
            "files": {
                "src/bug.py": (
                    "def last_n(items, n):\n"
                    "    if n <= 0:\n"
                    "        return []\n"
                    "    return items[-n - 1:]\n"
                ),
                "tests/test_bug.py": (
                    "import sys\nsys.path.insert(0, 'src')\n"
                    "from bug import last_n\n\n"
                    "def test_last_n():\n"
                    "    assert last_n([1, 2, 3, 4], 2) == [3, 4]\n"
                ),
            },
            "patch_text": (
                "### FILE: src/bug.py\n"
                "<<<<<<< SEARCH\n"
                "    return items[-n - 1:]\n"
                "=======\n"
                "    return items[-n:]\n"
                ">>>>>>> REPLACE\n"
            ),
            "test_cmd": "pytest -q",
        },
        {
            "name": "missing-key-default",
            "files": {
                "src/conf.py": (
                    "def get(config, key):\n"
                    "    return config[key]\n"
                ),
                "tests/test_conf.py": (
                    "import sys\nsys.path.insert(0, 'src')\n"
                    "from conf import get\n\n"
                    "def test_missing_key():\n"
                    "    assert get({'a': 1}, 'b') is None\n"
                ),
            },
            "patch_text": (
                "### FILE: src/conf.py\n"
                "<<<<<<< SEARCH\n"
                "def get(config, key):\n"
                "    return config[key]\n"
                "=======\n"
                "def get(config, key):\n"
                "    return config.get(key)\n"
                ">>>>>>> REPLACE\n"
            ),
            "test_cmd": "pytest -q",
        },
        {
            "name": "dead-condition",
            "files": {
                "src/limit.py": (
                    "def clamp(v, lo, hi):\n"
                    "    if v < lo:\n"
                    "        return lo\n"
                    "    if v > lo:\n"
                    "        return hi\n"
                    "    return v\n"
                ),
                "tests/test_limit.py": (
                    "import sys\nsys.path.insert(0, 'src')\n"
                    "from limit import clamp\n\n"
                    "def test_upper():\n"
                    "    assert clamp(10, 1, 5) == 5\n"
                ),
            },
            "patch_text": (
                "### FILE: src/limit.py\n"
                "<<<<<<< SEARCH\n"
                "    if v > lo:\n"
                "=======\n"
                "    if v > hi:\n"
                ">>>>>>> REPLACE\n"
            ),
            "test_cmd": "pytest -q",
        },
    ],
}


def _toolchain_present(language: str) -> bool:
    if language == "python":
        return True
    probes = {
        "javascript": "node", "typescript": "node", "java": "java",
        "go": "go", "ruby": "ruby", "rust": "cargo",
    }
    tool = probes.get(language)
    return bool(tool and shutil.which(tool))


def _run(cmd: list, cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True,
                          text=True, timeout=240)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def validate_language(language: str) -> dict:
    fixtures = FIXTURES.get(language)
    if not fixtures:
        return {"language": language, "status": "unknown_fixtures"}
    results = []
    for fixture in fixtures:
        row = {"name": fixture["name"], "status": "skipped"}
        if not _toolchain_present(language):
            row["status"] = "toolchain_missing"
            results.append(row)
            continue
        with tempfile.TemporaryDirectory(prefix=f"val-{language}-") as tmp:
            repo = Path(tmp)
            for rel, content in fixture["files"].items():
                dest = repo / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            try:
                repo_map = build_repo_map(repo)
                patch_result = apply_patch(repo, fixture["patch_text"])
                if patch_result.errors:
                    row.update(status="patch_failed", detail=patch_result.errors)
                    results.append(row)
                    continue
                for rel, content in patch_result.applied.items():
                    (repo / rel).write_text(content, encoding="utf-8")
                code, output = _run(fixture["test_cmd"].split(), repo)
                row.update(status="pass" if code == 0 else "fail",
                           map_files=len(repo_map.files),
                           test_output=output[-800:])
            except Exception as exc:  # noqa: BLE001
                row.update(status="error", detail=str(exc))
            results.append(row)
    proven = results and all(r["status"] == "pass" for r in results)
    status = "proven" if proven else "unproven"
    return {"language": language, "status": status, "fixtures": results}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="validate_language", description="W5 harness")
    parser.add_argument("--language", default="", help="one language key")
    parser.add_argument("--all", action="store_true", help="run every fixture")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)

    cap = json.loads(CAPABILITY_FILE.read_text(encoding="utf-8"))
    report = {"capability": {}}

    languages = (list(FIXTURES) if args.all
                 else ([args.language] if args.language else ["python"]))
    for lang in languages:
        row = validate_language(lang)
        report["language_results"] = report.get("language_results", [])
        report["language_results"].append(row)
        proven = row["status"] == "proven"
        cap["languages"].setdefault(lang, {}).update(status="proven" if proven else cap["languages"].get(lang, {}).get("status", "untested"))
        cap["languages"][lang]["status"] = "proven" if proven else "untested" if proven is False and any(f["status"] == "fail" for f in row.get("fixtures", [])) else cap["languages"][lang]["status"]
        print(f"{lang}: {row['status']}")

    try:
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError:
        pass
    if not args.report_only:
        CAPABILITY_FILE.write_text(json.dumps(cap, indent=2), encoding="utf-8")
    return 0 if all(r["status"] == "proven" for r in report.get("language_results", [])) else 2


if __name__ == "__main__":
    sys.exit(main())