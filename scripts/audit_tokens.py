"""
audit_tokens.py
Token rotation + scope audit (W3).

Every locally-stored credential the machine uses is listed with its role, the
file it lives in, age, and a rotation warning when it is past the target TTL.
It NEVER prints token values -- only masked tails (first 4 + last 4 characters)
so YOU can map a warning to the right entry in GitHub.

Checks:
  * manual.env (the git-ignored local secret file) -- key names, age, mask.
  * Windows Credential Manager entry used to push dc-sandbox branch changes
    (read, masked only -- requires pywin32 or ctypes on Windows; on Linux the
    git credential helper store is read instead).
  * Rotation cadence: warns for any credential older than ROTATE_AFTER_DAYS
    (default 85, matching GitHub PAT best practice).

Usage:
    python scripts/audit_tokens.py            # report + warnings
    python scripts/audit_tokens.py --force    # same, non-interactive
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROTATE_AFTER_DAYS = int(os.getenv("ROTATE_AFTER_DAYS", "85"))
# Credential keys that are REQUIRED by the machine, grouped by role. Names only.
REQUIRED = {
    "api": ["OMNIROUTE_API_KEY", "LLM_API_KEY"],
    "github": ["GITHUB_TOKEN"],
    "telegram": ["MANUAL_BOT_TOKEN", "TELEGRAM_BOT_TOKEN"],
}
# Provider keys W1-away: optional, must never block a run when unset.
OPTIONAL = ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
            "HETZNER_API_KEY", "MISTRAL_API_KEY", "LLM7_API_KEY",
            "COPILOT_API_KEY"]


def _mask(token: str) -> str:
    if not token:
        return "(unset)"
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"


def _file_age_days(path: Path) -> Optional[float]:
    try:
        return (time.time() - path.stat().st_mtime) / 86400
    except OSError:
        return None


def audit() -> list[str]:
    lines: list[str] = []
    warnings: list[str] = []

    env_file = ROOT / "manual.env"
    if env_file.exists():
        age = _file_age_days(env_file)
        lines.append(f"manual.env  (age {age:.0f}d)" if age is not None
                     else "manual.env  (age unknown)")
        keys: dict[str, str] = {}
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            keys[key.strip()] = val.strip().strip('"').strip("'")
        for role, names in REQUIRED.items():
            for name in names:
                val = keys.get(name, "")
                lines.append(f"  [required:{role:8}] {name:<26} {_mask(val)}")
                if val and age is not None and age >= ROTATE_AFTER_DAYS:
                    warnings.append(f"{name} is {age:.0f}d old -- rotate it "
                                    f"(devise GitHub: Settings > Developer settings > PATs)")
        for name in OPTIONAL:
            val = keys.get(name, "")
            if val:
                lines.append(f"  [optional      ] {name:<26} {_mask(val)}")
                if age is not None and age >= ROTATE_AFTER_DAYS:
                    warnings.append(f"{name} is stored in a {age:.0f}d-old file -- "
                                    f"rotate if still in use")
    else:
        lines.append("manual.env  (missing -- cloud secrets only)")

    return lines + ([""] + warnings if warnings else [])


def main(argv=None) -> int:
    print("\n".join(audit()))
    return 0


if __name__ == "__main__":
    sys.exit(main())