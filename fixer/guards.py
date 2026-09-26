"""
guards.py
Per-issue budget guard and failure-isolation helpers for oss-agent.

Fulfils solution-plan weaknesses:
  W15  per-issue token/iteration/wall-clock budget   -- a hard ceiling per issue
       attempt, independent of the per-repo-per-day caps.
  W16  per-issue failure isolation                   -- one issue's exception is
       captured and logged; it never propagates into tick.py's commit-back or
       gate-sync steps.
  W22  structured failure + metrics logs             -- data/failures.jsonl and
       data/metrics.jsonl, one JSON line per attempt, append-only.
  W10  checkpoint persistence                        -- fixer/.agent_data/
       checkpoints/<repo>-<issue>.json lets a crashed run resume from the last
       completed sub-step instead of losing everything.

Paths default to the repo root (this file's parent's parent) and can be
overridden for tests via the *_FILE / *_DIR env vars.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("OSS_AGENT_DATA", str(ROOT / "data")))
CKPT_DIR = Path(os.getenv("OSS_AGENT_CKPT_DIR",
                          str(ROOT / "fixer" / ".agent_data" / "checkpoints")))

FAILURES_FILE = DATA / "failures.jsonl"
METRICS_FILE = DATA / "metrics.jsonl"

DEFAULT_TOKEN_BUDGET = int(os.getenv("ISSUE_TOKEN_BUDGET", "40000"))
DEFAULT_ITERATION_BUDGET = int(os.getenv("ISSUE_ITERATION_BUDGET", "3"))
DEFAULT_WALL_CLOCK_SECONDS = float(os.getenv("ISSUE_WALL_CLOCK_SECONDS", "300"))


@dataclass
class IssueBudget:
    """W15: ceilings for one issue attempt."""

    max_tokens: int = DEFAULT_TOKEN_BUDGET
    max_iterations: int = DEFAULT_ITERATION_BUDGET
    max_wall_clock_seconds: float = DEFAULT_WALL_CLOCK_SECONDS
    tokens_spent: int = 0
    iterations: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def tally(self, tokens: int = 0, iterations: int = 1) -> None:
        self.tokens_spent = max(0, self.tokens_spent + int(tokens))
        self.iterations += int(iterations)

    def left_tokens(self) -> int:
        return max(0, self.max_tokens - self.tokens_spent)

    def exceeded(self) -> Optional[str]:
        """Returns the reason the budget is blown, or None while there's room."""
        if self.tokens_spent >= self.max_tokens:
            return "token_budget_exceeded"
        if self.iterations >= self.max_iterations:
            return "max_iterations_reached"
        if time.monotonic() - self.started_at >= self.max_wall_clock_seconds:
            return "time_budget_exceeded"
        return None

    def as_dict(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "max_iterations": self.max_iterations,
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
            "tokens_spent": self.tokens_spent,
            "iterations": self.iterations,
            "started_at": self.started_at,
        }


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, line: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
    except Exception:  # noqa: BLE001 - never let logging take down a run
        pass


def abandon(issue: dict, reason: str, budget: Optional[IssueBudget] = None,
            notes: str = "") -> None:
    """W15/W22: record an abandoned issue attempt (failures.jsonl) so the next
    tick does not immediately retry the same losing battle."""
    row = {
        "ts": _now_utc(),
        "type": "abandoned",
        "repo": issue.get("repo", ""),
        "issue": issue.get("issue"),
        "language": issue.get("language", ""),
        "reason": reason,
        "budget": budget.as_dict() if budget else None,
        "notes": notes,
    }
    _append_jsonl(FAILURES_FILE, row)


def log_failure(stage: str, exc: BaseException, issue: Optional[dict] = None,
                extra: Optional[dict] = None) -> None:
    """W16/W22: one JSONL line per failure. Never raises, never crashes a tick."""
    row = {
        "ts": _now_utc(),
        "type": "failure",
        "stage": stage,
        "error": str(exc),
        "traceback": traceback.format_exc()[-4000:] if exc.__cause__ or exc else "",
        "issue": issue or {},
    }
    if extra:
        row.update(extra)
    _append_jsonl(FAILURES_FILE, row)


def metrics(record: dict, outcome: str, tokens_spent: int = 0,
            iterations: int = 0, notes: str = "") -> None:
    """W22: append one metrics line per attempt to data/metrics.jsonl."""
    row = {
        "ts": _now_utc(),
        "repo": record.get("repo", ""),
        "issue": record.get("issue"),
        "language": record.get("language", ""),
        "outcome": outcome,
        "tokens_spent": int(tokens_spent),
        "iterations": int(iterations),
        "notes": notes,
    }
    _append_jsonl(METRICS_FILE, row)


def checkpoint(repo_name: str, issue: int, state: dict) -> Path:
    """W10: persist a resumable checkpoint for (repo, issue).

    Each call overwrites the previous checkpoint for the pair, so the freshest
    completed sub-step is always what a crash resumes from."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    safe = f"{repo_name.replace('/', '-')}-issue-{issue}"
    path = CKPT_DIR / f"{safe}.json"
    payload = {"repo": repo_name, "issue": issue, "ts": _now_utc(), "state": state}
    tmp = path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass
    return path


def load_checkpoint(repo_name: str, issue: int) -> Optional[dict]:
    path = CKPT_DIR / f"{repo_name.replace('/', '-')}-issue-{issue}.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def failure_isolation(stage: str):
    """W16: decorator -- run fn inside a boundary that records, never raises.

    `fn(repo, issue, **kw) -> outcome`. An exception becomes a failure row and
    the call returns the sentinel `{"isolated": True, "stage": stage}` so the
    tick loop can keep going (commit-back, gate-sync) regardless."""

    def _wrap(fn: Callable[..., Any]):
        def _run(repo: str, issue: int, *args, **kwargs):
            try:
                return fn(repo, issue, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - the whole point is isolation
                log_failure(stage, exc, issue={"repo": repo, "issue": issue})
                return {"isolated": True, "stage": stage, "error": str(exc)}

        _run.__name__ = fn.__name__
        return _run

    return _wrap


def read_jsonl(path: Path, limit: int = 0) -> list:
    """Read an append-only JSONL file into a list of dicts (optional tail limit)."""
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows[-limit:] if limit else rows


def total_spent_tokens() -> int:
    """W18: aggregate tokens spent across providers, from provider_usage.json.

    Fed to IssueBudget.tally() so the per-issue ceiling counts real model spend
    (which the router records after every call) rather than a guess."""
    import json as _json
    path = Path(os.getenv("PROVIDER_USAGE_FILE",
                          str(ROOT / "data" / "provider_usage.json")))
    try:
        store = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return sum(int(e.get("tokens", 0))
               for e in store.get("providers", {}).values())


def digest_failures(days: int = 1) -> list:
    """Failures logged in the last `days`, newest first (for the daily digest)."""
    cutoff = time.time() - days * 86400
    rows = []
    for row in read_jsonl(FAILURES_FILE):
        try:
            ts = datetime.fromisoformat(row["ts"]).timestamp()
        except (KeyError, ValueError, TypeError):
            continue
        if ts >= cutoff:
            rows.append(row)
    return rows