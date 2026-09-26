"""
digest.py
Daily failure digest (W17 component + W22 output).

The solution plan split alerting in two:
  * tick-level crash  -> immediate Telegram ping (already in controller.yml).
  * per-issue failures -> silence during the tick, rolled into ONE daily
    digest so every routine failure doesn't ping the operator at 2am.

Reads data/failures.jsonl + data/metrics.jsonl (written by fixer/guards.py),
builds terse per-language/per-repo stats, and sends one Telegram message.
Designed to run as a separate daily job in the controller workflow.

Usage:
    python beacon/digest.py --dry-run   # print the digest locally
    python beacon/digest.py --send      # post to Telegram (env: TELEGRAM_*)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
FAILURES = ROOT / "data" / "failures.jsonl"
METRICS = ROOT / "data" / "metrics.jsonl"


def _read_jsonl(path: Path, since: float) -> list[dict]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                try:
                    ts = datetime.fromisoformat(row.get("ts", "")).timestamp()
                except (ValueError, TypeError):
                    ts = 0
                if ts >= since:
                    rows.append(row)
    except OSError:
        return []
    return rows


def build_digest(days: int = 1) -> str:
    since = time.time() - days * 86400
    failures = _read_jsonl(FAILURES, since)
    metrics = _read_jsonl(METRICS, since)

    lines = [f"oss-agent daily digest ({days}d)"]
    if not failures and not metrics:
        lines.append("no attempts in window")
        return "\n".join(lines)

    outcomes: dict[str, int] = {}
    tokens = 0
    for row in metrics:
        outcome = row.get("outcome", "?")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        tokens += int(row.get("tokens_spent", 0) or 0)
    if outcomes:
        lines.append("attempts: " + ", ".join(f"{k}={v}" for k, v in outcomes.items()))
        lines.append(f"tokens spent: {tokens}")

    by_repo: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    for row in failures:
        repo = row.get("repo") or (row.get("issue") or {}).get("repo", "?")
        by_repo[repo] = by_repo.get(repo, 0) + 1
        stage = row.get("stage") or row.get("reason") or row.get("type", "failure")
        by_stage[stage] = by_stage.get(stage, 0) + 1
    if by_repo:
        lines.append("failures by repo: " + ", ".join(f"{k}={v}" for k, v in sorted(by_repo.items())))
    if by_stage:
        lines.append("failures by stage: " + ", ".join(f"{k}={v}" for k, v in sorted(by_stage.items())))
    lines.append("details: data/failures.jsonl + data/metrics.jsonl")
    return "\n".join(lines)


def _send_telegram(text: str, token: str, chat_id: str) -> bool:
    body = json.dumps({"chat_id": chat_id, "text": text, "disable_web_page_preview": True}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=body, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return 200 <= resp.status < 300
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="digest", description="W17 daily digest")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)

    text = build_digest(args.days)
    if args.dry_run or not args.send:
        print(text)
        return 0
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not (token and chat):
        print("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set; nothing sent")
        return 2
    ok = _send_telegram(text, token, chat)
    print("sent" if ok else "send failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())