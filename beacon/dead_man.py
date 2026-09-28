"""Dead-man watchdog for the OSS controller.

Reads the per-tick outcome rows the controller writes to data/metrics.jsonl
and judges whether the machine is alive-and-progressing or silently dead:

  * NO tick heartbeat within DEAD_MAN_MAX_IDLE_MIN -> alert (tick crashed,
    cron stopped, repo went read-only, etc.).
  * A STREAK of infra no-op outcomes (gh_dead / providers_down) -> alert:
    each single tick is bounded and non-fatal, but N in a row means the bot
    is breathing without doing its job.

Pure stdlib, one git-committed file in, one GitHub/Telegram call out: this is
meant to run as its own Actions step in the same workflow that runs the tick
cron, so "the controller is silently dead" is visible in the run list RED
instead of as an easy-to-ignore green tick.

Exit codes intentionally mirror infra severity:
  0  healthy (heartbeat fresh, no bad streak)
  1  dead-man fired -- needs a human
"""
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beacon_util as util  # noqa: E402

DEAD_MAN_MAX_IDLE_MIN = int(os.getenv("DEAD_MAN_MAX_IDLE_MIN", "20"))
BAD_STREAK = os.getenv("DEAD_MAN_BAD_STREAK", "gh_dead").split(",")
MAX_STREAK = int(os.getenv("DEAD_MAN_MAX_STREAK", "3"))


def _minutes_since(ts: str) -> float | None:
    if not ts:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(str(ts))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        return (now - parsed).total_seconds() / 60
    except ValueError:
        return None


def _bad_streak(rows: list) -> int:
    """Trailing count of tick rows whose outcome is in BAD_STREAK."""
    streak = 0
    for row in reversed(rows):
        if row.get("event") != "tick":
            continue
        if str(row.get("outcome", "")) in BAD_STREAK:
            streak += 1
        else:
            break
    return streak


def _tick_rows() -> list:
    try:
        lines = (util.DATA / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            import json
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def main() -> int:
    rows = _tick_rows()
    last = util.last_metric("tick")
    if last is None:
        msg = ("💀 OSS bot: NO tick heartbeat row exists in data/metrics.jsonl. "
               "The controller may never have run, or its metrics stream is "
               "gone. Check the tick cron / workflow.")
        util.log(msg)
        util.tg_send(msg)
        return 1

    idle_min = _minutes_since(last.get("ts"))
    streak = _bad_streak(rows)

    if idle_min is not None and idle_min > DEAD_MAN_MAX_IDLE_MIN:
        msg = (f"💀 OSS bot: no tick heartbeat for {idle_min:.0f} min "
               f"(last: {last.get('outcome')} at {last.get('ts')}). "
               "The controller is silent -- cron stopped or ticks are failing.")
        util.log(msg)
        util.tg_send(msg)
        return 1

    if streak >= MAX_STREAK:
        msg = (f"💀 OSS bot: {streak} consecutive infra no-op ticks "
               f"({', '.join(BAD_STREAK)}). Providers/token look dead -- "
               "the machine is breathing without hunting.")
        util.log(msg)
        util.tg_send(msg)
        return 1

    util.log(f"dead-man OK (last tick: {last.get('outcome')}, "
             f"idle {idle_min:.0f} min, bad-streak {streak})")
    return 0


if __name__ == "__main__":
    sys.exit(main())