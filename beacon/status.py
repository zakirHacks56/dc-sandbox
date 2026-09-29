"""Status summary for the Telegram master-switch card (stdlib only).

Reads only files the controller already commits (data/*, fixer/.agent_data) and
builds one Telegram-sized text block answering the operator's "what happened
since I looked?" question: issues encountered / solved, PRs raised, token burn,
remaining budgets, in-flight lanes, pending gates.

Deliberately dependency-free and never raises: if a file is missing/corrupt
its line is skipped. The numbers are best-effort diagnostics for a human, not
audit data.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beacon_util as util  # noqa: E402

# Issue-level fixer outcomes that mean the target was actually FIXED (vs.
# abandoned/error/noop). Mirrors fixer/guards.py's success vocabulary.
SOLVED_OUTCOMES = ("solved", "completed", "merged", "fix_merged")

# Record states that mean an attempt is still alive (vs terminal).
INFLIGHT_STATES = ("ANALYZING", "IMPLEMENTING", "TESTING", "REVIEWING",
                   "WAITING_FOR_FEEDBACK", "GATED", "PAUSED")


def _jsonl(path: Path) -> list:
    try:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    except OSError:
        return []


def _sum_int(rows, key: str) -> int:
    total = 0
    for r in rows:
        try:
            total += max(0, int(r.get(key, 0) or 0))
        except (TypeError, ValueError):
            continue
    return total


def _issue_outcomes() -> tuple[list, list, list, int]:
    """Return (unique_issues, issue_rows, all_rows, solved_by_metric).

    Tokens today use the ts prefix match so metrics rows and provider-usage
    events agree on the UTC day boundary used everywhere else."""
    rows = _jsonl(util.DATA / "metrics.jsonl")
    issue_rows = [r for r in rows if r.get("repo") and r.get("issue") is not None]
    seen = set()
    unique = []
    solved_by_metric = 0
    for r in issue_rows:
        key = (str(r.get("repo", "")), int(r.get("issue", -1)))
        if key not in seen:
            seen.add(key)
            unique.append(key)
        if str(r.get("outcome", "")).lower() in SOLVED_OUTCOMES:
            solved_by_metric += 1
    return unique, issue_rows, rows, solved_by_metric


def _record_states() -> tuple[list, int, int]:
    wf = util.FIXER / ".agent_data" / "workflows"
    records = []
    solved = 0
    prs = 0
    if wf.exists():
        for p in wf.glob("*/issue-*.json"):
            rec = util.load_json(p, None)
            if not rec:
                continue
            records.append(rec)
            if str(rec.get("state", "")) == "COMPLETED":
                solved += 1
            if rec.get("pr_number") is not None:
                prs += 1
    return records, solved, prs


def build_status() -> str:
    conf = util.load_json(util.CONFIG, {})
    board = util.load_json(util.BOARD, {})
    g = util.GATES

    unique, issue_rows, all_rows, solved_by_metric = _issue_outcomes()
    records, solved, prs = _record_states()
    solved = solved + solved_by_metric

    tokens_total = _sum_int(issue_rows, "tokens_spent")
    today = util.today_utc()
    tokens_today = _sum_int(
        [r for r in issue_rows if str(r.get("ts", "")).startswith(today)],
        "tokens_spent",
    )

    # Provider breakdown from provider_usage.json (period totals) plus today's
    # usage events, whichever the controller has recorded.
    usage = util.load_json(util.DATA / "provider_usage.json", {})
    prov_lines = []
    for name, st in (usage.get("providers") or {}).items():
        toks = int(st.get("tokens", 0) or 0)
        if toks:
            prov_lines.append(f" {name}: {toks:,}")
    events = _jsonl(util.DATA / "provider_usage_events.jsonl")
    events_today = [r for r in events if str(r.get("date", "")) == today]
    if events_today:
        prov_lines.append(f" today:{_sum_int(events_today, 'tokens'):,}")

    # Budgets straight from config + board, same names the tick uses.
    max_prs = int(conf.get("max_prs_per_day", 2))
    prs_today = int(board.get("prs_today", 0))
    max_pending = int(conf.get("max_pending_gates", 3))
    pending = sum(1 for p in g.glob("*.pending")) if g.exists() else 0
    decrees = sum(1 for p in g.glob("*.decree")) if g.exists() else 0
    max_attempts = int(conf.get("max_attempts_per_repo_day", 4))
    attempts_today = len({
        (str(r.get("repo", "")), int(r.get("issue", -1)))
        for r in issue_rows if str(r.get("ts", "")).startswith(today)
    })
    lanes = board.get("lanes") or {}
    inflight = sum(1 for l in lanes.values() if l.get("state") in INFLIGHT_STATES)
    rec_states = {}
    for r in records:
        st = str(r.get("state", "?"))
        rec_states[st] = rec_states.get(st, 0) + 1

    switch = util.master_switch()
    head = ("🟢 OSS controller RUNNING" if switch else
            "🔴 OSS controller PAUSED")
    lines = [head, ""]

    lines.append(f"📥 Issues encountered: {len(unique)}")
    lines.append(f"✅ Solved: {solved}")
    lines.append(f"🔀 PRs raised: {prs}")
    lines.append(f"🚧 In-flight lanes: {inflight}")
    if rec_states:
        parts = (f"{k.lower()}={v}" for k, v in sorted(rec_states.items()))
        lines.append("   lanes: " + ", ".join(parts))
    lines.append("")

    lines.append(f"💰 Tokens burned: {tokens_total:,}")
    lines.append(f"   today: {tokens_today:,}")
    if prov_lines:
        lines.append("   providers:" + "".join(prov_lines))
    lines.append("")

    lines.append(f"🎯 Budget: PRs {prs_today}/{max_prs}")
    lines.append(f"   pending gates {pending}/{max_pending}"
                 + (f" (+{decrees} waiting to apply)" if decrees else ""))
    lines.append(f"   attempts today {attempts_today}/{max_attempts} per repo")
    lines.append("")

    last_tick = board.get("last_tick") or "never"
    switch_state = "paused (no work)" if not switch else "live"
    lines.append(f"⚙️ last tick: {last_tick} [{switch_state}]")

    text = "\n".join(lines)
    return text[:3800]


if __name__ == "__main__":
    print(build_status())