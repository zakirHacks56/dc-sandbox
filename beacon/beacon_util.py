"""Shared helpers for the beacon scripts (pure stdlib, no third-party deps).

The whole orchestrator deliberately avoids PyGithub/openai/pyyaml so the
gate poller can run on a bare Actions runner with zero `pip install`. Only
the vendored fixer under fixer/ has real dependencies, and it is only ever
invoked as a subprocess from beacon/tick.py.
"""
import datetime
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "targets.json"
DATA = ROOT / "data"
GATES = DATA / "gates"
BOARD = DATA / "board.json"
FIXER = ROOT / "fixer"
FIXER_SCRIPT = FIXER / "oss_agent_v2.py"
# Each workflow logs to its OWN file (BEACON_LOG) so concurrent controller and
# gate-poll runs never write the same path and trip git merge conflicts.
LOG = Path(os.getenv("BEACON_LOG", str(DATA / "log.txt")))

# Fixed Telegram API IPv4s, used only when DNS cannot resolve api.telegram.org
# (broken ISP resolvers / NAT64-only networks). The API endpoints never move.
TG_API_HOST = "api.telegram.org"
TG_API_IPS = ("149.154.167.220", "149.154.167.99", "149.154.175.100", "149.154.166.110")


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def today_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, obj) -> None:
    """Atomic write (temp + os.replace) with a UNIQUE temp name and a small
    retry on Windows sharing violations -- stress tests hammer offset.json from
    several pollers at once, and a transient PermissionError here would abort
    a live workflow run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    for attempt in range(5):
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(payload)
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    # Re-raise the last error: five attempts of an atomic write all failed.
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def log(msg: str) -> None:
    line = f"{now_utc()}  {msg}"
    print(line, flush=True)
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def gh_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or ""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def gh_api(method: str, path: str, payload=None):
    """Minimal GitHub REST call. Returns parsed JSON or None on any failure."""
    import urllib.request

    url = f"https://api.github.com{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method.upper(), headers=gh_headers(),
    )
    _GH_API_ERRORS["calls"] += 1
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log(f"gh_api {method} {path} failed: {exc}")
        _GH_API_ERRORS["fails"] += 1
        return None


# Per-process counter so tick can distinguish "no issues found" (healthy) from
# "every API call failed" (dead token) and raise the alarm instead of silently
# doing nothing every ten minutes.
_GH_API_ERRORS = {"calls": 0, "fails": 0}


def gh_api_stats() -> tuple:
    """(total calls, failed calls) for GitHub REST calls this process."""
    return _GH_API_ERRORS["calls"], _GH_API_ERRORS["fails"]


def gh_paged(path: str, pages: int = 3) -> list:
    """Fetch several pages of a REST collection endpoint into one list."""
    results = []
    for page in range(1, pages + 1):
        sep = "&" if "?" in path else "?"
        batch = gh_api("GET", f"{path}{sep}per_page=100&page={page}")
        if not isinstance(batch, list):
            break
        results.extend(batch)
        if len(batch) < 100:
            break
    return results


def parse_gate_key(key: str):
    """Reverse the fixer's gate-key format (`owner-repo_issue42_human`).

    NOTE: the repo half is the fixer's safe form (``/`` replaced by ``-``),
    which is lossy for repos that already contain hyphens, so the parsed repo
    here is only ever cosmetic. Trust the metadata inside the .pending /
    .decree / .outcome files for the real ``owner/repo``. Returns
    (safe_repo, issue_number, gate) or None when it does not parse.
    """
    import re

    match = re.match(r"^(.*)_issue(\d+)_(human|final|close)$", key)
    if not match:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def env_for_fixer(extra=None) -> dict:
    """Env passed to the vendored fixer subprocess: cloud gate mode always on,
    GATE_DIR pointed at the committed data/gates tree.

    GATE_AUTO defaults to ON so a tick that produces a validated fix can land
    the draft PR without waiting for a Telegram button tap (the operator still
    sees the notification). Set GATE_AUTO=0 in extra to force parked gates."""
    env = dict(os.environ)
    env["GATE_ASYNC"] = "1"
    env.setdefault("GATE_AUTO", "1")
    env["GATE_DIR"] = str(GATES)
    # GATE_AUTO self-approves the quality gates, but distinct drafts across
    # repos are still bounded by hunter rotation + the daily max_pending_gates /
    # max_prs_per_day caps in tick.py.
    env["ALLOW_MULTI_PR_SAME_REPO"] = "1"
    if extra:
        env.update(extra)
    return env


# Tick outcomes, written one line per tick to data/metrics.jsonl so an operator
# (or the dead-man alert in the cloud workflow) can see -- from the git history
# alone -- whether the controller is making progress or quietly doing nothing.
TICK_OUTCOMES = (
    "decreed",          # applied a waiting gate decree
    "reconciled",       # re-filed parked lanes to match GitHub truth
    "swept",            # closed stale own PR(s) / abandoned in-flight lanes
    "reaped",           # abandoned leaked in-flight lanes at tick start
    "hunted",           # dispatched a fresh solve attempt
    "paused",           # master switch OFF -- operator stopped the machine
    "no_candidate",     # healthy tick, but nothing eligible to hunt
    "budget_limited",   # at PR / pending-gate / lane budget cap
    "providers_down",   # no LLM provider callable -- a real infra no-op
    "gh_dead",          # every GitHub API call failed -- token likely dead
)


def _offset_doc() -> dict:
    """The committed data/offset.json document (poll offset + operator state).

    This file is committed by BOTH the controller and gate-poll workflows, so
    it is the natural vehicle for the master switch + control card -- no
    workflow YAML changes needed for the state to survive runs."""
    return load_json(DATA / "offset.json", {})


def _save_offset_doc(doc: dict) -> None:
    save_json(DATA / "offset.json", doc)


def master_switch() -> bool:
    """True when the controller is allowed to work (default). A missing or
    unreadable switch key means enabled -- the switch is purely additive, so a
    repo without it keeps running as before."""
    try:
        return bool(_offset_doc().get("master_switch", {}).get("enabled", True))
    except Exception:  # noqa: BLE001
        return True


def set_master_switch(enabled: bool, by: str = "") -> None:
    """Persist the operator's Start/Stop state so ticks and the dead-man
    watchdog can see it. `by` records who flipped it (e.g. "telegram")."""
    doc = _offset_doc()
    doc["master_switch"] = {"enabled": bool(enabled), "by": by, "at": now_utc()}
    _save_offset_doc(doc)


def load_control_card() -> dict:
    return _offset_doc().get("control_card", {})


def save_control_card(**fields) -> None:
    doc = _offset_doc()
    card = doc.setdefault("control_card", {})
    card.update(fields)
    _save_offset_doc(doc)


def metric(event: str, **fields) -> None:
    """Append one JSONL row to data/metrics.jsonl (same stream the fixer's
    guards.metrics() writes). Never raises -- a metrics write must not crash
    a tick. `event` is the metric kind (e.g. "tick"); fields are free-form."""
    row = {"event": event, "ts": now_utc()}
    row.update(fields)
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = DATA / "metrics.jsonl"
    try:
        with open(tmp, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def last_metric(event: str) -> dict | None:
    """Newest row of a given event kind from data/metrics.jsonl, or None."""
    try:
        with open(DATA / "metrics.jsonl", "r", encoding="utf-8") as fh:
            last = None
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("event") == event:
                    last = row
            return last
    except OSError:
        return None


def tg_send(text: str, timeout: int = 20) -> bool:
    """Best-effort Telegram sendMessage (stdlib, zero pip). Returns True when
    Telegram acknowledged the message. Silently False on missing token or any
    transport failure, so alerts can never crash a tick. Falls back to the
    fixed Telegram IPv4 list when DNS is broken."""
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("MANUAL_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not (token and chat):
        return False
    body = json.dumps({"chat_id": chat, "text": text[:3900]}).encode("utf-8")

    def _post(host: str, with_host_header: bool) -> bool:
        url = f"https://{host}/bot{token}/sendMessage"
        headers = {"Content-Type": "application/json"}
        if with_host_header:
            headers["Host"] = TG_API_HOST
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            blob = resp.read().decode("utf-8")
            try:
                return bool(json.loads(blob).get("ok"))
            except ValueError:
                return True

    for host, with_header in [(TG_API_HOST, False)] + [(ip, True) for ip in TG_API_IPS]:
        try:
            if _post(host, with_header):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False