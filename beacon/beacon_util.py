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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(str(path.suffix) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
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
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log(f"gh_api {method} {path} failed: {exc}")
        return None


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