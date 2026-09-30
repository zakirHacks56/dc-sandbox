"""Candidate hunter: pick the next (repo, issue) to attempt this tick.

Pure stdout-free, stdlib-only GitHub REST. This is a PRE-FILTER only -- the
vendor fixer re-validates everything itself (classification, difficulty,
report-only routing, PR limits). The hunter just avoids obviously-wrong
targets (already attempted, hard-labelled, too many stars, locked, assigned,
already has one of our open PRs) and lets the fixer make the final call.
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beacon_util as util  # noqa: E402

# Labels that signal a much-bigger-than-bugfix task. The fixer has its own
# difficulty scoring; this blocklist just saves an LLM-solve on obvious no-gos.
_HARD_HINTS = ("hard", "advanced", "complex", "epic", "major", "big", "won't fix")

# Hunt-time body screen: an issue whose specification is this large is a
# design-doc/feature request, not a bugfix -- the fixer would burn a whole
# clone + baseline-test tick to read it, then classify it as needing a plan
# first (or as an ENHANCEMENT beyond its charter). The tokens alone (roughly
# body_chars//3) can eat the entire routine issue budget before ANY code is
# fetched. Screen it here, free, before the expensive solve path starts.
MAX_ISSUE_BODY_CHARS = int(os.getenv("MAX_ISSUE_BODY_CHARS", "12000"))

# Issue-selection preference: issues numbered 1..N are scored ahead of
# brand-new ones (additive on top of the existing easy-issue/token-burn /
# newest-creation-date scoring -- not a replacement).
PREFERRED_ISSUE_NUMBERS_MAX = int(os.getenv("PREFERRED_ISSUE_NUMBERS", "150").strip() or 150)
# A repo with no commits inside this window is stale and gets no hunt calls.
REPO_STALE_DAYS = int(os.getenv("REPO_STALE_DAYS", "90").strip() or 90)

# SQLite candidate store (issue-hunt back-end). The controller/workflow only
# READS this DB (mode=ro) -- it is maintained by the separate issue-hunter
# project. Write side (state commits, PRs) is always the local PAT path.
HUNT_DB = os.getenv("ISSUE_HUNTER_DB", r"D:\issue-hunter\data\issue_hunter.db")
HUNT_DB_LIVE_CHECKS = int(os.getenv("HUNT_DB_LIVE_CHECKS", "30").strip() or 30)
# Difficulty rank map: a config difficulty_cap of N allows ranks <= N.
_DIFFICULTY_RANK = {"trivial": 0, "easy": 1, "medium": 2, "hard": 3}
# enrich_verdict is the issue-hunter's own classifier verdict; 'skip' rows are
# never offered as candidates (only a handful are tagged today, but respect it).
_TERMINAL_PR_STATES = ("ABANDONED", "COMPLETED", "CLOSED", "MERGED")


def _body_too_big(issue: dict) -> bool:
    body = str(issue.get("body") or "")
    if len(body) <= MAX_ISSUE_BODY_CHARS:
        return False
    title = str(issue.get("title") or "")
    util.log(f"#{issue.get('number')}: skipping (body {len(body)} chars > "
             f"{MAX_ISSUE_BODY_CHARS} -- feature-request sized)")
    if title:
        util.log(f"   ↳ title: {title[:120]}")
    return True


def _is_hard(issue: dict) -> bool:
    for label in issue.get("labels", []) or []:
        name = str(label.get("name", "")).lower()
        if any(h in name for h in _HARD_HINTS):
            return True
    return False


def _eligible(issue: dict, attempts: set) -> bool:
    if issue.get("pull_request"):
        return False
    if issue.get("locked"):
        return False
    if issue.get("assignees"):
        return False
    if issue.get("state") != "open":
        return False
    if issue.get("number") in attempts:
        return False
    if _body_too_big(issue):
        return False
    return not _is_hard(issue)


def _stars_ok(repo_full: str, max_stars: int) -> bool:
    if max_stars <= 0:
        return True
    meta = util.gh_api("GET", f"/repos/{repo_full}")
    if not meta:
        return True  # let the fixer decide when the meta call fails
    return int(meta.get("stargazers_count", 0)) <= max_stars


def _safe_repo_dir(repo_full: str) -> str:
    """owner/repo -> owner-repo (matches fixer session_store.safe_repo_dir)."""
    return str(repo_full).replace("/", "-")


def _attempt_status(repo_full: str, issue_number: int) -> str:
    """Run-once guard: attempt_status from the vendored fixer's committed
    workflow store, or '' when the issue was never selected. These records are
    tracked in git, so the rule survives restarts and every runner sees the
    same answer: not_attempted -> attempted_success | attempted_failed |
    budget_exhausted; an issue that is anything but not_attempted (or a record
    still literally in flight) is consumed and never selected again."""
    base = util.FIXER / ".agent_data" / "workflows"
    new = base / _safe_repo_dir(repo_full) / f"issue-{issue_number}.json"
    rec_path = new if new.exists() \
        else base / f"{_safe_repo_dir(repo_full)}_issue{issue_number}.json"
    if not rec_path.exists():
        return ""
    try:
        record = json.loads(rec_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return "corrupt"
    status = record.get("attempt_status") or "not_attempted"
    if status != "not_attempted":
        return status
    if record.get("state") not in ("COMPLETED", "ABANDONED"):
        return "in_flight"
    return ""


def _repo_active(repo_full: str, board: dict) -> bool:
    """Whether a repo is worth hunting today: not archived, not disabled, has
    commits inside REPO_STALE_DAYS, owner not suspended/deleted. Checked once
    per calendar day and cached on the board (committed back with the rest),
    so an 8x-daily repeat scan never re-burns the repo/meta API budget."""
    today = util.today_utc()[:10]
    cache = board.setdefault("repo_activity", {})
    if cache.get("date") == today and repo_full in cache.get("results", {}):
        return bool(cache["results"][repo_full])
    meta = util.gh_api("GET", f"/repos/{repo_full}")
    active = True
    if not meta:
        active = False  # 404/renamed/private: don't hunt blind
    elif meta.get("archived") or meta.get("disabled"):
        active = False
    else:
        pushed = str(meta.get("pushed_at") or "")
        if not pushed:
            active = False
        else:
            try:
                pushed_dt = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
                active = (datetime.now(timezone.utc) - pushed_dt).days <= REPO_STALE_DAYS
            except ValueError:
                active = True  # unparseable timestamp: let the fixer decide
            owner = meta.get("owner") or {}
            if active and owner.get("suspended_at"):
                active = False
    results = cache.get("results", {})
    results[repo_full] = active
    cache["results"] = results
    cache["date"] = today
    board["repo_activity"] = cache
    return active


def _pref_bucket(number: int) -> int:
    n = int(number)
    if 1 <= n <= PREFERRED_ISSUE_NUMBERS_MAX:
        return 0
    return 1


def _pick_issue(issues: list, attempts: set, repo_full: str) -> tuple | None:
    """Score one batch of candidate issues: preference bucket (low numbers
    first) then newest-created (the pre-existing scoring). Returns
    (number, issue) for the best eligible issue, or None."""
    scored = []
    for issue in issues:
        num = issue.get("number")
        if num is None:
            continue
        if not _eligible(issue, attempts):
            continue
        status = _attempt_status(repo_full, num)
        if status:
            util.log(f"#{num}: skipping (attempt {status})")
            continue
        created = str(issue.get("created_at") or "")
        try:
            created_ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        except ValueError:
            created_ts = 0.0
        scored.append((_pref_bucket(num), -created_ts, num, issue))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return scored[0][2], scored[0][3]


def _tree_too_big(repo_full: str, max_file_bytes: int) -> bool:
    """Reject repos whose working tree contains a source file far bigger than a
    bug fix should touch. LLM editors routinely fail or produce hollow diffs on
    >600KB blobs (Posnic/POS src/main.js ~196KB, sales.js ~686KB), so this
    pre-filter exists precisely to avoid re-burning ticks there.

    Only SOURCE-looking blobs count: build artifacts, vendored/third-party
    trees, test fixtures and report/result dumps are skipped, otherwise a
    single fat `results/tasks.jsonl` would freeze an otherwise perfect repo."""
    if max_file_bytes <= 0:
        return False
    _SKIP_DIRS = (
        "node_modules", "vendor", "dist", "build", "target", ".git",
        "coverage", "test-results", "playwright-report", "results", "reports",
        "assets", "static", "docs", "examples", "fixtures", "third_party",
        "third-party", "dataset", "data/",
    )
    _SKIP_EXT = (".jsonl", ".lock", ".html", ".zip", ".json", ".csv", ".gz",
                 ".png", ".jpg", ".jpeg", ".svg", ".woff", ".woff2", ".min.js")
    meta = util.gh_api("GET", f"/repos/{repo_full}")
    if not meta:
        return False
    branch = meta.get("default_branch", "master")
    tree = util.gh_api("GET", f"/repos/{repo_full}/git/trees/{branch}?recursive=1")
    if not tree:
        return False
    for item in tree.get("tree", []):
        if item.get("type") != "blob":
            continue
        path = str(item.get("path", ""))
        low = path.lower()
        if any(seg in low.split("/") for seg in _SKIP_DIRS):
            continue
        if low.endswith(_SKIP_EXT):
            continue
        if int(item.get("size", 0)) > max_file_bytes:
            util.log(f"{repo_full}: skipping (source file {path} is "
                     f"{item.get('size')}B > {max_file_bytes}B)")
            return True
    return False


def _day_attempt_budget_ok(repo_full: str, board: dict, max_per_day: int) -> bool:
    """Cap how many distinct issues one repo can burn per day across the board
    so a single unlucky target can't starve the rest of the machine.

    Only lanes touched TODAY that represent a REAL attempt count: a lane where
    actual LLM tokens were spent. Preflight refusals (token budget refused
    before a single call), infra abandons and pure stale sweeps spend nothing,
    so they don't burn a slice of the daily cap -- otherwise a bot stuck behind
    a bad context step fills every lane with IMPLEMENTING no-spend stalls and
    then reports "no candidate" for hours.

    The lane's 'spent' field carries the token spend recorded by the fixer; a
    lane without it (legacy) is treated as an attempt (counts) to stay
    conservative. The board keeps old lanes around as a recoverable record."""
    if max_per_day <= 0:
        return True
    today = util.today_utc()
    lanes = board.get("lanes", {})
    count = sum(
        1 for lane_key, lane in lanes.items()
        if lane_key.startswith(f"{repo_full}#")
        and str(lane.get("updated", "")).startswith(today)
        and int(lane.get("spent", 1)) > 0
    )
    return count < max_per_day


def _open_pr_count(repo_full: str, login: str) -> int:
    pulls = util.gh_paged(f"/repos/{repo_full}/pulls?state=open")
    return sum(1 for pr in pulls if (pr.get("user") or {}).get("login") == login)


# --- SQLite-backed candidate store (issue_hunter.db) -------------------------

def _db_connect():
    """Read-only connection to the issue-hunter store, or None. Never writes:
    the workflow only reads this DB."""
    if not os.path.exists(HUNT_DB):
        util.log(f"issue db: {HUNT_DB} missing")
        return None
    try:
        return sqlite3.connect(Path(HUNT_DB).as_uri() + "?mode=ro",
                               uri=True, timeout=10)
    except Exception as exc:  # noqa: BLE001 - keep the tick alive
        util.log(f"issue db: open failed ({type(exc).__name__}: {exc})")
        return None


def _labels_from_db(labels_json) -> list:
    try:
        parsed = json.loads(labels_json or "[]")
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, TypeError):
        pass
    return []


def _is_hard_labels(names) -> bool:
    for name in names:
        low = name.lower()
        if any(h in low for h in _HARD_HINTS):
            return True
    return False


def _local_live_pr(repo_full: str, board: dict) -> bool:
    """Do we already have a non-terminal PR in this repo? Board lanes +
    workflow records only (no API call) -- a repo we already shipped to is
    skipped so one repo can't eat the whole machine's attention."""
    lanes = board.get("lanes", {})
    for key, lane in lanes.items():
        if key.startswith(f"{repo_full}#") and lane.get("pr"):
            if str(lane.get("state", "")).upper() not in _TERMINAL_PR_STATES:
                return True
    base = util.FIXER / ".agent_data" / "workflows" / _safe_repo_dir(repo_full)
    if base.exists():
        try:
            for rec_file in base.glob("issue-*.json"):
                try:
                    rec = json.loads(rec_file.read_text(encoding="utf-8")) or {}
                except (OSError, ValueError):
                    continue
                if rec.get("pr_number") and str(rec.get("state", "")).upper() not in _TERMINAL_PR_STATES:
                    return True
        except OSError:
            pass
    return False


def _find_from_db(conf: dict, board: dict, login: str) -> tuple | None:
    """Pick the next (repo, number) purely from the issue-hunter store.

    Local (no-API) filters run on every row: body size, hard labels,
    already-attempted / in-flight (workflow store), today's repo budget,
    and 'we already have a PR there'. Live GitHub checks (stars/activity,
    open-PR count) run only for repos that just passed everything, bounded
    by HUNT_DB_LIVE_CHECKS so a cold large store can't burn the API budget."""
    conn = _db_connect()
    if conn is None:
        return None
    cap = _as_int(conf, "difficulty_cap", 2)
    allowed = {lvl for lvl, rank in _DIFFICULTY_RANK.items() if rank <= cap}
    if not allowed:
        util.log(f"issue db: difficulty_cap {cap} admits no levels")
        conn.close()
        return None
    max_stars = _as_int(conf, "max_stars", 3000)
    max_per_day = _as_int(conf, "max_attempts_per_repo_day", 0)
    attempted_cache = {}
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT repo, number, body, labels_json, difficulty_level "
                "FROM issues "
                "WHERE status = 'active' "
                "  AND (enrich_verdict IS NULL OR enrich_verdict = '' "
                "       OR enrich_verdict != 'skip') "
                "  AND (tombstone_reason IS NULL OR tombstone_reason = '') "
                "ORDER BY "
                "  CASE difficulty_level "
                "    WHEN 'Trivial' THEN 0 WHEN 'Easy' THEN 1 "
                "    WHEN 'Medium' THEN 2 ELSE 3 END, "
                "  repo, rowid "
                "LIMIT 5000"
            )
        except sqlite3.OperationalError as exc:
            util.log(f"issue db: query failed ({str(exc)[:120]})")
            conn.close()
            return None
        live = 0
        for repo_full, num, body, labels_json, level in cur.fetchall():
            num = int(num)
            if (level or "").lower() not in allowed:
                continue
            if repo_full not in attempted_cache:
                attempted_cache[repo_full] = set(board.get("attempted", {}).get(repo_full, []))
            if num in attempted_cache[repo_full]:
                continue
            try:
                if _attempt_status(repo_full, num):
                    continue
            except Exception:  # noqa: BLE001 - a bad record must not crash the hunt
                continue
            if _body_too_big({"number": num, "body": body or "", "title": str(body or "")[:80]}):
                continue
            if _is_hard_labels(_labels_from_db(labels_json)):
                continue
            if not _day_attempt_budget_ok(repo_full, board, max_per_day):
                continue
            if _local_live_pr(repo_full, board):
                continue
            if live >= HUNT_DB_LIVE_CHECKS:
                util.log(f"issue db: stopped at {live} live repo checks "
                         "(HUNT_DB_LIVE_CHECKS)")
                break
            live += 1
            try:
                if not _repo_active(repo_full, board):
                    util.log(f"{repo_full}: issue-db skip (inactive repo)")
                    continue
                if not _stars_ok(repo_full, max_stars):
                    util.log(f"{repo_full}: issue-db skip (stars > {max_stars})")
                    continue
                if login and _open_pr_count(repo_full, login) >= 1:
                    util.log(f"{repo_full}: issue-db skip (we already have an open PR there)")
                    continue
            except Exception as exc:  # noqa: BLE001 - one bad repo must not kill the hunt
                util.log(f"{repo_full}: hunt live-check error: {type(exc).__name__}: "
                         f"{str(exc)[:120]} -- skipping")
                continue
            util.log(f"issue db: pick {repo_full}#{num} ({level})")
            conn.close()
            return repo_full, num
        conn.close()
        return None
    except Exception as exc:  # noqa: BLE001 - the hunt must never crash the tick
        util.log(f"issue db hunt error: {type(exc).__name__}: {str(exc)[:150]}")
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        return None


def find_candidate(conf: dict, board: dict) -> tuple | None:
    """Return (repo_full_name, issue_number) or None if nothing is worth a try.

    Never raises: a hunt runs 8x a day on a fixed controller slot, so a single
    misconfigured target or unexpected data shape must not kill the whole tick.
    Any unexpected exception is logged and treated as 'no candidate'."""
    try:
        return _find_candidate_impl(conf, board)
    except Exception as exc:  # noqa: BLE001 - a hunt must never crash the tick
        util.log(f"hunt error: {type(exc).__name__}: {str(exc)[:200]} -- "
                 "treating as no candidate")
        return None


def _as_int(conf: dict, key: str, default: int) -> int:
    """Tolerate a bad/missing int config so one typo can't kill the hunt."""
    try:
        return int(conf.get(key, default))
    except (TypeError, ValueError):
        util.log(f"hunt config: '{key}' is not an int ({conf.get(key)!r}) -- "
                 f"using {default}")
        return default


def _find_candidate_impl(conf: dict, board: dict) -> tuple | None:
    """Inner candidate search; per-target exceptions are absorbed so one bad
    target is skipped while the other repos keep being scanned.

    The candidate SOURCE is configurable: 'db' hunts purely from the
    issue-hunter SQLite store (read-only), 'targets' keeps the legacy
    GitHub-live target walk, 'both' tries the DB first then targets. If the
    DB is missing when 'db'/'both' is requested we fall back to the target
    walk with a loud log so the controller never starves."""
    login = (util.gh_api("GET", "/user") or {}).get("login", "")
    source = str(conf.get("hunt_source") or os.getenv("HUNT_SOURCE") or "targets").lower()
    if source in ("db", "both"):
        db_candidate = _find_from_db(conf, board, login)
        if db_candidate:
            return db_candidate
        if source == "db":
            return None
        util.log("hunt: DB returned no candidate -- falling back to targets")

    max_stars = _as_int(conf, "max_stars", 3000)
    max_file_bytes = _as_int(conf, "max_file_bytes", 0)
    max_per_day = _as_int(conf, "max_attempts_per_repo_day", 0)
    labels = conf.get("default_labels", ["good first issue"])

    targets = [t for t in conf.get("targets", []) if t.get("enabled", True)]
    if not targets:
        return None

    for target in targets:
        repo_full = target.get("repo", "")
        if not repo_full:
            util.log(f"hunt: skipping a target with no repo name ({str(target)[:80]})")
            continue
        try:
            attempts = set(board.get("attempted", {}).get(repo_full, []))
            if not _stars_ok(repo_full, max_stars):
                util.log(f"{repo_full}: skipping (stars > {max_stars})")
                continue
            if not _repo_active(repo_full, board):
                util.log(f"{repo_full}: skipping (inactive: archived/disabled/quiet/suspended)")
                continue
            if _tree_too_big(repo_full, max_file_bytes):
                continue
            if not _day_attempt_budget_ok(repo_full, board, max_per_day):
                util.log(f"{repo_full}: skipping (hit {max_per_day}-lane/day budget)")
                continue
            if login and _open_pr_count(repo_full, login) >= 1:
                util.log(f"{repo_full}: skipping (we already have an open PR there)")
                continue

            repo_labels = target.get("labels")
            if repo_labels is None:
                path = f"/repos/{repo_full}/issues?state=open"
                picked = _pick_issue(util.gh_paged(path), attempts, repo_full)
                if picked:
                    return repo_full, picked[0]
                util.log(f"{repo_full}: no eligible open issue (unlabeled hunt)")
                continue
            for label in repo_labels:
                path = f"/repos/{repo_full}/issues?state=open&labels={quote(label)}"
                picked = _pick_issue(util.gh_paged(path), attempts, repo_full)
                if picked:
                    return repo_full, picked[0]
                util.log(f"{repo_full}: no eligible issue under label '{label}'")
        except Exception as exc:  # noqa: BLE001 - one bad repo must not kill the hunt
            util.log(f"{repo_full}: hunt target error: {type(exc).__name__}: "
                     f"{str(exc)[:200]} -- skipping this repo")
            continue
    return None