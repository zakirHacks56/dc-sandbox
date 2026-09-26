"""Controller tick: one bounded unit of work per invocation.

Decision order per tick:
  1. Apply any waiting gate decree (highest priority -- an operator already
     tapped a button). Runs the fixer with --gate-sync so the decree is
     consumed exactly once and the PR (or decline) becomes real.
  2. Otherwise, if we are below the pending-gate and daily-PR budget, hunt a
     fresh candidate and run a solve; the fixer parks its human gate for the
     operator instead of crashing.
  3. Otherwise just housekeeping.

Only ONE unit per tick, on purpose: the controller cron fires every ~10min,
so the cloud spends a bounded, predictable amount of LLM time. Everything it
produces (board, fixer record, gate files, logs) is committed by the workflow
step that runs this script, which is what makes the whole machine recoverable
from nothing but the git history.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beacon_util as util  # noqa: E402
import hunter_stage  # noqa: E402

RUN_TIMEOUT_SECONDS = 55 * 60  # Actions default job timeout is generous
PENDING_MAX_AGE_DAYS = 14
# A draft PR we opened and that has sat unmerged, with its workflow stuck in
# a pre-merge state for this many days, is a stale own-draft. It blocks the
# hunter (open-PR guard) and burns the repo's daily lane budget, so the sweep
# closes it via the fixer's --force -close (auto-approving the close gate).
STALE_PR_AGE_DAYS = 3


def _run_fixer(args: list, extra_env: dict | None = None) -> str:
    cmd = [sys.executable, str(util.FIXER_SCRIPT), *args]
    util.log(f"fixer: {' '.join(cmd)}  (cwd={util.FIXER})")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(util.FIXER),
            env=util.env_for_fixer(extra_env),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=RUN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        util.log("fixer TIMED OUT after %s s" % RUN_TIMEOUT_SECONDS)
        return "timeout"
    tail = ((proc.stdout or "")[-1500:] + "\n" + (proc.stderr or "")[-500:]).strip()
    util.log(f"fixer exit={proc.returncode}\n--tail--\n{tail}")
    return "ok" if proc.returncode == 0 else f"exit{proc.returncode}"


def _workflow_record(repo_name: str, issue_number: int):
    safe = repo_name.replace("/", "-")
    rec = util.FIXER / ".agent_data" / "workflows" / safe / f"issue-{issue_number}.json"
    if not rec.exists():
        return None
    try:
        return util.load_json(rec, None)
    except Exception:
        return None


def _attempted(board, repo_name):
    board.setdefault("attempted", {})
    board["attempted"].setdefault(repo_name, [])
    return board["attempted"][repo_name]


def _pending_keys() -> list:
    if not util.GATES.exists():
        return []
    return sorted(p.stem for p in util.GATES.glob("*.pending"))


def _decree_keys() -> list:
    if not util.GATES.exists():
        return []
    return sorted(p.stem for p in util.GATES.glob("*.decree"))


def _apply_decree(board, key: str) -> bool:
    parsed = util.parse_gate_key(key)
    if not parsed:
        util.log(f"decree {key}: unparseable -- deleting")
        (util.GATES / f"{key}.decree").unlink(missing_ok=True)
        return False
    # The key is lossy for repo names containing hyphens, so trust the rich
    # metadata the poller copied out of the .pending file; the parsed safe
    # name is only a fallback (may be missing the forward slash).
    decree_path = util.GATES / f"{key}.decree"
    meta = util.load_json(decree_path, {})
    repo_name = meta.get("repo") or parsed[0]
    issue_number = meta.get("issue") if meta.get("issue") is not None else parsed[1]
    gate = meta.get("gate") or parsed[2]
    util.log(f"APPLYING decree {key} ({repo_name}#{issue_number}, gate={gate})")
    _run_fixer(["--repo", repo_name, "--issue", str(issue_number), "--gate-sync"])

    if decree_path.exists():
        util.log(f"decree {key} still present: fixer did not reach the gate this run")
        return False

    outcome = util.GATES / f"{key}.outcome"
    decision = None
    if outcome.exists():
        payload = util.load_json(outcome, {})
        decision = bool(payload.get("decision"))
        util.log(f"decree {key} resolved: {'APPROVE' if decision else 'DECLINE'}")

    rec = _workflow_record(repo_name, issue_number) or {}
    if decision:
        board["prs_today"] = int(board.get("prs_today", 0)) + 1
        board.setdefault("prs", []).append({
            "repo": repo_name, "issue": issue_number, "gate": gate,
            "pr": rec.get("pr_number"), "url": rec.get("pr_url"),
            "time": util.now_utc(),
        })
    board.setdefault("lanes", {})[f"{repo_name}#{issue_number}"] = {
        "state": rec.get("state"), "pr": rec.get("pr_number"),
        "gate": gate, "decision": decision, "updated": util.now_utc(),
    }
    return True


def _housekeeping(board) -> None:
    swept = 0
    for pending in util.GATES.glob("*.pending") if util.GATES.exists() else []:
        age_days = (time.time() - pending.stat().st_mtime) / 86400
        if age_days > PENDING_MAX_AGE_DAYS:
            pending.unlink(missing_ok=True)
            swept += 1
    if swept:
        util.log(f"housekeeping: swept {swept} stale .pending file(s)")
    board["last_tick"] = util.now_utc()


# Workflow states that are "in flight": a PR exists or is being built but nothing
# has landed. A lane stuck in any of these for STALE_PR_AGE_DAYS is sweepable.
_INFLIGHT_STATES = ("ANALYZING", "IMPLEMENTING", "WAITING_FOR_FEEDBACK",
                    "REVIEWING", "GATED", "PAUSED")


def _stale_days(updated: str | None) -> float | None:
    """Whole days a timestamp is past (None when it can't be read or is missing)."""
    import datetime as _dt
    if not updated:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(str(updated))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.timezone.utc)
        return (_dt.datetime.now(_dt.timezone.utc) - parsed).total_seconds() / 86400
    except Exception:
        return None


def _sweep_stale(board, enabled_repos: set) -> int:
    """Close own draft PRs that have been stuck in-flight longer than
    STALE_PR_AGE_DAYS.

    Scans the fixer's workflow records (the authoritative store for the PRs we
    opened -- the board lanes only track hunter attempts and carry no pr). The
    open-PR guard sees these as "we already have an open PR there"
    (sandbox#33/#34), so until they are closed the repo is permanently frozen.
    GATE_AUTO_CLOSE=1 makes the fixer's close gate auto-approve -- safe because:
      * the sweep only targets OUR OWN in-flight records past the age gate
      * the PR is a draft we opened and never got merged, not someone else's
    The fixer's -close releases the issue claim and logs the lane ABANDONED.

    The same pass also abandons stale in-flight lanes that never opened a PR
    (crashed attempts), so their board slots no longer count against the repo's
    daily lane budget. Returns the number of lanes resolved.
    Sweep is scoped to repos currently enabled as targets: a repo somebody
    retired from the rotation must keep whatever PR it already has, since a
    maintainer may still be reviewing it and we no longer own the slot."""
    closed = 0
    lanes = board.setdefault("lanes", {})
    wf_root = util.FIXER / ".agent_data" / "workflows"
    if wf_root.exists():
        for rec_path in sorted(wf_root.glob("*/issue-*.json")):
            rec = util.load_json(rec_path, None)
            if not rec:
                continue
            state = str(rec.get("state", ""))
            if state not in _INFLIGHT_STATES:
                continue
            updated = rec.get("updated_at") or rec.get("updated")
            days = _stale_days(updated)
            if days is None or days <= STALE_PR_AGE_DAYS:
                continue
            repo_name = str(rec.get("repo", ""))
            try:
                issue_num = int(rec["issue"])
            except (KeyError, TypeError, ValueError):
                continue
            if not repo_name or "/" not in repo_name:
                continue
            if repo_name not in enabled_repos:
                continue
            lane_key = f"{repo_name}#{issue_num}"
            if rec.get("pr_number"):
                util.log(f"sweep: closing stale own PR #{rec['pr_number']} for "
                         f"{lane_key} (state={state}, age={days:.1f}d)")
                try:
                    result = _run_fixer(
                        ["--repo", repo_name, "--issue", str(issue_num),
                         "--force", "-close"],
                        extra_env={"GATE_AUTO_CLOSE": "1"},
                    )
                except subprocess.TimeoutExpired:
                    util.log(f"sweep: close of {lane_key} TIMED OUT -- leaving lane")
                    continue
                util.log(f"sweep: fixer close of {lane_key} -> {result}")
                closed += 1
            else:
                util.log(f"sweep: abandoning in-flight lane {lane_key} "
                         f"(state={state}, age={days:.1f}d, no PR ever opened)")
                closed += 1
            # Keep the ABANDONED lane's ORIGINAL updated timestamp: the daily
            # lane budget counts lanes touched today, and preserving the old
            # timestamp makes a swept lane fall out of today's count instead
            # of burning brand-new budget (the "frees their daily budget" bit).
            lanes[lane_key] = {
                "state": "ABANDONED",
                "decision": False,
                "updated": rec.get("updated_at") or rec.get("updated") or util.now_utc(),
            }
            # Make the sweep CONVERGE: persist the workflow record as ABANDONED
            # too, so the next tick no longer sees an in-flight record here and
            # stops re-abandoning the same lane every run. Before this, a stale
            # no-PR lane was abandoned in the board forever but the record kept
            # its old PAUSED state + old timestamp, so the sweep re-fired on the
            # same lane each tick and ate the tick's single unit of work,
            # starving the hunter. ABANDONED is not in _INFLIGHT_STATES, so the
            # record will not be swept again. Keep the record's ORIGINAL
            # updated/updated_at so the lane stays out of today's budget.
            try:
                rec["state"] = "ABANDONED"
                rec["swept_at"] = util.now_utc()
                util.save_json(rec_path, rec)
            except Exception as sweep_err:
                util.log(f"sweep: could not persist ABANDONED state for "
                         f"{lane_key}: {sweep_err}")
    return closed


def main() -> int:
    util.DATA.mkdir(parents=True, exist_ok=True)
    conf = util.load_json(util.CONFIG, {})
    if not conf:
        util.log("no config/targets.json -- aborting")
        return 1
    board = util.load_json(util.BOARD, {"date": util.today_utc()})
    if board.get("date") != util.today_utc():
        util.log("new day: resetting PR budget")
        board = {"date": util.today_utc(), "prs_today": 0, "prs": [],
                 "attempted": board.get("attempted", {}),
                 "lanes": board.get("lanes", {})}

    acted = False

    decrees = _decree_keys()
    if decrees:
        key = decrees[0]
        _apply_decree(board, key)
        acted = True

    if not acted:
        enabled_repos = {
            t["repo"] for t in conf.get("targets", []) if t.get("enabled", True)
        }
        stale_resolved = _sweep_stale(board, enabled_repos)
        if stale_resolved:
            util.log(f"sweep: resolved {stale_resolved} stale in-flight lane(s)")
            acted = True  # a close/abandon is this tick's single unit of work

        pending = _pending_keys()
        prs_today = int(board.get("prs_today", 0))
        max_pending = int(conf.get("max_pending_gates", 3))
        max_prs = int(conf.get("max_prs_per_day", 2))
        if not acted and len(pending) < max_pending and prs_today < max_prs:
            candidate = hunter_stage.find_candidate(conf, board)
            if candidate:
                repo_name, issue_number = candidate
                util.log(f"HUNTING: trying {repo_name}#{issue_number}")
                _run_fixer(["--repo", repo_name, "--issue", str(issue_number), "--gate-sync"])
                attempts = _attempted(board, repo_name)
                if issue_number not in attempts:
                    attempts.append(issue_number)
                    if len(attempts) > 50:
                        del attempts[: len(attempts) - 50]
                rec = _workflow_record(repo_name, issue_number)
                if rec:
                    board.setdefault("lanes", {})[f"{repo_name}#{issue_number}"] = {
                        "state": rec.get("state"), "pr": rec.get("pr_number"),
                        "updated": util.now_utc(),
                    }
                acted = True
            else:
                calls, fails = util.gh_api_stats()
                if calls >= 3 and fails == calls:
                    util.log(
                        f"FATAL: every GitHub API call failed ({fails}/{calls}) -- "
                        "the token (PR_PAT) is dead (revoked/expired?) and the "
                        "machine is running blind. Sounding the alarm."
                    )
                    return 1
                util.log("no candidate found this tick")
        else:
            util.log(f"budget: {prs_today}/{max_prs} PRs, {len(pending)}/{max_pending} pending gates")

    _housekeeping(board)
    util.save_json(util.BOARD, board)
    util.log("tick done")
    return 0


if __name__ == "__main__":
    sys.exit(main())