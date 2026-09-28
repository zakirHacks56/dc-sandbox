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
import json
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
# Reaper age: a solve run is expected to finish (or park) well inside the
# per-issue wall clock; an in-flight record this old with no PR and no pending
# gate is a leaked/stuck run (crash between "IMPLEMENTING" and a terminal
# write), so the reaper abandons it at the start of a tick. This is the
# backstop for the guaranteed-terminal-state fix: a finally in the fixer can't
# catch SIGKILL/OOM, so the controller catches those here instead.
REAP_MAX_AGE_SECONDS = 40 * 60


def _providers_callable() -> bool:
    """Preflight for the fresh-hunt path: is any registered LLM provider
    callable right now?

    If every provider is OPEN (cooldown) or over budget, a new hunt would only
    burn CI minutes + baseline-test time for an immediate
    AllProvidersExhaustedError -- skip scanning until something recovers.
    Fail-open: no keys or an import hiccup lets the fixer run, since it
    reports its own diagnosis either way. Never touches the network."""
    try:
        if str(util.FIXER) not in sys.path:
            sys.path.insert(1, str(util.FIXER))
        import llm_router as router
    except Exception:  # noqa: BLE001 -- preflight must never crash the tick
        return True
    try:
        providers = router.load_providers()
    except Exception:  # noqa: BLE001
        return True
    if not providers:
        return True
    return any(router.is_callable(p)[0] for p in providers)


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


def _lane_spent(repo_name: str, issue_number: int) -> int:
    """Total tokens spent on (repo, issue) today, from the fixer metrics.

    The day lane budget only counts lanes with a REAL attempt (spent > 0), so a
    preflight-budget or infra abandonment -- which spends nothing -- can't fill
    every lane with no-spend stalls and starve other repos."""
    mfile = util.DATA / "metrics.jsonl"
    if not mfile.exists():
        return 0
    today = util.today_utc()
    total = 0
    try:
        with open(mfile, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001 -- one bad row must not stop us
                    continue
                if (str(row.get("repo", "")), row.get("issue")) != (repo_name, issue_number):
                    continue
                if str(row.get("ts", "")).startswith(today):
                    total += int(row.get("tokens_spent", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0
    return total


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
        "gate": gate, "decision": decision,
        "spent": _lane_spent(repo_name, issue_number),
        "updated": util.now_utc(),
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

# States that mean a solve run is ACTIVELY consuming a lane (vs. parked behind
# a human or deliberately paused). Only these are fast-reaped: a run stuck in
# ANALYZING/IMPLEMENTING/TESTING for over an hour is leaked, full stop. PAUSED
# is an explicit operator state, and WAITING_FOR_FEEDBACK/GATED mean a human is
# (or was meant to be) involved -- neither gets fast-reaped.
_REAPABLE_STATES = ("ANALYZING", "IMPLEMENTING", "TESTING", "REVIEWING")

# Records we will NOT re-file based on GitHub truth -- their terminal state is
# already the honest reflection of the PR's fate.
_RECONCILE_SKIP_STATES = ("COMPLETED", "ABANDONED", "CANCELLED")


def _reconcile_prs(board, enabled_repos: set) -> int:
    """GitHub-truth reconcile: resolve parked lanes whose PR (or the issue it
    fixes) has ALREADY been resolved upstream -- without waiting for the
    3-day sweep.

    The open-PR guard reads GitHub live, so a repo whose PR the maintainer
    CLOSED or MERGED is already unblocked -- but the lane record keeps sitting
    WAITING_FOR_FEEDBACK forever, booking today's budget and (for closed) never
    getting the courteous close + thanks the maintainer earned. Ask GitHub what
    actually happened to each parked PR and re-file the lane to match:

      * PR MERGED       -> COMPLETED + a courteous thank-you comment (retired
                           by the fixer's own merge detection on the next
                           -conversation; doing it here frees the slot now).
      * PR CLOSED unmerge -> ABANDONED (the maintainer already made the call;
                           we don't give up on it -- it's already over).
      * PR still open but the issue it fixes is CLOSED -> courteously close
                           OUR OWN draft via the fixer -close (auto-approved),
                           ABANDONED -- the slot is burned on a done issue.

    Cheap: one GET per parked record (two for the issue-closed check), and a
    courtesy comment / fixer-close subprocess is only spawned for the last
    case, once per tick. Pure re-files (merged / closed-unmerged) are only
    local JSON writes and run in bulk. Returns the number of lanes re-filed."""
    re_filed = 0
    wf_root = util.FIXER / ".agent_data" / "workflows"
    if not wf_root.exists():
        return 0
    lanes = board.setdefault("lanes", {})
    closed_one = False
    for rec_path in sorted(wf_root.glob("*/issue-*.json")):
        rec = util.load_json(rec_path, None)
        if not rec:
            continue
        if str(rec.get("state", "")) in _RECONCILE_SKIP_STATES:
            continue
        repo_name = str(rec.get("repo", ""))
        try:
            issue_num = int(rec["issue"])
            pr_num = int(rec["pr_number"])
        except (KeyError, TypeError, ValueError):
            continue
        if not repo_name or "/" not in repo_name or repo_name not in enabled_repos:
            continue
        lane_key = f"{repo_name}#{issue_num}"
        if lane_key in lanes and lanes[lane_key].get("state") in ("COMPLETED", "ABANDONED"):
            continue
        pr = util.gh_api("GET", f"/repos/{repo_name}/pulls/{pr_num}")
        if not pr:
            continue  # API hiccup -- sweep will catch this lane later anyway
        merged = bool(pr.get("merged"))
        pr_state = str(pr.get("state", ""))
        keep = rec.get("updated_at") or rec.get("updated") or util.now_utc()
        if merged:
            util.log(f"reconcile: PR #{pr_num} for {lane_key} MERGED upstream "
                     f"-- marking COMPLETED")
            rec["state"] = "COMPLETED"
            rec["outcome"] = "merged upstream"
            rec["reconcile_at"] = util.now_utc()
            util.save_json(rec_path, rec)
            lanes[lane_key] = {"state": "COMPLETED", "decision": True,
                               "pr": pr_num, "updated": keep}
            _thank_merge(repo_name, pr_num)
            re_filed += 1
        elif pr_state == "closed":
            util.log(f"reconcile: PR #{pr_num} for {lane_key} closed WITHOUT "
                     f"merge -- marking ABANDONED")
            rec["state"] = "ABANDONED"
            rec["outcome"] = "maintainer closed PR without merge"
            rec["reconcile_at"] = util.now_utc()
            util.save_json(rec_path, rec)
            lanes[lane_key] = {"state": "ABANDONED", "decision": False,
                               "pr": pr_num, "updated": keep}
            re_filed += 1
        else:
            # PR is still open. If the issue it fixes is DONE upstream, keep
            # the repo slot moving instead of parking forever on a corpse.
            # Only one courteous close per tick -- a subprocess is a real
            # unit of work, and the rest wait for the following ticks.
            if closed_one:
                continue
            issue = util.gh_api("GET", f"/repos/{repo_name}/issues/{issue_num}")
            if issue and str(issue.get("state", "")) != "open":
                util.log(f"reconcile: PR #{pr_num} for {lane_key} still open but "
                         f"issue #{issue_num} is {issue.get('state')} upstream -- "
                         f"courteously closing our own PR")
                try:
                    result = _run_fixer(
                        ["--repo", repo_name, "--issue", str(issue_num),
                         "--force", "-close"],
                        extra_env={"GATE_AUTO_CLOSE": "1"},
                    )
                except subprocess.TimeoutExpired:
                    util.log(f"reconcile: courteous close of {lane_key} TIMED OUT "
                             f"-- leaving lane to the sweep")
                    continue
                util.log(f"reconcile: fixer close of {lane_key} -> {result}")
                rec["state"] = "ABANDONED"
                rec["reconcile_at"] = util.now_utc()
                util.save_json(rec_path, rec)
                lanes[lane_key] = {"state": "ABANDONED", "decision": False,
                                   "updated": keep}
                closed_one = True
                re_filed += 1
    return re_filed


def _thank_merge(repo_name: str, pr_num: int) -> None:
    """Best-effort thank-you on a merged PR; never raises. The maintainer read
    the draft, reviewed and merged it -- a one-line thanks is the courteous
    part of the reconcile, and it also nudges flaky merge paths to converge."""
    try:
        util.gh_api(
            "POST",
            f"/repos/{repo_name}/issues/{pr_num}/comments",
            {"body": "Thanks for reviewing and merging this!"},
        )
    except Exception:  # noqa: BLE001 -- courtesy, never a blocker
        pass


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


def _stale_seconds(updated: str | None) -> float | None:
    """Seconds a timestamp is past (None when it can't be read or is missing)."""
    import datetime as _dt
    if not updated:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(str(updated))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.timezone.utc)
        return (_dt.datetime.now(_dt.timezone.utc) - parsed).total_seconds()
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


def _reap_stale(board, enabled_repos: set) -> int:
    """Abandon in-flight lanes whose runs leaked out of the fixer.

    Fast backstop for the guaranteed-terminal-state fix. The fixer's wrapper
    writes a terminal record before every exit, but a SIGKILL/OOM/CI-host kill
    can cut a solve run off mid-IMPLEMENTING with no terminal record and a
    live lane. The controller commits state each tick, so a leaked lane sits
    stuck in an in-flight state for the next tick -- and every tick -- eating a
    repo's daily lane budget forever (or until a human notices).

    So at the START of each tick, any lane in an ACTIVELY-RUNNING state
    (ANALYZING/IMPLEMENTING/TESTING/REVIEWING -- not PAUSED, not gated, not
    awaiting feedback) older than REAP_MAX_AGE_SECONDS (no PR, no pending
    gate) is abandoned, freeing its budget slice for a real attempt. This is
    fast and free (no GitHub calls): it writes the board lane and the workflow
    record in place and converges.

    Skipped when the record has a real PR (the 3-day sweep owns those) or a
    human gate is still pending (that run is parked, not leaked). Returns the
    number of lanes reaped this tick."""
    reaped = 0
    lanes = board.setdefault("lanes", {})
    wf_root = util.FIXER / ".agent_data" / "workflows"
    if not wf_root.exists():
        return 0
    gates_dir = util.GATES
    for rec_path in sorted(wf_root.glob("*/issue-*.json")):
        rec = util.load_json(rec_path, None)
        if not rec:
            continue
        state = str(rec.get("state", ""))
        if state not in _REAPABLE_STATES:
            continue
        updated = rec.get("updated_at") or rec.get("updated")
        age_secs = _stale_seconds(updated)
        if age_secs is None or age_secs <= REAP_MAX_AGE_SECONDS:
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
        # Parked at a human gate is a legitimately live lane -- never reap it.
        if rec.get("pr_number"):
            continue
        lane_key = f"{repo_name}#{issue_num}"
        # A pending gate means the run is waiting on a human decree, broker
        # outage or not; the reaper must not kill a lane that may still resolve.
        pending = list((gates_dir.glob(f"*{repo_name.replace('/', '-')}*{issue_num}*.pending"))
                       ) if gates_dir.exists() else []
        if pending:
            util.log(f"reap: skipping {lane_key} (state={state}, "
                     f"age={age_secs/60:.0f}m, gate pending -- parked)")
            continue
        util.log(f"reap: abandoning leaked in-flight lane {lane_key} "
                 f"(state={state}, age={age_secs/60:.0f}m, no PR, no gate)")
        reaped += 1
        keep = rec.get("updated_at") or rec.get("updated") or util.now_utc()
        # Preserve the ORIGINAL updated timestamp: the daily lane budget counts
        # lanes touched today, so a reaped lane must keep its old `updated` to
        # fall out of today's count (it never got a real attempt's spend).
        lanes[lane_key] = {
            "state": "ABANDONED",
            "decision": False,
            "updated": keep,
            "reaped_at": util.now_utc(),
        }
        try:
            rec["state"] = "ABANDONED"
            rec["reaped_at"] = util.now_utc()
            util.save_json(rec_path, rec)
        except Exception as reap_err:
            util.log(f"reap: could not persist ABANDONED state for "
                     f"{lane_key}: {reap_err}")
    return reaped


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

    enabled_repos = {
        t["repo"] for t in conf.get("targets", []) if t.get("enabled", True)
    }

    # First and cheapest: abandon in-flight lanes whose runs leaked. No GitHub
    # calls, no fixer subprocess -- just frees budget for a real attempt below.
    reaped = _reap_stale(board, enabled_repos)
    if reaped:
        util.log(f"reaped {reaped} leaked lane(s) -- budget freed for a real attempt")

    decrees = _decree_keys()
    if decrees:
        key = decrees[0]
        _apply_decree(board, key)
        acted = True

    if not acted:
        reconciled = _reconcile_prs(board, enabled_repos)
        if reconciled:
            util.log(f"reconcile: re-filed {reconciled} parked lane(s) to match GitHub truth")
            acted = True  # a subprocess close counts as this tick's unit of work

        stale_resolved = _sweep_stale(board, enabled_repos)
        if stale_resolved:
            util.log(f"sweep: resolved {stale_resolved} stale in-flight lane(s)")
            acted = True  # a close/abandon is this tick's single unit of work

        pending = _pending_keys()
        prs_today = int(board.get("prs_today", 0))
        max_pending = int(conf.get("max_pending_gates", 3))
        max_prs = int(conf.get("max_prs_per_day", 2))
        if not acted and len(pending) < max_pending and prs_today < max_prs:
            if _providers_callable():
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
                            "spent": _lane_spent(repo_name, issue_number),
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
                util.log("preflight: no LLM provider callable -- skipping hunt")
        else:
            util.log(f"budget: {prs_today}/{max_prs} PRs, {len(pending)}/{max_pending} pending gates")

    _housekeeping(board)
    util.save_json(util.BOARD, board)
    util.log("tick done")
    return 0


if __name__ == "__main__":
    sys.exit(main())