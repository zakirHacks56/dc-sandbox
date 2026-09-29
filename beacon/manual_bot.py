"""manual_bot.py
Personal Telegram control bot for the fixer, meant to run on YOUR OWN PC
alongside the cloud controller (which keeps doing its daily work on its own
bot token).

Design:
  * Listens on a SEPARATE bot token (MANUAL_BOT_TOKEN) so it never steals the
    cloud gate poller's getUpdates offset. Token/chat come from env:
        MANUAL_BOT_TOKEN      (required; your personal bot token)
        TELEGRAM_CHAT_ID or TELEGRAM_ALLOWED_IDS  (who may talk to this bot)
        LLM_API_KEY           (required for the OpenRouter fallback path)
        GITHUB_TOKEN or GH_TOKEN (required; your PAT for the target repos)
        OMNIROUTE_TIMEOUT / OMNIROUTE_STALL_SECONDS  (optional tuning)
  * Allowed chats default to TELEGRAM_CHAT_ID / TELEGRAM_ALLOWED_IDS.
  * Commands:
        /hunt                       pick the next candidate via the SAME issue
                                    hunter the cloud tick uses, then solve it
        /run <owner/repo> <issue>   solve an issue -> draft PR
        /conversation <owner/repo> <issue>   one maintainer-feedback round
        /finalize <owner/repo> <issue>  arm finalization (human gate stays)
        /resume [issue]             restore a saved workflow and continue
        /close <owner/repo> <issue> close the draft PR + un-claim
        /leave [issue]              park the workflow locally
        /status [issue]             board + gates (no issue) / one workflow
        /gates                      what needs a human decision right now
        /list                       table of every tracked workflow
        /conv-info [issue]          transcript + round history dump
        /review [issue]             read-only GitHub maintainer comments
        /finish [issue]             mark workflow finished locally
        /stop                       kill the active fixer run
        /help                       this text
  * Backend per /run: if OmniRoute is up at localhost:20128 => use it.
    Otherwise fall back to OpenRouter (free-tier chain; needs LLM_API_KEY).
  * Command button taps (decree:...) are written as .decree files under
    data/gates/ just like the cloud poller, and a --gate-sync fixer run is
    kicked off so the parked PR advances or closes. GATE_AUTO stays OFF here
    so every draft really does wait for your tap.

stdlib-only on purpose so it runs on a bare box with no pip install.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beacon_util as util  # noqa: E402
import hunter_stage  # noqa: E402

BOT_TOKEN = os.getenv("MANUAL_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED = [
    c.strip() for c in os.getenv(
        "TELEGRAM_ALLOWED_IDS", os.getenv("TELEGRAM_CHAT_ID", "")
    ).split(",") if c.strip()
]
OFFSET_FILE = util.DATA / "manual_offset.json"
LOCAL_GATEWAY = os.getenv("OMNIROUTE_LOCAL_URL", "http://localhost:20128")
FALLBACK_BASE = os.getenv(
    "OMNIROUTE_BASE_URL", "https://openrouter.ai/api/v1"
)

_active = {"proc": None, "lock": threading.Lock(), "label": None}


def _tg_raw(method: str, payload: dict, timeout: int = 30):
    if not BOT_TOKEN:
        return None
    body = json.dumps(payload).encode("utf-8")

    def _post(host: str, with_host_header: bool):
        url = f"https://{host}/bot{BOT_TOKEN}/{method}"
        headers = {"Content-Type": "application/json"}
        if with_host_header:
            headers["Host"] = util.TG_API_HOST
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    for host, header in [(util.TG_API_HOST, False), *((ip, True) for ip in util.TG_API_IPS)]:
        try:
            data = _post(host, header)
            if data is not None:
                return data
        except Exception:
            continue
    return None


def _send(chat_id: str, text: str) -> None:
    _tg_raw("sendMessage", {"chat_id": chat_id, "text": text[:3900]})


def _answer(callback_id: str, text: str) -> None:
    _tg_raw("answerCallbackQuery", {"callback_query_id": callback_id, "text": text})


def _get_updates(offset: int) -> list:
    data = _tg_raw("getUpdates", {"timeout": 8, "offset": offset, "limit": 20}, timeout=20)
    if not data:
        return []
    return data.get("result") or []


def _is_allowed(chat_id) -> bool:
    chat_id = str(chat_id)
    return not ALLOWED or chat_id in ALLOWED


def _sentinel_chat() -> str:
    return ALLOWED[0] if ALLOWED else ""


# --------------------------------------------------------------- commands
def _cmd_help() -> str:
    return (
        "dc-sandbox manual bot -- full control of the fixer from Telegram\n\n"
        "/hunt -- pick the next candidate via the same issue hunter the cloud "
        "tick uses, then solve it to a draft PR\n"
        "/run <owner/repo> <issue> -- solve an issue -> draft PR\n"
        "/conversation <owner/repo> <issue> -- one maintainer-feedback round "
        "on the existing draft PR\n"
        "/finalize <owner/repo> <issue> -- arm finalization (final green test "
        "+ human approval still required)\n"
        "/resume [issue] -- restore a saved workflow and continue\n"
        "/close <owner/repo> <issue> -- close the draft PR + un-claim issue\n"
        "/leave [issue] -- park workflow locally\n"
        "/status [issue] -- board/gates, or one workflow's report\n"
        "/gates -- what needs a human decision right now\n"
        "/list -- table of every tracked workflow\n"
        "/conv-info [issue] -- transcript + round history\n"
        "/review [issue] -- read-only GitHub comments/reviews\n"
        "/finish [issue] -- mark workflow finished locally\n"
        "/stop -- kill the active fixer run\n\n"
        "Backend: local OmniRoute at localhost:20128 if up, else the "
        "OpenRouter free chain (needs LLM_API_KEY in manual.env)."
    )


def _hunt_candidate() -> tuple | None:
    """Same pick the cloud tick makes: targets.json + board.json through
    hunter_stage.find_candidate(). Returns (repo, issue) or None."""
    conf = util.load_json(util.CONFIG, {})
    if not conf:
        return None
    board = util.load_json(util.BOARD, {"date": util.today_utc()})
    return hunter_stage.find_candidate(conf, board)


def _cmd_status(issue=None) -> str:
    if issue is not None:
        return _run_readonly(["--status", str(issue)]) or "(no output)"
    board = util.load_json(util.BOARD, {})
    lines = [
        f"date={board.get('date')}  prs_today={board.get('prs_today', 0)}",
        f"max_pending={(util.load_json(util.CONFIG, {}) or {}).get('max_pending_gates', 3)}",
    ]
    lanes = board.get("lanes", {})
    if lanes:
        lines.append("lanes:")
        for key, ls in sorted(lanes.items()):
            lines.append(f"  {key}: {ls.get('state')} (PR #{ls.get('pr')})")
    else:
        lines.append("lanes: none")
    pending = _pending_keys()
    lines.append(f"pending gates: {len(pending)}")
    if pending:
        lines.append("  " + ", ".join(pending[:10]))
    return "\n".join(lines)


def _cmd_gates() -> str:
    pending = _pending_keys()
    if not pending:
        return "no gates waiting on a human right now"
    out = ["gates waiting on you:"]
    for key in pending:
        meta = util.load_json(util.GATES / f"{key}.pending", {})
        repo = meta.get("repo") or key
        issue = meta.get("issue") or ""
        gate = meta.get("gate") or ""
        out.append(f"  {repo} #{issue} [{gate}]")
    return "\n".join(out)


def _pending_keys() -> list:
    if not util.GATES.exists():
        return []
    return sorted(p.stem for p in util.GATES.glob("*.pending"))


def _repo_for_issue(issue: str) -> str:
    """Resolve `owner/repo` for a bare issue number from the fixer's saved
    workflow index, so /resume N works without typing the repo."""
    index = util.load_json(util.FIXER / ".agent_data" / "state" / "index.json", {})
    workflows = index.get("workflows", {}) or {}
    for key, rec in workflows.items():
        if str(rec.get("issue")) == str(issue):
            return rec.get("repo") or ""
        key_issue = key.rsplit("#", 1)[-1] if "#" in key else ""
        if key_issue == str(issue) and rec.get("repo"):
            return rec["repo"]
    return ""


def _gateway_up() -> str:
    """Return the base URL that can ACTUALLY generate text.

    A gateway that answers /v1/models with 200 but has no provider credentials
    connected (``No active credentials for provider``) would make the fixer
    spin through its whole fallback chain for 40+ minutes, so the probe must
    be a real 1-token chat completion, not a reachability ping. Non-fatal:
    on any failure we return the fallback provider base instead.
    The probed models are the ones the fixer will actually use (auto/* combos
    served by the gateway, never valid on the OpenRouter fallback), tried
    with a generous timeout because the first combo the gateway routes can be
    slow to warm up."""
    import json as _json

    for model in ("auto/best-chat", "auto/best-coding"):
        payload = _json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }).encode()
        try:
            req = urllib.request.Request(
                f"{LOCAL_GATEWAY}/v1/chat/completions",
                data=payload, headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                if resp.status in (200, 201):
                    util.log(f"gateway probe OK via {model}")
                    return f"{LOCAL_GATEWAY}/v1"
        except Exception as exc:
            util.log(f"gateway probe {model} failed: {exc}")
            continue
    return FALLBACK_BASE


def _fixer_env(base: str, extra=None) -> dict:
    env = util.env_for_fixer({
        "GATE_AUTO": "0",  # manual mode: always wait for a button tap
        "OMNIROUTE_BASE_URL": base,
        "BEACON_LOG": str(util.DATA / "log-manual.txt"),
    })
    if extra:
        env.update(extra)
    return env


def _spawn_fixer(args: list, chat_id: str) -> None:
    base = _gateway_up()
    if base == FALLBACK_BASE and not (
        os.getenv("OMNIROUTE_API_KEY") or os.getenv("LLM_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    ):
        _send(
            chat_id,
            "no working LLM backend: the local gateway at "
            f"{LOCAL_GATEWAY} has no provider credentials connected, and no "
            "OMNIROUTE_API_KEY/LLM_API_KEY is set for the fallback. "
            "Connect a provider at http://localhost:20128/dashboard (or paste "
            "a key into manual.env), then retry. Not starting a run that "
            "would spin forever.",
        )
        return
    env = _fixer_env(base)
    label = " ".join(args)
    util.log(f"manual fixer: {label} via {base}")
    _send(chat_id, f"starting fixer `{label}` via {base} ...")
    cmd = [sys.executable, str(util.FIXER_SCRIPT), *args]
    with _active["lock"]:
        if _active["proc"] is not None and _active["proc"].poll() is None:
            _send(chat_id, "another fixer run is already active -- /stop it first")
            return
        proc = subprocess.Popen(
            cmd, cwd=str(util.FIXER), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        _active["proc"] = proc
        _active["label"] = label
    util.log(f"fixer pid={proc.pid} spawned")
    _send(chat_id, f"fixer pid={proc.pid} started; I will report when it finishes.")


def _run_readonly(args: list, timeout: int = 180) -> str:
    """Run a fixer read-only command (--status/--list-workflows/--conv-info/
    --review-feedback) synchronously and return its output. Safe to run while
    another fixer is active: these commands take no locks and write nothing."""
    base = _gateway_up()
    env = _fixer_env(base)
    try:
        proc = subprocess.run(
            [sys.executable, str(util.FIXER_SCRIPT), *args],
            cwd=str(util.FIXER), env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"(read-only command timed out after {timeout}s)"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    return out or err or f"(exit {proc.returncode}, no output)"


def _run_thread(args: list, chat_id: str) -> None:
    try:
        _spawn_fixer(args, chat_id)
    except Exception as exc:
        _send(chat_id, f"failed to start fixer: {exc}")
        return
    with _active["lock"]:
        proc = _active["proc"]
        label = _active["label"]
    assert proc is not None
    tail = []
    try:
        for line in proc.stdout or []:
            tail.append(line)
            if len(tail) > 200:
                tail = tail[-200:]
        proc.wait()
    except Exception:
        pass
    with _active["lock"]:
        if _active["proc"] is proc:
            _active["proc"] = None
            _active["label"] = None
    body = "".join(tail[-24:]).strip()
    _send(chat_id, f"done: {label} exit={proc.returncode}\n---tail---\n{body}")


def _stop_thread() -> None:
    with _active["lock"]:
        proc = _active["proc"]
        label = _active["label"]
        _active["proc"] = None
        _active["label"] = None
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        util.log(f"manual /stop killed pid={proc.pid}")
    chat = _sentinel_chat()
    if chat:
        _send(chat, f"stopped {label or 'active run'}")


# ------------------------------------------------------------ master control
def _master_text() -> str:
    running = util.master_switch()
    dot = "🟢" if running else "🔴"
    state = "RUNNING" if running else "PAUSED"
    return (f"OSS controller: {dot} {state}\n\n"
            "Tap Start to let it hunt again, Stop to freeze it "
            "(no more fixer runs / tokens spent while paused).")


def _master_markup() -> dict:
    return {
        "inline_keyboard": [[
            {"text": "Start", "callback_data": "master:on"},
            {"text": "Stop", "callback_data": "master:off"},
            {"text": "Status", "callback_data": "master:status"},
        ]]
    }


def _update_master_card() -> None:
    """Keep the Start/Stop/Status card pinned on the MANUAL bot. Uses its own
    state key (manual_card.json) so it never clobbers the cloud poller's
    control_card in offset.json."""
    chat = _sentinel_chat()
    if not chat:
        return
    card = util.load_json(util.DATA / "manual_card.json", {})
    text = _master_text()
    if card.get("message_id") and card.get("text") == text:
        return  # nothing changed -- do not touch Telegram
    if card.get("message_id"):
        data = _tg_raw("editMessageText", {
            "chat_id": chat, "message_id": card["message_id"],
            "text": text, "reply_markup": _master_markup(),
        })
        desc = (data or {}).get("description", "") if data else ""
        if data and data.get("ok") is True:
            util.save_json(util.DATA / "manual_card.json", {**card, "text": text})
            return
        if "message is not modified" in desc:
            util.save_json(util.DATA / "manual_card.json", {**card, "text": text})
            return
        util.save_json(util.DATA / "manual_card.json", {})  # stale -- repost
    data = _tg_raw("sendMessage", {
        "chat_id": chat, "text": text, "reply_markup": _master_markup(),
    })
    if data and data.get("ok"):
        msg_id = (data.get("result") or {}).get("message_id")
        util.save_json(util.DATA / "manual_card.json",
                       {"chat_id": chat, "message_id": msg_id, "text": text})
        util.log(f"control card posted (message_id={msg_id})")


def _push_state() -> None:
    """Commit + push data/offset.json (and any gates) so the CLOUD controller
    sees a manual Stop/Start on its next tick. Best-effort, never raises."""
    try:
        tok = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN", "")
        if not tok:
            return
        env = dict(os.environ)
        env["GITHUB_TOKEN"] = tok
        env["GH_TOKEN"] = tok
        root = util.ROOT
        git = ["git", "-C", str(root)]
        msg = "manual-bot: state " + util.now_utc()
        subprocess.run(git + ["add", "data/offset.json", "data/gates"],
                       env=env, capture_output=True)
        subprocess.run(git + ["commit", "-m", msg, "--allow-empty"],
                       env=env, capture_output=True)
        for _ in range(3):
            subprocess.run(git + ["pull", "--no-rebase", "--no-edit", "-X", "ours",
                                  "origin", "master"], env=env, capture_output=True)
            subprocess.run(git + ["add", "data/offset.json", "data/gates"],
                           env=env, capture_output=True)
            subprocess.run(git + ["commit", "-m", msg, "--allow-empty"],
                           env=env, capture_output=True)
            result = subprocess.run(git + ["push", "origin", "master"],
                                    env=env, capture_output=True)
            if result.returncode == 0:
                break
    except Exception as exc:  # noqa: BLE001
        util.log(f"push_state failed: {exc}")


def _handle_master(query: dict) -> bool:
    data = query.get("data") or ""
    if data == "master:status":
        _answer(query.get("id", ""), "Here's the status")
        _send(_sentinel_chat(), _cmd_status())
        return True
    if data not in ("master:on", "master:off"):
        return False
    enabled = data == "master:on"
    util.set_master_switch(enabled, by="telegram-manual")
    label = "RUNNING" if enabled else "PAUSED"
    _answer(query.get("id", ""), f"Controller {label}")
    _update_master_card()
    _send(_sentinel_chat(),
          f"OSS controller {label}.\n"
          + ("Resuming work -- I'll hunt again on the next tick. "
             "(This also syncs the cloud controller.)"
             if enabled else
             "Frozen -- no fixer runs / tokens while stopped. Tap Start to resume."))
    _push_state()
    return True


# ------------------------------------------------------------ decrees
def _handle_callback(query: dict) -> bool:
    data = query.get("data") or ""
    if data.startswith("master:"):
        return _handle_master(query)
    match = re.match(r"^decree:(.+):([01])$", data)
    if not match:
        return False
    key, decision = match.group(1), match.group(2) == "1"
    parsed = util.parse_gate_key(key)
    pending = util.GATES / f"{key}.pending"
    meta = util.load_json(pending, {}) if pending.exists() else {}
    repo = meta.get("repo") or (parsed[0] if parsed else key)
    issue = meta.get("issue") if meta.get("issue") is not None else (parsed[1] if parsed else None)
    gate = meta.get("gate") or (parsed[2] if parsed else None)
    (util.GATES / f"{key}.decree").write_text(
        json.dumps({
            "key": key, "decision": decision, "repo": repo,
            "issue": issue, "gate": gate, "decided_at": util.now_utc(),
        }, ensure_ascii=False), encoding="utf-8",
    )
    if pending.exists():
        pending.unlink(missing_ok=True)
    label = "APPROVE" if decision else "DECLINE"
    util.log(f"manual decree: {repo}#{issue} [{gate}] -> {label}")
    _answer(query.get("id", ""), f"Recorded: {label}")
    chat = _sentinel_chat()
    if chat:
        _send(chat, f"Gate {repo}#{issue} [{gate}]: {label} recorded.")
    if repo and issue is not None:
        target_chat = chat or _sentinel_chat()
        threading.Thread(
            target=_run_thread,
            args=(["--repo", repo, "--issue", str(issue), "--gate-sync"], target_chat),
            daemon=True,
        ).start()
    return True


def _handle_text(chat_id: str, text: str) -> None:
    text = (text or "").strip()
    low = text.lower()
    starts = low.split()
    cmd = starts[0] if starts else ""
    rest = text[len(cmd):].strip() if cmd else ""

    if cmd == "/run":
        m = re.match(r"^([\w.-]+/[\w.-]+)\s+(\d+)\b", rest)
        if not m:
            _send(chat_id, "usage: /run <owner/repo> <issue>")
            return
        args = ["--repo", m.group(1), "--issue", m.group(2), "--gate-sync"]
        threading.Thread(target=_run_thread, args=(args, chat_id), daemon=True).start()
    elif cmd == "/hunt":
        try:
            candidate = _hunt_candidate()
        except Exception as exc:
            _send(chat_id, f"hunt failed: {exc}")
            return
        if not candidate:
            _send(chat_id, "hunter found no candidate right now "
                           "(budget/gates/stars/filters -- /status for lanes)")
            return
        repo_name, issue = candidate
        args = ["--repo", repo_name, "--issue", str(issue), "--gate-sync"]
        _send(chat_id, f"/hunt picked {repo_name}#{issue} -- solving...")
        threading.Thread(target=_run_thread, args=(args, chat_id), daemon=True).start()
    elif cmd == "/conversation":
        m = re.match(r"^([\w.-]+/[\w.-]+)\s+(\d+)\b", rest)
        if not m:
            _send(chat_id, "usage: /conversation <owner/repo> <issue>")
            return
        args = ["--repo", m.group(1), "--issue", m.group(2), "-conversation",
                "--gate-sync"]
        threading.Thread(target=_run_thread, args=(args, chat_id), daemon=True).start()
    elif cmd == "/finalize":
        m = re.match(r"^([\w.-]+/[\w.-]+)\s+(\d+)\b", rest)
        if not m:
            _send(chat_id, "usage: /finalize <owner/repo> <issue>")
            return
        args = ["--repo", m.group(1), "--issue", m.group(2), "-conversation",
                "--finalize", "--gate-sync"]
        threading.Thread(target=_run_thread, args=(args, chat_id), daemon=True).start()
    elif cmd == "/close":
        m = re.match(r"^([\w.-]+/[\w.-]+)\s+(\d+)\b", rest)
        if not m:
            _send(chat_id, "usage: /close <owner/repo> <issue>")
            return
        args = ["--repo", m.group(1), "--issue", m.group(2), "--force", "-close"]
        threading.Thread(target=_run_thread, args=(args, chat_id), daemon=True).start()
    elif cmd == "/resume":
        issue = rest.split()[0] if rest.split() else None
        if issue and re.fullmatch(r"\d+", issue):
            repo = _repo_for_issue(issue)
            if repo:
                args = ["--repo", repo, "--issue", issue, "--resume", issue]
            else:
                args = ["--resume", issue]
        else:
            args = ["--resume"]
        threading.Thread(target=_run_thread, args=(args, chat_id), daemon=True).start()
    elif cmd == "/leave":
        issue = rest.split()[0] if rest.split() else None
        args = ["--leave"] + ([issue] if issue else [])
        threading.Thread(target=_run_thread, args=(args, chat_id), daemon=True).start()
    elif cmd == "/finish":
        issue = rest.split()[0] if rest.split() else None
        args = ["--force", "--finish"] + ([issue] if issue else [])
        threading.Thread(target=_run_thread, args=(args, chat_id), daemon=True).start()
    elif cmd in ("/status", "/conv-info", "/review"):
        issue = rest.split()[0] if rest.split() else None
        flag = {"status": "--status", "conv-info": "--conversation-info",
                "review": "--review-feedback"}[cmd[1:]]
        args = [flag] + ([issue] if issue else [])
        threading.Thread(
            target=lambda: _send(chat_id, _run_readonly(args) or "(no output)"),
            daemon=True,
        ).start()
    elif cmd == "/list":
        threading.Thread(
            target=lambda: _send(chat_id, _run_readonly(["--list-workflows"]) or "(no output)"),
            daemon=True,
        ).start()
    elif cmd == "/gates":
        _send(chat_id, _cmd_gates())
    elif cmd in ("/stop", "/stopall"):
        _stop_thread()
    else:
        _send(chat_id, _cmd_help())


def main() -> int:
    util.DATA.mkdir(parents=True, exist_ok=True)
    if not BOT_TOKEN:
        print("neither MANUAL_BOT_TOKEN nor TELEGRAM_BOT_TOKEN set -- aborting")
        return 1
    me = _tg_raw("getMe", {})
    util.log(f"manual bot online: {(me or {}).get('result', {}).get('username', '?')}")
    chat = _sentinel_chat()
    if chat:
        _send(chat, "manual bot online. /help for commands.")
        _update_master_card()
    offset = util.load_json(OFFSET_FILE, {}).get("offset", 0)
    while True:
        try:
            updates = _get_updates(offset)
        except Exception as exc:
            util.log(f"getUpdates failed: {exc}")
            time.sleep(3)
            continue
        if not updates:
            continue
        for update in updates:
            update_id = int(update.get("update_id", 0))
            if update_id >= offset:
                offset = update_id + 1
            query = update.get("callback_query")
            if query is not None:
                if _is_allowed(query.get("from", {}).get("id")):
                    _handle_callback(query)
                continue
            msg = update.get("message") or {}
            chat_id = str(msg.get("chat", {}).get("id", "") or "")
            if chat_id and _is_allowed(chat_id):
                _handle_text(chat_id, msg.get("text") or "")
        util.save_json(OFFSET_FILE, {"offset": offset, "t": util.now_utc()})


if __name__ == "__main__":
    sys.exit(main())