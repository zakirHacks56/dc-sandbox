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
        /run <owner/repo> <issue>   solve an issue -> draft PR
        /status                     board + gates + workflows summary
        /gates                      what needs a human decision right now
        /help                       this text
  * Backend per /run: if OmniRoute is up at localhost:20128 => use it.
    Otherwise fall back to OpenRouter (free-tier chain; needs LLM_API_KEY).
  * Command button taps (decree:...) are written as .decree files under
    data/gates/ just like the cloud poller, and a --gate-sync fixer run is
    kicked off so the parked PR advances or closes. GATE_AUTO stays OFF here
    so every draft really does wait for your tap.

stdlib-only on purpose so it runs on a bare box with no pip install.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beacon_util as util  # noqa: E402

BOT_TOKEN = os.getenv("MANUAL_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED = [
    c.strip() for c in os.getenv(
        "TELEGRAM_ALLOWED_IDS", os.getenv("TELEGRAM_CHAT_ID", "")
    ).split(",") if c.strip()
]
OFFSET_FILE = util.DATA / "manual_offset.json"
LOCAL_GATEWAY = os.getenv("OMNIROUTE_LOCAL_URL", "http://localhost:20128")
FALLBACK_BASE = os.getenv("OMNIROUTE_BASE_URL", "https://openrouter.ai")

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
        "dc-sandbox manual bot\n"
        "/run <owner/repo> <issue>  -> solve to a draft PR\n"
        "    (uses local OmniRoute if it is up, else OpenRouter free chain)\n"
        "/status                    -> board + budget + lanes\n"
        "/gates                     -> decisions waiting on a human\n"
        "/help                      -> this text\n"
        "Draft PRs wait for your Approve/Decline button tap."
    )


def _cmd_status() -> str:
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


def _gateway_up() -> str:
    """Return the base URL to use. Prefers a live local OmniRoute gateway,
    else the fallback provider (OpenRouter). Probe is short and non-fatal."""
    probe = f"{LOCAL_GATEWAY}/v1/models"
    try:
        req = urllib.request.Request(probe, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return f"{LOCAL_GATEWAY}/v1"
    except Exception:
        pass
    return FALLBACK_BASE


def _spawn_fixer(repo_name: str, issue_number: int, chat_id: str) -> None:
    base = _gateway_up()
    args = ["--repo", repo_name, "--issue", str(issue_number), "--gate-sync"]
    env = util.env_for_fixer({
        "GATE_AUTO": "0",  # manual mode: always wait for a button tap
        "OMNIROUTE_BASE_URL": base,
        "BEACON_LOG": str(util.DATA / "log-manual.txt"),
    })
    util.log(f"manual /run {repo_name}#{issue_number} via {base}")
    _send(chat_id, f"starting {repo_name}#{issue_number} via {base} ...")
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
        _active["label"] = f"{repo_name}#{issue_number}"
    util.log(f"fixer pid={proc.pid} spawned")
    _send(chat_id, f"fixer pid={proc.pid} started; I will report when it finishes.")


def _run_thread(repo_name: str, issue_number: int, chat_id: str) -> None:
    try:
        _spawn_fixer(repo_name, issue_number, chat_id)
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


# ------------------------------------------------------------ decrees
def _handle_callback(query: dict) -> bool:
    data = query.get("data") or ""
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
            target=_run_thread, args=(repo, int(issue), target_chat), daemon=True
        ).start()
    return True


def _handle_text(chat_id: str, text: str) -> None:
    text = (text or "").strip()
    low = text.lower()
    if low.startswith("/run"):
        m = re.match(r"^/run\s+([\w.-]+/[\w.-]+)\s+(\d+)\b", text)
        if not m:
            _send(chat_id, "usage: /run <owner/repo> <issue>")
            return
        repo_name, issue_number = m.group(1), int(m.group(2))
        threading.Thread(
            target=_run_thread, args=(repo_name, issue_number, chat_id), daemon=True
        ).start()
    elif low == "/status":
        _send(chat_id, _cmd_status())
    elif low == "/gates":
        _send(chat_id, _cmd_gates())
    elif low in ("/stop", "/stopall"):
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