"""Gate poller: pumps the Telegram bot for Approve/Decline button presses and
materialises them as `.decree` files under data/gates/.

The vendored fixer weaves its own pump/decree convention:
  data/gates/<owner>-<repo>_issue<N>_<gate>.pending   -> a gate is parked
  data/gates/<owner>-<repo>_issue<N>_<gate>.decree    -> operator decided
    written by THIS poller when a button is tapped; consumed exactly once by
    the fixer on its next run (which then writes a matching .outcome file).

stdlib-only on purpose: this job's whole reason to exist is the 5-minute
cloud poll, so it must not need `pip install`.
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beacon_util as util  # noqa: E402

OFFSET = util.DATA / "offset.json"

# The fixer prefers the manual (operator-facing) bot when present, so taps the
# operator makes locally must be polled with the SAME token precedence or the
# poller never sees them. Only TELEGRAM_BOT_TOKEN is set in the cloud.
POLL_LEASE_SECONDS = 45  # one long-poller per window avoids Telegram 409.
# 120s made button taps wait almost 10 minutes (it suppresses the *paired*
# cron run of controller.yml, ~2 min later). 45s still prevents two pollers
# overlapping a single getUpdates window (~8-20s) without eating a whole
# schedule interval, so the newest Start/Stop/Status tap is polled sooner.


def _tg_call(method: str, payload: dict, timeout: int = 30):
    token = os.getenv("MANUAL_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return None
    host = util.TG_API_HOST
    body = json.dumps(payload).encode("utf-8")

    def _post(host: str, with_host_header: bool):
        url = f"https://{host}/bot{token}/{method}"
        headers = {"Content-Type": "application/json"}
        if with_host_header:
            headers["Host"] = util.TG_API_HOST
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            blob = resp.read().decode("utf-8")
            try:
                return json.loads(blob)
            except ValueError:
                return None

    for host, header in [(host, False), *( (ip, True) for ip in util.TG_API_IPS )]:
        try:
            data = _post(host, header)
            if data is not None and data.get("ok") is not False:
                return data
        except Exception:
            continue
    return None


def _get_updates(offset: int):
    payload = {"timeout": 8, "offset": offset, "limit": 20}
    data = _tg_call("getUpdates", payload, timeout=20)
    if not data:
        return []
    return data.get("result") or []


def _answer_callback(callback_id: str, text: str) -> None:
    _tg_call("answerCallbackQuery", {"callback_query_id": callback_id, "text": text})


def _notify(text: str) -> None:
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if chat:
        _tg_call("sendMessage", {"chat_id": chat, "text": text})


def _control_markup() -> dict:
    """Inline-keyboard payload for the Start/Stop control card."""
    return {
        "inline_keyboard": [[
            {"text": "Start", "callback_data": "master:on"},
            {"text": "Stop", "callback_data": "master:off"},
            {"text": "Status", "callback_data": "master:status"},
        ]]
    }


def _control_text() -> str:
    running = util.master_switch()
    state = "RUNNING" if running else "PAUSED"
    dot = "🟢" if running else "🔴"
    return (f"OSS controller: {dot} {state}\n\n"
            "Tap Start to let it hunt again, Stop to freeze it "
            "(no more fixer runs / tokens spent while paused).")


def _ensure_control_card() -> None:
    """Post (or re-post) the master-switch card in the chat and keep it
    reflecting the current state. Best-effort: a dead/purged message is
    forgotten so the next poll reposts a fresh card."""
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not chat:
        return
    card = util.load_control_card()
    if chat == card.get("chat_id") and card.get("message_id"):
        payload = {
            "chat_id": chat,
            "message_id": card["message_id"],
            "text": _control_text(),
            "reply_markup": _control_markup(),
        }
        if _tg_call("editMessageText", payload) is None:
            util.log("control card purged -- forgetting it (will repost next poll)")
            util.save_control_card(**{"chat_id": chat, "message_id": None})
        return
    data = _tg_call("sendMessage", {
        "chat_id": chat,
        "text": _control_text(),
        "reply_markup": _control_markup(),
    })
    if data and data.get("ok"):
        msg_id = (data.get("result") or {}).get("message_id")
        util.save_control_card(**{"chat_id": chat, "message_id": msg_id})
        util.log(f"control card posted (message_id={msg_id})")


def _handle_master(query: dict) -> bool:
    """Master-switch Start/Stop/Status button taps. Returns True when consumed."""
    data = query.get("data") or ""
    if data == "master:status":
        try:
            import status as status_mod  # noqa: PLC0415
            util.log("status requested via Telegram")
            _answer_callback(query.get("id", ""), "Here's the status")
            _notify(status_mod.build_status())
        except Exception as exc:  # noqa: BLE001 -- a broken status must not crash the poller
            util.log(f"status failed: {exc}")
            _answer_callback(query.get("id", ""), "Status unavailable")
        return True
    if data not in ("master:on", "master:off"):
        return False
    enabled = data == "master:on"
    util.set_master_switch(enabled, by="telegram")
    label = "RUNNING" if enabled else "PAUSED"
    util.log(f"master switch -> {label} (via Telegram)")
    _answer_callback(query.get("id", ""), f"Controller {label}")
    _ensure_control_card()
    _notify(
        f"OSS controller {label}.\n"
        + ("Resuming work -- I'll hunt again on the next tick."
           if enabled else
           "Frozen -- no fixer runs / tokens while stopped. Tap Start to resume.")
    )
    if enabled:
        try:
            import status as status_mod  # noqa: PLC0415
            _notify(status_mod.build_status())
        except Exception as exc:  # noqa: BLE001
            util.log(f"status on start failed: {exc}")
    return True


def _handle_callback(query: dict) -> bool:
    data = query.get("data") or ""
    if _handle_master(query):
        return True
    match = re.match(r"^decree:(.+):([01])$", data)
    if not match:
        return False
    key, decision = match.group(1), match.group(2) == "1"

    # Idempotency: an update Telegram re-delivers after an offset regression
    # must never re-write a decree for a gate that was already decided and
    # consumed (the fixer wrote a .outcome). Spinner still gets answered.
    if (util.GATES / f"{key}.outcome").exists():
        util.log(f"decree skipped (already decided): {key}")
        _answer_callback(query.get("id", ""), "Already decided")
        return True

    parsed = util.parse_gate_key(key)

    # Prefer the rich metadata the fixer wrote when it parked the gate; the
    # key itself is lossy for repo names containing hyphens.
    pending = util.GATES / f"{key}.pending"
    meta = util.load_json(pending, {}) if pending.exists() else {}
    repo_name = meta.get("repo") or (parsed[0] if parsed else None)
    issue_number = meta.get("issue") if meta.get("issue") is not None else (parsed[1] if parsed else None)
    gate = meta.get("gate") or (parsed[2] if parsed else None)

    decree = util.GATES / f"{key}.decree"
    decree.write_text(
        json.dumps({
            "key": key,
            "decision": decision,
            "repo": repo_name,
            "issue": issue_number,
            "gate": gate,
            "decided_at": util.now_utc(),
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    if pending.exists():
        pending.unlink()
    label = "APPROVE" if decision else "DECLINE"
    target = f"{repo_name}#{issue_number}" if repo_name else key
    util.log(f"decree written: {target} ({gate or 'gate'}) -> {label}")
    _answer_callback(query.get("id", ""), f"Recorded: {label}")
    _notify(f"Gate {target}: {label} recorded.")
    return True


def _is_master_toggle(query: dict) -> bool:
    return (query.get("data") or "").startswith("master:")


def main() -> int:
    doc = util.load_json(OFFSET, {})
    offset = int(doc.get("offset", 0) or 0)
    confirmed = int(doc.get("confirmed_offset", 0) or 0)
    # Never re-poll below an offset we have already confirmed -- the concurrent
    # controller runs commit offset.json and `git pull -X ours` can momentarily
    # clobber it back to 0, which would make Telegram re-deliver the whole
    # 24h queue, starving the newest button tap and re-firing every old decree.
    start = max(offset, confirmed)
    master_seen = int(doc.get("master_seen", 0) or 0)
    offset = start  # never poll below the last confirmed update id
    util.log(f"gate poller start (offset={start})")
    # Only ONE long-poller per window: both the controller and gate-poll
    # workflows run this script, and two concurrent getUpdates on the same bot
    # make Telegram answer callbacks with "409 Conflict: terminated by other
    # getUpdates request" -- taps then appear dead. The lease rides in the same
    # committed offset.json, so both workflows see the most recent winner.
    lease = doc.get("poll_lease", 0) or 0
    held = lease and (time.time() - float(lease)) < POLL_LEASE_SECONDS
    if held:
        util.log("poll lease held by a concurrent poller -- skipping getUpdates")
    handled = 0
    if not held:
        # Drain the WHOLE backlog, not just two passes: after an offset
        # regression the newest tap sits behind every queued update, and the
        # repeat of a same update is skipped further down.
        for _ in range(16):
            updates = _get_updates(offset)
            if not updates:
                break
            got = len(updates)
            for update in updates:
                update_id = int(update.get("update_id", 0))
                if update_id <= start:
                    continue  # already consumed before -- do not re-handle
                offset = max(offset, update_id + 1)
                query = update.get("callback_query")
                if not query:
                    continue
                # A stale re-delivered tap must not re-toggle the master switch
                # (or re-notify) after the first time it was applied.
                if _is_master_toggle(query) and update_id <= master_seen:
                    _answer_callback(query.get("id", ""), "Already applied")
                    continue
                master_seen = max(master_seen, update_id)
                if _handle_callback(query):
                    handled += 1
            if got < 20:
                break
        doc["poll_lease"] = time.time()
        util.log(f"poll lease taken for {POLL_LEASE_SECONDS}s")
    # Keep the Start/Stop card pinned and current (also posts it if Telegram
    # purged it, or for the very first run -- where the card file is empty).
    if os.getenv("TELEGRAM_CHAT_ID"):
        _ensure_control_card()
    if held or doc.get("poll_lease"):
        # Merge, don't clobber: _handle_master/_ensure_control_card each reload
        # offset.json fresh, so writing our stale `doc` snapshot back would
        # silently drop the control_card / master_switch they just saved.
        fresh = util.load_json(OFFSET, doc)
        # Another poller may have advanced the committed offset/seen markers
        # since we loaded; a stale run must NEVER write a smaller value back,
        # or its progress would be silently lost (this is the one-way door that
        # keeps the poll offset monotonic under concurrent controller runs).
        offset = max(offset,
                     int(fresh.get("offset", 0) or 0),
                     int(fresh.get("confirmed_offset", 0) or 0))
        fresh["offset"] = offset
        fresh["confirmed_offset"] = max(int(fresh.get("confirmed_offset", 0) or 0), offset)
        fresh["master_seen"] = max(int(fresh.get("master_seen", 0) or 0), master_seen)
        fresh["poll_lease"] = doc.get("poll_lease", time.time())
        util.save_json(OFFSET, fresh)
    util.log(f"gate poller done: {handled} callback(s), new offset={offset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())