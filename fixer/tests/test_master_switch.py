"""Tests for the Telegram master switch (Start/Stop stop-the-tokens button).

Covers:
  * beacon_util.master_switch()/set_master_switch() defaults and persistence
  * gate_poller Start/Stop callback handling (no network -- _tg_call stubbed)
  * tick.py hard no-op when the switch is OFF (no GitHub/fixer/token work)

The control-card Telegram calls are stubbed via monkeypatch, so these tests
never touch the network or Telegram API.
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

BEACON = Path(__file__).resolve().parent.parent.parent / "beacon"
sys.path.insert(0, str(BEACON))


def _beacon_util(monkeypatch, tmp_path):
    import beacon_util as util
    monkeypatch.setattr(util, "DATA", tmp_path)
    return util


@pytest.mark.parametrize("missing", [True, False])
def test_master_switch_defaults_to_enabled(monkeypatch, tmp_path, missing):
    util = _beacon_util(monkeypatch, tmp_path)
    offset = tmp_path / "offset.json"
    if not missing:
        offset.write_text(json.dumps({"offset": 1}), encoding="utf-8")
    assert util.master_switch() is True


def test_master_switch_reads_written_state(monkeypatch, tmp_path):
    util = _beacon_util(monkeypatch, tmp_path)
    (tmp_path / "offset.json").write_text(
        json.dumps({"offset": 1, "master_switch": {"enabled": False}}),
        encoding="utf-8")
    assert util.master_switch() is False


def test_set_master_switch_persists(monkeypatch, tmp_path):
    util = _beacon_util(monkeypatch, tmp_path)
    (tmp_path / "offset.json").write_text(json.dumps({"offset": 42}), encoding="utf-8")
    util.set_master_switch(False, by="telegram")
    assert util.master_switch() is False
    doc = json.loads((tmp_path / "offset.json").read_text(encoding="utf-8"))
    assert doc["master_switch"]["enabled"] is False
    assert doc["master_switch"]["by"] == "telegram"
    util.set_master_switch(True, by="telegram")
    assert util.master_switch() is True
    # The poll offset survives a switch write (merge, not clobber)
    doc = json.loads((tmp_path / "offset.json").read_text(encoding="utf-8"))
    assert doc["offset"] == 42


def test_master_switch_bad_file_defaults_enabled(monkeypatch, tmp_path):
    util = _beacon_util(monkeypatch, tmp_path)
    (tmp_path / "offset.json").write_text("not json", encoding="utf-8")
    assert util.master_switch() is True


def _gate_poller(monkeypatch, tmp_path):
    import beacon_util as util
    import gate_poller
    monkeypatch.setattr(util, "DATA", tmp_path)
    calls = {"tg": [], "answered": [], "notify": []}
    monkeypatch.setattr(
        gate_poller, "_tg_call",
        lambda method, payload, timeout=30: calls["tg"].append((method, payload)) or {
            "ok": True, "result": {"message_id": 12345,
                                     "chat": {"id": payload.get("chat_id", "")}},
        },
    )
    monkeypatch.setattr(gate_poller, "_answer_callback",
                        lambda cid, text: calls["answered"].append((cid, text)))
    monkeypatch.setattr(gate_poller, "_notify",
                        lambda text: calls["notify"].append(text))
    return gate_poller, util, calls


def test_gate_poller_start_stop_buttons(monkeypatch, tmp_path):
    gate_poller, util, calls = _gate_poller(monkeypatch, tmp_path)
    os.environ["TELEGRAM_CHAT_ID"] = "-100123"

    # Stop tap flips the switch OFF
    query_stop = {"id": "q1", "data": "master:off"}
    assert gate_poller._handle_callback(query_stop) is True
    assert util.master_switch() is False
    assert ("q1", "Controller PAUSED") in calls["answered"]
    assert any("Frozen" in t for t in calls["notify"])

    # Start tap flips it back ON
    query_start = {"id": "q2", "data": "master:on"}
    assert gate_poller._handle_callback(query_start) is True
    assert util.master_switch() is True
    assert any("Resuming" in t for t in calls["notify"])

    # Unrelated callbacks are not consumed by the master handler
    assert gate_poller._handle_master({"id": "q3", "data": "decree:x:1"}) is False


def test_gate_poller_posts_control_card_once(monkeypatch, tmp_path):
    gate_poller, util, calls = _gate_poller(monkeypatch, tmp_path)
    os.environ["TELEGRAM_CHAT_ID"] = "-100123"
    gate_poller._ensure_control_card()
    sends = [p for m, p in calls["tg"] if m == "sendMessage"]
    assert len(sends) == 1
    assert "master:on" in json.dumps(sends[0])
    card = util.load_control_card()
    assert card.get("message_id") == 12345
    # A second, unchanged poll must NOT touch Telegram again (no edit, no post).
    before = len(calls["tg"])
    gate_poller._ensure_control_card()
    assert len(calls["tg"]) == before, "idle poll must skip the no-op card edit"
    assert util.load_control_card().get("message_id") == 12345


def test_control_card_not_modified_keeps_same_card(monkeypatch, tmp_path):
    """Telegram's 'message is not modified' (state changed, text raced) must not
    be mistaken for a purged card: same message id is kept, no repost."""
    import beacon_util as util
    import gate_poller
    monkeypatch.setattr(util, "DATA", tmp_path)
    calls = []

    def _stub(method, payload, timeout=30):
        calls.append((method, payload))
        if method == "sendMessage":
            return {"ok": True, "result": {"message_id": 12345}}
        if method == "editMessageText":
            return {"ok": False, "description": "Bad Request: message is not modified"}
        return {"ok": True}

    monkeypatch.setattr(gate_poller, "_tg_call", _stub)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    gate_poller._ensure_control_card()  # posts the fresh card (RUNNING)
    assert util.load_control_card().get("message_id") == 12345
    util.set_master_switch(False, by="test")  # rendered text becomes PAUSED
    gate_poller._ensure_control_card()
    posts = [m for m, _ in calls if m == "sendMessage"]
    edits = [m for m, _ in calls if m == "editMessageText"]
    assert len(posts) == 1, "must not repost a card that still exists"
    assert len(edits) == 1, "edit attempted for the changed text"
    card = util.load_control_card()
    assert card.get("message_id") == 12345, "same card kept on not-modified"
    assert "PAUSED" in card.get("text", ""), "adopted the synced text"
    n = len(calls)
    gate_poller._ensure_control_card()
    assert len(calls) == n, "idle once the adopted text matches"


def test_tick_paused_runs_no_work(monkeypatch, tmp_path):
    import beacon_util as util
    import tick
    monkeypatch.setattr(util, "DATA", tmp_path)
    monkeypatch.setattr(util, "BOARD", tmp_path / "board.json")
    monkeypatch.setattr(util, "GATES", tmp_path / "gates")
    # Pause it, then make sure the tick does not call the fixer or GitHub.
    util.set_master_switch(False, by="telegram")
    calls = []
    monkeypatch.setattr(tick, "_run_fixer", lambda *a, **k: calls.append(a) or "ok")
    monkeypatch.setattr(tick, "hunter_stage",
                        type("h", (), {"find_candidate": lambda *a, **k: None}))
    rc = tick.main()
    assert rc == 0
    assert calls == [], "paused tick must not launch the fixer"
    board = json.loads((tmp_path / "board.json").read_text(encoding="utf-8"))
    assert board.get("last_tick"), "paused tick still stamps a heartbeat"
    rows = [json.loads(l) for l in
            (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows and rows[-1]["event"] == "tick" and rows[-1]["outcome"] == "paused"


def _seed_status_data(tmp_path, monkeypatch):
    """Point beacon_util.DATA/CONFIG at a tiny fake dataset for the status."""
    import beacon_util as util
    fake_data = tmp_path / "data"
    fake_data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(util, "DATA", fake_data)
    real_config = util.CONFIG
    monkeypatch.setattr(util, "CONFIG", real_config)
    (fake_data / "metrics.jsonl").write_text(
        "\n".join([
            '{"ts": "2026-09-29T01:00:00+00:00", "repo": "a/b", "issue": 1, '
            '"outcome": "error:RuntimeError", "tokens_spent": 5000}',
            '{"ts": "2026-09-29T02:00:00+00:00", "repo": "a/b", "issue": 2, '
            '"outcome": "error:RuntimeError", "tokens_spent": 7000}',
            '{"ts": "2026-09-29T03:00:00+00:00", "repo": "c/d", "issue": 3, '
            '"outcome": "solved", "tokens_spent": 3000}',
            '{"event": "tick", "ts": "2026-09-29T04:00:00+00:00", "outcome": "hunted"}',
        ]) + "\n", encoding="utf-8")
    (fake_data / "board.json").write_text(json.dumps({
        "date": "2026-09-29", "prs_today": 1, "prs": [],
        "lanes": {"a/b#1": {"state": "GATED"}, "c/d#3": {"state": "COMPLETED"}},
        "last_tick": "2026-09-29T04:00:00+00:00",
    }), encoding="utf-8")
    return util


def test_build_status_sums_metrics_and_budget(monkeypatch, tmp_path):
    import status as status_mod
    util = _seed_status_data(tmp_path, monkeypatch)
    # Point the status module at the tmp data dir too.
    monkeypatch.setattr(status_mod.util, "DATA", util.DATA)
    monkeypatch.setattr(status_mod.util, "BOARD", util.DATA / "board.json")
    monkeypatch.setattr(status_mod.util, "GATES", util.DATA / "gates")
    text = status_mod.build_status()
    assert "Issues encountered: 3" in text
    assert "Solved: 1" in text
    assert "Tokens burned: 15,000" in text
    assert "PRs 1/2" in text
    assert "RUNNING" in text or "PAUSED" in text


def test_status_button_sends_summary(monkeypatch, tmp_path):
    util = _seed_status_data(tmp_path, monkeypatch)
    gate_poller, _, calls = _gate_poller(monkeypatch, tmp_path)
    os.environ["TELEGRAM_CHAT_ID"] = "-100123"
    query = {"id": "qs", "data": "master:status"}
    assert gate_poller._handle_callback(query) is True
    assert any("Issues encountered" in t for t in calls["notify"]), \
        "Status tap must send the summary to the chat"


def test_poll_lease_skips_getUpdates_when_held(monkeypatch, tmp_path):
    """A lease fresh enough must keep a second poller from long-polling the
    same bot (the two workflows both run gate_poller.py)."""
    import beacon_util as util
    import gate_poller
    monkeypatch.setattr(util, "DATA", tmp_path)
    monkeypatch.setattr(gate_poller, "OFFSET", tmp_path / "offset.json")
    (tmp_path / "offset.json").write_text(json.dumps({
        "offset": 5, "poll_lease": time.time() - 10,
    }), encoding="utf-8")
    updates_called = {"n": 0}
    monkeypatch.setattr(gate_poller, "_get_updates",
                        lambda offset: updates_called.update(n=updates_called["n"] + 1) or [])
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    rc = gate_poller.main()
    assert rc == 0
    assert updates_called["n"] == 0, "held lease must skip getUpdates"
    doc = json.loads((tmp_path / "offset.json").read_text(encoding="utf-8"))
    assert doc["offset"] == 5, "offset preserved while skipping"
    assert doc["poll_lease"], "lease persisted"


def test_poll_lease_taken_when_stale(monkeypatch, tmp_path):
    """An expired lease lets the poller long-poll and stamps a new lease."""
    import beacon_util as util
    import gate_poller
    monkeypatch.setattr(util, "DATA", tmp_path)
    monkeypatch.setattr(gate_poller, "OFFSET", tmp_path / "offset.json")
    (tmp_path / "offset.json").write_text(json.dumps({
        "offset": 0, "poll_lease": time.time() - 1000,
    }), encoding="utf-8")
    monkeypatch.setattr(gate_poller, "_get_updates", lambda offset: [])
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    rc = gate_poller.main()
    assert rc == 0
    doc = json.loads((tmp_path / "offset.json").read_text(encoding="utf-8"))
    assert doc["poll_lease"], "stale lease replaced"


def test_tg_call_prefers_manual_bot_token(monkeypatch):
    """gate_poller must poll the SAME bot the fixer notifies on (the fixer
    prefers MANUAL_BOT_TOKEN), or local taps would be invisible to it."""
    import gate_poller
    import urllib.request

    class _Resp:
        def __init__(self, url=None):
            self._url = url

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    urls = []
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=30: urls.append(req.full_url) or _Resp(req.full_url))
    monkeypatch.setenv("MANUAL_BOT_TOKEN", "manual:123")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram:456")
    assert gate_poller._tg_call("getMe", {}) is not None
    assert any("manual:123" in u for u in urls), "must use MANUAL_BOT_TOKEN first"


def test_tg_call_falls_back_to_telegram_token(monkeypatch):
    """Without MANUAL_BOT_TOKEN (the cloud default) the poller uses the
    TELEGRAM_BOT_TOKEN the workflows set."""
    import gate_poller
    import urllib.request

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    urls = []
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=30: urls.append(req.full_url) or _Resp())
    monkeypatch.delenv("MANUAL_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram:456")
    assert gate_poller._tg_call("getMe", {}) is not None
    assert any("telegram:456" in u for u in urls), "must fall back to TELEGRAM_BOT_TOKEN"


def test_target_flags_reads_require_approval(tmp_path):
    import beacon_util as util
    conf = {
        "targets": [
            {"repo": "a/b", "enabled": True, "require_approval": True},
            {"repo": "c/d", "enabled": True},
        ]
    }
    from tick import _target_flags
    assert _target_flags(conf, "a/b").get("require_approval") is True
    assert _target_flags(conf, "c/d").get("require_approval") is None
    assert _target_flags(conf, "missing/x") == {}


def test_offset_regression_skips_redelivered_but_still_gets_newest(
        monkeypatch, tmp_path):
    """When offset.json is clobbered back to 0 but confirmed_offset records the
    last seen update, a re-delivered OLD tap must be ignored while the newest
    (newer than confirmed) tap is still handled -- the button must not appear
    dead behind a replayed backlog."""
    import beacon_util as util
    import gate_poller
    monkeypatch.setattr(util, "DATA", tmp_path)
    monkeypatch.setattr(gate_poller, "OFFSET", tmp_path / "offset.json")
    (tmp_path / "offset.json").write_text(json.dumps({
        "offset": 0, "confirmed_offset": 50,  # clobbered by a bad merge
    }), encoding="utf-8")
    old = {"update_id": 40, "callback_query": {"id": "qold", "data": "master:off"}}
    new = {"update_id": 60, "callback_query": {"id": "qnew", "data": "master:on"}}
    monkeypatch.setattr(gate_poller, "_get_updates", lambda offset: [old, new])
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    rc = gate_poller.main()
    assert rc == 0
    # The OLD re-delivered Stop tap (uid 40 <= 50) never reaches the handlers,
    # so the switch stays whatever it was; the NEW Start tap (uid 60) lands.
    doc = json.loads((tmp_path / "offset.json").read_text(encoding="utf-8"))
    assert doc["offset"] == 61, "offset advances past the newest handled update"
    assert doc["confirmed_offset"] >= 61
    assert util.master_switch() is True
    assert doc.get("master_seen", 0) >= 60


def test_stale_master_toggle_not_reapplied(monkeypatch, tmp_path):
    """A master switch tap with update_id <= master_seen (an old tap replayed
    after an offset reset) must not re-toggle the switch or re-notify."""
    import beacon_util as util
    import gate_poller
    monkeypatch.setattr(util, "DATA", tmp_path)
    monkeypatch.setattr(gate_poller, "OFFSET", tmp_path / "offset.json")
    (tmp_path / "offset.json").write_text(json.dumps({
        "offset": 60, "confirmed_offset": 60, "master_seen": 70,
        "master_switch": {"enabled": True},
    }), encoding="utf-8")
    ans = []
    monkeypatch.setattr(gate_poller, "_answer_callback",
                        lambda cid, text: ans.append((cid, text)))
    # A re-delivered old Stop tap (uid 65 < master_seen 70, but > offset 60).
    monkeypatch.setattr(gate_poller, "_get_updates",
                        lambda offset: [{"update_id": 65,
                                         "callback_query": {"id": "qstale",
                                                            "data": "master:off"}}])
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    gate_poller.main()
    assert util.master_switch() is True, "stale Stop must not pause the controller"
    assert ans and ans[0][1] == "Already applied"


def test_redelivered_decree_with_outcome_is_skipped(monkeypatch, tmp_path):
    """A decree button re-delivered after an offset reset must not re-write a
    decree for a gate the fixer already decided (outcome exists)."""
    import beacon_util as util
    import gate_poller
    monkeypatch.setattr(util, "DATA", tmp_path)
    monkeypatch.setattr(util, "GATES", tmp_path / "gates")
    (tmp_path / "gates").mkdir(parents=True, exist_ok=True)
    (tmp_path / "gates" / "a-b_issue42_human.outcome").write_text(
        json.dumps({"decision": True}), encoding="utf-8")
    ans = []
    monkeypatch.setattr(gate_poller, "_answer_callback",
                        lambda cid, text: ans.append((cid, text)))
    query = {"id": "q", "data": "decree:a-b_issue42_human:1"}
    assert gate_poller._handle_callback(query) is True
    assert not (tmp_path / "gates" / "a-b_issue42_human.decree").exists(), \
        "already-decided gate must not get a second decree"
    assert ans and ans[0][1] == "Already decided"


def test_main_persist_does_not_drop_control_card(monkeypatch, tmp_path):
    """main() must merge into a FRESH offset.json at persist time, or the card
    / switch written by _ensure_control_card/_handle_master mid-run gets
    silently clobbered by the stale snapshot loaded at the top."""
    import beacon_util as util
    import gate_poller
    monkeypatch.setattr(util, "DATA", tmp_path)
    monkeypatch.setattr(gate_poller, "OFFSET", tmp_path / "offset.json")
    (tmp_path / "offset.json").write_text(json.dumps({
        "offset": 0, "confirmed_offset": 10,
    }), encoding="utf-8")
    handled = []
    monkeypatch.setattr(gate_poller, "_tg_call", lambda method, payload, timeout=30: {
        "ok": True, "result": {"message_id": 12345}})
    monkeypatch.setattr(gate_poller, "_get_updates",
                        lambda offset: handled.append(offset) or [])
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    gate_poller.main()
    doc = json.loads((tmp_path / "offset.json").read_text(encoding="utf-8"))
    assert doc["control_card"]["message_id"] == 12345, \
        "card saved mid-run must survive the persist merge"


def _paged(updates):
    """A Telegram-like getUpdates stub: only returns updates newer than the
    offset, 20 at a time, so a backlog drains across several passes."""
    def _get(offset):
        return [u for u in updates if u["update_id"] > offset][:20]
    return _get


def test_stress_replay_flood_applies_only_new(monkeypatch, tmp_path):
    """Hammer the poller with a full 24h-style replay (offset regressed to 0):
    100 already-decided decrees, 50 stale Stop taps, a handful of genuinely NEW
    apples -- only the new input must have any effect."""
    import beacon_util as util
    import gate_poller
    monkeypatch.setattr(util, "DATA", tmp_path)
    monkeypatch.setattr(util, "GATES", tmp_path / "gates")
    monkeypatch.setattr(gate_poller, "OFFSET", tmp_path / "offset.json")
    (tmp_path / "gates").mkdir(parents=True, exist_ok=True)
    updates = []
    uid = 1000
    # 100 replayed Approve buttons for gates the fixer already decided.
    for i in range(100):
        key = f"a-b_issue{i}_human"
        (tmp_path / "gates" / f"{key}.outcome").write_text(
            '{"decision": true}', encoding="utf-8")
        updates.append({"update_id": uid,
                        "callback_query": {"id": f"old-d{i}", "data": f"decree:{key}:1"}})
        uid += 1
    # 50 stale Stop taps, all older than the master_seen watermark.
    for i in range(50):
        updates.append({"update_id": uid,
                        "callback_query": {"id": f"old-m{i}", "data": "master:off"}})
        uid += 1
    # A genuinely NEW gate (no outcome yet) and a fresh Status tap.
    updates.append({"update_id": uid,
                    "callback_query": {"id": "new-d", "data": "decree:c-d_issue9_human:1"}})
    uid += 1
    updates.append({"update_id": uid,
                    "callback_query": {"id": "new-s", "data": "master:status"}})
    uid += 1
    (tmp_path / "offset.json").write_text(json.dumps({
        "offset": 0, "confirmed_offset": 995, "master_seen": 1149,
    }), encoding="utf-8")
    monkeypatch.setattr(gate_poller, "_get_updates", _paged(updates))
    monkeypatch.setattr(gate_poller, "_answer_callback", lambda cid, text: None)
    monkeypatch.setattr(gate_poller, "_notify", lambda text: None)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")  # no card posting during flood
    gate_poller.main()
    # Only the one genuinely new gate got a decree; the 100 replayed ones did
    # not re-write (their .outcome already exists), and stale Stops did nothing.
    decrees = list((tmp_path / "gates").glob("*.decree"))
    assert [d.name for d in decrees] == ["c-d_issue9_human.decree"], \
        f"only the new gate may be re-decided, got {[d.name for d in decrees]}"
    assert util.master_switch() is True, "stale Stop taps must leave the switch on"
    doc = json.loads((tmp_path / "offset.json").read_text(encoding="utf-8"))
    assert doc["offset"] == uid, "offset advances to the very newest handled update"
    assert doc["confirmed_offset"] == uid


def test_stress_stale_persist_never_regresses_offset(monkeypatch, tmp_path):
    """A poller that loaded a stale low offset must never write it back over a
    higher value another (concurrent) poller just committed."""
    import beacon_util as util
    import gate_poller
    monkeypatch.setattr(util, "DATA", tmp_path)
    monkeypatch.setattr(gate_poller, "OFFSET", tmp_path / "offset.json")
    monkeypatch.setattr(gate_poller, "_get_updates", lambda offset: [])
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    # We loaded a stale copy with offset 5, but by persist time the committed
    # file has already advanced to 999 (another poller won the merge).
    (tmp_path / "offset.json").write_text(json.dumps({
        "offset": 0, "confirmed_offset": 5,
    }), encoding="utf-8")
    gate_poller.main()
    (tmp_path / "offset.json").write_text(json.dumps({
        "offset": 999, "confirmed_offset": 999,
    }), encoding="utf-8")
    gate_poller.main()
    doc = json.loads((tmp_path / "offset.json").read_text(encoding="utf-8"))
    assert doc["offset"] == 999, "stale run must not march the offset backwards"
    assert doc["confirmed_offset"] == 999


def test_stress_concurrent_pollers_keep_offset_monotonic(monkeypatch, tmp_path):
    """N pollers hammering offset.json at once (the two workflows + overlapping
    controller runs): the file must stay valid JSON and the offset must never
    fall below the highest any poller reached."""
    import threading

    import beacon_util as util
    import gate_poller
    monkeypatch.setattr(util, "DATA", tmp_path)
    monkeypatch.setattr(util, "GATES", tmp_path / "gates")
    monkeypatch.setattr(gate_poller, "OFFSET", tmp_path / "offset.json")
    (tmp_path / "gates").mkdir(parents=True, exist_ok=True)
    # A full batch of decrees for gates that are ALREADY decided, so a duplicate
    # concurrent application is a harmless no-op and the test stays deterministic.
    updates = []
    for i in range(20):
        key = f"a-b_issue{i}_human"
        (tmp_path / "gates" / f"{key}.outcome").write_text(
            '{"decision": true}', encoding="utf-8")
        updates.append({"update_id": 100 + i,
                        "callback_query": {"id": f"c{i}", "data": f"decree:{key}:1"}})
    monkeypatch.setattr(gate_poller, "_get_updates", _paged(updates))
    monkeypatch.setattr(gate_poller, "_answer_callback", lambda cid, text: None)
    monkeypatch.setattr(gate_poller, "_notify", lambda text: None)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    (tmp_path / "offset.json").write_text(json.dumps({
        "offset": 95, "confirmed_offset": 95,
    }), encoding="utf-8")
    errs = []

    def _hammer():
        try:
            gate_poller.main()
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)

    threads = [threading.Thread(target=_hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errs, f"concurrent pollers raised: {errs}"
    doc = json.loads((tmp_path / "offset.json").read_text(encoding="utf-8"))
    assert doc["offset"] == 120, f"offset must reach the newest update, got {doc}"
    assert doc["confirmed_offset"] == 120