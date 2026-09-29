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