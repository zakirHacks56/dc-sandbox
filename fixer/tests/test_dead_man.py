"""Tests for beacon/dead_man.py tick-heartbeat watchdog (no network)."""
import datetime
import sys
import json
from pathlib import Path

BEACON = Path(__file__).resolve().parent.parent.parent / "beacon"


def _deadman(monkeypatch, tmp_path, rows):
    sys.path.insert(0, str(BEACON))
    import dead_man
    import beacon_util as util
    monkeypatch.setattr(util, "DATA", tmp_path)
    lines = "\n".join(json.dumps(r) for r in rows) + "\n"
    (tmp_path / "metrics.jsonl").write_text(lines, encoding="utf-8")
    sent = []
    monkeypatch.setattr(util, "tg_send", lambda text: sent.append(text) or True)
    return dead_man, util, sent


def _tick(ts, outcome):
    return {"event": "tick", "ts": ts, "outcome": outcome}


def test_no_metrics_row_is_dead(tmp_path, monkeypatch):
    dead_man, util, sent = _deadman(monkeypatch, tmp_path, [])
    assert dead_man.main() == 1
    assert sent, "dead-man must alert when no heartbeat exists"


def test_fresh_healthy_tick_ok(tmp_path, monkeypatch):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # current row fresh; last one-hour-ago no-candidate = not a bad streak
    rows = [_tick(now, "no_candidate"),
            _tick((datetime.datetime.now(datetime.timezone.utc)
                   - datetime.timedelta(minutes=10)).isoformat(), "no_candidate")]
    dead_man, util, sent = _deadman(monkeypatch, tmp_path, rows)
    assert dead_man.main() == 0
    assert not sent


def test_stale_heartbeat_alerts(tmp_path, monkeypatch):
    old = (datetime.datetime.now(datetime.timezone.utc)
           - datetime.timedelta(minutes=90)).isoformat()
    rows = [_tick(old, "hunted")]
    dead_man, util, sent = _deadman(monkeypatch, tmp_path, rows)
    assert dead_man.main() == 1
    assert sent and "no tick heartbeat" in sent[0]


def test_gh_dead_streak_alerts(tmp_path, monkeypatch):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows = [_tick(now, "no_candidate")] + [_tick(now, "gh_dead")] * 3
    dead_man, util, sent = _deadman(monkeypatch, tmp_path, rows)
    assert dead_man.main() == 1
    assert sent and "consecutive infra no-op" in sent[0]


def test_short_streak_is_ok(tmp_path, monkeypatch):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows = [_tick(now, "gh_dead")] * 2
    dead_man, util, sent = _deadman(monkeypatch, tmp_path, rows)
    assert dead_man.main() == 0
    assert not sent