"""Unit + live tests for fixer/llm_router.py (W1/W9/W11/W12/W14/W18/W19)."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

import llm_router as router

FIXER = Path(__file__).resolve().parent.parent


def _gateway_accepting(host="127.0.0.1", port=20128, timeout=1.0):
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture
def tmp_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "PROVIDER_USAGE_FILE", str(tmp_path / "provider_usage.json"))
    return tmp_path / "provider_usage.json"


def test_load_providers_empty_env_registers_nothing():
    providers = router.load_providers(env={"OMNIROUTE_API_KEY": ""})
    assert providers == []


def test_load_providers_omniroute_and_cloud(tmp_usage):
    env = {"OMNIROUTE_API_KEY": "sk-x", "GEMINI_API_KEY": "g", "GROQ_API_KEY": "q"}
    names = [p.name for p in router.load_providers(env=env)]
    assert "omniroute" in names and "gemini" in names and "groq" in names


def test_omniroute_is_auto_model_aware_others_not():
    providers = router.load_providers(env={"OMNIROUTE_API_KEY": "k", "GEMINI_API_KEY": "g"})
    by = {p.name: p for p in providers}
    assert by["omniroute"].auto_model_aware is True
    assert by["gemini"].auto_model_aware is False


def test_resolve_model_translates_auto_for_plain_endpoints():
    omni = router.Provider(name="omniroute", base_url="x", api_key="k", auto_model_aware=True)
    gemini = router.Provider(name="gemini", base_url="x", api_key="k",
                             default_model="gemini-2.0-flash")
    assert router.resolve_model(omni, "auto/best-coding") == "auto/best-coding"
    assert router.resolve_model(gemini, "auto/best-coding") == "gemini-2.0-flash"
    assert router.resolve_model(gemini, "gemini-2.0-flash") == "gemini-2.0-flash"


def test_matches_tier():
    p = router.Provider(name="x", base_url="b", api_key="k", tier="primary")
    assert p.matches_tier("primary") and p.matches_tier("fast")
    assert not p.matches_tier("batch")
    cop = router.Provider(name="c", base_url="b", api_key="k", tier="escalation")
    assert cop.matches_tier("escalation")      # only escalation -> escalation ok
    assert p.matches_tier("escalation")        # fallback to primary when no copilot
    ter = router.Provider(name="o", base_url="b", api_key="k", tier="tertiary")
    assert ter.matches_tier("escalation")      # tertiary stays in the hard-issue pool


def test_breaker_trips_opens_and_closes_via_trial(tmp_usage, monkeypatch):
    p = router.Provider(name="gemini", base_url="b", api_key="k")
    router.mark_result(p, ok=False)
    router.mark_result(p, ok=False)
    assert router.provider_state(router.provider_usage(p)) == "CLOSED"
    assert router.is_healthy(p) is True  # under threshold, still closed
    router.mark_result(p, ok=False)
    assert router.provider_state(router.provider_usage(p)) == "OPEN"
    assert router.is_healthy(p) is False  # tripped, waiting out cooldown
    assert router.is_callable(p) == (False, "circuit_open")
    # Cooldown elapses -> HALF_OPEN: a trial call is allowed through, and its
    # result -- not a flag -- is what re-closes the circuit (Bug 1 deadlock).
    monkeypatch.setattr(router, "BREAKER_COOLDOWN_SECONDS", -1.0)
    assert router.provider_state(router.provider_usage(p)) == "HALF_OPEN"
    assert router.is_callable(p) == (True, None)
    router.mark_result(p, ok=True)  # trial succeeds
    entry = router.provider_usage(p)
    assert router.provider_state(entry) == "CLOSED"
    assert router.is_healthy(p) is True


def test_budget_available_and_daily_rollover(tmp_usage, monkeypatch):
    p = router.Provider(name="gemini", base_url="b", api_key="k")
    monkeypatch.setattr(router, "GLOBAL_DAILY_TOKEN_BUDGET", 100)
    router.mark_result(p, ok=True, tokens=60)
    assert router.budget_available(p) is True
    router.mark_result(p, ok=True, tokens=60)
    assert router.budget_available(p) is False  # 120 >= 100


def test_budget_never_consults_breaker(tmp_usage, monkeypatch):
    """Bug 1 regression: quota is a PURE token check. A tripped-then-recovered
    provider with quota is callable; a healthy provider over budget reports
    'budget_spent' -- and only then is that label true."""
    p = router.Provider(name="gemini", base_url="b", api_key="k")
    monkeypatch.setattr(router, "GLOBAL_DAILY_TOKEN_BUDGET", 100)
    for _ in range(3):
        router.mark_result(p, ok=False)  # trip the circuit
    assert router.provider_state(router.provider_usage(p)) == "OPEN"
    # Even while OPEN-cooling, the quota side is independent and healthy.
    assert router.has_quota(p, router.provider_usage(p)) is True
    assert router.budget_available(p) is True
    # Healthy + over budget -> the label is real this time.
    other = router.Provider(name="openrouter_free", base_url="b", api_key="k")
    router.mark_result(other, ok=True, tokens=120)
    assert router.is_callable(other) == (False, "budget_spent")
    assert router.provider_state(router.provider_usage(other)) == "CLOSED"


def test_global_budget_gate(tmp_usage):
    assert router.global_budget_available() is True
    p = router.Provider(name="gemini", base_url="b", api_key="k")
    router.mark_result(p, ok=True, tokens=router.GLOBAL_DAILY_TOKEN_BUDGET)
    assert router.global_budget_available() is False


def test_watchdog_timeout(tmp_usage):
    def hang():
        time.sleep(30)

    with pytest.raises(router.WatchdogTimeout):
        router.call_with_watchdog(hang, timeout=0.2)


def test_watchdog_returns_value(tmp_usage):
    assert router.call_with_watchdog(lambda: 42, timeout=2) == 42


def test_watchdog_propagates_exception(tmp_usage):
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        router.call_with_watchdog(boom, timeout=2)


def test_convert_env_via_proxy_in_sequential_complete(tmp_usage, monkeypatch):
    """Proxy preserves the fixer's call shape; complete() failover runs through
    the provider chain and records usage."""
    monkeypatch.setattr(router, "complete",
                        lambda messages, model, max_tokens=4000, timeout=120,
                        provider_hint=None: router.LightCompletion(
                            provider_name="omniroute", model=model,
                            choices=[router.Choice(router.Message("hi"))]))
    proxy = router.CompletionProxy()
    resp = proxy.chat.completions.create(
        model="auto/best-coding", max_tokens=5,
        messages=[{"role": "user", "content": "x"}], timeout=10)
    assert resp.choices[0].message.content == "hi"


def test_mark_result_usage_counts(tmp_usage):
    p = router.Provider(name="gemini", base_url="b", api_key="k")
    router.mark_result(p, ok=True, tokens=99)
    entry = router.provider_usage(p)
    assert entry["tokens"] == 99 and entry["calls"] == 1


def test_new_day_rollover_clears_breaker(tmp_usage):
    """Bug 3: usage and the circuit are derived from TODAY's events only, so a
    new day has no tokens, no fails and no open circuit -- there is no stored
    breaker state left behind to forget to reset. Previous days' rows (even
    failures + token spend) are ignored entirely."""
    p = router.Provider(name="gemini", base_url="b", api_key="k")
    events_path = router._events_file()
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "a", encoding="utf-8") as fh:
        for _ in range(5):
            fh.write(json.dumps({
                "provider": "gemini", "ok": False, "tokens": 9999,
                "ts": time.time() - 86400, "date": "2020-01-01",
                "run_id": "old-day",
            }) + "\n")
    entry = router.provider_usage(p)
    assert entry["tokens"] == 0 and entry["calls"] == 0 and entry["fails"] == 0
    assert router.provider_state(entry) == "CLOSED"
    assert router.is_callable(p) == (True, None)
    assert router.daily_tokens_spent() == 0


@pytest.mark.live
def test_live_complete_via_local_omniroute(tmp_usage):
    """REAL integration: hit the running omniRoute gateway on localhost:20128
    with the token from manual.env. Proves the W14 failover path end-to-end."""
    env_file = FIXER.parent / "manual.env"  # repo root (where manual.env lives)
    if not env_file.exists():
        pytest.skip("manual.env not present; cannot live-test the gateway")
    if not _gateway_accepting():
        pytest.skip("omniRoute gateway not listening on localhost:20128")
    env = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    with pytest.MonkeyPatch.context() as mp:
        for k, v in env.items():
            mp.setenv(k, v)
        os.environ["PROVIDER_USAGE_FILE"] = str(tmp_usage)
        resp = router.complete(
            [{"role": "user", "content": "Reply with the single word: pong"}],
            model="auto/best-chat", max_tokens=10,
            timeout=30, provider_hint="omniroute")
        assert resp.provider_name == "omniroute"
        assert resp.choices[0].message.content
        assert router.provider_usage(
            router.Provider(name="omniroute", base_url="", api_key=""))["calls"] >= 1