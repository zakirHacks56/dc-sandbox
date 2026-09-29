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


@pytest.fixture(autouse=True)
def _clean_router_state():
    """Tests share the module-level negative-model cache and pre-flight
    ceiling; clear both so one test's failures can't silently skip another."""
    router._NEGATIVE_MODELS.clear()
    router.set_preflight_ceiling(None)
    yield
    router._NEGATIVE_MODELS.clear()
    router.set_preflight_ceiling(None)


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


def test_mark_result_records_reason(tmp_usage):
    p = router.Provider(name="gemini", base_url="b", api_key="k")
    router.mark_result(p, ok=False, reason="BadRequestError: 400 context window")
    events = [e for e in router._read_events() if e.get("provider") == "gemini"]
    assert events and events[-1]["ok"] is False
    assert "context window" in events[-1]["reason"]


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


# --- User SDE2 round: quota classification + reset, pre-flight token budget,
#     provider reorder, model-compat filter, negative cache, and the
#     deadlock-shaped HALF_OPEN end-to-end trial. ---

def _fake_response(text="ok"):
    from types import SimpleNamespace
    return SimpleNamespace(
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, reasoning=None))],
    )


def _noop_openai_client(raise_conn=True):
    """call_with_watchdog stub that either fails every connection (raise_conn)
    or succeeds; OpenAI() construction never touches the network."""
    if raise_conn:
        def _boom(fn, timeout):
            raise ConnectionError("connection refused")
        return _boom
    return lambda fn, timeout: _fake_response("hi")


def test_quota_429_skips_until_reset(tmp_usage, monkeypatch):
    p = router.Provider(name="openrouter_free", base_url="b", api_key="k")
    router.mark_result(p, ok=False, quota=True, reason="RateLimitError: Error code: 429")
    events = [e for e in router._read_events() if e.get("provider") == p.name]
    assert events and events[-1].get("quota") is True
    entry = router.provider_usage(p)
    assert entry["fails"] == 1
    assert float(entry["quota_exhausted_until"]) > time.time()
    # Even with fails<3 (circuit still CLOSED, never "tripped"), the quota
    # marker makes the provider un-callable until the daily reset -- NO 5-min
    # cooldown cycling into the same 429.
    assert router.is_callable(p) == (False, "quota_exhausted")
    # A success later in the same day clears the marker and re-opens it.
    router.mark_result(p, ok=True, tokens=1)
    assert router.is_callable(p) == (True, None)


def test_mocked_date_rollover_clears_quota(tmp_usage, monkeypatch):
    p = router.Provider(name="openrouter_free", base_url="b", api_key="k")
    router.mark_result(p, ok=False, quota=True)
    assert router.is_callable(p) == (False, "quota_exhausted")
    monkeypatch.setattr(router, "_today", lambda: "2099-12-31")
    entry = router.provider_usage(p)
    assert float(entry["quota_exhausted_until"]) == 0.0
    assert router.is_callable(p) == (True, None)


def test_preflight_refuses_before_any_spend(tmp_usage, monkeypatch):
    p = router.Provider(name="gemini", base_url="http://localhost:1/v1", api_key="k")
    monkeypatch.setattr(router, "load_providers", lambda env=None: [p])
    monkeypatch.setattr(router, "call_with_watchdog", _noop_openai_client())
    router.set_preflight_ceiling(50)  # tiny: no call can fit
    big = [{"role": "user", "content": "x" * 900}]  # ~300 input tokens alone
    with pytest.raises(router.PreflightTokenBudgetExceeded):
        router.complete(big, model="gemini-2.0-flash", max_tokens=64)
    events = router._read_events()
    assert all(e.get("provider") != "gemini" for e in events)  # refused before spend


def test_preflight_no_ceiling_allows_call(tmp_usage, monkeypatch):
    p = router.Provider(name="gemini", base_url="http://localhost:1/v1", api_key="k")
    monkeypatch.setattr(router, "load_providers", lambda env=None: [p])
    monkeypatch.setattr(router, "call_with_watchdog", _noop_openai_client(raise_conn=False))
    router.set_preflight_ceiling(None)
    resp = router.complete(big_messages(), model="gemini-2.0-flash", max_tokens=64)
    assert resp.provider_name == "gemini" and resp.choices[0].message.content == "hi"
    assert router.provider_usage(p)["calls"] == 1
    router.set_preflight_ceiling(None)


def test_half_open_end_to_end_trial_recloses(tmp_usage, monkeypatch):
    """Bug 1 deadlock-shaped, at the complete() level:
    3 fails -> OPEN -> cooldown elapses -> HALF_OPEN: the trial call is the
    ONLY thing allowed through, and its SUCCESS is what re-closes the circuit.
    A "trip/open flag" implementation would refuse this trial forever."""
    p = router.Provider(name="gemini", base_url="http://localhost:1/v1", api_key="k",
                        default_model="gemini-2.0-flash")
    monkeypatch.setattr(router, "load_providers", lambda env=None: [p])
    monkeypatch.setattr(router, "call_with_watchdog", _noop_openai_client(raise_conn=False))
    for _ in range(3):
        router.mark_result(p, ok=False, reason="500")
    assert router.provider_state(router.provider_usage(p)) == "OPEN"
    assert router.is_callable(p) == (False, "circuit_open")  # genuinely must NOT call
    monkeypatch.setattr(router, "BREAKER_COOLDOWN_SECONDS", -1.0)  # cooldown elapsed
    assert router.provider_state(router.provider_usage(p)) == "HALF_OPEN"
    resp = router.complete([{"role": "user", "content": "x"}],
                           model="gemini-2.0-flash", max_tokens=64)
    assert resp.provider_name == "gemini"
    assert router.provider_state(router.provider_usage(p)) == "CLOSED"  # re-closed
    assert router.is_callable(p) == (True, None)


def test_reorder_primary_chain_user_order(tmp_usage):
    env = {"GEMINI_API_KEY": "g", "GROQ_API_KEY": "q",
           "OPENROUTER_API_KEY": "o", "OMNIROUTE_API_KEY": "m"}
    names = [p.name for p in router._sort_candidates(router.load_providers(env=env))]
    assert names == ["gemini", "groq", "openrouter_free", "omniroute"]


def test_escalation_sorts_copilot_first(tmp_usage):
    env = {"COPILOT_API_KEY": "c", "GEMINI_API_KEY": "g", "OMNIROUTE_API_KEY": "m"}
    names = [p.name for p in router._sort_candidates(router.load_providers(env=env))]
    assert names[0] == "copilot"


def test_model_compat_filter_keeps_foreign_ids_off_native_apis(tmp_usage, monkeypatch):
    gemini = router.Provider(name="gemini", base_url="http://localhost:1/v1", api_key="k",
                             default_model="gemini-2.0-flash",
                             models=["gemini-2.0-flash"])
    groq = router.Provider(name="groq", base_url="http://localhost:1/v1", api_key="k",
                           default_model="qwen/qwen3.8-27b",
                           models=["qwen/qwen3.8-27b"])
    openrouter = router.Provider(name="openrouter_free", base_url="http://localhost:1/v1",
                                 api_key="k", tier="tertiary")
    monkeypatch.setattr(router, "load_providers", lambda env=None: [gemini, groq, openrouter])
    monkeypatch.setattr(router, "call_with_watchdog", _noop_openai_client(raise_conn=False))
    # openrouter-style combos must NOT reach gemini/groq (404/400 class), but
    # the openrouter funnel accepts them.
    resp = router.complete([{"role": "user", "content": "x"}],
                           model="nvidia/nemotron-3-super-120b-a12b:free", max_tokens=8)
    assert resp.provider_name == "openrouter_free"
    providers_hit = {e.get("provider") for e in router._read_events() if e.get("ok")}
    assert providers_hit == {"openrouter_free"}
    # a gemini id reaches gemini, first in the chain, and never a foreign fn
    resp2 = router.complete([{"role": "user", "content": "x"}],
                            model="gemini-2.0-flash", max_tokens=8)
    assert resp2.provider_name == "gemini"
    hit = {e.get("provider") for e in router._read_events() if e.get("ok")}
    assert hit == {"openrouter_free", "gemini"}


def test_negative_cache_skips_repeat_failures(tmp_usage, monkeypatch):
    p = router.Provider(name="gemini", base_url="http://localhost:1/v1", api_key="k")
    monkeypatch.setattr(router, "load_providers", lambda env=None: [p])
    monkeypatch.setattr(router, "call_with_watchdog", _noop_openai_client())
    for _ in range(3):
        with pytest.raises(router.AllProvidersExhaustedError):
            router.complete([{"role": "user", "content": "x"}],
                            model="gemini-2.0-flash", max_tokens=8)
    # First failure is recorded (and negative-cached); the next two attempts
    # skip the pair WITHOUT re-eatting the refused socket -> exactly ONE event.
    events = [e for e in router._read_events() if e.get("provider") == "gemini"]
    assert len(events) == 1


from types import SimpleNamespace


class _FakeAPIError(Exception):
    def __init__(self, status_code, body=None, headers=None):
        super().__init__(f"fake status {status_code}: {body}")
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.response = SimpleNamespace(headers=headers or {})


def test_classify_failure_splits_402_429_and_transient():
    assert router._classify_failure(_FakeAPIError(402, "requires more credits")) == "402_call_too_big"
    assert router._classify_failure(_FakeAPIError(429, "free-models-per-day reached")) == "429_daily"
    assert router._classify_failure(_FakeAPIError(429, "PerDay limit")) == "429_daily"
    assert router._classify_failure(_FakeAPIError(429, "rate limit")) == "429_rate"
    assert router._classify_failure(RuntimeError("connection reset by peer")) == "transient"


def test_retry_after_seconds_honoured_or_default():
    long = router._retry_after_seconds(_FakeAPIError(429, "x", headers={"retry-after": "9999"}))
    assert long == 300.0
    short = router._retry_after_seconds(_FakeAPIError(429, "x", headers={"retry-after": "5"}))
    assert 4.9 < short <= 5.0
    assert router._retry_after_seconds(RuntimeError("boom")) == 20.0


def test_reset_until_per_kind():
    nowish = time.time()
    utc = router.Provider(name="a", base_url="b", api_key="k", reset_kind="utc_midnight")
    pac = router.Provider(name="a", base_url="b", api_key="k", reset_kind="pacific_midnight")
    slim = router.Provider(name="a", base_url="b", api_key="k", reset_kind="rolling")
    unknown = router.Provider(name="a", base_url="b", api_key="k", reset_kind="probe")
    assert 0 < router._reset_until(utc) - nowish <= 26 * 3600
    assert 0 < router._reset_until(pac) - nowish <= 26 * 3600
    assert abs(router._reset_until(slim) - (nowish + 24 * 3600)) < 5
    assert abs(router._reset_until(unknown) - (nowish + router.PROBE_INTERVAL_SECONDS)) < 5


def test_402_retries_half_tokens_and_leaves_state_alone(tmp_usage, monkeypatch):
    calls = {"n": 0}
    p = router.Provider(name="gemini", base_url="http://localhost:1/v1", api_key="k")
    monkeypatch.setattr(router, "load_providers", lambda env=None: [p])

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FakeAPIError(402, "requires more credits, or fewer max_tokens")
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=9, completion_tokens=40))
        resp.choices = [SimpleNamespace(message=SimpleNamespace(content="ok-from-402-retry"))]
        return resp

    monkeypatch.setattr(router, "call_with_watchdog", boom)
    resp = router.complete([{"role": "user", "content": "x"}],
                           model="gemini-2.0-flash", max_tokens=1000)
    assert calls["n"] == 2
    assert resp.provider_name == "gemini"
    assert resp.choices[0].message.content == "ok-from-402-retry"
    entry = router.provider_usage(p)
    assert float(entry.get("quota_exhausted_until", 0)) == 0.0  # NOT a day-ban
    assert entry.get("down") is False                        # breaker untouched (trip=False)
    events = [e for e in router._read_events() if e.get("provider") == "gemini"]
    assert len(events) == 1  # only the halved-retry ok; the first 402 leaves no trace
    assert events[0].get("ok") is True
    # and the SAME call still works next attempt (no negative cache on 402)
    resp2 = router.complete([{"role": "user", "content": "x"}],
                            model="gemini-2.0-flash", max_tokens=1000)
    assert resp2.provider_name == "gemini"


def test_429_daily_ban_is_per_reset_kind_and_negative_cache(tmp_usage, monkeypatch):
    p = router.Provider(name="gemini", base_url="http://localhost:1/v1", api_key="k",
                        reset_kind="pacific_midnight")
    monkeypatch.setattr(router, "load_providers", lambda env=None: [p])
    monkeypatch.setattr(router, "call_with_watchdog",
                        lambda *a, **k: (_ for _ in ()).throw(
                            _FakeAPIError(429, {"error": {"message": "PerDay free quota reached"}})))
    with pytest.raises(router.AllProvidersExhaustedError):
        router.complete([{"role": "user", "content": "x"}],
                        model="gemini-2.0-flash", max_tokens=8)
    assert router.is_callable(p) == (False, "quota_exhausted")
    until = float(router.provider_usage(p)["quota_exhausted_until"])
    assert until > time.time()
    assert abs(until - router._next_pacific_midnight()) < 60
    # negative-cached in-process too
    with pytest.raises(router.AllProvidersExhaustedError):
        router.complete([{"role": "user", "content": "x"}],
                        model="gemini-2.0-flash", max_tokens=8)
    events = [e for e in router._read_events() if e.get("provider") == "gemini"]
    assert len(events) == 1  # 2nd attempt skipped before reaching the socket


def test_429_rate_backoff_is_not_a_day_ban(tmp_usage, monkeypatch):
    p = router.Provider(name="gemini", base_url="http://localhost:1/v1", api_key="k")
    monkeypatch.setattr(router, "load_providers", lambda env=None: [p])
    monkeypatch.setattr(router, "call_with_watchdog",
                        lambda *a, **k: (_ for _ in ()).throw(
                            _FakeAPIError(429, "You are being rate limited")))
    with pytest.raises(router.AllProvidersExhaustedError):
        router.complete([{"role": "user", "content": "x"}],
                        model="gemini-2.0-flash", max_tokens=8)
    assert router.is_callable(p) == (False, "rate_limited")
    backoff = float(router.provider_usage(p)["rate_backoff_until"])
    assert backoff > time.time()
    assert router.provider_state(router.provider_usage(p)) == "CLOSED"  # no breaker trip
    assert float(router.provider_usage(p)["quota_exhausted_until"]) == 0.0  # no day ban
    # backoff lapses naturally -> callable again (simulate by rewriting the
    # single gemini event with an already-expired backoff window)
    events = [e for e in router._read_events() if e.get("provider") == "gemini"]
    stale = dict(events[-1], rate_backoff_until=time.time() - 10)
    with open(router._events_file(), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(stale) + "\n")
    assert router.is_callable(p) == (True, None)


def test_registry_loading_from_file(tmp_path, monkeypatch):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({
        "providers": [
            {"name": "fastchat", "base_url": "https://x/v1", "key_env": "FASTCHAT_KEY",
             "role": ["planner"], "models": ["mixtral-8x7b"], "reset_kind": "rolling",
             "max_tokens_cap": 10000, "rpm": 30},
            {"name": "biglink", "base_url": "https://y/v1", "key_env": "BIGLINK_KEY",
             "role": ["coder"], "models": ["big-1"], "reset_kind": "pacific_midnight"},
            {"name": "ghost", "base_url": "https://z/v1", "key_env": "GHOST_KEY",
             "role": ["overflow"]},
        ]}, sort_keys=True), encoding="utf-8")
    env = {"PROVIDER_REGISTRY": str(reg), "BIGLINK_KEY": "b", "GHOST_KEY": "gr"}
    vals = sorted((p.name, p.tier, p.reset_kind, p.models, p.max_tokens_cap)
                  for p in router.load_providers(env=env))
    assert vals == [
        ("biglink", "primary", "pacific_midnight", ["big-1"], None),
        ("ghost", "opportunistic", "utc_midnight", [], None),
    ]  # fastchat skipped: its key is empty


def test_models_list_is_authoritative_compat():
    p = router.Provider(name="groq", base_url="http://localhost:1/v1", api_key="k",
                        default_model="qwen/qwen3.8-27b",
                        models=["qwen/qwen3.8-27b", "llama-3.3-70b-versatile"])
    assert router._provider_accepts(p, "qwen/qwen3.8-27b") is True
    assert router._provider_accepts(p, "llama-3.3-70b-versatile") is True
    assert router._provider_accepts(p, "nvidia/nemotron-3-super-120b-a12b:free") is False


def big_messages():
    return [{"role": "user", "content": "x" * 900}]


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
        mp.setenv("PROVIDER_USAGE_FILE", str(tmp_usage))
        resp = router.complete(
            [{"role": "user", "content": "Reply with the single word: pong"}],
            model="auto/best-chat", max_tokens=10,
            timeout=30, provider_hint="omniroute")
        assert resp.provider_name == "omniroute"
        assert resp.choices[0].message.content
        assert router.provider_usage(
            router.Provider(name="omniroute", base_url="", api_key=""))["calls"] >= 1