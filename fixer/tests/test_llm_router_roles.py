"""Unit tests for the multi-provider role router + persistent budget windows:
role dispatch (coder/planner/background -> designated provider), overflow
fallthrough (budget-exhausted role provider reroutes to llm7 -> hetzner),
rolling rpm/rps, calendar-day (daily_tokens / daily_tokens_in / daily_neurons)
and calendar-month (monthly_tokens) budget windows, and the Cloudflare
Workers AI adapter translation. EVERY provider call is mocked -- no live calls.
"""
import json
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

import llm_router as router


@pytest.fixture
def tmp_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "PROVIDER_USAGE_FILE", str(tmp_path / "provider_usage.json"))
    return tmp_path / "provider_usage.json"


@pytest.fixture(autouse=True)
def _clean_router_state():
    router._NEGATIVE_MODELS.clear()
    router.set_preflight_ceiling(None)
    yield
    router._NEGATIVE_MODELS.clear()
    router.set_preflight_ceiling(None)


def _prov(name="gemini", base_url="https://example/v1", tier="primary", **kw):
    kw.setdefault("api_key", "k")
    kw.setdefault("roles", [])
    return router.Provider(name=name, base_url=base_url, api_key=kw.pop("api_key"),
                           tier=tier, roles=kw.pop("roles"), **kw)


def _fake_response(text="ok"):
    return SimpleNamespace(
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, reasoning=None))],
    )


# --- registry ---------------------------------------------------------------

def test_registry_registers_role_providers(tmp_usage):
    env = {
        "NVIDIA_API_KEY": "n", "CF_API_TOKEN": "c", "MISTRAL_API_KEY": "m",
        "LLM7_API_KEY": "l", "HETZNER_API_KEY": "h", "AION_LAB_API_KEY": "a",
        "GEMINI_API_KEY": "g", "GROQ_API_KEY": "q", "OPENROUTER_API_KEY": "o",
        "OMNIROUTE_API_KEY": "om", "COPILOT_API_KEY": "cp",
        "COPILOT_BASE_URL": "https://api.githubcopilot.com/",
    }
    providers = router.load_providers(env=env)
    by = {p.name: p for p in providers}
    assert by["nvidia_nim"].roles == ["coder"]
    assert by["nvidia_nim"].rpm == 40
    assert by["nvidia_nim"].base_url == "https://integrate.api.nvidia.com/v1"
    assert by["cloudflare_workers_ai"].adapter == "cloudflare_workers_ai"
    assert by["cloudflare_workers_ai"].roles == ["planner"]
    assert by["cloudflare_workers_ai"].daily_neurons == 10000
    assert by["mistral"].roles == ["background"]
    assert by["mistral"].rps == 1
    assert by["mistral"].monthly_tokens == 1000000000
    assert by["llm7"].roles == ["overflow"]
    assert by["llm7"].daily_tokens == 5000000
    assert by["hetzner"].roles == ["overflow"]
    assert by["hetzner"].daily_tokens_in == 500000000


def test_key_env_falls_back_to_alt(tmp_usage):
    env = {"NVIDIA_NIM_API_KEY": "legacy", "MISTRAL_API_KEY": "m"}
    providers = router.load_providers(env=env)
    nvidia = next((p for p in providers if p.name == "nvidia_nim"), None)
    assert nvidia is not None and nvidia.api_key == "legacy"


def test_cloudflare_registers_without_base_url(tmp_usage):
    env = {"CF_API_TOKEN": "c", "CF_ACCOUNT_ID": "acct"}
    providers = router.load_providers(env=env)
    cf = [p for p in providers if p.name == "cloudflare_workers_ai"]
    assert cf and cf[0].base_url == ""


def test_freellmapi_not_registered_without_url(tmp_usage):
    env = {"FREELLMAPI_API_KEY": "k"}
    providers = router.load_providers(env=env)
    assert all(p.name != "freellmapi" for p in providers)


def test_freellmapi_registers_with_env_url(tmp_usage):
    env = {
        "FREELLMAPI_BASE_URL": "https://router.example/v1",
        "FREELLMAPI_UNIFIED_KEY": "freellmapi-secret",
    }
    providers = router.load_providers(env=env)
    fm = next((p for p in providers if p.name == "freellmapi"), None)
    assert fm is not None
    assert fm.base_url == "https://router.example/v1"
    assert fm.api_key == "freellmapi-secret"
    assert fm.roles == ["overflow"]
    assert fm.default_model == "meta-llama/llama-3.3-70b-instruct"


def test_role_candidates_coder_overflow_ends_with_freellmapi():
    providers = [
        _prov("gemini", roles=["coder"]),
        _prov("nvidia_nim", base_url="https://integrate.api.nvidia.com/v1", roles=["coder"]),
        _prov("llm7", base_url="https://api.llm7.io/v1", roles=["overflow"]),
        _prov("hetzner", base_url="https://inference.hetzner.com/api/v1", roles=["overflow"]),
        _prov("aion_lab", roles=["overflow"]),
        _prov("freellmapi", roles=["overflow"]),
    ]
    names = [p.name for p in router._role_candidates(providers, "coder", "primary")]
    assert names == ["nvidia_nim", "gemini", "llm7", "hetzner", "aion_lab", "freellmapi"]


def test_role_candidates_coder_omniroute_first_then_others_freellmapi_last():
    providers = [
        _prov("gemini", roles=["coder"]),
        _prov("omniroute", roles=["coder"], auto_model_aware=True),
        _prov("nvidia_nim", base_url="https://integrate.api.nvidia.com/v1", roles=["coder"]),
        _prov("llm7", base_url="https://api.llm7.io/v1", roles=["overflow"]),
        _prov("hetzner", base_url="https://inference.hetzner.com/api/v1", roles=["overflow"]),
        _prov("aion_lab", roles=["overflow"]),
        _prov("freellmapi", roles=["overflow"]),
    ]
    names = [p.name for p in router._role_candidates(providers, "coder", "primary")]
    assert names == ["omniroute", "nvidia_nim", "gemini",
                     "llm7", "hetzner", "aion_lab", "freellmapi"]


# --- role dispatch -----------------------------------------------------------

def test_role_candidates_coder_nvidia_first_then_overflow():
    providers = [
        _prov("gemini", roles=["coder"]),
        _prov("nvidia_nim", base_url="https://integrate.api.nvidia.com/v1", roles=["coder"]),
        _prov("groq", roles=["planner"]),
        _prov("llm7", base_url="https://api.llm7.io/v1", roles=["overflow"]),
        _prov("hetzner", base_url="https://inference.hetzner.com/api/v1", roles=["overflow"]),
        _prov("aion_lab", roles=["overflow"]),
    ]
    names = [p.name for p in router._role_candidates(providers, "coder", "primary")]
    assert names == ["nvidia_nim", "gemini", "llm7", "hetzner", "aion_lab"]


def test_role_candidates_planner_cloudflare_first():
    providers = [
        _prov("groq", roles=["planner"]),
        _prov("cloudflare_workers_ai", base_url="", roles=["planner"],
              adapter="cloudflare_workers_ai"),
        _prov("openrouter_free", roles=["planner"]),
        _prov("llm7", base_url="https://api.llm7.io/v1", roles=["overflow"]),
        _prov("hetzner", base_url="https://inference.hetzner.com/api/v1", roles=["overflow"]),
    ]
    names = [p.name for p in router._role_candidates(providers, "planner", "fast")]
    assert names == ["cloudflare_workers_ai", "groq", "openrouter_free", "llm7", "hetzner"]


def test_role_candidates_background_mistral_only_plus_overflow():
    providers = [
        _prov("mistral", roles=["background"]),
        _prov("gemini", roles=["coder"]),
        _prov("llm7", base_url="https://api.llm7.io/v1", roles=["overflow"]),
    ]
    names = [p.name for p in router._role_candidates(providers, "background", "batch")]
    assert names == ["mistral", "llm7"]


def test_get_client_unknown_role_rejected():
    with pytest.raises(ValueError):
        router.get_client("nonexistent")


def test_get_client_forwards_role(tmp_usage, monkeypatch):
    seen = {}

    def fake_complete(messages, model, max_tokens=4000, timeout=120,
                      provider_hint=None, role=None):
        seen["role"] = role
        return router.LightCompletion(provider_name="nvidia_nim", model=model,
                                      choices=[router.Choice(router.Message("hi"))])

    monkeypatch.setattr(router, "complete", fake_complete)
    resp = router.get_client("coder").chat.completions.create(
        model="auto/best-coding", max_tokens=5,
        messages=[{"role": "user", "content": "x"}], timeout=10)
    assert seen["role"] == "coder"
    assert resp.provider_name == "nvidia_nim"
    assert resp.choices[0].message.content == "hi"


def test_overflow_fallthrough_when_role_budget_exhausted(tmp_usage, monkeypatch):
    """coder -> nvidia_nim, but its rolling rpm is spent, so complete() must
    reroute to the overflow pool (llm7) and report THAT provider served."""
    nvidia = _prov("nvidia_nim", base_url="https://integrate.api.nvidia.com/v1",
                   roles=["coder"], rpm=1)
    llm7 = _prov("llm7", base_url="https://api.llm7.io/v1", roles=["overflow"],
                 default_model="GLM-5.3-Flash")
    monkeypatch.setattr(router, "load_providers", lambda env=None: [nvidia, llm7])
    monkeypatch.setattr(router, "call_with_watchdog", lambda fn, timeout: _fake_response("hi"))
    router.mark_result(nvidia, ok=True, tokens=1, requests=1)
    resp = router.complete([{"role": "user", "content": "x"}], "auto/best-coding",
                           max_tokens=5, timeout=10, role="coder")
    assert resp.provider_name == "llm7"
    assert resp.choices[0].message.content == "hi"


# --- budget windows -----------------------------------------------------------

def test_rolling_rpm_exceeds_then_rolls_off(tmp_usage, monkeypatch):
    p = _prov("nvidia_nim", base_url="https://integrate.api.nvidia.com/v1",
              roles=["coder"], rpm=2)
    router.mark_result(p, ok=True, tokens=1, requests=1)
    router.mark_result(p, ok=True, tokens=1, requests=1)
    assert router.is_callable(p) == (False, "rpm_exceeded")
    real_time = time.time
    monkeypatch.setattr(router.time, "time", lambda: real_time() + 120)
    assert router.is_callable(p) == (True, None)


def test_rolling_rps_exceeds(tmp_usage):
    p = _prov("mistral", roles=["background"], rps=1)
    router.mark_result(p, ok=True, tokens=1, requests=1)
    assert router.is_callable(p) == (False, "rps_exceeded")


def test_calendar_day_tokens_exceeded_then_rollover(tmp_usage, monkeypatch):
    p = _prov("llm7", base_url="https://api.llm7.io/v1", roles=["overflow"],
              daily_tokens=100)
    router.mark_result(p, ok=True, tokens=100)
    assert router.is_callable(p) == (False, "daily_tokens_exceeded")
    monkeypatch.setattr(router, "_today", lambda: "2099-12-31")
    assert router.is_callable(p) == (True, None)


def test_calendar_day_input_tokens_exceeded(tmp_usage):
    p = _prov("hetzner", base_url="https://inference.hetzner.com/api/v1",
              roles=["overflow"], daily_tokens_in=500000000)
    router.mark_result(p, ok=True, tokens=10, input_tokens=500000001)
    assert router.is_callable(p) == (False, "daily_tokens_in_exceeded")


def test_calendar_day_neurons_exceeded(tmp_usage):
    p = _prov("cloudflare_workers_ai", base_url="", roles=["planner"],
              adapter="cloudflare_workers_ai", daily_neurons=10000)
    router.mark_result(p, ok=True, tokens=1, neurons=10000)
    assert router.is_callable(p) == (False, "daily_neurons_exceeded")


def test_calendar_month_tokens_exceeded(tmp_usage, monkeypatch):
    p = _prov("mistral", roles=["background"], monthly_tokens=10)
    router.mark_result(p, ok=True, tokens=6)
    router.mark_result(p, ok=True, tokens=6)
    assert router.is_callable(p) == (False, "monthly_tokens_exceeded")
    monkeypatch.setattr(router, "_today", lambda: "2099-12-31")
    assert router.is_callable(p) == (True, None)


def test_unset_budgets_never_bind(tmp_usage):
    p = _prov("gemini", roles=["coder"])
    router.mark_result(p, ok=True, tokens=999)
    assert router.is_callable(p) == (True, None)


def test_reset_until_monthly_and_daily():
    monthly = _prov("mistral", reset_kind="monthly")
    ts = router._reset_until(monthly)
    assert ts > time.time() and ts < time.time() + 40 * 86400
    daily = _prov("llm7", base_url="https://api.llm7.io/v1", reset_kind="daily")
    assert router._reset_until(daily) == router._next_utc_midnight()


def test_mark_result_records_input_and_neurons(tmp_usage):
    p = _prov("cloudflare_workers_ai", base_url="", roles=["planner"],
              adapter="cloudflare_workers_ai")
    router.mark_result(p, ok=True, tokens=99, input_tokens=50, neurons=7)
    entry = router.provider_usage(p)
    assert entry["day_input_tokens"] == 50
    assert entry["day_neurons"] == 7


# --- Cloudflare adapter -------------------------------------------------------

def test_cloudflare_adapter_translates_envelope(tmp_usage, monkeypatch):
    captured = {}

    class _Resp:
        def __init__(self, data):
            self._data = data.encode()

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=120):
        captured["method"] = req.method
        captured["host"] = urlparse(req.full_url).netloc
        captured["body"] = json.loads(req.data)
        payload = {
            "success": True,
            "result": {"usage": {"total_tokens": 30, "prompt_tokens": 22},
                       "choices": [{"message": {"content": "cf says hello"}}]},
        }
        return _Resp(json.dumps(payload))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    p = _prov("cloudflare_workers_ai", base_url="https://test.local", roles=["planner"],
              adapter="cloudflare_workers_ai")
    resp = router._cloudflare_chat(
        p, "@cf/meta/llama-3.1-8b-instruct",
        [{"role": "user", "content": "hi"}], 50, timeout=5)
    assert captured["body"]["model"] == "@cf/meta/llama-3.1-8b-instruct"
    assert captured["body"]["max_tokens"] == 50
    assert resp.choices[0].message.content == "cf says hello"
    assert resp.usage.total_tokens == 30
    assert resp.provider_name == "cloudflare_workers_ai"


def test_cloudflare_adapter_builds_url_from_account_id(tmp_usage, monkeypatch):
    monkeypatch.setenv("CF_ACCOUNT_ID", "myacct")
    captured = {}

    class _Resp:
        def __init__(self, data):
            self._data = data.encode()

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=120):
        captured["url"] = req.full_url
        payload = {"success": True, "result": {"response": "legacy envelope"}}
        return _Resp(json.dumps(payload))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    p = _prov("cloudflare_workers_ai", base_url="", roles=["planner"],
              adapter="cloudflare_workers_ai")
    resp = router._cloudflare_chat(p, "m", [{"role": "user", "content": "hi"}], 10, 5)
    assert "myacct" in captured["url"]
    assert resp.choices[0].message.content == "legacy envelope"


def test_cloudflare_adapter_requires_account_id(tmp_usage, monkeypatch):
    monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
    p = _prov("cloudflare_workers_ai", base_url="", roles=["planner"],
              adapter="cloudflare_workers_ai")
    with pytest.raises(RuntimeError):
        router._cloudflare_chat(p, "m", [{"role": "user", "content": "hi"}], 10, 5)