import sys
import pytest
from types import SimpleNamespace

import llm_router as router
import oss_agent_v2 as o


@pytest.fixture(autouse=True)
def _clear_ceiling():
    router.set_preflight_ceiling(None)
    yield
    router.set_preflight_ceiling(None)


def test_get_preflight_ceiling_roundtrip():
    assert router.get_preflight_ceiling() is None
    router.set_preflight_ceiling(12345)
    assert router.get_preflight_ceiling() == 12345
    router.set_preflight_ceiling(0)
    assert router.get_preflight_ceiling() is None


def test_cheapest_first_reorders_omniroute_combos():
    chain = ["auto/best-coding", "auto/best-chat",
             "auto/best-reasoning", "auto/best-fast"]
    out = o._cheapest_first(chain)
    assert out[0] == "auto/best-fast"
    assert out[1:] == ["auto/best-coding", "auto/best-chat", "auto/best-reasoning"]


def test_cheapest_first_concrete_hosted_models():
    chain = ["gemini-3.6-pro", "gemini-3.6-flash", "qwen-small"]
    out = o._cheapest_first(chain)
    assert out == ["gemini-3.6-flash", "qwen-small", "gemini-3.6-pro"]


def test_cheapest_first_no_hints_preserves_order():
    chain = ["foo", "bar"]
    assert o._cheapest_first(chain) == ["foo", "bar"]
    assert o._cheapest_first([]) == []


def test_generate_fix_tight_budget_shrinks_and_uses_fast_first(monkeypatch, _clear_ceiling):
    captured = {}

    def fake_call_model(prompt, max_tokens=4000, retry_variant=False,
                        fast=False, cheap_first=False):
        captured.update(max_tokens=max_tokens, cheap_first=cheap_first)
        return "# full-file rewrite\n"

    monkeypatch.setattr(o, "call_model", fake_call_model)
    monkeypatch.setattr(o, "find_similar_experiences", lambda *a, **k: [])
    issue = SimpleNamespace(title="Fix a traceback", body="short repro")
    router.set_preflight_ceiling(1000)  # tiny: system prompts alone eat it
    o.generate_fix(issue, [("a.py", "def a():\n    pass\n")], language="python")
    assert captured["cheap_first"] is True
    assert captured["max_tokens"] == o._MIN_FIX_OUTPUT_TOKENS


def test_generate_fix_roomy_budget_keeps_full_output(monkeypatch, _clear_ceiling):
    captured = {}

    def fake_call_model(prompt, max_tokens=4000, retry_variant=False,
                        fast=False, cheap_first=False):
        captured.update(max_tokens=max_tokens, cheap_first=cheap_first)
        return "ok"

    monkeypatch.setattr(o, "call_model", fake_call_model)
    monkeypatch.setattr(o, "find_similar_experiences", lambda *a, **k: [])
    issue = SimpleNamespace(title="Fix a traceback", body="short repro")
    router.set_preflight_ceiling(100000000)
    o.generate_fix(issue, [("a.py", "def a():\n    pass\n")], language="python")
    assert captured["cheap_first"] is False
    assert captured["max_tokens"] == int(__import__("os").getenv(
        "SOLVE_MAX_OUTPUT_TOKENS", "12000"))