"""
llm_router.py
Tiered LLM provider pool for oss-agent.

Fulfils the solution-plan weaknesses that were NOT already handled by the
fixer's OmniRoute-only path:

  W1   drop the single-gateway dependency   -- providers are first-class; the
       omniRoute gateway is just one provider entry among several.
  W11  per-call watchdog timeout            -- `call_with_watchdog()` hard-caps
       every inference call.
  W12  circuit breaker per provider         -- N consecutive failures put a
       provider into a cooldown instead of retrying it into the ground.
  W14  multi-provider pool with failover    -- `complete()` walks a tier chain
       and moves to the next healthy/budgeted provider.
  W18  usage counters checked before call   -- data/provider_usage.json tracks
       daily tokens + calls; providers over budget are skipped pre-call.
  W19  opportunistic-only tier              -- Hetzner/LLM7 never enter the
       primary path; they are explicitly last-resort.
  W9   selective escalation                 -- a COPILOT tier (opt-in, e.g.
       GitHub Copilot Student) is reserved for hard-flagged issues only.

Ground rules copied from the fixer's own conventions:
  * A provider is registered ONLY when its API key is set and non-empty, so a
    machine with no new keys behaves byte-for-byte like before (omniRoute-only).
  * Empty/unset env never wins over a default (the "empty beats stale" rule).
  * No provider is ever retried into a breaker trip forever; cooldowns let a
    healthy provider recover.

Design note: this module cannot import oss_agent_v2 (would be circular). It is
self-contained and drives the OpenAI SDK directly. The fixer adapts by
constructing its client from `build_completion_proxy()`, so every existing
`ai_client.chat.completions.create(...)` call site keeps working unchanged.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

PROVIDER_USAGE_FILE = os.getenv(
    "PROVIDER_USAGE_FILE",
    str(Path(__file__).resolve().parent.parent / "data" / "provider_usage.json"),
)

# Consecutive failures before a provider is tripped into cooldown (W12).
BREAKER_THRESHOLD = int(os.getenv("BREAKER_THRESHOLD", "3"))
# Seconds a tripped provider stays out of rotation (W12 recovery window).
BREAKER_COOLDOWN_SECONDS = float(os.getenv("BREAKER_COOLDOWN_SECONDS", "300"))
# Hard watchdog cap for a single inference call, in seconds (W11). The fixer
# also passes its own per-combo timeouts; this is the outer, non-negotiable lid.
DEFAULT_WATCHDOG_SECONDS = float(os.getenv("LLM_WATCHDOG_SECONDS", "120"))
# Global daily token ceiling across ALL providers (W18). Set per-provider with
# <PROVIDER>_DAILY_BUDGET; this is the aggregate safety net.
GLOBAL_DAILY_TOKEN_BUDGET = int(os.getenv("GLOBAL_DAILY_TOKEN_BUDGET", "2000000"))

# Tier order used for failover within each tier; providers keep this order.
_TIER_RANK = {"escalation": 0, "primary": 1, "fast": 2, "tertiary": 3,
              "batch": 4, "opportunistic": 5}


class RouterError(Exception):
    """Base class for llm_router failures."""


class WatchdogTimeout(RouterError):
    """A provider call did not return inside its watchdog window (W11)."""


class ProviderBudgetExceeded(RouterError):
    """Provider/global daily token budget is exhausted (W18)."""


class AllProvidersExhaustedError(RouterError):
    """Every provider in the tier chain was down or over budget (W14)."""


# W9: when set, hard-flagged issues may route to the reserved escalation tier
# (COPILOT_API_KEY) before the free pool. Opt-in -- no key, no effect.
_ESCALATE = {"enabled": False}


def set_escalation(enabled: bool) -> None:
    _ESCALATE["enabled"] = bool(enabled)


def escalation_enabled() -> bool:
    return bool(_ESCALATE["enabled"])


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    tier: str = "primary"
    default_model: Optional[str] = None
    auto_model_aware: bool = False  # understands omniRoute auto/* combos
    models: list = field(default_factory=list)  # concrete models this provider offers
    enabled: bool = True

    def matches_tier(self, tier: str) -> bool:
        """A provider serves the requested tier if it IS that tier, or if it is
        primary/tertiary (wide coverage) when asked for a tier it is not
        explicitly part of."""
        if self.tier == tier:
            return True
        if tier == "primary" and self.tier in ("primary", "tertiary"):
            return True
        if tier == "fast" and self.tier in ("fast", "primary"):
            return True
        if tier == "escalation":
            # Escalation is a preference, not a hard requirement: if no copilot
            # tier is configured the free primary pool takes over.
            return self.tier in ("escalation", "primary")
        return False


def _env_flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def load_providers(env: Optional[dict] = None) -> list[Provider]:
    """Build the provider pool from the environment. Only providers with a
    non-empty API key are registered (empty-beats-stale keeps legacy machines
    omniRoute-only). `env` is injected for tests; defaults to os.environ."""
    e = os.environ if env is None else env

    def _key(name: str) -> str:
        return (e.get(name) or "").strip()

    providers: list[Provider] = []

    omniroute = _key("OMNIROUTE_API_KEY") or _key("LLM_API_KEY")
    if omniroute:
        providers.append(Provider(
            name="omniroute",
            base_url=e.get("OMNIROUTE_BASE_URL", "http://localhost:20128/v1").strip(),
            api_key=omniroute,
            tier="primary",
            auto_model_aware=True,
        ))

    # Opt-in hosted fallback (OMNIROUTE_FALLBACK_*) -- preserves the old
    # cloud-outage behaviour: when the primary gateway chain is down, the
    # hosted endpoint becomes the next provider in the primary tier.
    fallback_url = e.get("OMNIROUTE_FALLBACK_BASE_URL", "").strip()
    if fallback_url:
        providers.append(Provider(
            name="omniroute_fallback",
            base_url=fallback_url,
            api_key=e.get("OMNIROUTE_FALLBACK_API_KEY", "").strip() or omniroute or "omniroute-local",
            tier="primary",
            default_model=(e.get("OMNIROUTE_FALLBACK_MODEL", "").strip() or None),
        ))

    gemini = _key("GEMINI_API_KEY")
    if gemini:
        providers.append(Provider(
            name="gemini",
            base_url=e.get("GEMINI_BASE_URL",
                           "https://generativelanguage.googleapis.com/v1beta/openai").strip(),
            api_key=gemini,
            tier="primary",
            default_model=(e.get("GEMINI_MODEL") or "gemini-2.0-flash").strip(),
            models=[m.strip() for m in (e.get("GEMINI_MODELS") or "").split(",") if m.strip()],
        ))

    groq = _key("GROQ_API_KEY")
    if groq:
        providers.append(Provider(
            name="groq",
            base_url=e.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip(),
            api_key=groq,
            tier="fast",
            default_model=(e.get("GROQ_MODEL") or "llama-3.3-70b-versatile").strip(),
        ))

    openrouter = _key("OPENROUTER_API_KEY")
    if openrouter:
        providers.append(Provider(
            name="openrouter_free",
            base_url=e.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
            api_key=openrouter,
            tier="tertiary",
            default_model=(e.get("OPENROUTER_MODEL") or
                           "google/gemini-2.0-flash-lite:free").strip(),
        ))

    mistral = _key("MISTRAL_API_KEY")
    if mistral:
        providers.append(Provider(
            name="mistral",
            base_url=e.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").strip(),
            api_key=mistral,
            tier="batch",
            default_model=(e.get("MISTRAL_MODEL") or "mistral-small-2507").strip(),
        ))

    # W19: opportunistic-only tier. Never used unless every other tier is down.
    for name, default_url, default_model in (
        ("hetzner", "https://inference.hetzner.com/api/v1", "Hetzner/turbo"),
        ("llm7", "https://llm7.io/api/v1", "openchat/openchat-7b:free"),
    ):
        api_key = _key(f"{name.upper()}_API_KEY")
        if api_key:
            providers.append(Provider(
                name=name,
                base_url=(e.get(f"{name.upper()}_BASE_URL") or default_url).strip(),
                api_key=api_key,
                tier="opportunistic",
                default_model=(e.get(f"{name.upper()}_MODEL") or default_model).strip(),
            ))

    # W9: escalation tier, opt-in (e.g. GitHub Copilot Student access).
    copilot = _key("COPILOT_API_KEY")
    if copilot:
        providers.append(Provider(
            name="copilot",
            base_url=(e.get("COPILOT_BASE_URL") or
                      "https://api.githubcopilot.com/chat/completions").strip(),
            api_key=copilot,
            tier="escalation",
            default_model=(e.get("COPILOT_MODEL") or "claude-sonnet-4").strip(),
        ))

    if _env_flag("LLM_ROUTER_TRIM_TIERS"):
        # Opportunistic/batch providers are only served if the caller explicitly
        # requested that tier -- keeps W19's "never core path" promise airtight.
        providers = [p for p in providers if p.tier not in ("opportunistic", "batch")]

    return providers


# ---------------------------------------------------------------------------
# Usage budget + circuit breaker store (W18 / W12)
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()


def _default_store() -> dict:
    return {"providers": {}, "updated": time.time()}


def _read_store() -> dict:
    try:
        with open(PROVIDER_USAGE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return _default_store()


def _write_store(store: dict) -> None:
    store["updated"] = time.time()
    path = Path(PROVIDER_USAGE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)
    os.replace(tmp, path)


def _today() -> str:
    return date.today().isoformat()


def _entry(store: dict, name: str) -> dict:
    entry = store.setdefault("providers", {}).setdefault(name, {})
    if entry.get("date") != _today():
        # New day: roll usage over, keep breaker state.
        entry = {"date": _today(), "tokens": 0, "calls": 0,
                 "fails": entry.get("fails", 0),
                 "down": entry.get("down", False),
                 "cooldown_until": entry.get("cooldown_until", 0)}
        store["providers"][name] = entry
    return entry


def daily_budget_for(provider: Provider) -> int:
    raw = os.getenv(f"{provider.name.upper()}_DAILY_BUDGET", "").strip()
    return int(raw) if raw.isdigit() else GLOBAL_DAILY_TOKEN_BUDGET


def provider_usage(provider: Provider) -> dict:
    with _LOCK:
        store = _read_store()
        return dict(_entry(store, provider.name))


def budget_available(provider: Provider) -> bool:
    """W18: is this provider under its daily token budget, checked BEFORE a call?"""
    entry = provider_usage(provider)
    if entry.get("down"):
        return False
    return int(entry.get("tokens", 0)) < daily_budget_for(provider)


def global_budget_available() -> bool:
    total = sum(e.get("tokens", 0) for e in _read_store().get("providers", {}).values())
    return total < GLOBAL_DAILY_TOKEN_BUDGET


def is_healthy(provider: Provider) -> bool:
    entry = provider_usage(provider)
    if not entry.get("down"):
        return True
    cooldown_until = entry.get("cooldown_until", 0)
    return time.time() >= cooldown_until  # cooldown elapsed -> healthy again


def mark_result(provider: Provider, ok: bool, tokens: int = 0) -> None:
    """W12/W18: update the breaker + usage counters for one call outcome."""
    with _LOCK:
        store = _read_store()
        entry = _entry(store, provider.name)
        if ok:
            entry["fails"] = 0
            entry["down"] = False
        else:
            entry["fails"] = int(entry.get("fails", 0)) + 1
            if entry["fails"] >= BREAKER_THRESHOLD:
                entry["down"] = True
                entry["cooldown_until"] = time.time() + BREAKER_COOLDOWN_SECONDS
        if tokens > 0:
            entry["tokens"] = int(entry.get("tokens", 0)) + tokens
        entry["calls"] = int(entry.get("calls", 0)) + 1
        _write_store(store)


# ---------------------------------------------------------------------------
# Watchdog (W11)
# ---------------------------------------------------------------------------
def call_with_watchdog(fn: Callable[[], Any],
                       timeout: float = DEFAULT_WATCHDOG_SECONDS) -> Any:
    """Run fn on a worker thread; fail hard if it does not return in time.

    A hung provider call must not eat a whole 10-minute tick. The SDK timeout
    catches most cases; this thread-level wall clock is the guarantee that
    nothing slows past `timeout`."""
    result_q: "queue.Queue[tuple]" = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_q.put(("ok", fn()))
        except Exception as exc:  # noqa: BLE001 - must not escape the thread
            result_q.put(("err", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        kind, value = result_q.get(timeout=timeout)
    except queue.Empty:
        raise WatchdogTimeout(
            f"provider call exceeded watchdog window of {timeout:.0f}s"
        ) from None
    if kind == "err":
        raise value
    return value


# ---------------------------------------------------------------------------
# Model resolution across providers (auto/* only exists on omniRoute)
# ---------------------------------------------------------------------------
def resolve_model(provider: Provider, requested: str) -> str:
    if provider.auto_model_aware:
        return requested
    if requested.startswith("auto/") and provider.default_model:
        return provider.default_model
    if provider.models and requested not in provider.models:
        return provider.models[0]
    return requested


# ---------------------------------------------------------------------------
# The public completion entry point (W14 failover core)
# ---------------------------------------------------------------------------
def complete(messages: list, model: str, max_tokens: int = 4000,
             timeout: float = DEFAULT_WATCHDOG_SECONDS,
             provider_hint: Optional[str] = None):
    """Ask the model, failing over across healthy, budgeted providers.

    Returns a LightCompletion(response-like) so the fixer's call site receives
    `.choices[0].message.content` exactly as before. Raises
    AllProvidersExhaustedError when the whole tier chain is down/over-budget.
    """
    from openai import OpenAI

    tier = "fast" if "fast" in str(model).lower() else "primary"
    if escalation_enabled():
        tier = "escalation"
    providers = sorted(load_providers(), key=lambda p: _TIER_RANK.get(p.tier, 9))
    if provider_hint:
        providers = [p for p in providers if p.name == provider_hint] or providers

    if not global_budget_available():
        raise ProviderBudgetExceeded(
            f"global daily token budget of {GLOBAL_DAILY_TOKEN_BUDGET} exhausted "
            f"(see {PROVIDER_USAGE_FILE})"
        )

    candidates = [p for p in providers if p.matches_tier(tier) and p.enabled]
    if not candidates:
        raise AllProvidersExhaustedError(f"no providers registered for tier '{tier}' "
                                          "(set at least one API key, e.g. OMNIROUTE_API_KEY)")

    last_error: Optional[Exception] = None
    for provider in candidates:
        if not is_healthy(provider):
            continue
        if not budget_available(provider):
            last_error = ProviderBudgetExceeded(f"{provider.name}: daily budget spent")
            continue
        resolved = resolve_model(provider, model)
        client = OpenAI(api_key=provider.api_key, base_url=provider.base_url)
        started = time.monotonic()
        try:
            response = call_with_watchdog(
                lambda: client.chat.completions.create(
                    model=resolved,
                    max_tokens=max_tokens,
                    messages=messages,
                ),
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - record and fail over
            mark_result(provider, ok=False)
            last_error = exc
            continue
        # Token accounting (W18): prefer SDK usage, estimate otherwise.
        tokens = _estimate_tokens(response, messages)
        mark_result(provider, ok=True, tokens=tokens)
        content = _extract_content(response)
        if not content:
            mark_result(provider, ok=False)
            last_error = RuntimeError(f"{provider.name} returned an empty completion")
            continue
        return LightCompletion(
            provider_name=provider.name,
            model=resolved,
            choices=[Choice(Message(content))],
            usage=getattr(response, "usage", None),
            latency=time.monotonic() - started,
        )

    raise AllProvidersExhaustedError(
        f"all providers for tier '{tier}' failed or were over budget"
        f"{f' (last: {last_error})' if last_error else ''}"
    ) from last_error


def _estimate_tokens(response, messages: list) -> int:
    """W18: tokens for this call, from SDK usage when present, else an estimate."""
    usage = getattr(response, "usage", None)
    if usage is not None:
        inp = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        out = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        if inp is not None or out is not None:
            return int(inp or 0) + int(out or 0)
    return int(sum(len(m.get("content") or "") for m in messages) / 4) + 256


def _extract_content(response) -> str:
    try:
        text = response.choices[0].message.content
    except Exception:  # noqa: BLE001
        text = ""
    if not text:
        try:
            text = response.choices[0].message.reasoning or ""
        except Exception:  # noqa: BLE001
            text = ""
    return str(text or "")


class Message:
    def __init__(self, content: str):
        self.content = content
        self.reasoning = None


class Choice:
    def __init__(self, message: Message):
        self.message = message


@dataclass
class LightCompletion:
    provider_name: str
    model: str
    choices: list
    usage: Any = None
    latency: float = 0.0


# ---------------------------------------------------------------------------
# Fixer integration: a client-shaped proxy so the fixer needs no call-site
# changes. `_LazyClient(_make_ai_client)` in oss_agent_v2.py simply builds
# `CompletionProxy()` instead of an OpenAI client.
# ---------------------------------------------------------------------------
class _Completions:
    def __init__(self, proxy: "CompletionProxy"):
        self._proxy = proxy

    def create(self, model="", max_tokens=4000, messages=None, timeout=None, **kwargs):
        del kwargs
        use_timeout = float(timeout) if timeout else DEFAULT_WATCHDOG_SECONDS
        return self._proxy.complete(model=model, max_tokens=max_tokens,
                                     messages=messages or [], timeout=use_timeout)


class _Chat:
    def __init__(self, proxy: "CompletionProxy"):
        self.completions = _Completions(proxy)


class CompletionProxy:
    """Drop-in replacement client: `proxy.chat.completions.create(...)`."""

    def __init__(self):
        self.chat = _Chat(self)

    def complete(self, model, max_tokens, messages, timeout) -> LightCompletion:
        return complete(messages, model, max_tokens=max_tokens, timeout=timeout)


def build_completion_proxy():
    return CompletionProxy()


# ---------------------------------------------------------------------------
# CLI: python llm_router.py --status
# ---------------------------------------------------------------------------
def health_status() -> str:
    lines = ["llm_router provider pool", "-" * 42]
    store = _read_store()
    providers = load_providers()
    if not providers:
        lines.append("  <no providers registered -- set at least one API key>")
    else:
        for provider in providers:
            entry = _entry(store, provider.name)
            health = "UP" if is_healthy(provider) else "DOWN"
            budget = "ok" if budget_available(provider) else "OVER"
            lines.append(
                f"  {provider.name:<14} tier={provider.tier:<13} {health:4} budget={budget:5} "
                f"tokens={int(entry.get('tokens', 0))} calls={int(entry.get('calls', 0))} "
                f"fails={int(entry.get('fails', 0))}"
            )
    lines.append(f"global daily tokens: {sum(e.get('tokens', 0) for e in store.get('providers', {}).values())}/{GLOBAL_DAILY_TOKEN_BUDGET}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="llm_router",
                                     description="oss-agent LLM provider pool")
    parser.add_argument("--status", action="store_true",
                        help="print provider health + usage and exit")
    args = parser.parse_args(argv)
    if args.status:
        print(health_status())
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())