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
  W18  usage counters checked before call   -- data/provider_usage_events.jsonl
       (append-only) tracks daily tokens + calls; providers over budget are
       skipped pre-call, and breaker state is a derived 3-state circuit.
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


def _int_setting(name: str, default: int) -> int:
    """Read an integer env setting; an EMPTY value behaves like unset.

    GitHub Actions exports missing secrets as the empty string, so a plain
    os.getenv(name, default) would crash int() once the secret existed and
    then went blank again (empty-beats-stale, 7f02810e class bug). A blank
    value must always fall back to the default instead of raising."""
    raw = (os.getenv(name) or "").strip()
    return int(raw) if raw else default


def _float_setting(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    return float(raw) if raw else default


# Consecutive failures before a provider is tripped into cooldown (W12).
BREAKER_THRESHOLD = _int_setting("BREAKER_THRESHOLD", 3)
# Seconds a tripped provider stays out of rotation (W12 recovery window).
BREAKER_COOLDOWN_SECONDS = _float_setting("BREAKER_COOLDOWN_SECONDS", 300)
# Hard watchdog cap for a single inference call, in seconds (W11). The fixer
# also passes its own per-combo timeouts; this is the outer, non-negotiable lid.
DEFAULT_WATCHDOG_SECONDS = _float_setting("LLM_WATCHDOG_SECONDS", 120)
# Global daily token ceiling across ALL providers (W18). Set per-provider with
# <PROVIDER>_DAILY_BUDGET; this is the aggregate safety net.
GLOBAL_DAILY_TOKEN_BUDGET = _int_setting("GLOBAL_DAILY_TOKEN_BUDGET", 2000000)

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


class PreflightTokenBudgetExceeded(RouterError):
    """A single call's estimated input + max_tokens exceeds the per-issue
    pre-flight ceiling, so it was refused before any token was spent."""


# W9: when set, hard-flagged issues may route to the reserved escalation tier
# (COPILOT_API_KEY) before the free pool. Opt-in -- no key, no effect.
_ESCALATE = {"enabled": False}

# Per-issue pre-flight ceiling (tokens): when set, complete() estimates the
# messages' input tokens BEFORE calling and refuses to send a call whose
# input + max_tokens would blow the remaining issue budget. This stops the
# "spend 99k on input, then discover the 40k issue budget is blown" pattern
# (the Rekin226/aquascope#449 kill) -- refusal happens before spend, not after.
_PREFLIGHT_CEILING: Optional[int] = None

# Within-run memo of (provider.name, resolved_model) pairs that already failed
# this process. A run re-tries the same combo repeatedly across attempts; once
# a pair has failed hard it is skipped for the rest of the process instead of
# re-eating the same 429/404/timeout on every hunt.
_NEGATIVE_MODELS: set = set()


def set_preflight_ceiling(tokens: Optional[int]) -> None:
    """Set/clear the per-issue pre-flight ceiling. The fixer calls this with
    the remaining issue budget at the start of every attempt."""
    global _PREFLIGHT_CEILING
    _PREFLIGHT_CEILING = (int(tokens) if tokens is not None and int(tokens) > 0 else None)


def get_preflight_ceiling() -> Optional[int]:
    """Remaining per-issue budget currently enforced by the pre-flight guard,
    or None when no ceiling is active. Lets the fixer shrink a call's requested
    output / pick a cheaper combo while the budget is tight instead of waiting
    for the pre-flight guard to refuse the whole call."""
    return _PREFLIGHT_CEILING


# Preferred failover order within the PRIMARY tier. gemini proved healthy
# today (102k real tokens, no shared quota with OpenRouter); groq is a
# separate free tier that shares nothing with the exhausted openrouter-free
# bucket; openrouter_free/omniroute funnel into that same exhausted bucket and
# are deliberately last. Anything not listed keeps its old relative position.
_PROVIDER_ORDER = {
    "gemini": 0,
    "groq": 1,
    "openrouter_free": 2,
    "omniroute": 3,
    "omniroute_fallback": 4,
    "copilot": 5,
    "mistral": 6,
}


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
    # Registry-supplied tuning (see providers_registry.json / load_providers):
    reset_kind: str = "utc_midnight"  # utc_midnight|pacific_midnight|rolling|daily|monthly|probe
    max_tokens_cap: Optional[int] = None  # hard cap per call send (<= fed to API)
    max_input_tokens: Optional[int] = None  # context window: skip when request would overflow
    rpm: Optional[float] = None  # requests per rolling 60s (rpm ceiling when set)
    rps: Optional[float] = None  # requests per rolling 1s (rps ceiling when set)
    roles: list = field(default_factory=list)  # coder|planner|background|overflow (role dispatch)
    adapter: str = ""  # non-OpenAI-schema provider (e.g. "cloudflare_workers_ai"); base_url may be blank
    key_env_alt: str = ""  # alternate env var name when key_env is empty (e.g. NVIDIA_NIM_API_KEY)
    daily_tokens: Optional[int] = None  # per-provider calendar-day token ceiling
    daily_tokens_in: Optional[int] = None  # calendar-day INPUT-token ceiling (Hetzner)
    daily_neurons: Optional[int] = None  # calendar-day neurons ceiling (Cloudflare Workers AI)
    monthly_tokens: Optional[int] = None  # calendar-month token ceiling

    def matches_tier(self, tier: str) -> bool:
        """A provider serves the requested tier if it IS that tier, or if it is
        primary/tertiary (wide coverage) when asked for a tier it is not
        explicitly part of."""
        if self.tier == tier:
            return True
        if tier == "primary" and self.tier in ("primary", "tertiary", "fast"):
            # fast (e.g. groq) is explicitly part of the primary chain -- it
            # shares no quota with the openrouter funnel and is cheap.
            return True
        if tier == "fast" and self.tier in ("fast", "primary"):
            return True
        if tier == "escalation":
            # Escalation is a preference, not a hard requirement: if no copilot
            # tier is configured the free primary pool takes over. Tertiary
            # (e.g. OpenRouter) is part of that wide-coverage pool, and fast
            # (groq) shares no quota with the openrouter funnel, so a "hard"
            # issue must not lose the only providers that are actually healthy.
            return self.tier in ("escalation", "primary", "tertiary", "fast")
        return False


def _env_flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _tier_for_role(roles: list) -> str:
    """Role -> serving tier, so the registry can express intent without tiers:
       coder (stronger models) -> primary; planner (cheap fast) -> fast;
       background (slow orderly) -> batch (NOT served by the default path);
       overflow/opportunistic (experimental) -> never the core path;
       copilot -> escalation."""
    r = {str(x).strip().lower() for x in (roles or [])}
    if "copilot" in r:
        return "escalation"
    if "overflow" in r or "opportunistic" in r:
        return "opportunistic"
    if "coder" in r:
        return "primary"
    if "planner" in r:
        return "fast"
    if "background" in r:
        return "batch"
    return "primary"


# Role dispatch (get_client(role)): within a role, the designated provider(s)
# are tried in this order BEFORE the overflow pool, so a role request never
# bleeds into a random provider. Providers not listed keep their day-to-day
# order. The heart of the new routing: coder -> nvidia_nim, planner ->
# cloudflare_workers_ai, background -> mistral; overflow = llm7 -> hetzner.
_ROLE_ORDER = {
    "coder": ["omniroute", "nvidia_nim", "gemini"],
    "planner": ["cloudflare_workers_ai", "groq", "openrouter_free", "omniroute"],
    "background": ["mistral"],
    "overflow": ["llm7", "hetzner", "aion_lab", "freellmapi"],
}


def _has_role(provider: "Provider", role: str) -> bool:
    return role in {str(x).strip().lower() for x in (provider.roles or [])}


def _registry_path(env: Optional[dict]) -> Path:
    raw = env.get("PROVIDER_REGISTRY") or os.getenv("PROVIDER_REGISTRY", "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent / "providers_registry.json"


def load_registry(env: Optional[dict] = None) -> Optional[list]:
    """Read the provider registry (a JSON list of entries). Returns None when
    no registry file exists -- the env-derived default chain applies. Registry
    entries carry NO secrets: the API key always comes from key_env."""
    e = os.environ if env is None else env
    path = _registry_path(e)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("providers") or []


def _provider_from_registry(entry: dict, e: dict) -> Optional[Provider]:
    """Build one Provider from a registry entry; None when its key_env (and
    key_env_alt, if set) is empty (empty-beats-stale, same rule as the
    env-derived chain), its base_url is unset AND it has no adapter (registry
    entry OR env OVERRIDE must provide it, then it may be an ops-only endpoint
    like omniroute_fallback), or it's disabled."""
    if not bool(entry.get("enabled", True)):
        return None
    name = str(entry.get("name") or "").strip()
    if not name:
        return None
    key_env = str(entry.get("key_env") or f"{name.upper()}_API_KEY")
    api_key = (e.get(key_env) or "").strip()
    key_env_alt = str(entry.get("key_env_alt") or "").strip()
    if not api_key and key_env_alt:
        api_key = (e.get(key_env_alt) or "").strip()
    if not api_key:
        return None
    adapter = str(entry.get("adapter") or "").strip()
    base_url = str(e.get(f"{name.upper()}_BASE_URL") or "").strip() \
        or str(entry.get("base_url") or "").strip()
    if not base_url and not adapter:
        # Registered key but nowhere to call (espera omniroute_fallback whose
        # URL is only supplied via the OMNIROUTE_FALLBACK_BASE_URL override).
        return None
    cap = entry.get("max_tokens_cap")
    in_cap = entry.get("max_input_tokens")
    rpm = entry.get("rpm")
    rps = entry.get("rps")
    d_tokens = entry.get("daily_tokens")
    d_in = entry.get("daily_tokens_in")
    d_neurons = entry.get("daily_neurons")
    m_tokens = entry.get("monthly_tokens")
    return Provider(
        name=name,
        base_url=base_url,
        api_key=api_key,
        tier=str(entry.get("tier") or _tier_for_role(entry.get("role") or [])),
        default_model=entry.get("default_model") or None,
        auto_model_aware=bool(entry.get("auto_model_aware", False)),
        models=list(entry.get("models") or []),
        enabled=bool(entry.get("enabled", True)),
        reset_kind=str(entry.get("reset_kind") or "utc_midnight"),
        max_tokens_cap=(int(cap) if cap else None),
        max_input_tokens=(int(in_cap) if in_cap else None),
        rpm=(float(rpm) if rpm is not None else None),
        rps=(float(rps) if rps is not None else None),
        roles=[str(x).strip().lower() for x in (entry.get("role") or []) if str(x).strip()],
        adapter=adapter,
        key_env_alt=key_env_alt,
        daily_tokens=(int(d_tokens) if d_tokens is not None else None),
        daily_tokens_in=(int(d_in) if d_in is not None else None),
        daily_neurons=(int(d_neurons) if d_neurons is not None else None),
        monthly_tokens=(int(m_tokens) if m_tokens is not None else None),
    )


def _apply_env_overrides(providers: list, e: dict) -> None:
    """Allow env to override registry defaults without editing the registry
    (OMNIROUTE_BASE_URL, GEMINI_MODEL, ...). Empty env never wins."""
    for p in providers:
        base = (e.get(f"{p.name.upper()}_BASE_URL") or "").strip()
        if base:
            p.base_url = base
        model = (e.get(f"{p.name.upper()}_MODEL") or "").strip()
        if model:
            p.default_model = model


def load_providers(env: Optional[dict] = None) -> list[Provider]:
    """Build the provider pool. Registry-first: when fixer/providers_registry.json
    exists, only providers whose key_env holds a non-empty key are registered
    (empty-beats-stale keeps legacy machines omniRoute-only). Without a registry
    the env-derived chain below applies unchanged. `env` is injected for tests;
    defaults to os.environ."""
    e = os.environ if env is None else env
    registry = load_registry(e)
    if registry is not None:
        providers = [p for entry in registry
                     if (p := _provider_from_registry(entry, e)) is not None]
        _apply_env_overrides(providers, e)
        if _env_flag("LLM_ROUTER_TRIM_TIERS", default=False):
            providers = [p for p in providers
                         if p.tier not in ("opportunistic", "batch")]
        return providers

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
            default_model=(e.get("GEMINI_MODEL") or "gemini-3.8-flash").strip(),
            models=[m.strip() for m in (e.get("GEMINI_MODELS") or "").split(",") if m.strip()],
        ))

    groq = _key("GROQ_API_KEY")
    if groq:
        providers.append(Provider(
            name="groq",
            base_url=e.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip(),
            api_key=groq,
            tier="fast",
            default_model=(e.get("GROQ_MODEL") or "qwen/qwen3.8-27b").strip(),
        ))

    openrouter = _key("OPENROUTER_API_KEY")
    if openrouter:
        providers.append(Provider(
            name="openrouter_free",
            base_url=e.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
            api_key=openrouter,
            tier="tertiary",
            default_model=(e.get("OPENROUTER_MODEL") or
                           "meta-llama/llama-3.1-8b-instruct").strip(),
        ))

    mistral = _key("MISTRAL_API_KEY")
    if mistral:
        providers.append(Provider(
            name="mistral",
            base_url=e.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").strip(),
            api_key=mistral,
            tier="batch",
            default_model=(e.get("MISTRAL_MODEL") or "mistral-small-latest").strip(),
        ))

    # W19: opportunistic-only tier. Never used unless every other tier is down.
    for name, default_url, default_model in (
        ("hetzner", "https://inference.hetzner.com/api/v1", "Hetzner/turbo"),
        ("llm7", "https://api.llm7.io/v1", "GLM-5.3-Flash"),
        ("aion_lab", "https://api.aionlabs.ai/v1", "aion-labs/aion-3.0-mini"),
        ("nvidia_nim", "https://integrate.api.nvidia.com/v1",
         "nvidia/nemotron-3-super-120b-a12b"),
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
    # NOTE: the SDK appends `/chat/completions` to base_url, so the default
    # is the API root (not the full path); Copilot also requires the
    # Editor-Version headers or it answers 400, which complete() adds for us.
    copilot = _key("COPILOT_API_KEY")
    if copilot:
        providers.append(Provider(
            name="copilot",
            base_url=(e.get("COPILOT_BASE_URL") or
                      "https://api.githubcopilot.com/").strip(),
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
# Usage budget + circuit breaker (W18 / W12)
#
# State is DERIVED from an append-only events log, not stored as overwritable
# JSON. Every call appends one line to provider_usage_events.jsonl
# ({provider, ok, tokens, ts, date, run_id}); per-provider usage (tokens,
# calls) and the 3-state circuit (CLOSED/OPEN/HALF_OPEN) are recomputed from
# today's events at read time. Append-only is the defense-in-depth half of the
# concurrency fix: the gate-poll and controller workflows each commit the whole
# worktree with `pull -X ours`, so a snapshot file silently loses whichever
# run committed second -- but no commit can roll another run's tokens back out
# of a log, and day-rollover (Bug 3) is emergent: tomorrow has no events, so
# no tokens, no fails, and no open circuit.
#
# Breaker rules (Bug 1): is_healthy()/budget_available() used to disagree about
# `down`, which deadlocked recovery -- a fired provider could never be retried
# because budget_available() short-circuited on `down`, yet only a successful
# call can clear it. Now one is_callable() owns the decision: OPEN rejects,
# HALF_OPEN (cooldown elapsed) is allowed through for a real trial call, and
# quota (has_quota) is a PURE token-count check that never consults the
# breaker, so "budget_spent" is only reported when it is actually true.
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()

RUN_ID = (
    os.getenv("GITHUB_RUN_ID", "")
    or os.getenv("RUN_ID", "")
    or f"proc-{os.getpid()}"
)


def _events_file() -> Path:
    # PROVIDER_USAGE_FILE is read from the env at call time (not just the
    # import-time constant) so tests and live runs can redirect it cleanly.
    usage_path = os.getenv("PROVIDER_USAGE_FILE", PROVIDER_USAGE_FILE)
    default = str(Path(usage_path).with_name("provider_usage_events.jsonl"))
    return Path(os.getenv("PROVIDER_USAGE_EVENTS_FILE", default))


def _read_events() -> list:
    """All lines from the append-only events log (malformed lines skipped)."""
    path = _events_file()
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def _append_event(provider: Provider, ok: bool, tokens: int, reason: str = "",
                  quota: bool = False, rate_backoff_until: float = 0.0,
                  trip: bool = True, requests: int = 1,
                  input_tokens: int = 0, neurons: int = 0) -> None:
    path = _events_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "provider": provider.name,
        "ok": bool(ok),
        "tokens": max(0, int(tokens)),
        "ts": time.time(),
        "date": _today(),
        "run_id": RUN_ID,
    }
    if requests:
        event["requests"] = max(1, int(requests))
    if input_tokens:
        event["input_tokens"] = max(0, int(input_tokens))
    if neurons:
        event["neurons"] = max(0, int(neurons))
    if reason:
        event["reason"] = reason[:240]
    if quota:
        event["quota"] = True
    if rate_backoff_until:
        event["rate_backoff_until"] = float(rate_backoff_until)
    if not trip:
        event["trip"] = False
    with _LOCK:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        except OSError:
            return


def _today() -> str:
    return date.today().isoformat()


def _next_utc_midnight() -> float:
    """Seconds-since-epoch of the next UTC midnight. OpenRouter's
    'free-models-per-day' quota resets then."""
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    return (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    ).timestamp()


def _next_pacific_midnight() -> float:
    """Seconds-since-epoch of the next US Pacific midnight (Gemini free tier
    resets at midnight PT, not UTC)."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return nxt.astimezone(ZoneInfo("UTC")).timestamp()


PROBE_INTERVAL_SECONDS = int(os.getenv("LLM_QUOTA_PROBE_SECONDS", "10800").strip() or 10800)


def _reset_until(provider: Provider) -> float:
    """When does this provider's quota re-open? Per the registry's reset_kind
    (defaults to utc_midnight). 'probe' means we DON'T know the reset cadence,
    so instead of sleeping to a guessed midnight we re-probe after a few hours.
    'daily' resets at the next UTC midnight, 'monthly' at the first of next
    month (both calendar-based, matching the budget windows)."""
    now = time.time()
    kind = (provider.reset_kind or "utc_midnight").lower()
    if kind == "pacific_midnight":
        return _next_pacific_midnight()
    if kind == "monthly":
        from datetime import datetime, timezone, timedelta
        now_utc = datetime.now(timezone.utc)
        first_next = (now_utc + timedelta(days=32)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
        return first_next.timestamp()
    if kind == "rolling":
        return now + 24 * 3600
    if kind == "probe":
        return now + PROBE_INTERVAL_SECONDS
    return _next_utc_midnight()


def _response_headers(exc: BaseException) -> dict:
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
        if isinstance(headers, dict):
            return headers
        if hasattr(headers, "items"):
            try:
                return {str(k).lower(): str(v) for k, v in headers.items()}
            except Exception:  # noqa: BLE001
                return {}
    return {}


def _is_daily_exhausted(exc: BaseException) -> bool:
    """429 IS 'quota spent for the day' only on the markers providers attach;
    a bare rate-limit 429 must be handled as a short backoff instead."""
    low = str(exc).lower()
    if any(m in low for m in ("perday", "per_day", "free-models-per-day", "daily")):
        return True
    for v in _response_headers(exc).values():
        vl = str(v).lower()
        if "perday" in vl or "per-day" in vl or "free-models-per-day" in vl:
            return True
        if v == "daily" or vl == "daily":
            return True
    return False


def _classify_failure(exc: BaseException) -> str:
    """One of '402_call_too_big' | '429_daily' | '429_rate' | 'transient'."""
    status = getattr(exc, "status_code", None)
    text = str(exc)
    low = text.lower()
    if status == 402 or "more credits" in low:
        return "402_call_too_big"
    if status == 429 or "rate limit" in low:
        return "429_daily" if _is_daily_exhausted(exc) else "429_rate"
    return "transient"


def _retry_after_seconds(exc: BaseException) -> float:
    """Honour Retry-After when the provider sent it; default 20s, capped 300s."""
    try:
        raw = _response_headers(exc).get("retry-after")
    except Exception:  # noqa: BLE001
        raw = None
    if raw:
        try:
            return min(300.0, max(1.0, float(raw)))
        except (TypeError, ValueError):
            pass
    return 20.0


def _is_quota_error(exc: BaseException) -> bool:
    """Back-compat (callers outside complete()) -- classification re-runs
    _classify_failure so the fast-path consumer sees the same split the
    complete() loop does."""
    return _classify_failure(exc) in ("402_call_too_big", "429_daily")


def _budget_windows(provider: Provider) -> dict:
    """Rolling + calendar-budget counters for a provider, derived from the
    append-only event log (persists across restarts -- no in-memory budget).

    - window_requests_1s / window_requests_60s: ALL calls (ok or failed) whose
      ts fell inside the trailing window -- the rpm/rps ceilings count requests
      made, not successful ones.
    - day_input_tokens: successful calls' input_tokens on the calendar day.
    - day_neurons: successful calls' neurons on the calendar day.
    - month_tokens: successful calls' tokens in the calendar month.

    A default of 0 for a missing field means "no budget pressure"; the checks
    in is_callable only bind when the provider actually declares a ceiling."""
    now = time.time()
    today = _today()
    month = today[:7]
    recents_1s = recents_60s = 0
    day_input = day_neurons = 0
    month_tokens = 0
    for e in _read_events():
        if e.get("provider") != provider.name:
            continue
        ts = float(e.get("ts") or 0)
        if ts >= now - 60:
            recents_60s += 1
            if ts >= now - 1:
                recents_1s += 1
        if e.get("ok") and e.get("date") == today:
            day_input += int(e.get("input_tokens") or 0)
            day_neurons += int(e.get("neurons") or 0)
        if e.get("ok") and str(e.get("date") or "")[:7] == month:
            month_tokens += int(e.get("tokens") or 0)
    return {
        "window_requests_1s": recents_1s,
        "window_requests_60s": recents_60s,
        "day_input_tokens": day_input,
        "day_neurons": day_neurons,
        "month_tokens": month_tokens,
    }


def provider_usage(provider: Provider) -> dict:
    """Per-provider daily usage + breaker position, derived from today's events.

    tokens/calls count only today. The circuit is computed from the fails
    tallied since the last success; a provider that tripped can only be
    re-enabled by an actual successful call (see provider_state()). A provider
    whose LATEST event is a quota/credit failure is additionally marked
    quota_exhausted_until = its provider-specific reset instant (per the
    registry's reset_kind -- UTC/Pacific midnight, rolling 24h, or a periodic
    probe). A 429 that is merely rate-limiting (no daily marker) sets a short
    rate_backoff_until instead -- NOT a day-long ban. Failures recorded with
    trip=False (e.g. the 402-too-big retry) never flip the breaker."""
    events = [e for e in _read_events()
              if e.get("provider") == provider.name and e.get("date") == _today()]
    tokens = 0
    calls = 0
    fails = 0
    down = False
    down_since = 0.0
    quota_exhausted_until = 0.0
    rate_backoff_until = 0.0
    latest_ok = None  # None (no events) | True | False
    for e in events:
        calls += 1
        latest_ok = bool(e.get("ok"))
        if e.get("ok"):
            tokens += int(e.get("tokens") or 0)
            fails = 0
            down = False
            down_since = 0.0
        else:
            if e.get("trip", True):
                fails += 1
            if e.get("quota"):
                quota_exhausted_until = _reset_until(provider)
            if float(e.get("rate_backoff_until") or 0):
                rate_backoff_until = max(rate_backoff_until,
                                         float(e["rate_backoff_until"]))
            if not down and fails >= BREAKER_THRESHOLD:
                down = True
                down_since = float(e.get("ts") or 0)
    if not events or latest_ok is True:
        quota_exhausted_until = 0.0
        rate_backoff_until = 0.0
    cooldown_until = down_since + BREAKER_COOLDOWN_SECONDS if down else 0.0
    entry = {
        "date": _today(),
        "tokens": tokens,
        "calls": calls,
        "fails": fails,
        "down": down,
        "down_since": down_since,
        "cooldown_until": cooldown_until,
        "quota_exhausted_until": quota_exhausted_until,
        "rate_backoff_until": rate_backoff_until,
    }
    entry.update(_budget_windows(provider))
    return entry


def provider_state(entry: dict) -> str:
    """CLOSED (healthy) | OPEN (tripped, reject) | HALF_OPEN (trial allowed)."""
    if not entry.get("down"):
        return "CLOSED"
    if time.time() >= float(entry.get("cooldown_until", 0)):
        return "HALF_OPEN"
    return "OPEN"


def daily_budget_for(provider: Provider) -> int:
    raw = os.getenv(f"{provider.name.upper()}_DAILY_BUDGET", "").strip()
    return int(raw) if raw.isdigit() else GLOBAL_DAILY_TOKEN_BUDGET


def has_quota(provider: Provider, entry: dict) -> bool:
    """Pure token-count check. NEVER consults the breaker, so an OPEN circuit
    can still come back through HALF_OPEN and be re-probed by a real call."""
    return int(entry.get("tokens", 0)) < daily_budget_for(provider)


def is_callable(provider: Provider) -> tuple:
    """(callable, reason). reason is None when the provider may be called now;
    'quota_exhausted' when its quota is spent until its provider-specific reset
    (skipped -- NO cooldown cycling), 'rate_limited' when a merely-throttled 429
    is still inside its short Retry-After window, 'circuit_open' when its
    cooldown is still pending, 'budget_spent' only when its daily token ceiling
    is genuinely exhausted. Registry-declared ceilings bind BEFORE the daily
    check: rpm_exceeded (rolling minute), rps_exceeded (rolling second),
    daily_tokens_exceeded, daily_tokens_in_exceeded, daily_neurons_exceeded,
    monthly_tokens_exceeded. A field that is None never binds."""
    entry = provider_usage(provider)
    now = time.time()
    if float(entry.get("quota_exhausted_until", 0)) > now:
        return False, "quota_exhausted"
    if float(entry.get("rate_backoff_until", 0)) > now:
        return False, "rate_limited"
    if provider_state(entry) == "OPEN":
        return False, "circuit_open"
    if provider.rps is not None and int(entry.get("window_requests_1s", 0)) >= int(provider.rps):
        return False, "rps_exceeded"
    if provider.rpm is not None and int(entry.get("window_requests_60s", 0)) >= int(provider.rpm):
        return False, "rpm_exceeded"
    if provider.daily_neurons is not None \
            and int(entry.get("day_neurons", 0)) >= int(provider.daily_neurons):
        return False, "daily_neurons_exceeded"
    if provider.daily_tokens_in is not None \
            and int(entry.get("day_input_tokens", 0)) >= int(provider.daily_tokens_in):
        return False, "daily_tokens_in_exceeded"
    if provider.daily_tokens is not None \
            and int(entry.get("tokens", 0)) >= int(provider.daily_tokens):
        return False, "daily_tokens_exceeded"
    if provider.monthly_tokens is not None \
            and int(entry.get("month_tokens", 0)) >= int(provider.monthly_tokens):
        return False, "monthly_tokens_exceeded"
    if not has_quota(provider, entry):
        return False, "budget_spent"
    return True, None


def is_healthy(provider: Provider) -> bool:
    """Back-compat alias: callable except a still-cooling-down OPEN circuit."""
    return provider_state(provider_usage(provider)) != "OPEN"


def budget_available(provider: Provider) -> bool:
    """Back-compat alias: pure quota check (never consults the breaker)."""
    return has_quota(provider, provider_usage(provider))


def mark_result(provider: Provider, ok: bool, tokens: int = 0, reason: str = "",
                quota: bool = False, rate_backoff_until: float = 0.0,
                trip: bool = True, requests: int = 1,
                input_tokens: int = 0, neurons: int = 0) -> None:
    """Record one call outcome. Append-only; usage and the breaker re-derive
    themselves from the log, so concurrent runners cannot clobber each other.
    `reason` is a short human-readable failure text captured for the event log.
    `quota=True` marks a quota/credit ceiling (429 with a daily marker / 402) --
    the provider is then skipped until its provider-specific reset instead of
    being cooldown-retried. `rate_backoff_until` is the instant a 429-rate
    backoff clears (NOT a day ban). `trip=False` records a failure that must
    NOT push the circuit breaker (e.g. the 402-too-big halved retry).
    `requests`/`input_tokens`/`neurons` feed the rolling (rpm/rps) and
    calendar (daily_*/monthly) budget windows."""
    _append_event(provider, bool(ok), max(0, int(tokens)), reason,
                  quota=bool(quota), rate_backoff_until=rate_backoff_until,
                  trip=bool(trip), requests=requests,
                  input_tokens=int(input_tokens), neurons=int(neurons))


def daily_tokens_spent() -> int:
    return sum(int(e.get("tokens") or 0) for e in _read_events()
               if e.get("ok") and e.get("date") == _today())


def global_budget_available() -> bool:
    return daily_tokens_spent() < GLOBAL_DAILY_TOKEN_BUDGET


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
    # Concrete ids pass through untouched -- the models-list compat filter
    # (_provider_accepts) decides whether this provider may serve them. We do
    # NOT silently substitute models[0]: a caller that named a specific model
    # gets that exact id or a skip, never a quiet swap that could mask a leak.
    return requested


def _provider_accepts(provider: Provider, requested: str) -> bool:
    """Whether this provider can realistically serve this model ID, so we
    never burn a call on a doomed pairing. Authoritative source is the
    provider's OWN whitelist (registry `models` entry -- native ids only):
    an id not on the list is skipped. When a provider lists no models it
    accepts everything (openrouter-style funnels / auto-model-aware routers,
    and tests)."""
    if provider.auto_model_aware:
        return True
    requested = str(requested or "").strip()
    if requested.startswith("auto/"):
        return True
    if provider.models:
        return requested in provider.models
    return True


def _sort_candidates(providers: Iterable[Provider]) -> list:
    """Order providers for the serving tier. Three groups, in priority:

      0  escalation (e.g. copilot) -- ALWAYS first on hard-flagged issues so
         the reserved tier is never starved of a turn,
      1  the user-approved primary chain, in explicit order
         (gemini -> groq -> openrouter_free -> omniroute -> ...): gemini is
         first because it proved healthy yesterday with no shared quota;
         groq and openrouter* follow the funnel preference; omniroute -- which
         funnels into the same exhausted openrouter-free bucket -- is last,
         despite being tier 'primary', because its primary rank is deliberately
         overridden here.
      2  anything else, sorted by tier rank then name."""
    def _key(p: Provider):
        if p.tier == "escalation":
            return (0, _PROVIDER_ORDER.get(p.name, 0))
        base = _PROVIDER_ORDER.get(p.name)
        if base is not None:
            return (1, base)
        return (2, _TIER_RANK.get(p.tier, 9), p.name)

    return sorted(providers, key=_key)


# ---------------------------------------------------------------------------
# The public completion entry point (W14 failover core)
# ---------------------------------------------------------------------------
def estimate_input_tokens(messages: list) -> int:
    """Cheap deterministic pre-flight estimate of a messages list's input size.
    Conservative (chars//3, code-heavy English texts run ~3-4 chars/token), so
    the guard errs toward refusal rather than silent overspend."""
    total_chars = sum(len(m.get("content") or "") for m in messages)
    return max(0, int(total_chars / 3))


def _preflight_check(messages: list, max_tokens: int) -> None:
    """Refuse a call BEFORE it spends anything when input + output would blow
    the remaining per-issue ceiling. This is what stops the 99k-input call
    that previously atomically exceeded the issue budget mid-attempt."""
    ceiling = _PREFLIGHT_CEILING
    if ceiling is None:
        return
    est = estimate_input_tokens(messages) + int(max_tokens)
    if est > ceiling:
        raise PreflightTokenBudgetExceeded(
            f"pre-flight: call needs ~{est} tokens (input {estimate_input_tokens(messages)} "
            f"+ max_tokens {max_tokens}) but only {ceiling} remain in the issue budget "
            f"-- refusing before spend"
        )


def _role_candidates(providers: list, role: str, tier: str) -> list:
    """Candidate chain for role dispatch (get_client(role)/complete(role=...)).
    Explicit-role providers first, in the per-role order (coder -> nvidia_nim
    first, planner -> cloudflare_workers_ai first, background -> mistral);
    overflow-role providers (`llm7 -> hetzner -> ...`) appended last so a
    budget-exhausted role provider falls through to the overflow pool exactly
    as the operator specified instead of hitching a ride on another role."""
    order = {n: i for i, n in enumerate(_ROLE_ORDER.get(role, []))}
    of_order = {n: i for i, n in enumerate(_ROLE_ORDER.get("overflow", []))}
    explicit = [p for p in providers if _has_role(p, role) and p.enabled]
    overflow = [p for p in providers
                if _has_role(p, "overflow") and p.name not in {q.name for q in explicit}]
    explicit.sort(key=lambda p: (order.get(p.name, 10 ** 9),
                                 _PROVIDER_ORDER.get(p.name, 0), p.name))
    overflow.sort(key=lambda p: (of_order.get(p.name, 10 ** 9),
                                 _PROVIDER_ORDER.get(p.name, 0), p.name))
    return explicit + overflow


def _cloudflare_chat(provider: Provider, model: str, messages: list,
                     max_tokens: int, timeout: float):
    """Cloudflare Workers AI does NOT speak the OpenAI schema natively: its
    base endpoint returns its own {success,result} envelope. The opinionated
    OpenAI-compatible route lives at
    https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/...
    which accepts OpenAI-shaped chat requests but still wraps the reply. This
    adapter translates request + response both ways so complete()'s existing
    accounting/failover path consumes it unchanged. Needs CF_ACCOUNT_ID in
    addition to the CF_API_TOKEN when no base_url is configured."""
    import urllib.request  # noqa: PLC0415 - stdlib, no openai SDK round-trip

    if not provider.base_url:
        account_id = os.getenv("CF_ACCOUNT_ID", "").strip()
        if not account_id:
            raise RuntimeError(
                "cloudflare_workers_ai: CF_ACCOUNT_ID unset "
                "(adapter builds the URL from it; nothing to call)"
            )
        url = (f"https://api.cloudflare.com/client/v4/accounts/"
               f"{account_id}/ai/v1/chat/completions")
    else:
        url = provider.base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": int(max_tokens)}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {provider.api_key}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("success") is False:
        raise RuntimeError(f"cloudflare_workers_ai: {payload.get('errors')}")
    result = payload.get("result", payload)
    choices = result.get("choices")
    if choices:
        content = str(choices[0].get("message", {}).get("content", "") or "")
    else:
        content = str(result.get("response", "") or "")
    usage = None
    u = result.get("usage") or payload.get("usage")
    if u:
        from types import SimpleNamespace  # noqa: PLC0415
        usage = SimpleNamespace(
            prompt_tokens=int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0),
            completion_tokens=int(u.get("completion_tokens", 0) or 0),
            total_tokens=int(u.get("total_tokens", 0) or 0),
            neurons=int(u.get("neurons", 0) or 0),
        )
    return LightCompletion(
        provider_name=provider.name,
        model=model,
        choices=[Choice(Message(content))],
        usage=usage,
    )


def _neurons_for(response, input_tokens: int) -> int:
    """Neurons billed (Cloudflare Workers AI unit). Use the response's own
    figure when present, else an input-token estimate so the daily_neurons
    ceiling still binds even when a provider omits usage."""
    usage = getattr(response, "usage", None)
    if usage is not None:
        raw = getattr(usage, "neurons", None)
        if raw:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    return int(input_tokens)


def complete(messages: list, model: str, max_tokens: int = 4000,
             timeout: float = DEFAULT_WATCHDOG_SECONDS,
             provider_hint: Optional[str] = None,
             role: Optional[str] = None):
    """Ask the model, failing over across healthy, budgeted providers.

    Returns a LightCompletion(response-like) so the fixer's call site receives
    `.choices[0].message.content` exactly as before. Raises
    AllProvidersExhaustedError when the whole tier chain is down/over-budget,
    or PreflightTokenBudgetExceeded when this single call would blow the
    remaining per-issue ceiling BEFORE any token is spent.

    `role` (coder/planner/background) dispatches through that role's designated
    provider first (per _ROLE_ORDER) then the overflow pool; when NO provider
    declares the role the legacy tier chain applies unchanged, so machines
    without the new role registry behave byte-for-byte as before.
    """
    from openai import OpenAI  # noqa: PLC0415

    _preflight_check(messages, max_tokens)

    tier = "fast" if "fast" in str(model).lower() else "primary"
    if escalation_enabled():
        tier = "escalation"
    providers = _sort_candidates(load_providers())
    if provider_hint:
        providers = [p for p in providers if p.name == provider_hint] or providers

    if not global_budget_available():
        raise ProviderBudgetExceeded(
            f"global daily token budget of {GLOBAL_DAILY_TOKEN_BUDGET} exhausted "
            f"(see {PROVIDER_USAGE_FILE})"
        )

    if role:
        role_candidates = _role_candidates(providers, role, tier)
        if role_candidates and any(_has_role(p, role) for p in role_candidates):
            candidates = role_candidates
        else:
            candidates = [p for p in providers if p.matches_tier(tier) and p.enabled]
    else:
        candidates = [p for p in providers if p.matches_tier(tier) and p.enabled]
    if not candidates:
        raise AllProvidersExhaustedError(f"no providers registered for tier '{tier}' "
                                          f"or role '{role}' "
                                          "(set at least one API key, e.g. OMNIROUTE_API_KEY)")

    last_error: Optional[Exception] = None
    for provider in candidates:
        callable_now, reason = is_callable(provider)
        if reason == "circuit_open":
            continue  # OPEN: cooldown still pending -- try the next provider
        if reason == "quota_exhausted":
            # Daily quota spent (429-with-daily-marker / 402 balance). Do NOT
            # cooldown-retry: skipped instantly until this provider's OWN
            # reset instant instead of re-eating the same refusal every cycle.
            continue
        if reason == "rate_limited":
            # 429 without a daily marker: short Retry-After backoff, not a
            # day-long ban. Try the next provider now; it self-heals quickly.
            continue
        if reason == "budget_spent":
            last_error = ProviderBudgetExceeded(f"{provider.name}: daily budget spent")
            continue
        if reason in ("rpm_exceeded", "rps_exceeded"):
            # Rolling rate window -- clears in under a minute. Skip quietly to
            # the next provider instead of setting a scary last_error.
            continue
        if reason in ("daily_tokens_exceeded", "daily_tokens_in_exceeded",
                      "daily_neurons_exceeded", "monthly_tokens_exceeded"):
            last_error = ProviderBudgetExceeded(f"{provider.name}: {reason}")
            continue
        # CLOSED or HALF_OPEN with quota: HALF_OPEN means the cooldown has
        # elapsed, so we MUST actually call -- its result is what decides
        # whether the circuit re-closes or re-trips (Bug 1 deadlock escape).
        resolved = resolve_model(provider, model)
        # Never let a foreign model ID reach a provider whose native API can't
        # serve it (the registry's models whitelist is authoritative).
        if not _provider_accepts(provider, resolved) and not provider.auto_model_aware:
            continue
        if (provider.name, resolved) in _NEGATIVE_MODELS:
            continue  # already failed this process -- don't re-eat it
        # Context-window guard: a provider whose declared input window cannot
        # hold this request would fail (400/413/context-overflow) and waste the
        # call on a doomed pairing. Estimated cheaply and conservatively from
        # the request itself -- no network. An unset max_input_tokens means
        # "unknown, accept" (fail-open) so legacy/registry-less setups are
        # unaffected; a set-but-too-small window skips cleanly to the next
        # candidate exactly like a model-whitelist miss.
        if provider.max_input_tokens is not None:
            est_input = estimate_input_tokens(messages)
            if est_input > provider.max_input_tokens:
                print(f"[!] provider {provider.name} skipped: request input "
                      f"~{est_input} tokens > its {provider.max_input_tokens}-token "
                      f"window -- trying the next candidate", flush=True)
                continue
        client_kwargs: dict = {"api_key": provider.api_key, "base_url": provider.base_url}
        if provider.name == "copilot":
            # Copilot's /chat/completions rejects requests that lack the
            # editor integration metadata (400 otherwise).
            client_kwargs["default_headers"] = {
                "Copilot-Integration-Id": "oss-agent",
                "Editor-Version": "oss-agent-1.0",
                "Editor-Plugin-Version": "copilot-chat-1.0",
            }
        client = (None if provider.adapter
                  else OpenAI(**client_kwargs))

        def _do_call(call_mt: int):
            if provider.adapter:
                return _cloudflare_chat(provider, resolved, messages,
                                        int(call_mt), timeout)
            return client.chat.completions.create(
                model=resolved,
                max_tokens=int(call_mt),
                messages=messages,
            )

        call_max_tokens = max_tokens
        if provider.max_tokens_cap is not None:
            # Registry cap: some free endpoints hard-fail >N output tokens
            # (openrouter-free 429s at 2048) -- clamp so the budget fits.
            call_max_tokens = min(max_tokens, int(provider.max_tokens_cap))
        started = time.monotonic()
        try:
            response = call_with_watchdog(
                lambda: _do_call(call_max_tokens),
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - record and fail over
            cohort = provider if isinstance(provider, Provider) else provider
            kind = _classify_failure(exc)

            def _record_failure(fail_kind: str, err: BaseException) -> BaseException:
                """Mark one classified failure on the events log + (per kind)
                the negative cache, then hand back err for last_error."""
                if fail_kind == "402_call_too_big":
                    mark_result(cohort, ok=False, reason=f"{type(err).__name__}: {err}",
                                quota=True)
                elif fail_kind == "429_daily":
                    mark_result(cohort, ok=False, reason=f"{type(err).__name__}: {err}",
                                quota=True)
                    _NEGATIVE_MODELS.add((provider.name, resolved))
                elif fail_kind == "429_rate":
                    backoff = _retry_after_seconds(err)
                    mark_result(cohort, ok=False, reason=f"{type(err).__name__}: {err}",
                                rate_backoff_until=time.time() + backoff)
                    print(f"[!] provider {provider.name} rate-limited "
                          f"(backoff {backoff:.0f}s)", flush=True)
                else:
                    mark_result(cohort, ok=False, reason=f"{type(err).__name__}: {err}")
                    _NEGATIVE_MODELS.add((provider.name, resolved))
                if fail_kind != "429_rate":
                    print(f"[!] provider {provider.name} failed: "
                          f"{f'{type(err).__name__}: {err}'[:240]}", flush=True)
                return err

            if kind == "402_call_too_big":
                # 402 = "requires more credits OR fewer max_tokens": one call
                # blew the daily cap on tokens. Retry ONCE with halved output
                # on the SAME provider; the cap is what overran, not the quota.
                # trip=False: this is not circuit noise. quota/negative-cache
                # stays untouched unless the halved retry ALSO 402s.
                if call_max_tokens > 1:
                    print(f"[!] provider {provider.name} 402 w/ max_tokens={call_max_tokens} "
                          f"-- retrying once at {max(1, call_max_tokens // 2)}", flush=True)
                    try:
                        response = call_with_watchdog(
                            lambda: _do_call(max(1, call_max_tokens // 2)),
                            timeout=timeout,
                        )
                    except Exception as exc2:  # noqa: BLE001
                        last_error = _record_failure(_classify_failure(exc2), exc2)
                        continue
                else:
                    last_error = _record_failure("402_call_too_big", exc)
                    continue
            elif kind == "429_daily":
                last_error = _record_failure("429_daily", exc)
                continue
            elif kind == "429_rate":
                last_error = _record_failure("429_rate", exc)
                continue
            else:
                last_error = _record_failure("transient", exc)
                continue
        # Token accounting (W18): prefer SDK usage, estimate otherwise. The
        # input_tokens/neurons feeds feed the per-provider calendar/neurons
        # budget walls so they bind BEFORE the next call on that provider.
        tokens = _estimate_tokens(response, messages)
        input_tokens = estimate_input_tokens(messages)
        neurons = _neurons_for(response, input_tokens)
        mark_result(provider, ok=True, tokens=tokens,
                    input_tokens=input_tokens, neurons=neurons)
        content = _extract_content(response)
        if not content:
            reason = "returned an empty completion"
            mark_result(provider, ok=False, reason=reason)
            _NEGATIVE_MODELS.add((provider.name, resolved))
            print(f"[!] provider {provider.name} failed: {reason}", flush=True)
            last_error = RuntimeError(f"{provider.name} returned an empty completion")
            continue
        _NEGATIVE_MODELS.discard((provider.name, resolved))
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
        role = kwargs.pop("role", None)
        del kwargs
        use_timeout = float(timeout) if timeout else DEFAULT_WATCHDOG_SECONDS
        return self._proxy.complete(model=model, max_tokens=max_tokens,
                                     messages=messages or [], timeout=use_timeout,
                                     role=role)


class _Chat:
    def __init__(self, proxy: "CompletionProxy"):
        self.completions = _Completions(proxy)


class CompletionProxy:
    """Drop-in replacement client: `proxy.chat.completions.create(...)`."""

    def __init__(self):
        self.chat = _Chat(self)

    def complete(self, model, max_tokens, messages, timeout, role=None) -> LightCompletion:
        # Only forward `role` when the caller actually used it, so legacy
        # call sites / monkeypatched `complete` (old signature) stay untouched.
        if role is None:
            return complete(messages, model, max_tokens=max_tokens, timeout=timeout)
        return complete(messages, model, max_tokens=max_tokens, timeout=timeout,
                        role=role)


def build_completion_proxy():
    return CompletionProxy()


class _RoleProxy(CompletionProxy):
    """A CompletionProxy pinned to one role: every create() routes through
    that role's designated provider (then the overflow pool), and reports
    which provider actually served the call via `provider_name`."""

    def __init__(self, role: str):
        if role not in ("coder", "planner", "background", "overflow"):
            raise ValueError(f"unknown llm role: {role!r} "
                             "(use coder|planner|background|overflow)")
        super().__init__()
        self.role = role

    def complete(self, model, max_tokens, messages, timeout, role=None) -> LightCompletion:
        return complete(messages, model, max_tokens=max_tokens, timeout=timeout,
                        role=self.role if role is None else role)


def get_client(role: str):
    """Role-routed client shaped like the legacy proxy -- the role picks the
    provider (coder->nvidia_nim, planner->cloudflare_workers_ai,
    background->mistral) with overflow fallback, so callers get the same
    `obj.chat.completions.create(...)` surface as before.

    Every call consults the persistent budget tracker first and reroutes when
    the designated provider is budget-exhausted; which provider actually
    served is on `resp.provider_name`. When no provider declares `role`, the
    legacy tier chain applies (no behavioural change on unconfigured boxes).
    """
    return _RoleProxy(role)


# ---------------------------------------------------------------------------
# CLI: python llm_router.py --status
# ---------------------------------------------------------------------------
def health_status() -> str:
    lines = ["llm_router provider pool", "-" * 42]
    providers = load_providers()
    if not providers:
        lines.append("  <no providers registered -- set at least one API key>")
    else:
        for provider in providers:
            entry = provider_usage(provider)
            state = provider_state(entry)
            if state == "CLOSED":
                health = "UP"
            elif state == "HALF_OPEN":
                health = "trial"
            else:
                health = "DOWN"
            budget = "ok" if has_quota(provider, entry) else "OVER"
            qmark = " QUOTA" if float(entry.get("quota_exhausted_until", 0)) > time.time() else ""
            roles = ",".join(provider.roles) if provider.roles else provider.tier
            extras = ""
            if provider.rpm is not None:
                extras += f" rpm={int(entry.get('window_requests_60s', 0))}/{int(provider.rpm)}"
            if provider.daily_tokens is not None:
                extras += f" day={int(entry.get('tokens', 0))}/{int(provider.daily_tokens)}"
            if provider.monthly_tokens is not None:
                extras += f" mo={int(entry.get('month_tokens', 0))}/{int(provider.monthly_tokens)}"
            lines.append(
                f"  {provider.name:<14} role={roles:<12} {health:4} budget={budget:5}"
                f"{qmark} tokens={int(entry.get('tokens', 0))} calls={int(entry.get('calls', 0))} "
                f"fails={int(entry.get('fails', 0))} state={state}{extras}"
            )
    lines.append(f"global daily tokens: {daily_tokens_spent()}/{GLOBAL_DAILY_TOKEN_BUDGET}")
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