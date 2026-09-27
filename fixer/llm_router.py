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
                  quota: bool = False) -> None:
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
    if reason:
        event["reason"] = reason[:240]
    if quota:
        event["quota"] = True
    with _LOCK:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        except OSError:
            return


def _today() -> str:
    return date.today().isoformat()


def _next_daily_reset() -> float:
    """Seconds-since-epoch of the next UTC midnight. OpenRouter's
    'free-models-per-day' quota resets then; providers marked 429 stay out of
    rotation until this instant instead of being cooldown-cycled every 5 min."""
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _is_quota_error(exc: BaseException) -> bool:
    """True when the failure is a quota/credit ceiling, not a transient fault.
    429 = rate/per-day limit; 402 'requires more credits' = balance ceiling.
    Both say 'do not retry on cooldown, this will not succeed until reset'."""
    status = getattr(exc, "status_code", None)
    if status in (429, 402):
        return True
    text = str(exc)
    return "rate limit exceeded" in text.lower() or "more credits" in text.lower()


def provider_usage(provider: Provider) -> dict:
    """Per-provider daily usage + breaker position, derived from today's events.

    tokens/calls count only today. The circuit is computed from the fails
    tallied since the last success; a provider that tripped can only be
    re-enabled by an actual successful call (see provider_state()). A provider
    whose LATEST event is a quota failure is additionally marked
    quota_exhausted_until = next UTC midnight -- it is skipped instantly for
    the rest of the day instead of being cooldown-retried into the same 429."""
    events = [e for e in _read_events()
              if e.get("provider") == provider.name and e.get("date") == _today()]
    tokens = 0
    calls = 0
    fails = 0
    down = False
    down_since = 0.0
    quota_exhausted_until = 0.0
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
            fails += 1
            if e.get("quota"):
                quota_exhausted_until = _next_daily_reset()
            if not down and fails >= BREAKER_THRESHOLD:
                down = True
                down_since = float(e.get("ts") or 0)
    if not events or latest_ok is True:
        quota_exhausted_until = 0.0
    cooldown_until = down_since + BREAKER_COOLDOWN_SECONDS if down else 0.0
    return {
        "date": _today(),
        "tokens": tokens,
        "calls": calls,
        "fails": fails,
        "down": down,
        "down_since": down_since,
        "cooldown_until": cooldown_until,
        "quota_exhausted_until": quota_exhausted_until,
    }


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
    'quota_exhausted' when its daily free quota is spent (skipped until the
    next UTC reset -- NO cooldown cycling), 'circuit_open' when its cooldown is
    still pending, 'budget_spent' only when its daily token ceiling is
    genuinely exhausted."""
    entry = provider_usage(provider)
    if float(entry.get("quota_exhausted_until", 0)) > time.time():
        return False, "quota_exhausted"
    if provider_state(entry) == "OPEN":
        return False, "circuit_open"
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
                quota: bool = False) -> None:
    """Record one call outcome. Append-only; usage and the breaker re-derive
    themselves from the log, so concurrent runners cannot clobber each other.
    `reason` is a short human-readable failure text captured for the event log.
    `quota=True` marks a quota/credit ceiling (429/402) -- the provider is then
    skipped until the next daily reset instead of being cooldown-retried."""
    _append_event(provider, bool(ok), max(0, int(tokens)), reason, quota=bool(quota))


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
    if provider.models and requested not in provider.models:
        return provider.models[0]
    return requested


_OPENROUTER_SLUGS = ("nvidia/", "cohere/", "dots-studio/", "google/gemma",
                     "meta-llama/", "openrouter/", "anthropic/", "mistralai/")


def _looks_foreign(requested: str) -> bool:
    """True for vendor-prefixed OpenRouter model IDs ('nvidia/…:free'). These
    only belong on the openrouter funnel, not on gemini/groq's native APIs."""
    requested = str(requested or "").strip()
    low = requested.lower()
    return ((":free" in low and "/" in low)
            or any(low.startswith(slug) for slug in _OPENROUTER_SLUGS)
            or low.startswith("google/"))


def _provider_accepts(provider: Provider, requested: str) -> bool:
    """Whether this provider can realistically serve this model ID, so we
    never burn a call on a doomed pairing. gemini only takes its own
    'gemini-*' family (the fixer's openrouter-style combos leak into it and
    404 -- the events log showed 8 identical 404s); groq takes anything that
    is NOT an openrouter vendor slug; the openrouter funnel/omniroute take
    everything (they are auto-model-aware)."""
    requested = str(requested or "").strip()
    if provider.auto_model_aware:  # omniroute: resolves auto/* combos itself
        return True
    if requested.startswith("auto/"):
        return True
    if provider.name == "gemini":
        return requested.startswith("gemini-") or requested.startswith("models/gemini")
    if provider.name == "groq":
        return not _looks_foreign(requested)
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


def complete(messages: list, model: str, max_tokens: int = 4000,
             timeout: float = DEFAULT_WATCHDOG_SECONDS,
             provider_hint: Optional[str] = None):
    """Ask the model, failing over across healthy, budgeted providers.

    Returns a LightCompletion(response-like) so the fixer's call site receives
    `.choices[0].message.content` exactly as before. Raises
    AllProvidersExhaustedError when the whole tier chain is down/over-budget,
    or PreflightTokenBudgetExceeded when this single call would blow the
    remaining per-issue ceiling BEFORE any token is spent.
    """
    from openai import OpenAI

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

    candidates = [p for p in providers if p.matches_tier(tier) and p.enabled]
    if not candidates:
        raise AllProvidersExhaustedError(f"no providers registered for tier '{tier}' "
                                          "(set at least one API key, e.g. OMNIROUTE_API_KEY)")

    last_error: Optional[Exception] = None
    for provider in candidates:
        callable_now, reason = is_callable(provider)
        if reason == "circuit_open":
            continue  # OPEN: cooldown still pending -- try the next provider
        if reason == "quota_exhausted":
            # Daily free quota spent (429/402). Do NOT cooldown-retry: skipped
            # instantly until the next UTC reset instead of re-eating the same
            # refusal every 5 minutes.
            continue
        if reason == "budget_spent":
            last_error = ProviderBudgetExceeded(f"{provider.name}: daily budget spent")
            continue
        # CLOSED or HALF_OPEN with quota: HALF_OPEN means the cooldown has
        # elapsed, so we MUST actually call -- its result is what decides
        # whether the circuit re-closes or re-trips (Bug 1 deadlock escape).
        resolved = resolve_model(provider, model)
        # Bug-1 companion: never let a foreign model ID reach a provider whose
        # native API can't serve it (gemini got 8 pointless 404s yesterday).
        if not _provider_accepts(provider, resolved) and not provider.auto_model_aware:
            continue
        if (provider.name, resolved) in _NEGATIVE_MODELS:
            continue  # already failed this process -- don't re-eat it
        client_kwargs: dict = {"api_key": provider.api_key, "base_url": provider.base_url}
        if provider.name == "copilot":
            # Copilot's /chat/completions rejects requests that lack the
            # editor integration metadata (400 otherwise).
            client_kwargs["default_headers"] = {
                "Copilot-Integration-Id": "oss-agent",
                "Editor-Version": "oss-agent-1.0",
                "Editor-Plugin-Version": "copilot-chat-1.0",
            }
        client = OpenAI(**client_kwargs)
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
            cohort = provider if isinstance(provider, Provider) else provider
            quota = _is_quota_error(exc)
            reason = f"{type(exc).__name__}: {exc}"
            mark_result(cohort, ok=False, reason=reason, quota=quota)
            _NEGATIVE_MODELS.add((provider.name, resolved))
            print(f"⚠️ provider {provider.name} failed: {reason[:240]}", flush=True)
            last_error = exc
            continue
        # Token accounting (W18): prefer SDK usage, estimate otherwise.
        tokens = _estimate_tokens(response, messages)
        mark_result(provider, ok=True, tokens=tokens)
        content = _extract_content(response)
        if not content:
            reason = "returned an empty completion"
            mark_result(provider, ok=False, reason=reason)
            _NEGATIVE_MODELS.add((provider.name, resolved))
            print(f"⚠️ provider {provider.name} failed: {reason}", flush=True)
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
            lines.append(
                f"  {provider.name:<14} tier={provider.tier:<13} {health:4} budget={budget:5}"
                f"{qmark} tokens={int(entry.get('tokens', 0))} calls={int(entry.get('calls', 0))} "
                f"fails={int(entry.get('fails', 0))} state={state}"
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