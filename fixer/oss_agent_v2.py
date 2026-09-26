"""
oss_agent_v2.py
Personal Autonomous Open-Source Contribution Engine
Implements the 10-step flow from the PRD/DRD, with the gaps found during
verification fixed: PR limits, test timeout, claim comment, and the
verification step are now actually implemented (the original reference
code only sketched these in the architecture diagram).

Runs synchronously, one command per issue. No scheduler/cron by design --
that would conflict with the "Zero Unattended Operations" safety rule,
since a human has to be present to approve step 9 anyway.

Usage:
    python oss_agent_v2.py --repo owner/repo --issue 42        # solve -> draft PR
    python oss_agent_v2.py --repo owner/repo --issue 42 -conversation
    python oss_agent_v2.py --leave 42                          # park it, keep everything
    python oss_agent_v2.py --resume 42                         # pick it back up
    python oss_agent_v2.py --list-workflows                    # what am I tracking?
"""

import argparse
import difflib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from github import Auth, Github
from openai import OpenAI  # used to talk to OmniRoute's OpenAI-compatible endpoint

try:
    # PyGithub raises this on 403/429 secondary-rate-limit responses.
    from github import RateLimitExceededException
except ImportError:  # older/newer PyGithub layouts -- degrade gracefully

    class RateLimitExceededException(Exception):
        pass


load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def _git_auth_args() -> list:
    """Auth args for git-over-HTTPS commands (clone/push/fetch).

    The token is NOT embedded in the remote URL: that persists it in the clone's
    .git/config and splashes it across `git remote -v`, transcripts and backups.
    Instead the same basic-auth credential is sent per-invocation through
    `http.extraheader`, which git uses for HTTPS regardless of the URL, and
    which leaves nothing behind on disk. `x-access-token:` is the username
    GitHub accepts for PAT auth; the token is the password.
    """
    if not GITHUB_TOKEN:
        return []
    import base64

    token_b64 = base64.b64encode(f"x-access-token:{GITHUB_TOKEN}".encode()).decode()
    return ["-c", f"http.extraheader=AUTHORIZATION: basic {token_b64}"]
# OmniRoute runs locally and exposes an OpenAI-compatible endpoint that
# internally rotates across whatever providers you connected in its
# dashboard (Gemini, DeepSeek, etc.) with automatic fallback on
# quota/rate-limit -- this replaces the manual MODEL_CHAIN we had before.
OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
# The auto-combo call_model() uses, named once so the workflow record and the
# conversation metadata can report which model produced a transcript.
#
# DEFAULTS ARE HARD-CODED to OmniRoute virtual combos (auto/*) that resolve
# against whatever providers are connected in the gateway's dashboard. These
# are the SAME defaults that run the local machine, and the fix for 7f02810e:
# a stale OMNIROUTE_MODEL env value (e.g. an OpenRouter slug routed to a
# provider with no credentials) silently won over these and crashed every lane
# with "No active credentials for provider".
#
# Because some CLOUD deployments point OMNIROUTE_BASE_URL at a plain
# OpenAI-compatible endpoint that does not understand auto/* combos, each value
# here may be overridden through the environment -- but ONLY when the env var
# is set to a NON-EMPTY value. An empty/unset variable (which is what a stale
# secret, missing secret, or the '' manual.env comment-out yields) keeps the
# hard-coded auto/* default, so the old failure mode cannot resurrect itself
# silently: the only way the chain changes is a deliberate, non-empty override.
OMNIROUTE_MODEL = os.getenv("OMNIROUTE_MODEL", "").strip() or "auto/best-coding"
# Fallback combos tried (in order) when a combo fails hard INSTEAD of timing
# out. OmniRoute returns "Maximum combo retry limit reached" (503) when every
# model inside a combo has failed; its own recovery hint is to switch combos,
# so we walk this list rather than dying. Comma-separated when read from the
# environment (cloud secret), defaults to the auto/* chain when empty.
OMNIROUTE_MODEL_FALLBACKS = [
    m.strip() for m in os.getenv("OMNIROUTE_MODEL_FALLBACKS", "").split(",") if m.strip()
] or ["auto/best-chat", "auto/best-reasoning", "auto/best-fast"]
# Seconds before an OmniRoute call times out instead of hanging forever.
# A silent, provider-less gateway (TCP open, never responding) used to make
# every run freeze at its first verify/generate step with no visible error.
# Kept generous (600s): the default "auto/coding" combo is a reasoning-tier
# combo and a large generate_fix prompt + 4000 output tokens can legitimately
# exceed a 2-minute budget; call_model() also retries with a longer timeout
# before giving up, so this is the floor, not the ceiling.
OMNIROUTE_TIMEOUT = float(os.getenv("OMNIROUTE_TIMEOUT", "600"))
# Cheap steps (issue classification, AI file pre-selection) do not need the
# reasoning-tier combo -- they are short, structured, tolerance-heavy decisions.
# Routing them through a fast auto-combo makes the front of every run cheaper.
# Env-overridable exactly like the coding chain above, and only when non-empty;
# empty/unset keeps the auto/* default (see OMNIROUTE_MODEL for the rationale).
OMNIROUTE_FAST_MODEL = os.getenv("OMNIROUTE_FAST_MODEL", "").strip() or "auto/best-chat"
OMNIROUTE_FAST_MODEL_FALLBACKS = [
    m.strip() for m in os.getenv("OMNIROUTE_FAST_MODEL_FALLBACKS", "").split(",") if m.strip()
] or ["auto/best-coding", "auto/best-fast"]
# Stall guard for the reasoning combo: cap the FIRST attempt of each combo at
# this many seconds. A dead/silent provider is abandoned after ~this instead
# of burning the full budget, and the next fallback combo gets the same quick
# probe. Only when every combo has stalled does the LAST combo stretch to the
# full budget (so a slow-but-healthy reasoning model is not starved).
OMNIROUTE_STALL_SECONDS = float(os.getenv("OMNIROUTE_STALL_SECONDS", "90"))

# ============================================================
# Fallback endpoint (opt-in hosted OpenAI-compatible LLM)
# ============================================================
# omniRoute is the PRIMARY endpoint everywhere: local runs resolve it on
# localhost, and a cloud deployment can reach it through a tunnel. But if
# omniRoute is down (PC off, gateway stopped) a cloud run would otherwise
# hard-fail every step -- so call_model() falls back to this hosted endpoint
# when the primary is unreachable OR every primary combo fails hard.
#
# The fallback is DISABLED by default (all envs empty) so local behavior is
# byte-for-byte unchanged. Configure it only where you want outage tolerance
# (e.g. GitHub Actions secrets). Each setting is only consulted when non-empty,
# matching the deliberate non-empty-override rule of the model chain above.
OMNIROUTE_FALLBACK_BASE_URL = os.getenv("OMNIROUTE_FALLBACK_BASE_URL", "").strip()
# API key for the fallback endpoint. Defaults to the primary key (many hosted
# gateways share one key); set separately when the endpoints have different auth.
OMNIROUTE_FALLBACK_API_KEY = os.getenv("OMNIROUTE_FALLBACK_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
# Model chain for the fallback endpoint: a hosted OpenAI-compatible endpoint
# typically does NOT understand omniRoute's auto/* virtual combos, so the
# fallback chain must be concrete model names (e.g. "gemini-3.6-flash").
# Empty/unset keeps the primary chain so the fallback still works for the
# common case where the hosted endpoint mirrors omniRoute's model names.
OMNIROUTE_FALLBACK_MODEL = os.getenv("OMNIROUTE_FALLBACK_MODEL", "").strip()
OMNIROUTE_FALLBACK_MODEL_FALLBACKS = [
    m.strip() for m in os.getenv("OMNIROUTE_FALLBACK_MODEL_FALLBACKS", "").split(",") if m.strip()
]
OMNIROUTE_FALLBACK_FAST_MODEL = os.getenv("OMNIROUTE_FALLBACK_FAST_MODEL", "").strip()
OMNIROUTE_FALLBACK_FAST_MODEL_FALLBACKS = [
    m.strip() for m in os.getenv("OMNIROUTE_FALLBACK_FAST_MODEL_FALLBACKS", "").split(",") if m.strip()
]
# Seconds a primary-endpoint probe may take before we declare omniRoute down.
# Kept small: this probe runs once at startup, not per call.
OMNIROUTE_FALLBACK_PROBE_SECONDS = float(os.getenv("OMNIROUTE_FALLBACK_PROBE_SECONDS", "5"))

# System prompt attached to every model call. It steers BOTH the fix content
# and the prose around it (PR bodies, comments, commit messages) towards a
# plain human engineering note instead of machine-authored filler. Keep it
# short: it is a style guardrail, not a task spec -- the task lives in the
# per-call user prompt.
SYSTEM_PROMPT = (
    "You are a senior software developer finishing a real fix, not an "
    "AI assistant demonstrating itself.\n"
    "Write like a developer would: direct, concrete, no filler, no "
    "self-reference ('I', 'as an AI', 'hope this helps', 'feel free to'), "
    "no exclamation marks, no canned praise, no summary-of-what-I-did "
    "preamble. State the change and, only when it is non-obvious, why.\n"
    "Code, comments, commit messages and PR text must contain only what is "
    "necessary: no boilerplate placeholders, no TO-DO theater, no "
    "re-stating the issue title, and no line that exists purely to sound "
    "helpful. If a caller could not reasonably need a line, drop it."
)

# Second system prompt: the SDE-2 review discipline. It applies BEFORE the PR
# is created -- self-review the change, run what validation is available, and
# do not hand over something that is knowingly broken. It is advisory framing
# for the model; the workflow/operator still approve and publish the PR.
SDE2_SYSTEM_PROMPT = (
    "You are an SDE-2 engineer. Before creating a PR, review the complete "
    "change and relevant codebase.\n\n"
    "* Understand architecture, dependencies, data/control flow, and existing "
    "conventions.\n"
    "* Review the full diff and all affected code paths.\n"
    "* Check correctness, requirements, edge cases, errors, regressions, "
    "compatibility, security, performance, maintainability, and tests.\n"
    "* Run relevant tests, lint, type checks, and builds when available.\n"
    "* Fix issues you find, then re-run validation.\n"
    "* If anything fails or the implementation is NOT READY, do not stop: "
    "diagnose the root cause, fix it, and validate again.\n"
    "* Repeat the review -> fix -> test cycle until no significant issues "
    "remain.\n"
    "* Re-check the complete change after every meaningful fix to catch "
    "regressions.\n"
    "* Do not commit, push, or create the PR unless explicitly instructed.\n\n"
    "Final report:\n"
    "* Issues fixed\n"
    "* Tests/checks run and results\n"
    "* Remaining concerns\n"
    "* Final status: `READY`, `READY WITH NOTES`, or `NOT READY`\n\n"
    "Never mark `READY` while known blocking errors or failing relevant "
    "checks remain."
)

# All generated/runtime data lives under one folder, kept separate from
# the code itself -- clones, logs, and state never mix with the script.
AGENT_HOME = Path(".agent_data")
WORKSPACE = AGENT_HOME / "workspace"
LOGS_DIR = AGENT_HOME / "logs"
# Read-only investigation reports (--analyze). Kept under AGENT_HOME like every
# other piece of generated data; never mixed with workflow/PR state.
REPORTS_DIR = AGENT_HOME / "reports"
STATE_FILE = AGENT_HOME / "state.json"
EXPERIENCE_LOG = AGENT_HOME / "experience_log.jsonl"
# One persisted workflow record per (repo, issue). A draft PR is NOT the end
# of the job -- the record keeps the review/iteration loop resumable across
# separate process invocations (this CLI runs once per invocation by design).
WORKFLOWS_DIR = AGENT_HOME / "workflows"

# Local, dependency-free session store: paths, the (repo, issue) index,
# conversation transcripts, and workspace-ownership safety. Imported as `store`
# so this file stays the CLI/orchestrator. It holds no GitHub or LLM client,
# which is what makes --leave/--status/--list-workflows structurally incapable
# of changing anything remote.
import session_store as store  # noqa: E402

store.configure(AGENT_HOME)

# FileLock and _atomic_write_json below are thin wrappers over the store's
# implementations, keeping the names this file and the test suites already use
# while guaranteeing there is only one lock and one atomic write in the system.

# Opt out of the one-PR-per-repo rule when you genuinely want several open at
# once in the same repo. The per-DAY cap still applies -- this widens
# parallelism, it does not remove the brakes. Parallel work across DIFFERENT
# repos never needed this.
ALLOW_MULTI_PR_SAME_REPO = os.getenv("ALLOW_MULTI_PR_SAME_REPO", "").strip().lower() in (
    "1", "true", "yes", "on",
)

for _dir in (WORKSPACE, LOGS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- NFR constants, taken directly from the PRD safety section ---
MAX_ATTEMPTS = 5
TEST_TIMEOUT_SECONDS = 120
MAX_PRS_PER_DAY = 2
MAX_ACTIVE_PRS_PER_REPO = 1

# Every fix must ship a test that would fail without it. Free: the test is
# requested in the same generate_fix call, and the evidence comes from the diff
# plus the suite run that already happens. Set False to accept untested fixes.
REQUIRE_REGRESSION_TEST = True
# Strict proof: revert the code half, re-run, confirm the new test goes RED.
# Off by default because it costs one extra full suite run per attempt.
VERIFY_TEST_FAILS_WITHOUT_FIX = False

# --- Difficulty-aware escalation -----------------------------------------
# A HARD issue (AI/ML math, cross-cutting integrations, deep refactors, large
# frontend tables) gets more attempts, more context files and bigger file
# slices than a typo. This is the "solve higher problems smoothly" knob: the
# agent raises its OWN budget from the classification instead of dying inside
# the flat 5-attempt box.
MAX_HARD_ATTEMPTS = int(os.getenv("MAX_HARD_ATTEMPTS", "8"))
HARD_CONTEXT_FILES = int(os.getenv("HARD_CONTEXT_FILES", "10"))
HARD_FILE_CHARS = int(os.getenv("HARD_FILE_CHARS", "24000"))
# Hard tasks also get a bigger answer budget: a multi-file patch can exceed
# the 4000-token default cap and truncate, which then fails the format parser
# and costs a full retry. 8000 tokens covers most cross-cutting patch shapes.
HARD_MAX_TOKENS = int(os.getenv("HARD_MAX_TOKENS", "8000"))
# "enhancement"-labelled issues are accepted when the model judges them SCOPED
# or MODERATE (a contained improvement to existing code: a new option, a
# missing code path, an extra API field + test, one new UI element in an
# existing screen). BROAD/architectural features stay rejected -- that keeps
# the old "not a feature factory" rule while actually serving enhancement
# labels. Set to "0"/"false" to restore pure-bugs-only behaviour.
ACCEPT_SCOPED_ENHANCEMENTS = (
    os.getenv("ACCEPT_SCOPED_ENHANCEMENTS", "1").strip().lower()
    in ("1", "true", "yes", "on")
)

SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build"}
CODE_EXTENSIONS = {
    # Programming languages
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rb",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".vue",
    ".svelte",
    # Styles
    ".css",
    ".scss",
    ".sass",
    ".less",
    # Markup / templates
    ".html",
    ".htm",
    ".xml",
    ".ejs",
    ".hbs",
    ".pug",
    # Config / data (often where bugs actually live -- build config, schemas)
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".env.example",
}
MAX_CONTEXT_FILES = 6
MAX_FILE_CHARS = 18000
# Raised from 4000 after repeated diff-application failures (blank-line
# prefixes, hunk math, cross-contamination between multiple diff blocks
# in one response). Diffs are more fragile in practice than full-file
# rewrites, so the priority is to avoid needing them at all -- most
# individual source files fit comfortably under 12000 chars, so this
# keeps far more fixes on the reliable full-file path.
# Separate, larger limit specifically for surfacing a file's REAL content
# when correcting a hallucinated guess (see apply_fix) -- truncating this
# too aggressively defeats the purpose, since the relevant section (e.g.
# line 133 of a long stylesheet) can easily fall past the normal limit.
SURFACE_CONTENT_CHARS = 16000
MAX_ERROR_CHARS = 1500


class _LazyClient:
    """Defers construction of an API client until first use.

    Building the GitHub/OpenAI clients at import time forced a valid token
    just to `import oss_agent_v2` -- which blocked unit tests and any tooling
    that merely imports the module. This proxy constructs the real client on
    first attribute access instead, so importing is side-effect-free while
    every existing `gh.foo()` / `ai_client.bar()` call site keeps working
    unchanged. Tests simply monkeypatch the module attribute with a fake."""

    __slots__ = ("_factory", "_obj")

    def __init__(self, factory):
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_obj", None)

    def _resolve(self):
        if object.__getattribute__(self, "_obj") is None:
            object.__setattr__(self, "_obj", object.__getattribute__(self, "_factory")())
        return object.__getattribute__(self, "_obj")

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


gh = _LazyClient(lambda: Github(auth=Auth.Token(GITHUB_TOKEN)))
# API key for the omniRoute PRIMARY endpoint. The OpenAI client requires a
# non-empty string; omniRoute itself validates it against its api_keys store
# only when REQUIRE_API_KEY is enabled. Dedicated OMNIROUTE_API_KEY keeps the
# omniRoute token separate from LLM_API_KEY (the hosted fallback key), so a
# local PC with auth enforced keeps working while the fallback stays keyed by
# LLM_API_KEY. Unset -> falls back to LLM_API_KEY for backward compatibility.
LLM_API_KEY = os.getenv("LLM_API_KEY", "omniroute-local")
OMNIROUTE_API_KEY = os.getenv("OMNIROUTE_API_KEY", "").strip() or LLM_API_KEY


def _endpoint_is_reachable(base_url: str, api_key: str, timeout: float = OMNIROUTE_FALLBACK_PROBE_SECONDS) -> bool:
    """Cheap probe: does an OpenAI-compatible server answer at base_url?

    Uses the OpenAI SDK client so the same auth/TLS path the real calls use,
    but only lists models (no token spend, no provider involvement). A
    reachable endpoint answers even with no/bogus credentials (omniRoute
    returns 200 without auth; hosted gateways return 200/401 both of which
    mean "a server is there"). Network errors/timeouts -> not reachable.
    """
    if not base_url:
        return False
    try:
        client = OpenAI(api_key=api_key or "omniroute-local", base_url=base_url)
        client.models.list(timeout=timeout)
        return True
    except Exception:
        return False


def _primary_unreachable() -> bool:
    """True while a live omniRoute cannot be reached.

    Probes are cached for OMNIROUTE_FALLBACK_PROBE_SECONDS so a healthy run
    never wastes time probing before every call; a down gateway is re-probed
    after that window so a fixer session lasting many minutes can pick the
    gateway back up if it recovers mid-run. Each probe is a short models.list
    call, not a model inference."""
    now = time.monotonic()
    last = getattr(_primary_unreachable, "_t", 0.0)
    if now - last >= OMNIROUTE_FALLBACK_PROBE_SECONDS:
        _primary_unreachable._t = now
        _primary_unreachable._down = not _endpoint_is_reachable(OMNIROUTE_BASE_URL, OMNIROUTE_API_KEY)
    return getattr(_primary_unreachable, "_down", True)


def _make_ai_client():
    """Primary is omniRoute; fall back to a hosted endpoint only when both
    configured AND omniRoute cannot be reached. This keeps every local run on
    the free local gateway (~890 models, auto/* combos) while letting a cloud
    deployment survive an offline PC."""
    if OMNIROUTE_FALLBACK_BASE_URL and _primary_unreachable():
        print(f"WARNING: omniRoute unreachable at {OMNIROUTE_BASE_URL}; "
              f"falling back to hosted endpoint {OMNIROUTE_FALLBACK_BASE_URL}")
        return OpenAI(api_key=OMNIROUTE_FALLBACK_API_KEY or "omniroute-local",
                      base_url=OMNIROUTE_FALLBACK_BASE_URL)
    return OpenAI(api_key=OMNIROUTE_API_KEY, base_url=OMNIROUTE_BASE_URL)


ai_client = _LazyClient(_make_ai_client)


# ============================================================
# State tracking (fixes the missing PR-limit enforcement)
# ============================================================
def _default_state():
    return {"date": str(date.today()), "prs_today": 0, "active_prs_by_repo": {}}


class FileLock(store.FileLock):
    """Cross-platform advisory lock via an atomic O_CREAT|O_EXCL lockfile.

    fcntl is POSIX-only and msvcrt is Windows-only; an exclusive-create
    lockfile behaves identically on both, which matters because this agent
    runs on Windows. A stale lock left by a crashed run (older than
    `stale_after` seconds) is broken automatically so the agent can never
    permanently wedge itself.

    The implementation now lives in session_store so the store and the agent
    cannot drift into two subtly different locks over the same files; this
    subclass keeps the historical name and signature."""


def _state_lock():
    return FileLock(STATE_FILE)


def _migrate_state(raw):
    """Normalise any older/partial/corrupt-but-parseable schema to the
    current shape so downstream key access can never KeyError."""
    base = _default_state()
    if isinstance(raw, dict):
        if isinstance(raw.get("date"), str):
            base["date"] = raw["date"]
        if isinstance(raw.get("prs_today"), int):
            base["prs_today"] = raw["prs_today"]
        if isinstance(raw.get("active_prs_by_repo"), dict):
            base["active_prs_by_repo"] = {
                k: v for k, v in raw["active_prs_by_repo"].items() if isinstance(v, int)
            }
    return base


def load_state():
    """Corruption-tolerant load. A truncated/garbage file (e.g. a run killed
    mid-write before atomic saves existed) is preserved as *.corrupt for
    forensics, then we fall back to a fresh default instead of crashing every
    future run."""
    if not STATE_FILE.exists():
        return _default_state()
    try:
        return _migrate_state(json.loads(STATE_FILE.read_text()))
    except (json.JSONDecodeError, ValueError, OSError):
        try:
            STATE_FILE.replace(STATE_FILE.with_suffix(".corrupt"))
        except OSError:
            pass
        return _default_state()


def _atomic_write_json(path, data) -> None:
    """Atomic write: unique temp file in the same dir -> fsync -> os.replace.
    A crash mid-write can no longer leave a half-written file behind. Shared
    by save_state and the workflow-record persistence below.

    Delegates to session_store.atomic_write_json -- one implementation, so the
    store's index/conversation files and this file's state get identical
    crash-safety guarantees."""
    store.atomic_write_json(path, data)


def save_state(state):
    """Atomic write of the daily PR-limit state (see _atomic_write_json)."""
    _atomic_write_json(STATE_FILE, state)


def get_active_pr_count(repo_name):
    """How many OPEN PRs the authenticated user actually has on `repo_name`.

    The local counter is only ever incremented (on PR creation), never
    decremented, so it goes stale the instant a PR is merged/closed/deleted.
    Reconciling against GitHub -- the source of truth -- prevents a stale
    counter from blocking the repo forever. Returns None on any transient
    failure (rate limit, network) so the caller falls back to local state
    instead of crashing the run."""
    try:
        login = gh.get_user().login
        return gh.search_issues(f"repo:{repo_name} is:pr is:open author:{login}").totalCount
    except RateLimitExceededException:
        print(f"⚠️  Rate-limited verifying PRs on {repo_name}; using local state.")
        return None
    except Exception as e:
        print(f"⚠️  Could not verify active PRs on {repo_name} ({e}); using local state.")
        return None


def _apply_daily_rollover(state):
    if state["date"] != str(date.today()):
        state["date"] = str(date.today())
        state["prs_today"] = 0


def check_pr_limits(state, repo_name):
    """Startup gate. The whole read-modify-write runs under a file lock so a
    second concurrent run can't slip between our check and our self-heal
    write. Fast path: only spend a (slow, rate-limited) GitHub search when the
    local counter already says we're blocked -- then confirm before refusing.
    The caller's `state` dict is resynced to the authoritative on-disk copy."""
    with _state_lock():
        fresh = load_state()
        _apply_daily_rollover(fresh)
        if fresh["prs_today"] >= MAX_PRS_PER_DAY:
            save_state(fresh)
            state.clear()
            state.update(fresh)
            raise RuntimeError(
                f"Daily PR limit reached ({MAX_PRS_PER_DAY}/day). Try again tomorrow."
            )

        active = fresh["active_prs_by_repo"].get(repo_name, 0)
        if active >= _active_pr_cap() and not ALLOW_MULTI_PR_SAME_REPO:
            actual = get_active_pr_count(repo_name)  # confirm the block is real
            if actual is not None:
                fresh["active_prs_by_repo"][repo_name] = actual
                active = actual
        save_state(fresh)
        state.clear()
        state.update(fresh)
        if active >= _active_pr_cap():
            raise RuntimeError(
                f"Already have an active PR on {repo_name}. Wait for it to close/merge.\n"
                f"   Working on several issues at once is supported -- use DIFFERENT "
                f"repos, which needs no override.\n"
                f"   To allow {MAX_ACTIVE_PRS_PER_REPO}+ open PRs in this same repo, set "
                f"ALLOW_MULTI_PR_SAME_REPO=1 (the {MAX_PRS_PER_DAY}/day cap still applies)."
            )


def _active_pr_cap():
    """The per-repo cap in force. MAX_ACTIVE_PRS_PER_REPO stays the default;
    ALLOW_MULTI_PR_SAME_REPO=1 lifts it for people who deliberately want several
    PRs open in one repo. The daily cap is never lifted -- that is the brake that
    stops a runaway loop spamming maintainers."""
    return float("inf") if ALLOW_MULTI_PR_SAME_REPO else MAX_ACTIVE_PRS_PER_REPO


def record_pr_created(repo_name):
    """Call immediately AFTER a PR is created. Re-reads the freshest on-disk
    state under lock and increments both counters atomically -- closing the
    (potentially minutes-long) TOCTOU window between the startup check and the
    actual PR creation, and guaranteeing concurrent runs can't lose an
    increment. Returns the updated state."""
    with _state_lock():
        fresh = load_state()
        _apply_daily_rollover(fresh)
        fresh["prs_today"] += 1
        fresh["active_prs_by_repo"][repo_name] = fresh["active_prs_by_repo"].get(repo_name, 0) + 1
        save_state(fresh)
        return fresh


# ============================================================
# Iterative-workflow state machine + resumable, auditable record.
#
# A draft PR is the START of a review/iteration cycle, not the end.
# Because this CLI runs once per invocation, "staying alive to wait for
# feedback" is implemented as a persisted, resumable record: each run
# picks up the workflow where the last one left off, processes any new
# maintainer/user feedback, updates the PR, and persists again.
# ============================================================
class WF:
    """Explicit workflow states (string constants -> JSON-friendly)."""

    TASK_RECEIVED = "TASK_RECEIVED"
    ANALYZING = "ANALYZING"
    IMPLEMENTING = "IMPLEMENTING"
    TESTING = "TESTING"
    DRAFT_PR_CREATED = "DRAFT_PR_CREATED"
    WAITING_FOR_FEEDBACK = "WAITING_FOR_FEEDBACK"
    PROCESS_FEEDBACK = "PROCESS_FEEDBACK"
    OPTIMIZING = "OPTIMIZING"
    UPDATE_PR = "UPDATE_PR"
    READY_FOR_HUMAN_APPROVAL = "READY_FOR_HUMAN_APPROVAL"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    FINAL_VALIDATION = "FINAL_VALIDATION"
    COMMIT = "COMMIT"
    SIGN_OFF = "SIGN_OFF"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"
    # Parked by --leave (or by a crash/Ctrl+C). A real state, not a flag, so the
    # transition guard protects it like any other: nothing may move a paused
    # workflow forward without an explicit resume. LEAVING and RESUMING from the
    # spec are moments rather than resting places -- they are recorded as
    # `last_transition_marker` and in the transcript, because a state you can
    # enter and never leave is how a state machine deadlocks.
    PAUSED = store.PAUSED


# Which state may follow which. Guards against illegal jumps (e.g. going
# straight from WAITING_FOR_FEEDBACK to COMPLETED without the approval gate).
_WF_TRANSITIONS = {
    WF.TASK_RECEIVED: {WF.ANALYZING, WF.ABANDONED},
    WF.ANALYZING: {WF.IMPLEMENTING, WF.ABANDONED},
    WF.IMPLEMENTING: {WF.TESTING, WF.WAITING_FOR_FEEDBACK, WF.ABANDONED},
    WF.TESTING: {WF.DRAFT_PR_CREATED, WF.OPTIMIZING, WF.IMPLEMENTING, WF.ABANDONED},
    WF.DRAFT_PR_CREATED: {WF.WAITING_FOR_FEEDBACK, WF.ABANDONED},
    WF.WAITING_FOR_FEEDBACK: {
        WF.PROCESS_FEEDBACK,
        WF.READY_FOR_HUMAN_APPROVAL,
        WF.ABANDONED,
    },
    WF.PROCESS_FEEDBACK: {WF.IMPLEMENTING, WF.OPTIMIZING, WF.WAITING_FOR_FEEDBACK, WF.ABANDONED},
    WF.OPTIMIZING: {WF.UPDATE_PR, WF.IMPLEMENTING, WF.ABANDONED},
    WF.UPDATE_PR: {WF.WAITING_FOR_FEEDBACK, WF.ABANDONED},
    WF.READY_FOR_HUMAN_APPROVAL: {WF.HUMAN_APPROVED, WF.WAITING_FOR_FEEDBACK, WF.ABANDONED},
    WF.HUMAN_APPROVED: {WF.FINAL_VALIDATION, WF.ABANDONED},
    WF.FINAL_VALIDATION: {WF.COMMIT, WF.WAITING_FOR_FEEDBACK, WF.ABANDONED},
    WF.COMMIT: {WF.SIGN_OFF, WF.ABANDONED},
    WF.SIGN_OFF: {WF.COMPLETED, WF.ABANDONED},
    WF.COMPLETED: set(),
    WF.ABANDONED: set(),
}

# States behind the mandatory human gate. Named once, used twice: a paused
# workflow may not resume INTO one of them (you cannot pause your way past a
# gate you never passed), and _WF_WALKABLE below excludes them too.
_WF_GATED = {
    WF.READY_FOR_HUMAN_APPROVAL,
    WF.HUMAN_APPROVED,
    WF.FINAL_VALIDATION,
    WF.COMMIT,
    WF.SIGN_OFF,
}

# --leave parks ANY non-terminal state; resuming returns to where it paused.
# Wired programmatically rather than typed into each entry above so a state
# added later cannot silently become un-pausable. PAUSED is deliberately kept
# out of _WF_WALKABLE, so no automatic repair/bookkeeping walk can park a
# workflow -- only an explicit --leave (or the crash handler) may.
_WF_PAUSABLE = {s for s in _WF_TRANSITIONS if s not in (WF.COMPLETED, WF.ABANDONED)}
for _src in _WF_PAUSABLE:
    _WF_TRANSITIONS[_src].add(WF.PAUSED)
_WF_TRANSITIONS[WF.PAUSED] = (_WF_PAUSABLE - _WF_GATED) | {WF.ABANDONED}


def wf_can_transition(src: str, dst: str) -> bool:
    return dst in _WF_TRANSITIONS.get(src, set())


def wf_advance(record: dict, dst: str, note: str = "") -> dict:
    """Move the workflow to `dst`, validating the transition, timestamping it,
    and persisting. Raises RuntimeError on an illegal transition so a logic bug
    can't silently skip the human-approval gate."""
    src = record.get("state", WF.TASK_RECEIVED)
    if src == dst:
        return record
    if not wf_can_transition(src, dst):
        raise RuntimeError(f"Illegal workflow transition {src} -> {dst}")
    record["state"] = dst
    record.setdefault("transitions", []).append(
        {"from": src, "to": dst, "at": datetime.now().isoformat(timespec="seconds"), "note": note}
    )
    record["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_workflow(record)
    return record


# States a tolerant walk/repair is allowed to enter. Everything from
# READY_FOR_HUMAN_APPROVAL onwards is deliberately EXCLUDED: those states are
# behind the mandatory human gate, and no automatic bookkeeping helper may
# slide into them.
_WF_WALKABLE = {
    WF.TASK_RECEIVED,
    WF.ANALYZING,
    WF.IMPLEMENTING,
    WF.TESTING,
    WF.DRAFT_PR_CREATED,
    WF.WAITING_FOR_FEEDBACK,
    WF.PROCESS_FEEDBACK,
    WF.OPTIMIZING,
    WF.UPDATE_PR,
}


def wf_walk(record: dict, chain: list, note: str = "", allow_gated: bool = False) -> str:
    """Advance through `chain`, taking only the steps that are LEGAL from the
    current state and silently skipping the rest.

    This exists for bookkeeping that runs AFTER an irreversible side effect (a
    real draft PR on GitHub). A state-machine mismatch there must never raise
    and orphan the PR -- which is exactly what happened to Posnic/POS#52: the
    PR was created, the very next `wf_advance` was illegal, the process died,
    and the record stayed in ANALYZING with an open PR it no longer tracked.

    Gated states (human approval onwards) are rejected unless `allow_gated` is
    set, which only the "PR was merged upstream" path uses.
    """
    for dst in chain:
        if not allow_gated and dst not in _WF_WALKABLE:
            raise ValueError(f"wf_walk may not enter approval-gated state {dst}")
        if wf_can_transition(record.get("state", WF.TASK_RECEIVED), dst):
            wf_advance(record, dst, note)
    return record.get("state")


def _wf_path(src: str, dst: str) -> list:
    """Shortest legal path src -> dst using only _WF_WALKABLE states (BFS).
    Returns the states to enter in order, or [] if unreachable. Because the
    search space excludes the approval-gated states, no path it finds can
    bypass the human gate."""
    if src == dst:
        return []
    queue, seen = [(src, [])], {src}
    while queue:
        node, path = queue.pop(0)
        for nxt in _WF_TRANSITIONS.get(node, set()):
            if nxt in seen or nxt not in _WF_WALKABLE:
                continue
            if nxt == dst:
                return path + [nxt]
            seen.add(nxt)
            queue.append((nxt, path + [nxt]))
    return []


def wf_repair_for_iteration(record: dict) -> bool:
    """Self-heal a record that owns a real PR but whose state was left behind
    (crash between "PR created" and the state advance). Walks the shortest legal
    path to WAITING_FOR_FEEDBACK so a conversation round can proceed.

    Refuses to act on terminal records, and never invents a PR: with no
    `pr_number` there is nothing to repair. Returns True if it moved the state.
    """
    state = record.get("state", WF.TASK_RECEIVED)
    if state in (WF.WAITING_FOR_FEEDBACK, WF.COMPLETED, WF.ABANDONED):
        return False
    if not record.get("pr_number"):
        return False
    # Prefer a single legal step when one exists; otherwise route through
    # DRAFT_PR_CREATED, because a record that owns a PR should have passed
    # through it -- the repaired history then tells the truth about the PR.
    if wf_can_transition(state, WF.WAITING_FOR_FEEDBACK):
        path = [WF.WAITING_FOR_FEEDBACK]
    else:
        via = _wf_path(state, WF.DRAFT_PR_CREATED)
        path = (via + [WF.WAITING_FOR_FEEDBACK]) if via else _wf_path(
            state, WF.WAITING_FOR_FEEDBACK
        )
    if not path:
        return False
    for dst in path:
        wf_advance(record, dst, f"repair: PR #{record['pr_number']} exists, state was {state}")
    print(
        f"🔧 Repaired workflow state {state} -> {WF.WAITING_FOR_FEEDBACK} "
        f"(PR #{record['pr_number']} is already open)."
    )
    return True


def _sync_store_home() -> None:
    """Keep session_store pointed at the same tree as WORKFLOWS_DIR.

    WORKFLOWS_DIR is a module global that the test suites (and anyone relocating
    AGENT_HOME) patch to a temp directory. Syncing here means the store's index,
    conversations and workspaces follow it instead of quietly writing into the
    real .agent_data next to the code. In production the two are already equal,
    so this is a no-op."""
    home = Path(WORKFLOWS_DIR).parent
    if Path(store.AGENT_HOME) != home:
        store.configure(home)


def _workflow_path(repo_name: str, issue_number: int) -> Path:
    """Where a record is WRITTEN: workflows/<owner-repo>/issue-<N>.json.

    Nested per repo so a machine tracking dozens of workflows stays browsable,
    and derived from WORKFLOWS_DIR (not from the store) so patching that global
    keeps working."""
    _sync_store_home()
    return WORKFLOWS_DIR / store.safe_repo_dir(repo_name) / f"issue-{issue_number}.json"


def _legacy_workflow_path(repo_name: str, issue_number: int) -> Path:
    """The flat layout used before the session store existed. Still READ, so an
    upgrade mid-review never orphans an open PR; the next save moves the record
    to the nested path."""
    return WORKFLOWS_DIR / f"{store.safe_repo_dir(repo_name)}_issue{issue_number}.json"


def load_workflow(repo_name: str, issue_number: int):
    """Return the persisted workflow record for (repo, issue), or None. Tolerant
    of a corrupt file (backs it up to *.corrupt and returns None). Checks the
    legacy flat path as a fallback -- that check IS the migration."""
    for path in (_workflow_path(repo_name, issue_number),
                 _legacy_workflow_path(repo_name, issue_number)):
        if not path.exists():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError, OSError, UnicodeDecodeError):
            try:
                path.replace(path.with_suffix(".corrupt"))
                print(
                    f"⚠️  Corrupt workflow record {path.name} -> {path.stem}.corrupt\n"
                    f"    Nothing was lost on GitHub. Inspect any open PR with:\n"
                    f"      python oss_agent_v2.py --status {issue_number} "
                    f"--repo {repo_name}"
                )
            except OSError:
                pass
            return None
        if isinstance(record, dict):
            record.setdefault("repo", repo_name)
            record.setdefault("issue", issue_number)
            return _migrate_workflow_record(record)
    return None


def _migrate_workflow_record(record: dict) -> dict:
    """Fill in fields added by the session store so a record written by an older
    version is usable immediately, without a migration command. Only absent keys
    are touched -- existing values are never rewritten."""
    template = new_workflow(record["repo"], record["issue"], record.get("issue_title", ""))
    for key, default in template.items():
        record.setdefault(key, default)
    record["schema"] = store.SCHEMA_VERSION
    record["status"] = store.derive_status(record)
    return record


def save_workflow(record: dict) -> None:
    record["updated_at"] = datetime.now().isoformat(timespec="seconds")
    record["status"] = store.derive_status(record)
    _atomic_write_json(_workflow_path(record["repo"], record["issue"]), record)
    _retire_legacy_record(record)
    try:
        store.index_upsert(record)
    except (OSError, TimeoutError) as exc:
        # The index is a cache rebuilt from the records themselves, so a failure
        # here costs a lookup convenience, never the workflow.
        print(f"⚠️  Could not update the workflow index: {exc} (records are intact)")


def _retire_legacy_record(record: dict) -> None:
    """Once a record has been written to the nested path, rename any flat
    predecessor to *.migrated. Renamed rather than deleted: reversible by hand,
    and it stops the old file being re-read as a second, stale workflow."""
    old = _legacy_workflow_path(record["repo"], record["issue"])
    if not old.exists():
        return
    try:
        old.replace(old.with_suffix(".json.migrated"))
        print(f"📦 Migrated {old.name} -> {_workflow_path(record['repo'], record['issue']).name}")
    except OSError:
        pass


# --- the active session ---------------------------------------------------
# call_model() is a single-shot completion with no memory, which is exactly why
# "conversation state" used to evaporate between invocations. These two globals
# are the seam: whatever run is in progress attaches its conversation and record
# here, and every model turn is appended to a durable transcript on the way
# past. Nothing else about call_model changes -- the provider stays stateless,
# the DURABILITY is ours.
_active_conversation = None
_active_workflow = None


def attach_conversation(record: dict, provider="omniroute", model="auto/coding",
                        purpose="a run"):
    """Open (or reopen) this workflow's conversation and make it the active one.
    Returns (conversation, created). Purely local: no GitHub, no LLM.

    `purpose` names WHY the conversation was attached, because the transcript is
    what a person reads after a crash: a `--leave` that logged "resumed" would
    read as if work had continued when in fact it stopped."""
    global _active_conversation, _active_workflow
    _sync_store_home()
    conv, created = store.Conversation.open_or_create(
        record["repo"], record["issue"], provider=provider, model=model
    )
    _active_conversation, _active_workflow = conv, record
    record["conversation_id"] = conv.id
    record["conversation_dir"] = store.rel_to_home(conv.dir)
    record["provider"], record["model"] = provider, model
    conv.note(
        f"{'opened' if created else 'reopened'} for {purpose} at "
        f"state={record.get('state')} status={store.derive_status(record)} "
        f"pr={record.get('pr_number')}"
    )
    return conv, created


def detach_conversation() -> None:
    global _active_conversation, _active_workflow
    _active_conversation, _active_workflow = None, None


def _transcript(text: str) -> None:
    """One line into the active conversation's human transcript, if there is
    one. Silent no-op otherwise, so every call site can stay unconditional."""
    if _active_conversation is not None:
        _active_conversation.note(text)


def _log_turn(role: str, content: str, kind="message", **meta) -> None:
    if _active_conversation is not None:
        _active_conversation.append(role, content, kind=kind, meta=meta)


def new_workflow(repo_name: str, issue_number: int, issue_title: str) -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "schema": store.SCHEMA_VERSION,
        "repo": repo_name,
        "issue": issue_number,
        "issue_title": issue_title,
        "issue_url": "",
        "issue_state": "",
        "state": WF.TASK_RECEIVED,
        "status": "new",  # derived label, kept in sync by save_workflow
        "branch": None,
        "base_branch": None,
        "pr_number": None,
        "pr_url": None,
        "created_at": now,
        "updated_at": now,
        "processed_comment_ids": [],  # dedup: feedback we've already acted on
        "iterations": [],  # auditable history (see append_iteration)
        "unresolved_feedback": [],  # checklist items not yet satisfied
        "transitions": [],
        # --- session/resume fields (see session_store) ---
        "conversation_id": None,
        "conversation_dir": store.rel_to_home(store.conversation_dir(repo_name, issue_number)),
        "workspace": store.rel_to_home(store.workspace_dir(repo_name, issue_number)),
        "provider": "omniroute",
        "model": "",
        "last_test_status": "",
        "pause_count": 0,
        "resume_count": 0,
        "pause_history": [],
    }


def get_or_create_workflow(repo_name: str, issue_number: int, issue_title: str) -> dict:
    return load_workflow(repo_name, issue_number) or new_workflow(
        repo_name, issue_number, issue_title
    )


# --- pause / resume -------------------------------------------------------
# The two halves of "walk away and come back". Both are LOCAL ONLY: they read
# and write files under .agent_data and nothing else. No GitHub call, no push,
# no branch switch, no cleanup -- which is the entire safety promise of --leave.
TERMINAL_STATES = (WF.COMPLETED, WF.ABANDONED)


def pause_workflow(record: dict, reason: str = "", trigger: str = "--leave") -> bool:
    """Persist enough context to resume later, then park the record in PAUSED.

    Returns False (and changes nothing) for a terminal workflow -- there is no
    such thing as pausing something already finished or abandoned."""
    if record.get("state") in TERMINAL_STATES:
        return False
    _sync_store_home()
    payload = store.pause_payload(
        record,
        reason=reason,
        conversation=_active_conversation,
        workspace=store.home_to_abs(record["workspace"]) if record.get("workspace") else None,
        provider=record.get("provider", ""),
        model=record.get("model", ""),
        trigger=trigger,
    )
    previous = record.get("state")
    record.update({k: v for k, v in payload.items() if k != "state"})
    _transcript(
        f"PAUSED from {previous} via {trigger}: {payload['paused_reason']} "
        f"(dirty={payload['workspace_dirty']}, branch={payload.get('branch')})"
    )
    if previous == store.PAUSED:
        save_workflow(record)          # re-pausing an already-paused record
    else:
        wf_advance(record, WF.PAUSED, payload["paused_reason"])
    return True


def resume_workflow_record(record: dict, note: str = "resumed") -> str:
    """Bring a PAUSED record back to its safe resume point. A no-op (returning
    the current state) for anything not paused, so callers need not check."""
    if record.get("state") != store.PAUSED:
        return record.get("state", WF.TASK_RECEIVED)
    payload = store.resume_payload(record)
    target = payload["state"]
    record.update({k: v for k, v in payload.items() if k != "state"})
    # PAUSED -> target is legal for every non-gated state by construction, but a
    # hand-edited record could name anything, so fall back rather than crash.
    if wf_can_transition(store.PAUSED, target):
        wf_advance(record, target, note)
    else:
        print(f"⚠️  {target} is not resumable from PAUSED; going back to ANALYZING instead.")
        wf_advance(record, WF.ANALYZING, f"{note} (fallback: {target} unreachable)")
    _transcript(f"RESUMING into {record['state']} ({note})")
    print(f"↩️  Resumed {record['repo']}#{record['issue']}: PAUSED -> {record['state']}")
    return record["state"]


def append_iteration(record: dict, iteration: dict) -> dict:
    """Append one auditable iteration to the history. Records exactly what the
    prompt asked for: feedback received, action taken, tests added/failed, what
    was fixed, what remains unresolved, and the resulting PR state."""
    iteration = {
        "n": len(record["iterations"]) + 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trigger": iteration.get("trigger", "feedback"),
        "feedback": iteration.get("feedback", []),
        "plan": iteration.get("plan", ""),
        "checklist": iteration.get("checklist", []),
        "action_taken": iteration.get("action_taken", ""),
        "tests_added": iteration.get("tests_added", []),
        "tests_result": iteration.get("tests_result", ""),
        "fixed": iteration.get("fixed", ""),
        "unresolved": iteration.get("unresolved", []),
        "pr_state": iteration.get("pr_state", record.get("state")),
    }
    record["iterations"].append(iteration)
    record["unresolved_feedback"] = iteration["unresolved"]
    save_workflow(record)
    return record


# --- Completion / approval signal detection -------------------------------
# The workflow must NOT infer completion from "a PR exists" or "tests pass".
# It closes only on an EXPLICIT signal. We deliberately distinguish a USER
# completion signal (ends the iteration loop) from a maintainer APPROVAL
# signal (satisfies the human-approval gate) -- they are not equivalent.
_USER_COMPLETION_SIGNALS = ("done", "finish", "finished", "complete", "completed", "exit")
_APPROVAL_SIGNALS = ("approved", "approve", "lgtm", "ship it", "shipit")


def classify_signal(text: str):
    """Return 'completion', 'approval', or None for a piece of feedback text.
    Word-boundary matched so 'abandoned' does not match 'done' and 'incomplete'
    does not match 'complete'."""
    if not text:
        return None
    low = text.lower()
    for phrase in _APPROVAL_SIGNALS:
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", low):
            return "approval"
    for phrase in _USER_COMPLETION_SIGNALS:
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", low):
            return "completion"
    return None


_REVIEW_APPROVED = "APPROVED"
_REVIEW_CHANGES_REQUESTED = "CHANGES_REQUESTED"


def signal_for_item(item: dict):
    """The signal carried by ONE feedback item, review verdicts included.

    A review submitted as APPROVED counts as approval even with an empty body --
    the common "just click Approve" case, which pure text matching misses. A
    CHANGES_REQUESTED review is never an approval however it is worded, so
    "lgtm apart from the naming" cannot finalize anything."""
    if item.get("kind") == "review":
        state = (item.get("state") or "").upper()
        if state == _REVIEW_CHANGES_REQUESTED:
            return None
        if state == _REVIEW_APPROVED:
            return "approval"
    return classify_signal(item.get("body", ""))


def arm_finalize(record: dict, reason: str) -> None:
    """Remember that an explicit finalize signal was seen, and why.

    Persisted deliberately: a conversation round only ever sees NEW comments, so
    an "approved" that arrived while unresolved work remained would be marked
    processed and then lost forever, leaving the workflow parked with nothing
    left to trigger the gate."""
    record["pending_finalize"] = True
    record["pending_finalize_reason"] = reason
    save_workflow(record)


def disarm_finalize(record: dict, reason: str) -> None:
    """Clear a pending finalize (human declined, or it already ran)."""
    record["pending_finalize"] = False
    record["pending_finalize_reason"] = reason
    save_workflow(record)


def finalize_armed(record: dict) -> bool:
    return bool(record.get("pending_finalize"))


def dedupe_feedback(items: list) -> list:
    """Drop already-processed and exact-duplicate feedback. Keyed by comment id
    first, then by normalised body so a maintainer re-posting the same text
    doesn't trigger a second identical iteration (a reliability requirement)."""
    seen_ids, seen_bodies, out = set(), set(), []
    for it in items:
        cid = it.get("id")
        body_key = " ".join((it.get("body") or "").lower().split())
        if cid is not None and cid in seen_ids:
            continue
        if body_key and body_key in seen_bodies:
            continue
        if cid is not None:
            seen_ids.add(cid)
        if body_key:
            seen_bodies.add(body_key)
        out.append(it)
    return out


def filter_unprocessed(record: dict, items: list) -> list:
    """Feedback we haven't acted on yet (id not in the processed set), our own
    bot comments removed, then de-duplicated."""
    processed = set(record.get("processed_comment_ids", []))
    bot_login = (record.get("bot_login") or "").lower()
    fresh = []
    for it in items:
        if it.get("id") in processed:
            continue
        if bot_login and (it.get("author") or "").lower() == bot_login:
            continue
        fresh.append(it)
    return dedupe_feedback(fresh)


def mark_feedback_processed(record: dict, items: list) -> None:
    ids = record.setdefault("processed_comment_ids", [])
    for it in items:
        if it.get("id") is not None and it["id"] not in ids:
            ids.append(it["id"])
    save_workflow(record)


# ============================================================
# Experience Log -- practical "learn from past attempts" without
# claiming to do real reinforcement learning (which needs far more
# episodes and training infra than a personal project has). Every
# attempt gets logged; before generating a new fix, we retrieve the
# most similar past entries and give them to the model as precedent.
# ============================================================
def log_experience(repo_name, issue, language, outcome, error_category=None, notes=""):
    entry = {
        "repo": repo_name,
        "issue_title": issue.title,
        "language": language,
        "outcome": outcome,  # "success" | "failed" | "skipped"
        "error_category": error_category,  # A/B/C/D/E from the failure architecture
        "notes": notes[:500],
        "timestamp": str(date.today()),
    }
    with open(EXPERIENCE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def find_similar_experiences(issue_title: str, language: str, limit: int = 3) -> list:
    if not EXPERIENCE_LOG.exists():
        return []
    keywords = set(re.findall(r"[a-zA-Z_]{4,}", issue_title.lower()))
    scored = []
    with open(EXPERIENCE_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_keywords = set(re.findall(r"[a-zA-Z_]{4,}", entry["issue_title"].lower()))
            score = len(keywords & entry_keywords)
            if entry.get("language") == language:
                score += 2  # same-language precedent is worth more
            if score > 0:
                scored.append((score, entry))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:limit]]


def format_failure_history(history: list, max_chars: int = MAX_ERROR_CHARS) -> str:
    """Compact, deduped summary of every failed attempt, for the next retry's
    prompt. Each entry is a dict {"attempt": n, "changed": [paths],
    "signal": str, "error": str}. Only the last 5 attempts are shown, and
    consecutive attempts whose failure signal repeats are collapsed, so the
    model sees the SEQUENCE of distinct failures instead of a wall of text or
    the same error twice. Pure formatting -- never raises, never touches
    files."""
    if not history:
        return ""
    seen, out = set(), []
    for h in history[-5:]:
        sig = str(h.get("signal") or "")[:160].strip()
        if sig and sig in seen:
            continue
        if sig:
            seen.add(sig)
        changed = ", ".join(h.get("changed") or ["?"])[:120] or "?"
        detail = (str(h.get("error") or "").strip()) or "no detail"
        out.append(f"  Attempt {h.get('attempt', '?')}: touched {changed} -> {detail[-240:]}")
    return "\n".join(out)


# ============================================================
# Step 1: Discovery (plain GitHub search -- no AI needed here,
# this was incorrectly assigned to an LLM in the original DRD)
# ============================================================
# A repo spec is exactly "owner/repo". Everything else in a --repos-file --
# the file's own `#` header comments, blank lines, a pasted browser URL --
# must be normalised or dropped BEFORE it reaches the search query, because
# GitHub answers a nonsense qualifier like `repo:# Ek repo per line` with a
# 422 "resources do not exist or you do not have permission" that used to
# abort the entire scan.
_REPO_SPEC_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def parse_repo_line(line: str) -> str | None:
    """Normalise one line of a --repos-file into `owner/repo`.

    Returns None when the line isn't a repo at all (blank, comment, prose,
    malformed), so the caller can skip it instead of building an invalid query.
    Accepts the forms people actually paste: a full https/ssh URL, a trailing
    `.git`, a trailing slash, a deep link like `owner/repo/issues/42`, and a
    trailing inline `# comment`."""
    # '#' can't appear in an owner or repo name, so cutting there handles both
    # whole-line and inline comments.
    spec = (line or "").split("#", 1)[0].strip()
    if not spec:
        return None
    spec = spec.removeprefix("git@github.com:")
    for scheme in ("https://", "http://", "git://", "ssh://"):
        spec = spec.removeprefix(scheme)
    spec = spec.removeprefix("www.").removeprefix("github.com/")
    spec = spec.strip("/").removesuffix(".git")
    # Keep only owner/repo, so a pasted issue or tree link still resolves.
    parts = [p for p in spec.split("/") if p][:2]
    spec = "/".join(parts)
    return spec if _REPO_SPEC_RE.match(spec) else None


def _explain_search_failure(exc: Exception) -> str:
    """Turn a search exception into something a human can act on."""
    status = getattr(exc, "status", None)
    if status == 422:
        return (
            "GitHub refused the query (422). That repo can't be searched: it doesn't "
            "exist, was renamed, is private, or your token can't see it. Check the "
            "spelling in the repo list."
        )
    if status in (401, 403):
        return f"access denied ({status}) -- check GITHUB_TOKEN's scope, or you're rate-limited."
    return f"{type(exc).__name__}: {exc}"


def discover_valid_issue(repo_names: list, label: str = "good first issue"):
    """Scan a list of repos for a candidate issue that's both unclaimed
    and in a language this workflow actually supports (per
    capability_map.json), pre-verified as accepted before returning it.
    Returns (repo_name, issue_number) or (None, None) if nothing found.

    `label` accepts a comma-separated list ("bug,enhancement,area:frontend")
    -- each label is tried in order, so you can hunt enhancement/frontend/
    areas the way you asked for instead of being locked to one label."""
    capability_map = load_capability_map()
    supported_extensions = set()
    for cfg in capability_map.values():
        supported_extensions.update(cfg["extensions"])

    labels = [item.strip() for item in (label or "").split(",") if item.strip()]
    labels = labels or ["good first issue"]

    for raw_line in repo_names:
        repo_name = parse_repo_line(raw_line)
        if repo_name is None:
            stripped = (raw_line or "").strip()
            # Silent on comments/blanks; loud on a line that looked like it was
            # meant to be a repo, since that's a typo the user needs to see.
            if stripped and not stripped.startswith("#"):
                print(f"   ↳ Skipping unusable repo-list line: {stripped[:60]!r}")
            continue
        print(f"🔎 Scanning {repo_name} (labels: {', '.join(labels)})...")
        first_failure = None
        any_query_ran = False
        for lbl in labels:
            try:
                query = f'repo:{repo_name} label:"{lbl}" state:open no:assignee'
                # PyGithub's search returns a LAZY PaginatedList -- no HTTP
                # happens until it's iterated. Materialising the first page
                # INSIDE this try is the whole point: otherwise a 422 or
                # rate-limit is raised later, outside the guard, and one bad
                # repo kills the entire scan.
                candidates = list(gh.search_issues(query=query, sort="created", order="desc")[:5])
                any_query_ran = True
            except Exception as e:
                if first_failure is None:
                    first_failure = e
                continue

            for issue in candidates:
                if issue.assignees or issue.locked:
                    continue
                print(f"   ↳ Candidate: #{issue.number} - {issue.title} (label={lbl})")
                if not verify_issue_is_genuine(issue):
                    print("     ✗ Not a workable issue, skipping.")
                    continue
                print(
                    f"   ✅ Found a valid, compatible issue: {repo_name}#{issue.number}"
                    f" (label={lbl})"
                )
                return repo_name, issue.number
        # Only shout about a failed search when EVERY label on this repo
        # failed -- a partial failure among several labels is normal.
        if not any_query_ran and first_failure is not None:
            print(f"   ↳ Could not search {repo_name}: {_explain_search_failure(first_failure)}")

    print("❌ No valid compatible issue found across the given repos.")
    return None, None


# ============================================================
# Step 2: Verification -- is this a genuine, fixable code bug
# (or a scoped enhancement), or a user/environment-specific
# issue / broad feature? (was missing from the original
# reference code entirely)
# ============================================================
# area/frontend/ai/integration hint map: a label is the cheapest, most reliable
# domain signal a repo maintainer gives us. It reaches the fix prompt and the
# file picker so a "frontend" task and an "ai" task are approached differently.
_LABEL_DOMAINS = {
    "frontend": "frontend",
    "ui": "frontend",
    "ux": "frontend",
    "css": "frontend",
    "scss": "frontend",
    "design": "frontend",
    "web": "frontend",
    "integration": "integration",
    "api": "integration",
    "backend": "integration",
    "service": "integration",
    "connector": "integration",
    "rest": "integration",
    "ai": "ai/ml",
    "ml": "ai/ml",
    "machine-learning": "ai/ml",
    "model": "ai/ml",
    "llm": "ai/ml",
    "nlp": "ai/ml",
    "neural": "ai/ml",
    "data": "data",
    "database": "data",
    "db": "data",
    "sql": "data",
    "testing": "testing",
    "test": "testing",
    "docs": "docs",
    "documentation": "docs",
}


def extract_labels(issue) -> list:
    """The issue's labels as plain lowercase strings, tolerating PyGithub
    Label objects, dict-shaped fakes, bare strings, or no labels at all."""
    raw = getattr(issue, "labels", None) or []
    out = []
    for item in raw:
        if isinstance(item, str):
            out.append(item.lower())
        elif isinstance(item, dict):
            out.append(str(item.get("name", "")).lower())
        else:
            name = getattr(item, "name", None)
            if name:
                out.append(str(name).lower())
    return [entry for entry in out if entry]


def domain_from_labels(labels: list) -> str:
    """First label that maps to a known domain ('frontend', 'integration',
    'ai/ml', 'data', 'testing', 'docs'), else ''. Multi-label issues take the
    first match so the signal stays unambiguous."""
    for label in labels or []:
        label = label.lower().strip()
        # "area:integration" and plain "integration" both match.
        for token in label.replace(":", "-").split("-"):
            if token in _LABEL_DOMAINS:
                return _LABEL_DOMAINS[token]
    return ""


# --- General task taxonomy -------------------------------------------------
# What KIND of work an issue is (beyond the GFI kind/scope/difficulty gate).
# This is what lets the agent treat a security review, a repo-exploration ask,
# or a multi-file refactor as its own kind of job instead of forcing every
# issue into the "produce a small fix + PR" box.
_TASK_TYPES = frozenset({
    "BUG_FIX", "FEATURE", "REFACTOR", "INVESTIGATION", "CODE_REVIEW",
    "SECURITY", "PERFORMANCE", "DOCS", "DEPENDENCY", "TEST_IMPROVEMENT",
    "ENVIRONMENT",
})
# Task types whose deliverable is analysis/investigation rather than a code
# change. These never go down the fix->PR pipeline: they short-circuit to the
# read-only --analyze investigation flow.
REPORT_ONLY_TASK_TYPES = frozenset({
    "INVESTIGATION", "CODE_REVIEW", "SECURITY", "PERFORMANCE",
})


def _default_task_type(kind: str) -> str:
    """Sensible task_type when the model omits/normalises the TASK line
    (keeps a one-line answer from derailing routing)."""
    return {
        "GENERIC": "BUG_FIX",
        "ENHANCEMENT": "FEATURE",
        "FEATURE": "FEATURE",
        "ENVIRONMENT": "ENVIRONMENT",
    }.get(kind, "BUG_FIX")


def is_report_only_task(task_type: str) -> bool:
    """Should this task deliver a findings report instead of a code change?"""
    return (task_type or "").upper() in REPORT_ONLY_TASK_TYPES


def classify_issue(issue) -> dict:
    """Structured classification used by the whole pipeline (as opposed to the
    old bool). Returns:

        kind:       GENERIC | ENHANCEMENT | FEATURE | ENVIRONMENT
        scope:      SCOPED | MODERATE | BROAD
        difficulty: easy | medium | hard
        domain:     frontend | integration | ai/ml | data | testing | docs | ""
        task_type:  bug_fix | feature | refactor | investigation | code_review |
                    security | performance | docs | dependency | test_improvement |
                    environment
        labels:     the raw label list
        accepted:   proceed or skip
        reason:     one-line justification (for human-facing logs/gates)

    Acceptance rule:
      GENERIC                    -> accepted (a real bug)
      ENHANCEMENT                -> accepted iff scope is SCOPED/MODERATE AND
                                    ACCEPT_SCOPED_ENHANCEMENTS is on
      FEATURE / ENVIRONMENT      -> rejected (broad/architectural work, or not
                                    a code problem -- the old FEATURE rule,
                                    now reachable via an enhancement label too)
    """
    labels = extract_labels(issue)
    domain = domain_from_labels(labels)
    label_note = f"LABELS ON THE ISSUE: {', '.join(labels) or '(none)'}\n" if labels else ""
    prompt = f"""Classify this GitHub issue. Answer with EXACTLY four lines:

KIND: GENERIC, ENHANCEMENT, FEATURE, or ENVIRONMENT
SCOPE: SCOPED, MODERATE, or BROAD
DIFFICULTY: EASY, MEDIUM, or HARD
TASK: BUG_FIX, FEATURE, REFACTOR, INVESTIGATION, CODE_REVIEW, SECURITY, PERFORMANCE, DOCS, DEPENDENCY, TEST_IMPROVEMENT, or ENVIRONMENT

then a single sentence starting with REASON: justifying the choices.

KIND definitions:
GENERIC = a small, scoped code defect in the repository that any user would
          hit, with clear reproduction steps or a stack trace pointing into
          the codebase. Fixable as a surgical, targeted change to one or a
          few existing files.
ENHANCEMENT = an improvement/extention of EXISTING functionality that fits the
          current architecture: a new option, a missing code path, an extra
          API field, better error handling, one new UI element inside an
          existing screen, a new endpoint on an existing service. Contained.
FEATURE = requires building substantial NEW functionality (new API
          integrations, new components/systems, multi-file new architecture)
          rather than fixing/extending something that already exists. Also
          use for broad/architectural asks ("restyle all links site-wide")
          that touch many unrelated files.
ENVIRONMENT = caused by the reporter's local setup, network, OS-specific
          config, or a duplicate/vague/unclear report.

SCOPE definitions:
SCOPED = one or a few existing files, or one small new file plus tests.
MODERATE = a handful of files across one layer (one endpoint + its tests, one
          component + its styles, one module + its callers).
BROAD = spans multiple layers/subsystems or many unrelated files (new
          architecture, site-wide restyle, new integration subsystem).

DIFFICULTY (how much retry budget this deserves):
EASY = single obvious change. MEDIUM = needs care, several files, subtle
          logic. HARD = cross-cutting domain logic (integrations, AI/ML math,
          big existing code, deep refactor, distributed/concurrent behaviour).

TASK definitions (what kind of WORK this is, independent of the PR gate):
BUG_FIX = fix defective existing behaviour; reproduction is concrete.
FEATURE = build new capability on top of existing code (multi-file ok).
REFACTOR = rework existing structure without changing visible behaviour.
INVESTIGATION = diagnose/understand/report on something; may not need a
          code change at all (root-cause, repo exploration, design study).
CODE_REVIEW = assess an existing diff/PR for correctness and risks.
SECURITY = hunt vulnerabilities or evaluate security posture.
PERFORMANCE = find/quantify a performance problem.
DOCS = documentation/comments/examples only.
DEPENDENCY = dependency/version/build/config fixes.
TEST_IMPROVEMENT = write/harden tests without changing app behaviour.
ENVIRONMENT = local setup / OS / network / tooling problem, not repo code.

Title: {issue.title}
Body: {issue.body or "(no description)"}
{label_note}"""
    try:
        # Structured, short, tolerance-heavy decision -- the fast combo is
        # enough and keeps the front of every run cheap (#2).
        response = call_model(prompt, max_tokens=120, fast=True).strip()
    except Exception as e:
        # A gateway outage must not look like "not a bug" -- the old behaviour
        # crashed the whole run here anyway; failing open to GENERIC/medium
        # keeps the pipeline moving and the human gate still guards the PR.
        print(f"   ↳ classification unavailable ({e}); assuming GENERIC/medium")
        return {
            "kind": "GENERIC", "scope": "MODERATE", "difficulty": "medium",
            "domain": domain, "labels": labels, "accepted": True,
            "task_type": "BUG_FIX",
            "reason": "(model unavailable during classification)",
        }

    def _field(name: str) -> str:
        m = re.search(rf"^{name}:\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip().upper() if m else ""

    kind = _field("KIND")
    scope = _field("SCOPE")
    difficulty = _field("DIFFICULTY")
    task_type = _field("TASK")
    reason_m = re.search(r"REASON:\s*(.+)", response, re.IGNORECASE | re.MULTILINE)
    reason = reason_m.group(1).strip() if reason_m else response[:200]

    # Normalise/tolerate slightly-off model wording.
    kind = next((k for k in ("GENERIC", "ENHANCEMENT", "FEATURE", "ENVIRONMENT") if k in kind), "GENERIC")
    scope = next((s for s in ("SCOPED", "MODERATE", "BROAD") if s in scope), "MODERATE")
    difficulty = difficulty.lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    task_type = next(
        (t for t in _TASK_TYPES if t in task_type),
        _default_task_type(kind),
    )

    if kind == "ENHANCEMENT":
        accepted = bool(ACCEPT_SCOPED_ENHANCEMENTS) and scope in ("SCOPED", "MODERATE")
    else:
        accepted = kind == "GENERIC"

    print(
        f"   ↳ verification verdict: {kind}/{scope} difficulty={difficulty}"
        + (f" domain={domain}" if domain else "")
    )
    return {
        "kind": kind, "scope": scope, "difficulty": difficulty,
        "domain": domain, "labels": labels, "accepted": accepted,
        "task_type": task_type, "reason": reason,
    }


def verify_issue_is_genuine(issue) -> bool:
    """Back-compat bool wrapper used by discovery mode."""
    return classify_issue(issue)["accepted"]


def escalation_budget(kind: str, difficulty: str) -> dict:
    """How hard this issue is allowed to work. Flat 5/6/18000 for routine
    work; HARD issues (and scope-flagged enhancements) get a bigger retry
    budget, more context files, larger file slices and a mandatory plan-first
    step. Returns a dict consumed by main() and the iteration loop."""
    hard = difficulty == "hard"
    if hard:
        print(f"   ↳ 🚀 Difficulty escalation: hard task -> {MAX_HARD_ATTEMPTS} attempts, "
              f"{HARD_CONTEXT_FILES} context files.")
    return {
        "attempts": MAX_HARD_ATTEMPTS if hard else MAX_ATTEMPTS,
        "context_files": HARD_CONTEXT_FILES if hard else MAX_CONTEXT_FILES,
        "file_chars": HARD_FILE_CHARS if hard else MAX_FILE_CHARS,
        "plan_first": hard or (kind == "ENHANCEMENT" and difficulty in ("medium", "hard")),
    }


# ============================================================
# ============================================================
# Step 3: Anti-collision guard (cheap, side-effect-free -- run early)
# Step 4: Claim comment (side-effecting -- deferred until AFTER we
#         actually have a validated fix, so we never claim an issue
#         we couldn't solve)
# ============================================================
_CLAIM_PHRASES = [
    "working on this",
    "i'll take this",
    "assigned to me",
    "pr open",
    "opening a pr",
]


def check_not_already_claimed(issue) -> bool:
    """Has someone ELSE already signalled they're on this issue?

    Read-only, so we call it early (to avoid wasting effort on a taken issue)
    AND again right before posting our own claim, since a lot of time passes
    while we generate and validate a fix. Returns True when the issue looks
    free to work on."""
    for c in issue.get_comments():
        body = (c.body or "").lower()
        if any(p in body for p in _CLAIM_PHRASES):
            print(f"   ↳ Already claimed in comments ('{c.body[:60]}...').")
            return False
    return True


def post_claim_comment(issue) -> bool:
    """Post our claim comment. Called only AFTER an optimal, validated fix is
    in hand -- so we never claim something we couldn't actually solve. Re-runs
    the collision check first, because someone may have claimed it while we
    were solving. Returns True once the claim is posted."""
    from github import GithubException

    if not check_not_already_claimed(issue):
        print("   ↳ Someone claimed it while we were solving -- backing off, no claim posted.")
        return False

    try:
        issue.create_comment(
            "I've prepared a fix for this and am opening a draft PR for review shortly. "
            "If you're already working on it, let me know and I'll back off / close mine."
        )
    except GithubException as e:
        if e.status == 403:
            print(
                "❌ Token can't post comments on this repo. You likely have a "
                "fine-grained token, which only writes to repos you own or "
                "collaborate on. For third-party repos, use a classic token "
                "with the 'public_repo' scope instead. Skipping this issue."
            )
        else:
            print(f"❌ GitHub API error while claiming: {e}")
        return False
    print("   ↳ Claim comment posted (fix ready).")
    return True


# ============================================================
# Step 4b: Fork, clone, branch (was missing from reference code)
# ============================================================
def workflow_workspace(repo_name: str, issue_number: int, force: bool = False) -> Path:
    """This workflow's OWN clone directory: workspace/<owner-repo>/issue-<N>/.

    Replaces the old `WORKSPACE / repo.name`, which used the bare repo name --
    no owner, no issue -- so every issue in a repo shared one clone and the
    unconditional `git reset --hard` below wiped whichever workflow got there
    first. That is precisely what made working on two issues at once unsafe.

    Claims the directory before returning it, so a second workflow gets a clear
    error instead of somebody else's half-finished branch."""
    _sync_store_home()
    path = store.workspace_dir(repo_name, issue_number)
    store.assert_workspace_owner(path, repo_name, issue_number)
    path.mkdir(parents=True, exist_ok=True)
    conv = _active_conversation
    store.write_owner_marker(
        path, repo_name, issue_number,
        conversation_id=getattr(conv, "id", "") or "",
        branch=f"fix-issue-{issue_number}",
    )
    if force:
        print(f"⚠️  --force-workspace: uncommitted changes in {path} may be discarded.")
    return path


def guard_workspace(repo_dir: Path, repo_name: str, issue_number: int, action: str,
                    force: bool = False) -> None:
    """Call immediately before anything destructive (reset/clean/checkout).
    Raises WorkspaceConflict with recovery instructions rather than clobbering."""
    discarded = store.guard_before_destructive(
        repo_dir, repo_name, issue_number, action, force=force
    )
    if discarded:
        print(f"   ↳ {action}: discarding {len(discarded)} uncommitted change(s) (--force).")
        _transcript(f"{action}: discarded {len(discarded)} uncommitted change(s) under --force")


def fork_clone_and_branch(repo, issue_number: int, force_workspace: bool = False) -> Path:
    fork = repo.create_fork() if repo.owner.login != gh.get_user().login else repo
    repo_dir = workflow_workspace(repo.full_name, issue_number, force=force_workspace)

    if not (repo_dir / ".git").exists():
        auth = _git_auth_args()
        # Bigger buffer + shallow clone + retries handle unstable connections
        # and large repos, which caused "RPC failed / early EOF" before.
        for attempt in range(1, 4):
            result = subprocess.run(
                ["git", *auth,
                    "-c",
                    "http.postBuffer=524288000",
                    "clone",
                    "--depth",
                    "50",
                    fork.clone_url,
                    str(repo_dir),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0:
                break
            print(f"⚠️  Clone attempt {attempt}/3 failed, retrying...")
            if repo_dir.exists():
                subprocess.run(
                    ["cmd", "/c", "rmdir", "/s", "/q", str(repo_dir)], capture_output=True
                ) if os.name == "nt" else subprocess.run(
                    ["rm", "-rf", str(repo_dir)], capture_output=True
                )
        else:
            raise RuntimeError(f"Clone failed after 3 attempts: {result.stderr[-500:]}")

    # Discard any uncommitted changes left over from a previous run --
    # this is what caused "branch already exists" / checkout conflicts.
    # Safe here because this is a disposable clone, never your only copy --
    # but ONLY after guard_workspace() confirms this clone belongs to this
    # workflow and holds nothing the user would miss.
    guard_workspace(
        repo_dir, repo.full_name, issue_number, "reset the clone", force=force_workspace
    )
    subprocess.run(["git", "reset", "--hard"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)

    branch_name = f"fix-issue-{issue_number}"
    # Go back to the default branch first, then either switch to an
    # existing fix branch or create a fresh one -- avoids the
    # "already on this branch with dirty state" conflict entirely.
    default_branch = repo.default_branch
    subprocess.run(["git", "checkout", default_branch], cwd=repo_dir, capture_output=True)

    result = subprocess.run(["git", "checkout", branch_name], cwd=repo_dir, capture_output=True)
    if result.returncode != 0:
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=repo_dir, check=True)
    else:
        # Existing branch found -- reset it to match default branch so
        # each run starts from a clean slate instead of stacking old fixes.
        subprocess.run(
            ["git", "reset", "--hard", default_branch], cwd=repo_dir, capture_output=True
        )

    print(f"✅ Forked, cloned, branched: {branch_name}")
    return repo_dir, branch_name


# ============================================================
# Step 4.5: Grounding & hallucination guards. Issues like
# "add resultpage in that thing" are vague and cross-cutting --
# exactly where a model invents file names, symbols and routes
# that do not exist. These three helpers are pure file I/O (no
# LLM), so they are deterministic, cheap, and always factual:
#   1. is_vague_issue             -- flag vague/short reports and
#      quietly hand them the HARD budget + plan-first step
#   2. collect_codebase_facts     -- real symbols / entry points /
#      closest matching files, fed into the plan + fix prompts
#   3. scan_for_hallucinated_     -- static check on each generated
#      symbols patch: fake repo-local imports and symbols get fed
#      back to the model as a retryable error instead of shipping
# ============================================================
_VAGUE_PHRASES = (
    "that thing", "this thing", "the thing", "and stuff", "and things",
    "and all", "you know", "etc", "something like", "similar to", "like the",
    "somewhere", "somehow", "probably", "maybe", "whatever", "basically",
    "a bit", "if possible", "as usual", "add a page", "add the page",
    "make it", "should be", "place holder", "placeholder", "do something",
    "whatever fits",
)


def is_vague_issue(issue) -> bool:
    """Cheap heuristic (no model call) for reports that force hallucination:
    no/missing description, very short body, or vague deictic phrasing
    ("add resultpage in that thing"). Such issues get the hard-task budget
    and a mandatory plan-first step, because guessing is exactly what burns
    attempts. Never raises."""
    raw = f"{issue.title} {issue.body or ''}"
    text = raw.strip()
    body = (issue.body or "").strip()
    vague = (
        len(text) < 60                                          # ultra-short report
        or (body and len(body) < 40)                            # bare title-level body
        or any(phrase in body.lower() for phrase in _VAGUE_PHRASES)
    )
    if vague:
        print("   ↳ ⚠️  Issue looks vague/underspecified -- treating as HARD: "
              "plan-first + wider context so it is grounded in real code, not guessed.")
    return vague


def _module_source_file(repo_dir: Path, dotted: str):
    """Resolve a dotted python module path ('app.models', 'pkg/sub') to the
    file(s) that implement it inside the repo, or None when it is not
    repo-local (third-party package -- out of our scanner's jurisdiction).
    Checks plain file, .pyi, and package __init__ layouts. Also accepts a
    slash-form path which some models emit for the same module."""
    dotted = dotted.replace("/", ".")
    candidates = []
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        prefix = parts[:i]
        tail = parts[i:]
        base = repo_dir.joinpath(*prefix)
        if len(tail) == 0:
            candidates.extend([
                base.with_suffix(".py"), base.with_suffix(".pyi"),
                base / "__init__.py", base / "__init__.pyi",
            ])
        else:
            candidates.append(base.joinpath(*tail).with_suffix(".py"))
            candidates.append(base.joinpath(*tail).with_suffix(".pyi"))
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


_PY_TOP_LEVEL_NAME = re.compile(
    r"^((?:async\s+)?def|class)\s+([_A-Za-z][_A-Za-z0-9]*)", re.MULTILINE
)
_PY_IMPORT_LINE = re.compile(
    r"^\s*from\s+([_.\w]+)\s+import\s+(.*)$", re.MULTILINE
)
_PY_IMPORT_STAR = re.compile(r"^\s*import\s+([_.\w]+)", re.MULTILINE)


def _defined_python_names(path: Path) -> set:
    """Top-level defs/classes plus names pulled in by imports -- the little
    export surface a fix is allowed to reference. Cheap and approximate on
    purpose: a false NEGATIVE here (missing a symbol) must never happen, so
    anything we are unsure about simply is not flagged upstream."""
    text = path.read_text(errors="ignore")
    names = set(_PY_TOP_LEVEL_NAME.findall(text))
    # `_PY_TOP_LEVEL_NAME.findall` returns (def|class, name) pairs.
    names = {name for _, name in names}
    for m in _PY_IMPORT_LINE.finditer(text):
        for part in m.group(2).split(","):
            alias = re.split(r"\s+as\s+", part.strip())[0]
            if alias and alias not in ("*",):
                names.add(alias.split(".")[0])
    return names


_TS_NAMED_EXPORTS = re.compile(
    r"export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+([_A-Za-z][_A-Za-z0-9]*)"
)
_TS_EXPORT_BRACE = re.compile(
    r"export\s*\{([^}]+)\}", re.MULTILINE
)


def _resolve_js_module(base_dir: Path, spec: str) -> Path | None:
    """Resolve a relative JS/TS import spec against the importing file's
    directory, trying the extension/`/index` layouts bundlers use."""
    target = (base_dir / spec).resolve()
    for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ""):
        cand = Path(str(target) + ext) if ext else target
        if cand.is_file():
            return cand
    if (target / "index.ts").is_file():
        return target / "index.ts"
    if (target / "index.js").is_file():
        return target / "index.js"
    if (target / "index.tsx").is_file():
        return target / "index.tsx"
    if (target / "index.jsx").is_file():
        return target / "index.jsx"
    return target if target.is_file() else None


def scan_for_hallucinated_symbols(repo_dir, changed_paths: list) -> list:
    """Conservative static guard: does the model's patch reference repo-local
    modules/symbols that do not exist? Returns a list of one-line findings
    ([] = clean).

    Only HIGH-precision signals are reported, so we never fight the model:
      * a repo-local absolute python import (`from app.models import X`) where
        the module file exists but does not define/import X
      * a RELATIVE JS/TS import (`import { X } from "./helpers"`) pointing at a
        non-existent file, or importing a name that module never exports
    Absolute/third-party packages and everything else are deliberately NOT
    flagged (we cannot know what they export). Unparseable or generated files
    are skipped. Never raises. `repo_dir` may be a string or Path."""
    findings = []
    if not changed_paths:
        return findings
    repo_root = Path(repo_dir)
    for rel in changed_paths:
        path = repo_root / rel
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        text = ""
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        if suffix not in (".py", ".js", ".ts", ".jsx", ".tsx"):
            continue

        if suffix == ".py":
            for m in _PY_IMPORT_LINE.finditer(text):
                mod, names = m.group(1), m.group(2)
                src = _module_source_file(repo_root, mod)
                if src is None:
                    continue  # third-party or virtual -- outside jurisdiction
                defined = _defined_python_names(src)
                for part in names.split(","):
                    name = re.split(r"\s+as\s+", part.strip())[0].strip()
                    if name and name != "*" and name not in defined:
                        findings.append(
                            f"{rel}: imports {name} from {mod!r}, but {src.relative_to(repo_root).as_posix()} "
                            f"defines no such name (real symbols: "
                            f"{', '.join(sorted(defined)[:6]) or '(none)'})."
                        )
        else:
            rel_imports = re.finditer(
                r"""import\s*\{([^}]+)\}\s*from\s+["']([^"']+)["']""", text
            )
            rel_imports = list(rel_imports)
            for m in rel_imports:
                spec = m.group(2)
                if not spec.startswith("."):
                    continue
                target = _resolve_js_module(path.parent, spec)
                if target is None:
                    findings.append(
                        f"{rel}: imports {spec!r}, but that relative module does not "
                        f"exist in the repository."
                    )
                    continue
                exports = set(_TS_NAMED_EXPORTS.findall(target.read_text(errors="ignore")))
                exported = target.read_text(errors="ignore")
                brace_match = _TS_EXPORT_BRACE.search(exported)
                named_braces = set()
                if brace_match:
                    named_braces = {
                        n.strip() for n in brace_match.group(1).split(",") if n.strip()
                    }
                for name in (n.strip() for n in m.group(1).split(",")):
                    base = re.split(r"\s+as\s+", name)[0].strip()
                    if base == "*":
                        continue
                    if base in exports or base in named_braces:
                        continue
                    if re.search(rf'\bexport\s*.*\b{re.escape(base)}\b', exported):
                        continue
                    findings.append(
                        f"{rel}: imports {base} from {spec!r}, which exports none of it."
                    )
    return findings


_ENTRY_DIRS = {"pages", "views", "templates", "routes", "controllers", "components", "router"}
_ENTRY_STEMS = {"main", "app", "index", "router", "routes", "urls", "pages"}


# ============================================================
# Change-impact and reverse-dependency analysis (pure logic,
# no LLM). Lets the agent answer "who else depends on the files
# I'm about to edit?" before it writes the fix, so a correct
# multi-file change stops surprising callers it never saw.
# ============================================================
def _code_file_paths(repo_dir: Path) -> list:
    """Every code file in the repo (same filtering as find_relevant_files),
    sorted by path so output is deterministic for tests."""
    paths = []
    for path in repo_dir.rglob("*"):
        if not path.is_file() or path.suffix not in CODE_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        paths.append(path)
    paths.sort(key=lambda p: p.relative_to(repo_dir).as_posix())
    return paths


def _python_module_rel(rel_path: str) -> str:
    """A python file's DOT-relative module name (``pkg/mod/py`` ->
    ``pkg.mod``), the form another file would use to import it."""
    parts = list(Path(rel_path.replace("\\", "/")).parts)
    if parts[-1] == "__init__.py":
        parts.pop()
        return ".".join(parts) if parts else ""
    parts[-1] = parts[-1][:-3] if parts[-1].endswith(".py") else parts[-1]
    return ".".join(parts)


def symbol_sources(repo_dir: Path, symbol: str) -> list:
    """Which repo files DEFINE ``symbol`` (top-level def/class/export/assignment).
    Deterministic, conservative: only module-level definitions count, so a file
    that merely mentions the name is not reported as its source."""
    sources = []
    for path in _code_file_paths(repo_dir):
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        rel = path.relative_to(repo_dir).as_posix()
        if path.suffix.lower() == ".py":
            if re.search(
                rf"^(?:def|class|async\s+def)\s+{re.escape(symbol)}\b",
                text, re.MULTILINE,
            ) or re.search(rf"^{re.escape(symbol)}\s*=", text, re.MULTILINE):
                sources.append(rel)
        elif path.suffix.lower() in (".js", ".jsx", ".ts", ".tsx"):
            if re.search(
                rf"(?:export\s+)?(?:const|let|var|function|class)\s+{re.escape(symbol)}\b",
                text,
            ) or re.search(rf"export\s*{{[^}}]*\b{re.escape(symbol)}\b", text):
                sources.append(rel)
    return sources


def who_imports(repo_dir: Path, target_rel: str) -> list:
    """Reverse dependency scan: which repo files import/reference ``target_rel``.
    Python matches by dotted module import or module-name literal; JS/TS matches
    by relative import specifier that actually resolves to that file. Returns
    [] when nobody depends on it, so callers can stay confident about isolated
    edits."""
    repo_root = Path(repo_dir).resolve()
    target_rel = target_rel.replace("\\", "/")
    callers = []
    dotted = _python_module_rel(target_rel) if target_rel.endswith(".py") else ""
    leaf_name = target_rel.rsplit("/", 1)[-1]
    stem = leaf_name[:-3] if leaf_name.endswith(".py") else ""
    for path in _code_file_paths(repo_root):
        if path.relative_to(repo_root).as_posix() == target_rel:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        suffix = path.suffix.lower()
        hit = False
        if suffix == ".py":
            if dotted and re.search(
                rf"^\s*(?:from\s+{re.escape(dotted)}\s+import|"
                rf"import\s+{re.escape(dotted)}\b)",
                text, re.MULTILINE,
            ) or stem and re.search(rf"[\"'](?:[\w./]*/)?{re.escape(stem)}[\"']", text):
                hit = True
        elif suffix in (".js", ".jsx", ".ts", ".tsx"):
            for m in re.finditer(r"""from\s+["']([^"']+)["']""", text):
                spec = m.group(1)
                if not spec.startswith("."):
                    continue
                target = _resolve_js_module(path.parent, spec)
                if target is None:
                    continue
                try:
                    target_rel_posix = target.relative_to(repo_root).as_posix()
                except ValueError:
                    continue  # escapes the repo root -- external dep, not a caller
                if target_rel_posix == target_rel:
                    hit = True
                    break
        if hit:
            callers.append(path.relative_to(repo_root).as_posix())
    return sorted(set(callers))


def change_impact(repo_dir: Path, target_paths: list) -> str:
    """Text block (for prompts/plans) flagging the caller-dependency surface of
    the files an edit would touch. '' when nothing real depends on them."""
    blocks = []
    for rel in dict.fromkeys(target_paths):  # dedupe, keep order
        callers = who_imports(repo_dir, str(rel))
        if callers:
            blocks.append(
                f"- {rel} is imported/referenced by: {', '.join(callers)}"
            )
    if not blocks:
        return ""
    return (
        "\nCHANGE-IMPACT WARNING -- these repo files DEPEND on the code you are "
        "editing, so they constrain what you can do without breaking them (keep "
        "signatures compatible, update callers when you must change them):\n"
        + "\n".join(blocks)
    )


# ============================================================
# Cheap static verification (no LLM, no test run): syntax-check
# the changed files before paying for a full test suite.
# ============================================================
_NODE_AVAILABLE = None


def _node_on_path() -> bool:
    global _NODE_AVAILABLE
    if _NODE_AVAILABLE is None:
        try:
            r = subprocess.run(
                ["node", "--version"], capture_output=True, timeout=10
            )
            _NODE_AVAILABLE = r.returncode == 0
        except Exception:
            _NODE_AVAILABLE = False
    return _NODE_AVAILABLE


def _node_check(path: Path):
    """Best-effort JS syntax check via `node --check`. Returns (ok, stderr)."""
    if not _node_on_path():
        return True, ""
    try:
        r = subprocess.run(
            ["node", "--check", str(path)], capture_output=True, text=True, timeout=30
        )
        return r.returncode == 0, r.stderr
    except Exception:
        return True, ""  # never fail the pipeline on tooling problems


def syntax_check(repo_dir: Path, changed_paths: list) -> list:
    """Pre-test verification of changed files. Returns [(rel_path, message)].
    Python is always checked (compile, no subprocess). JS is checked with node
    when node is on PATH; TypeScript needs a build step, not a syntax checker,
    so it is skipped here -- the suite still catches it. [] = clean."""
    problems = []
    repo_root = Path(repo_dir)
    for rel in changed_paths:
        path = repo_root / str(rel).replace("\\", "/")
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        suffix = path.suffix.lower()
        if suffix == ".py":
            try:
                compile(text, str(path), "exec")
            except SyntaxError as e:
                problems.append(
                    (str(rel).replace("\\", "/"), f"python syntax error: {e}")
                )
        elif suffix in (".js", ".jsx", ".mjs"):
            ok, err = _node_check(path)
            if not ok:
                problems.append(
                    (str(rel).replace("\\", "/"), (err or "node syntax error").strip()[:300])
                )
    return problems


def collect_codebase_facts(repo_dir: Path, issue, relevant_files: list,
                           max_chars: int = 4000) -> str:
    """Factual anchors (no LLM) for vague/conceptual issues, so the model never
    has to invent them:

      * CLOSEST EXISTING FILES -- the repo files whose name/content most
        resemble the issue's own words ("resultpage" -> the files that
        already spell result/page), resolved with difflib + keyword overlap
      * REAL SYMBOLS -- the top-level defs/classes/exports actually defined
        in each file the fix will edit
      * PAGE/ROUTE ENTRY POINTS -- where this repo conventionally wires up
        new pages/routes/views, so "add a page" lands in the right place

    Returns a formatted block ('' when nothing useful was found), capped so
    token use stays bounded."""
    text = f"{issue.title} {issue.body or ''}"
    tokens = set(re.findall(r"[a-zA-Z_]{4,}", text.lower()))
    if "the" in tokens:
        tokens.discard("the")

    rel_paths = []
    for path in repo_dir.rglob("*"):
        if not path.is_file() or path.suffix not in CODE_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel_paths.append(path.relative_to(repo_dir).as_posix())

    # --- closest files by name similarity + keyword overlap ---
    close = []
    if tokens and rel_paths:
        stems = {p.rsplit("/", 1)[-1] for p in rel_paths}
        for tok in sorted(tokens):
            for m in difflib.get_close_matches(tok, stems, n=3, cutoff=0.6):
                full = next(p for p in rel_paths if p.rsplit("/", 1)[-1] == m)
                close.append((0.9, full))
        for p in rel_paths:
            low = p.lower()
            hits = sum(1 for t in tokens if t in low)
            if hits:
                close.append((0.5 + 0.1 * hits, p))
        close.sort(key=lambda x: -x[0])
        closest = []
        seen = set()
        for _, p in close:
            if p not in seen:
                seen.add(p)
                closest.append(p)
            if len(closest) >= 8:
                break
    else:
        closest = rel_paths[:8]

    # --- real symbols per relevant file ---
    symbol_blocks = []
    for rel, _content in (relevant_files or [])[:6]:
        fp = repo_dir / rel
        if not fp.is_file():
            continue
        if fp.suffix == ".py":
            names = sorted(_defined_python_names(fp))[:12]
            if names:
                symbol_blocks.append(f"{rel}: {', '.join(names)}")
        elif fp.suffix in (".js", ".ts", ".jsx", ".tsx"):
            exports = sorted(_TS_NAMED_EXPORTS.findall(fp.read_text(errors="ignore")))[:12]
            if exports:
                symbol_blocks.append(f"{rel} (exports): {', '.join(exports)}")

    # --- page / route / view entry points ---
    entrants = []
    for p in rel_paths:
        low = p.lower()
        parts = low.split("/")
        if any(d in parts for d in _ENTRY_DIRS) or any(
            low.rsplit("/", 1)[-1].startswith(s + ".") for s in _ENTRY_STEMS
        ):
            entrants.append(p)
    entrants.sort()
    entrants = [e for e in entrants if not any(b in e for b in ("node_modules", "venv", ".git"))][:12]

    chunks = []
    if closest:
        chunks.append("CLOSEST EXISTING FILES (conceptually closest to the issue):\n" +
                      "\n".join(f"- {c}" for c in closest))
    if symbol_blocks:
        chunks.append("REAL SYMBOLS (verify every reference against these):\n" +
                      "\n".join(f"- {b}" for b in symbol_blocks))
    if entrants:
        chunks.append("PAGE/ROUTE ENTRY POINTS (where pages/routes/views live here):\n" +
                      "\n".join(f"- {e}" for e in entrants))
    if not chunks:
        return ""
    block = "\n".join(chunks)
    header = (
        "\nGROUNDING FACTS (facts from the ACTUAL repository -- resolve "
        "vague referents against these real files; never invent paths or "
        f"symbols):\n{block}\n"
    )
    return header[:max_chars]


# ============================================================
# Step 5: Relevant file discovery (keyword-scored, keeps token
# use bounded instead of ingesting the whole repo)
# ============================================================
def find_relevant_files(repo_dir: Path, issue_title: str, issue_body: str,
                        labels=None, max_files: int = None, max_chars: int = None):
    """Discover the files most likely to change for this issue.

    `labels` and `domain`-style hints are fed to the AI file picker so a
    'frontend' or 'ai' task steers toward the right layer. `max_files` /
    `max_chars` let the difficulty escalator widen the context for hard
    issues (defaults to the module constants, so plain callers stay cheap)."""
    max_files = max_files or MAX_CONTEXT_FILES
    max_chars = max_chars or MAX_FILE_CHARS
    # --- Method 0 (strongest signal): the issue text directly names a
    # real file path in the repo. This is what was missed on the
    # open-fixture-library case -- the issue mentioned the exact SCSS
    # file, but it was never picked up because CODE_EXTENSIONS didn't
    # include .scss, so the model never saw its real content and had
    # to guess (differently every attempt).
    explicit_hits = set()
    text = f"{issue_title} {issue_body or ''}"
    path_like = re.findall(r"[\w\-./]+\.[a-zA-Z]{1,5}", text)
    for candidate in path_like:
        candidate = candidate.strip("`'\" ")
        if (repo_dir / candidate).is_file():
            explicit_hits.add(candidate)

    GENERATED_MARKERS = (
        "code generated",
        "do not edit",
        "auto-generated",
        "autogenerated",
        "this file is generated",
        "@generated",
        "machine generated",
    )

    def is_generated(path: Path) -> bool:
        """Skip files that are auto-generated (from Go structs, protobuf,
        API doc generators, etc.) -- editing them directly is the wrong
        fix even when they contain the text an issue is about; the real
        fix belongs in the source that generates them. Repeated failures
        on Kubernetes/Envoy-style generated api-reference docs are what
        this catches."""
        try:
            head = path.read_text(errors="ignore")[:500].lower()
        except Exception:
            return False
        return any(marker in head for marker in GENERATED_MARKERS)

    explicit_hits = {p for p in explicit_hits if not is_generated(repo_dir / p)}

    all_paths = []
    for path in repo_dir.rglob("*"):
        if not path.is_file() or path.suffix not in CODE_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if is_generated(path):
            continue
        all_paths.append(path)

    # --- Method 1: keyword overlap (fast, works for literal-name matches) ---
    keywords = re.findall(r"[a-zA-Z_]{4,}", f"{issue_title} {issue_body or ''}")
    keywords = list(set(k.lower() for k in keywords))[:15]
    scored = []
    for path in all_paths:
        try:
            content = path.read_text(errors="ignore")
        except Exception:
            continue
        score = sum(content.lower().count(k) for k in keywords)
        score += sum(2 for k in keywords if k in path.name.lower())
        if score > 0:
            scored.append((score, path, content))
    scored.sort(key=lambda x: -x[0])
    keyword_hits = {p.relative_to(repo_dir).as_posix() for _, p, _ in scored[:max_files]}

    # --- Method 2: let the AI pick from the directory listing ---
    # Keyword matching fails on conceptual/algorithmic issues (e.g. "add a
    # NAESatisfiability to Satisfiability reduction") where the relevant
    # file's name or content doesn't literally contain the issue's words.
    # Showing the AI the file tree lets it reason about structure instead.
    # The issue's LABELS are shown too, so an area:frontend / ai-labelled
    # issue is steered toward that layer rather than guessed at.
    ai_hits = set()
    if len(all_paths) <= 500:  # skip on huge repos to control token use
        file_list = "\n".join(p.relative_to(repo_dir).as_posix() for p in all_paths)
        label_line = f"\nISSUE LABELS: {', '.join(labels) or '(none)'}" if labels else ""
        prompt = f"""Given this GitHub issue and this repository's file list,
which files most likely need to change? List ONLY the file paths, one per
line, exactly as they appear below. Pick at most {max_files}.
Consider the issue's labels/domain when choosing -- e.g. a frontend label
favours UI components/styles, an ai/integration label favours the model or
service layer.{label_line}

ISSUE: {issue_title}
{issue_body or ""}

FILES:
{file_list[:8000]}
"""
        try:
            response = call_model(prompt, max_tokens=300, fast=True)
            for line in response.splitlines():
                candidate = line.strip().lstrip("-").strip()
                if candidate and (repo_dir / candidate).is_file():
                    ai_hits.add(candidate)
        except Exception as e:
            print(f"   ↳ AI file selection unavailable ({e}), using keyword matching only.")

    # Merge all three signals, strongest first: explicit file mentions in
    # the issue text, then AI-selected (good for conceptual issues), then
    # keyword-scored. explicit_hits was being computed but never actually
    # used until now -- that was a real bug.
    combined = (
        list(explicit_hits)
        + [p for p in ai_hits if p not in explicit_hits]
        + [p for p in keyword_hits if p not in explicit_hits and p not in ai_hits]
    )
    combined = combined[:max_files]

    result = []
    for rel_path in combined:
        full_path = repo_dir / rel_path
        try:
            content = full_path.read_text(errors="ignore")
        except Exception:
            continue
        truncated = len(content) > max_chars
        content = content[:max_chars]
        if truncated:
            content += "\n\n... [TRUNCATED -- this is not the full file] ..."
        result.append((rel_path, content))

    print(
        f"✅ Found {len(result)} relevant files "
        f"({len(ai_hits)} via AI selection, {len(keyword_hits)} via keyword match)"
    )
    return result


# ============================================================
# Model routing: Gemini (permanent free tier) primary,
# OpenRouter/DeepSeek fallback if Gemini errors or rate-limits
# ============================================================
# ============================================================
# Model routing: try each model in order until one succeeds.
# Gemini first (free tier), then rotate through several models
# on OpenRouter -- one key covers all of these, no need for
# separate Claude/Grok/DeepSeek accounts.
#
# NOTE: OpenRouter model slugs change over time. Check the exact
# current names at https://openrouter.ai/models before relying on
# this list -- update MODEL_CHAIN below if any slug 404s.
# ============================================================
# ============================================================
# Model routing: OmniRoute handles the fallback chain internally
# (across whatever providers you connected in its dashboard), so
# the script just makes one call to its "best coding" auto-combo.
# ============================================================
def _using_fallback_endpoint() -> bool:
    """True when the hosted fallback endpoint is active for this process.

    The decision is made once and cached: ai_client also resolves once (the
    _LazyClient proxy caches the built client), so caching here guarantees the
    model chain always matches the endpoint the client actually talks to. A
    *new* fixer invocation re-decides, so a recovered omniRoute is picked up
    on the next run automatically."""
    if not getattr(_using_fallback_endpoint, "_decided", False):
        _using_fallback_endpoint._decided = True
        _using_fallback_endpoint._fallback = bool(OMNIROUTE_FALLBACK_BASE_URL) and _primary_unreachable()
    return _using_fallback_endpoint._fallback


def _model_chain(fast: bool = False) -> list:
    """Combos to try, user's choice first, deduped, always non-empty. The fast
    chain is used for cheap structured steps (classification, AI file pick).
    On the hosted fallback endpoint the fallback model chain is used instead,
    since hosted endpoints usually need concrete model names, not auto/*."""
    if _using_fallback_endpoint():
        if fast:
            primary = OMNIROUTE_FALLBACK_FAST_MODEL or OMNIROUTE_FALLBACK_MODEL or OMNIROUTE_FAST_MODEL
            fallbacks = (OMNIROUTE_FALLBACK_FAST_MODEL_FALLBACKS
                         or OMNIROUTE_FALLBACK_MODEL_FALLBACKS
                         or OMNIROUTE_FAST_MODEL_FALLBACKS)
        else:
            primary = OMNIROUTE_FALLBACK_MODEL or OMNIROUTE_MODEL
            fallbacks = OMNIROUTE_FALLBACK_MODEL_FALLBACKS or OMNIROUTE_MODEL_FALLBACKS
    elif fast:
        primary, fallbacks = OMNIROUTE_FAST_MODEL, OMNIROUTE_FAST_MODEL_FALLBACKS
    else:
        primary, fallbacks = OMNIROUTE_MODEL, OMNIROUTE_MODEL_FALLBACKS
    chain = []
    for m in [primary] + fallbacks:
        m = (m or "").strip()
        if m and m not in chain:
            chain.append(m)
    return chain or ["auto"]


def call_model(prompt: str, max_tokens: int = 4000, retry_variant: bool = False,
               fast: bool = False) -> str:
    # Documented OmniRoute auto-combo names (verified against their docs):
    # "auto/coding" = best for coding, "auto/fast" = fastest available.
    # On a retry after a malformed response, switch variants so we're not
    # necessarily hitting the exact same underlying model again.
    # Previously switched to "auto/fast" on retry for "variety" -- but
    # that model tier trades quality for speed, and a retry after a
    # failure needs MORE reliability, not less. Stick with the
    # coding-optimized combo first; if it exhausts (OmniRoute's 503
    # "Maximum combo retry limit reached" -- every model IN the combo is
    # down), move DOWN the fallback chain instead of crashing the run.
    #
    # Three retry axes:
    #   1. STALL probe -- the first attempt of each combo is capped at
    #      OMNIROUTE_STALL_SECONDS. A silent/dead provider is abandoned after
    #      ~that instead of burning the whole budget; the next combo gets the
    #      same quick probe. A healthy model that returns inside the window
    #      is unaffected.
    #   2. TIMEOUT retry -- same combo, longer timeout (1x/2x/4x), but only
    #      once the combo has survived its stall probe. This preserves the
    #      "healthy gateway + slow reasoning model" case without letting an
    #      unresponsive provider eat minutes per step.
    #   3. COMBO fallback -- a non-timeout failure (503 combo exhausted,
    #      auth, etc.) means retrying the same combo is futile; move to the
    #      next combo in the chain.
    chain = _model_chain(fast=fast)
    attempts = 2
    last_error = None
    for combo_index, model in enumerate(chain):
        for attempt in range(attempts):
            # First attempt of a combo: quick stall probe. Later attempts on
            # the same combo get the full growing budget.
            timeout = (
                OMNIROUTE_STALL_SECONDS if attempt == 0
                else OMNIROUTE_TIMEOUT * (2 ** (attempt - 1))
            )
            try:
                response = ai_client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "system", "content": SDE2_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    timeout=timeout,
                )
                reply = response.choices[0].message.content
                if not reply:
                    reply = getattr(response.choices[0].message, "reasoning", None) or ""
                    if not reply:
                        _log_turn(
                            "system",
                            f"model '{model}' returned empty content "
                            f"(choices={len(response.choices)})",
                            kind="error", model=model,
                        )
                        raise RuntimeError(
                            f"model '{model}' returned an empty completion "
                            f"(no content, no reasoning) -- nothing to apply"
                        )
                _log_turn("user", prompt, kind="prompt", model=model)
                _log_turn("assistant", reply or "", kind="completion", model=model)
                return reply
            except Exception as e:
                last_error = e
                _log_turn(
                    "system",
                    f"model call attempt {attempt + 1}/{attempts} on '{model}' FAILED: {e}",
                    kind="error", model=model,
                )
                timed_out = _is_timeout_error(e)
                stall_like = timed_out and attempt == 0
                if stall_like and combo_index < len(chain) - 1:
                    wait = 5 * (combo_index + 1)
                    nxt = chain[combo_index + 1]
                    print(
                        f"⚠️ Combo '{model}' gave no answer within {int(timeout)}s "
                        f"(stall probe). Trying fallback combo '{nxt}' in {wait}s..."
                    )
                    time.sleep(wait)
                    break  # out of attempt loop -> next combo
                if timed_out and attempt < attempts - 1:
                    wait = 5 * (2 ** attempt)
                    print(
                        f"⚠️ OmniRoute timed out after {int(timeout)}s "
                        f"(attempt {attempt + 1}) on '{model}'. "
                        f"Retrying with a longer timeout in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                if combo_index < len(chain) - 1:
                    wait = 5 * (combo_index + 1)
                    nxt = chain[combo_index + 1]
                    print(
                        f"⚠️ Combo '{model}' failed hard ({type(e).__name__}): {str(e)[:200]}\n"
                        f"   Trying fallback combo '{nxt}' in {wait}s..."
                    )
                    time.sleep(wait)
                    break  # out of attempt loop -> next combo
                raise RuntimeError(
                    f"OmniRoute call failed ({e}). Is 'omniroute' running in another "
                    f"terminal, and do you have at least one provider connected in "
                    f"its dashboard at http://localhost:20128/dashboard?"
                    + (
                        f" (fallback endpoint {OMNIROUTE_FALLBACK_BASE_URL} was "
                        f"configured but also failed)"
                        if OMNIROUTE_FALLBACK_BASE_URL else ""
                    )
                )
    raise RuntimeError(
        f"OmniRoute call failed after exhausting {attempts} timeout-retries per combo "
        f"across all [{', '.join(chain)}] (last error: {last_error}). Is 'omniroute' "
        f"running in another terminal, and do you have at least one provider "
        f"connected in its dashboard at http://localhost:20128/dashboard?"
        + (
            f" (fallback endpoint {OMNIROUTE_FALLBACK_BASE_URL} was configured "
            f"but also failed)"
            if OMNIROUTE_FALLBACK_BASE_URL else ""
        )
    )


def _is_timeout_error(e: Exception) -> bool:
    """True for socket/read/API timeouts. A timeout is transient -- the
    provider is just slow, so a longer-timeout retry is worth it."""
    if type(e).__name__ in ("ReadTimeout", "APITimeoutError", "TimeoutError",
                            "ConnectTimeout", "socket.timeout"):
        return True
    msg = str(e).lower()
    return "timed out" in msg or "timeout" in msg or "read timeout" in msg


FULL_FILE_ONLY_INSTRUCTIONS = """
Respond ONLY with one or more file blocks in this exact format, no other text:

FILE: <one of the exact paths from ALLOWED FILE PATHS below>
<<<CONTENT>>>
...complete new content of the file...
<<<END>>>

ALWAYS terminate every FILE block with <<<END>>> on its own line, then
start the next FILE block (or nothing) immediately after it. If the full
new content would be very large, PREFER a diff limited to the changed
regions instead of the full-file format, so your answer is never cut off
mid-block -- a truncated response cannot be applied and wastes attempts.

Do NOT write the literal text "relative/path/to/file.py" or any other
placeholder -- always substitute a real, exact path copied character-
for-character from the ALLOWED FILE PATHS list below.

Output the COMPLETE file content, not a diff or partial snippet --
every line of the original file that you're not intentionally changing
must still be present, unchanged, in your output.

CRITICAL: Use the EXACT relative path shown in the "--- path ---" header
of each file under RELEVANT FILES below. Do not guess an alternate path,
do not create a new top-level copy of an existing file, and do not
duplicate a file under a different directory. If a file you need isn't
shown below, only then create it at a sensible new path.
"""

DIFF_REQUIRED_INSTRUCTIONS = """
At least one relevant file is too large to show you in full (marked
"[TRUNCATED]" below). For THAT file, you MUST use a unified diff -- a
full rewrite would delete everything you didn't see. For any other,
fully-visible file, either format is fine, but the full-file format
below is simpler and preferred when the whole file was shown to you.

UNIFIED DIFF FORMAT (for truncated files) -- wrapped in a diff code
block. Below is an EXAMPLE ONLY, showing required syntax -- substitute
a REAL path from ALLOWED FILE PATHS and REAL line numbers/content,
never copy this example's path or numbers literally:

```diff
--- a/example_module.py
+++ b/example_module.py
@@ -10,3 +10,3 @@
 unchanged context line
-old line being removed
+new line being added
 unchanged context line
```

FULL-FILE FORMAT (only for files that were NOT marked TRUNCATED):

FILE: <one of the exact paths from ALLOWED FILE PATHS below>
<<<CONTENT>>>
...complete new content of the file...
<<<END>>>

ALWAYS terminate every FILE block with <<<END>>> on its own line, then
start the next FILE block (or nothing) immediately after it. If the full
new content would be very large, PREFER a diff limited to the changed
regions instead of the full-file format, so your answer is never cut off
mid-block -- a truncated response cannot be applied and wastes attempts.

Do NOT write the literal text "relative/path/to/file.py" or any other
placeholder -- always substitute a real, exact path copied character-
for-character from the ALLOWED FILE PATHS list below.

CRITICAL: Use the EXACT relative path shown in the "--- path ---" header
of each file under RELEVANT FILES below. Do not guess an alternate path,
do not create a new top-level copy of an existing file, and do not
duplicate a file under a different directory.

If a file is marked "[TRUNCATED -- this is not the full file]", it is
too large to see in full. Do NOT use the full-file format on that file
-- you would delete everything you didn't see. Use a diff instead.
"""


# ============================================================
# Step 7: Generate fix (Step 6 "software role guard" folded in
# as an instruction rather than a separate Kaggle GPU pass --
# Gemini/DeepSeek already read the surrounding code conventions
# from the relevant files you feed them)
# ============================================================
REGRESSION_TEST_INSTRUCTIONS = """
MANDATORY: your patch must ALSO add a regression test, in the same
response, using the same output format as the code change.

The test has to FAIL on the code as it is right now and PASS with your
fix -- that is the only thing that proves the bug is really gone, and
it is what a maintainer will ask for anyway. So:

- assert the specific behaviour described in the issue, not that some
  function merely runs;
- add a NEW test function/case -- editing an existing one, or only
  adding a fixture/helper, does not count;
- put it where this repository already keeps its tests, and follow the
  naming convention the runner discovers (see EXISTING TEST FILES
  below when present). If the language keeps tests inside the module
  under change (Rust's `#[cfg(test)]`, for example), that is fine too;
- do not weaken or delete an existing test to make the suite pass.

A patch with no new test is rejected before it is even run.
"""

# Iterations answer maintainer feedback, which is often cosmetic ("rename this",
# "add a docstring"). Demanding a brand-new test every round would deadlock the
# conversation, so the requirement softens to a nudge once a PR exists.
REGRESSION_TEST_SOFT_NOTE = """
If your change alters behaviour, add or extend a test that covers it in
the same response. Never weaken, skip, or delete an existing test to
make the suite green.
"""


def _label_ctx(labels, difficulty=None, domain=None) -> str:
    """Small context block injected into the fix prompt when the issue was
    classified. Gives the model the labels/domain so an 'area:integration' or
    'ai' issue is approached with the right lens, and the difficulty tension
    it should not cheat its way out of."""
    parts = []
    if labels:
        parts.append(f"ISSUE LABELS: {', '.join(labels)}")
    if domain:
        parts.append(
            f"DOMAIN: this issue lives in the {domain} area -- respect that "
            f"layer's conventions (e.g. typed API surfaces, framework idioms, "
            f"model/data contracts) and do not leak concerns across layers."
        )
    if difficulty == "hard":
        parts.append(
            "DIFFICULTY: this is a HARD, cross-cutting issue. Think before "
            "patching: identify every caller and every test that touches this "
            "behaviour, keep edge cases consistent, and do not paper over the "
            "root cause with a local special-case."
        )
    return ("\n" + "\n".join(parts) + "\n") if parts else ""


def generate_fix(
    issue, relevant_files, previous_error=None, retry_variant=False, language="unknown", guidance=None,
    test_layout=None, require_test=None, failure_history=None, memory_context=None,
    labels=None, difficulty=None, domain=None, grounding=None,
):
    context = "\n\n".join(f"--- {p} ---\n{c}" for p, c in relevant_files)
    allowed_paths = "\n".join(f"- {p}" for p, _ in relevant_files)

    similar = find_similar_experiences(issue.title, language)
    precedent = ""
    if similar:
        lines = []
        for e in similar:
            lines.append(
                f'- "{e["issue_title"]}" ({e["language"]}) -> {e["outcome"]}'
                + (f": {e['notes']}" if e.get("notes") else "")
            )
        precedent = (
            "\n\nRELEVANT PAST ATTEMPTS (from this tool's own history, "
            "for context only -- this repo/issue may differ):\n" + "\n".join(lines)
        )

    # Only ask for the more error-prone diff format when it's actually
    # necessary (a truncated file is in play). Diff parsing/application
    # has repeatedly proven fragile in practice -- default to the
    # simpler, already safety-checked full-file format whenever every
    # relevant file fits in full.
    any_truncated = any("[TRUNCATED" in c for _, c in relevant_files)
    # A diff failure on the previous attempt means the model's line
    # numbers/context didn't line up. Retrying with diffs just repeats the
    # same 'corrupt patch' loop that dominates the logs -- so as soon as a
    # diff has failed once, force the reliable full-file format instead.
    prev_diff_failed = bool(previous_error) and any(
        marker in previous_error.lower()
        for marker in ("diff", "corrupt patch", "could not be located", "hunk")
    )
    if any_truncated and not prev_diff_failed:
        format_instructions = DIFF_REQUIRED_INSTRUCTIONS
    else:
        format_instructions = FULL_FILE_ONLY_INSTRUCTIONS

    prompt = f"""You are fixing a GitHub issue. Follow the existing code style,
naming conventions, and architecture visible in the files below. Do not
introduce new dependencies or restructure unrelated code.

ISSUE TITLE: {issue.title}
ISSUE DESCRIPTION:
{issue.body}
{_label_ctx(labels, difficulty, domain)}
{grounding or ''}
RELEVANT FILES:
{context}

ALLOWED FILE PATHS (use EXACTLY these paths, do not invent alternates,
do not duplicate a file under .github/, .agents/, or any other mirrored
directory even if similar files exist there):
{allowed_paths}
{precedent}

{format_instructions}
"""
    if REQUIRE_REGRESSION_TEST and (REQUIRE_REGRESSION_TEST if require_test is None else require_test):
        prompt += REGRESSION_TEST_INSTRUCTIONS
        if test_layout:
            prompt += (
                "\nEXISTING TEST FILES in this repo (match this location and "
                "naming style, and note that these paths are also allowed "
                "targets even though they are not listed above):\n"
                f"{test_layout}\n"
            )
        else:
            prompt += (
                "\nNo existing test files were detected, so create one at the "
                "conventional location for this ecosystem.\n"
            )
    elif REQUIRE_REGRESSION_TEST:
        prompt += REGRESSION_TEST_SOFT_NOTE
    if memory_context:
        prompt += (
            "\n\nPRIOR CONVERSATION / ATTEMPTS FROM THIS WORKFLOW'S OWN HISTORY "
            "(context only -- build on what was already decided, do NOT repeat "
            "already-failed approaches):\n"
            f"{memory_context[:MAX_ERROR_CHARS]}"
        )
    if previous_error:
        prompt += (
            f"\n\nPrevious attempt FAILED tests:\n{previous_error[-MAX_ERROR_CHARS:]}\nFix it."
        )
    if failure_history:
        prompt += (
            "\n\nFAILED ATTEMPTS SO FAR (do NOT repeat these approaches blindly; "
            "analyze why each one failed and pick a DIFFERENT strategy):\n"
            f"{format_failure_history(failure_history)}"
        )
        if len(failure_history) >= 3 and len({str(h.get('signal') or '')[:160]
                                              for h in failure_history}) >= 2:
            prompt += (
                "\n\nYou have failed at least 3 times with different failures. STOP and "
                "re-analyze: what is the common root cause? If you've been editing the "
                "same file, consider that the fix belongs elsewhere. State your new "
                "hypothesis in one line, then produce the patch."
            )
        # Decompensation: the SAME file failing again and again with NEW causes
        # each time usually means the change is too atomic -- several unrelated
        # problems collapsed into one oversized patch that can't be debugged as
        # a unit. Explicitly offer a smaller split so the loop can converge on
        # the part that is actually stuck instead of re-burning the budget.
        repeat_counts: dict = {}
        for h in failure_history:
            for p in (h.get("changed") or []):
                repeat_counts[p] = repeat_counts.get(p, 0) + 1
        stuck_files = sorted(
            (p for p, n in repeat_counts.items() if n >= 2),
            key=lambda p: -repeat_counts[p],
        )
        if len(failure_history) >= 3 and stuck_files:
            prompt += (
                "\n\nThese files have now failed multiple times with different "
                f"causes: {', '.join(stuck_files[:5])}. That is the signature of "
                "a change that is too big to debug as one patch. Either (a) shrink "
                "this fix to the smallest self-contained sub-change, or (b) split "
                "it into ordered smaller steps and state the first one, which "
                "must leave the tree working on its own."
            )
    if guidance:
        # Iteration mode: maintainer/user feedback + the implementation plan
        # and review checklist derived from it (Prompts A & B). This steers
        # the change toward what the reviewer actually asked for, instead of
        # re-deriving the fix from the issue text alone.
        prompt += (
            "\n\nMAINTAINER/REVIEW GUIDANCE FOR THIS ITERATION -- your change "
            "MUST address these points and preserve existing behavior:\n"
            f"{guidance[:MAX_ERROR_CHARS]}"
        )

    flat_budget = int(os.getenv("SOLVE_MAX_OUTPUT_TOKENS", "12000"))
    token_budget = HARD_MAX_TOKENS if difficulty == "hard" else flat_budget
    return call_model(prompt, max_tokens=token_budget, retry_variant=retry_variant)


def _parse_diff_hunks(diff_text: str):
    """Split a unified diff into (target_path, body_lines) hunks. Unlike
    `git apply`, this ignores the @@ line numbers and hunk counts entirely
    -- LLMs get those wrong constantly (that is the #1 cause of the
    'corrupt patch' failures in the logs). We only trust the +/-/space
    prefixes and the actual line content."""
    hunks = []
    current_path = None
    current_body = None
    for line in diff_text.split("\n"):
        if line.startswith("+++ "):
            m = re.match(r"\+\+\+ [ab]/(.+)", line.rstrip())
            current_path = m.group(1).replace("\\", "/").strip() if m else None
            current_body = None
        elif line.startswith("--- "):
            current_body = None
        elif line.startswith("@@"):
            current_body = []
            hunks.append((current_path, current_body))
        elif current_body is not None:
            if line.startswith("\\"):  # "\ No newline at end of file"
                continue
            if line and line[0] in "+- ":
                current_body.append(line)
            else:
                # Blank or prefix-less line inside a hunk -> treat as blank
                # context. This is the exact malformation git apply chokes on.
                current_body.append(" " + line)
    return [(p, b) for p, b in hunks if p and b]


def _locate_block(file_lines: list, old_block: list) -> int:
    """Find where old_block occurs in file_lines, tolerant of the whitespace
    drift LLMs introduce. Returns the start index or -1. Progressively
    relaxes matching: exact -> ignore trailing ws -> ignore all edge ws."""
    if not old_block:
        return -1
    n, m = len(file_lines), len(old_block)
    for normalize in (lambda s: s, lambda s: s.rstrip(), lambda s: s.strip()):
        target = [normalize(x) for x in old_block]
        for i in range(n - m + 1):
            if [normalize(x) for x in file_lines[i : i + m]] == target:
                return i
    return -1


def _apply_diff_fuzzy(
    repo_dir: Path, diff_text: str, allowed_paths: set = None, relevant_files: list = None
):
    """Content-based unified-diff applier used as a robust fallback when
    `git apply` reports 'corrupt patch' / 'does not apply'. Locates each
    hunk by its context+removed lines (not by line number) and swaps in the
    context+added lines. Returns the list of touched paths, or raises
    ValueError with actionable feedback for the retry loop."""
    hunks = _parse_diff_hunks(diff_text)
    if not hunks:
        return []

    # Group hunks per file so we edit each file's content once.
    by_file = {}
    for path, body in hunks:
        by_file.setdefault(path, []).append(body)

    touched = []
    for rel_path, bodies in by_file.items():
        full_path = repo_dir / rel_path
        # Same scope guard as the full-file path: reject invented files, but
        # surface the real content of a genuine file we simply hadn't shown.
        if allowed_paths and rel_path not in allowed_paths:
            if full_path.exists():
                allowed_paths.add(rel_path)
                real_content = full_path.read_text(errors="ignore")[:SURFACE_CONTENT_CHARS]
                if relevant_files is not None:
                    relevant_files.append((rel_path, real_content))
                raise ValueError(
                    f"You tried to edit '{rel_path}', which exists but wasn't "
                    f"in your context, so your diff was a guess at its content. "
                    f"Here is its ACTUAL current content -- redo the fix using "
                    f"this:\n\n{real_content}"
                )
            if is_permitted_new_test_path(rel_path):
                allowed_paths.add(rel_path)
            else:
                raise ValueError(
                    f"Your diff targets '{rel_path}', which does not exist in the "
                    f"repo and wasn't provided. Only edit the exact paths listed "
                    f"under ALLOWED FILE PATHS."
                )
        if not full_path.exists():
            raise ValueError(f"Diff targets missing file '{rel_path}'.")

        raw = full_path.read_bytes()
        uses_crlf = b"\r\n" in raw[:4000]
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
        file_lines = text.split("\n")

        for body in bodies:
            old_block, new_block = [], []
            for l in body:
                tag, content = l[0], l[1:]
                if tag == " ":
                    old_block.append(content)
                    new_block.append(content)
                elif tag == "-":
                    old_block.append(content)
                elif tag == "+":
                    new_block.append(content)
            idx = _locate_block(file_lines, old_block)
            if idx == -1:
                real_content = "\n".join(file_lines)[:SURFACE_CONTENT_CHARS]
                raise ValueError(
                    f"A hunk for '{rel_path}' could not be located -- its "
                    f"context lines don't match the real file, so the change "
                    f"was based on a guessed version. Here is the ACTUAL "
                    f"current content of '{rel_path}' -- redo the fix using "
                    f"this exact content:\n\n{real_content}"
                )
            file_lines[idx : idx + len(old_block)] = new_block

        new_text = "\n".join(file_lines)
        if uses_crlf:
            new_text = new_text.replace("\n", "\r\n")
        with open(full_path, "w", encoding="utf-8", newline="") as f:
            f.write(new_text)
        touched.append(rel_path)

    print(f"✅ Applied diff via content-based matcher (line-number independent): {touched}")
    return touched


def _normalize_hybrid_blocks(fix_text: str) -> str:
    """The Gemini fallback models often MERGE the two output formats: they
    write 'FILE: <path>' then a unified diff (or the raw file content) then
    '<<<END>>>', skipping the '<<<CONTENT>>>' marker entirely. Neither the
    fenced-diff regex nor the strict full-file regex matches that, so every
    attempt dies with 'Could not parse'. Rewrite such blocks into the exact
    shapes the existing parsers understand:
      * FILE: <path> + unified-diff body  -> fenced ```diff block
      * FILE: <path> + content + <<<END>>> -> full-file block with <<<CONTENT>>>
    Blocks that already contain <<<CONTENT>>> are left untouched.
    An empty/None response is left empty so the retry loop can re-prompt
    instead of tracebacking on pattern.sub(None, ...)."""
    if not fix_text or not str(fix_text).strip():
        return ""
    def _rewrite(m):
        path = m.group(1).strip()
        body = m.group(2)
        if "<<<CONTENT>>>" in body:
            return m.group(0)
        body = re.sub(r"[ \t]*<<<END>>>[ \t]*$", "", body).rstrip("\n")
        inner = body.strip("\n")
        if not inner:
            return m.group(0)
        looks_like_diff = (
            "\n+++ " in ("\n" + inner)
            or inner.startswith("--- ")
            or inner.startswith("+++ ")
        )
        if looks_like_diff:
            if not inner.startswith("--- ") and not inner.startswith("+++ "):
                inner = f"--- a/{path}\n{inner}"
            if "\n+++ b/" not in ("\n" + inner):
                inner = f"{inner}\n+++ b/{path}"
            if "@@" not in inner:
                inner = f"@@ -1,1 +1,1 @@\n{inner}"
            return f"```diff\n{inner}\n```"
        return f"FILE: {path}\n<<<CONTENT>>>\n{inner}\n<<<END>>>"

    pattern = re.compile(
        r"(?m)^[ \t]*FILE:[ \t]*([^\s]+)[ \t]*\n(.*?)(?=^[ \t]*FILE:|\Z)",
        re.DOTALL,
    )
    return pattern.sub(_rewrite, fix_text)


def apply_fix(
    repo_dir: Path, fix_text: str, allowed_paths: set = None, relevant_files: list = None
):
    fix_text = _normalize_hybrid_blocks(fix_text)
    # --- Preferred path: unified diff (git apply) ---
    # Diffs only touch the lines that actually changed, so a model that
    # only saw a truncated file can never accidentally wipe the rest of
    # it -- this directly closes the -3,523-line data-loss hole that the
    # old full-file-overwrite approach had. Tried first; if the model
    # didn't produce a valid diff (LLMs often get diff line numbers
    # wrong), we fall back to the full-file method below.
    diff_blocks = re.findall(r"```(?:diff|patch)?\s*\n(---\s+a/.*?)\n```", fix_text, re.DOTALL)
    if diff_blocks:
        all_touched = []
        any_applied = False
        for diff_text in diff_blocks:
            diff_text = diff_text.rstrip("\n \t")
            # Defensive normalization: some models echo back Windows-style
            # backslash paths in the diff header (--- a/x\y\z.py), which
            # git apply can't parse -- it always requires forward slashes,
            # even on Windows. Fix just the header lines, not code content
            # that might legitimately contain backslashes.
            diff_text = re.sub(
                r"^(---|\+\+\+) ([ab])/(.+)$",
                lambda m: f"{m.group(1)} {m.group(2)}/{m.group(3).replace(chr(92), '/')}",
                diff_text,
                flags=re.MULTILINE,
            )
            # Reject a touched file only if it's BOTH outside our pre-selected
            # context AND doesn't actually exist in the repo -- that combination
            # is what "invented" paths (.github/skills/, .agents/skills/
            # duplicates) look like. A file that genuinely exists but wasn't in
            # our initial keyword/AI selection (like a correctly-identified
            # style.scss our discovery step simply missed) is real and safe to
            # touch -- rejecting it punishes the model for being right.
            touched = re.findall(r"^\+\+\+ b/(.+)$", diff_text, re.MULTILINE)
            invented = [
                t
                for t in touched
                if allowed_paths
                and t not in allowed_paths
                and not (repo_dir / t).exists()
                and not is_permitted_new_test_path(t)
            ]
            missing_context = [
                t
                for t in touched
                if allowed_paths and t not in allowed_paths and (repo_dir / t).exists()
            ]
            if invented:
                print(
                    f"⚠️  Diff touches invented/nonexistent files {invented}, skipping this block."
                )
                continue
            if missing_context:
                # The file is real but the model never saw its actual content --
                # every attempt so far has guessed different plausible content
                # for it, which risks writing wrong (not just deleted) code.
                # Surface the real content so the NEXT attempt uses facts, not
                # guesses, instead of blindly trusting a hallucinated diff.
                # Also permanently allow-list it (allowed_paths is the same
                # set object reused every attempt) so this doesn't repeat.
                if allowed_paths is not None:
                    allowed_paths.add(missing_context[0])
                real_content = (repo_dir / missing_context[0]).read_text(errors="ignore")[
                    :SURFACE_CONTENT_CHARS
                ]
                if relevant_files is not None:
                    relevant_files.append((missing_context[0], real_content))
                raise ValueError(
                    f"You tried to edit '{missing_context[0]}', which exists but "
                    f"wasn't in your context, so your diff was based on a guess "
                    f"at its content, not the real file. Here is its ACTUAL "
                    f"current content -- redo the fix using this:\n\n{real_content}"
                )

            # Sanitize common LLM diff mistakes before handing it to git:
            # blank lines inside a hunk with NO leading space/+/- character
            # are technically malformed unified-diff syntax (this is what
            # caused every "corrupt patch" error above) -- treat them as
            # blank context lines by adding the required leading space.
            sanitized_lines = []
            in_hunk = False
            for line in diff_text.split("\n"):
                if line.startswith("@@"):
                    in_hunk = True
                elif line.startswith(("---", "+++")):
                    in_hunk = False
                if in_hunk and line == "":
                    line = " "
                sanitized_lines.append(line)
            diff_text = "\n".join(sanitized_lines)

            patch_file = repo_dir / ".tmp_fix.patch"
            # CRITICAL: write with explicit newline="\n" -- Path.write_text()
            # on Windows silently translates \n to \r\n by default, which
            # corrupts the patch file's byte structure. git apply parses
            # patches exactly, so this was likely the root cause behind
            # many "corrupt patch" failures even when the diff LOOKED
            # syntactically correct in every printed debug output.
            with open(patch_file, "w", encoding="utf-8", newline="\n") as f:
                f.write(diff_text)
            # Use just the filename, NOT the full path -- cwd=repo_dir
            # already positions us there, so passing the full relative
            # path again doubles it up and git can't find the file.
            # --recount tells git to recompute hunk line counts itself
            # instead of trusting the model's @@ header math, which is
            # the other common source of "corrupt patch" failures.
            check = subprocess.run(
                ["git", "apply", "--check", "--recount", patch_file.name],
                cwd=repo_dir,
                capture_output=True,
                text=True,
            )
            if check.returncode == 0:
                subprocess.run(["git", "apply", "--recount", patch_file.name], cwd=repo_dir)
                all_touched.extend(touched)
                any_applied = True
                print(f"✅ Applied unified diff touching: {touched}")
            else:
                print(
                    f"⚠️  One diff block didn't apply cleanly ({check.stderr[-200:]}), "
                    f"skipping just that block."
                )
            patch_file.unlink(missing_ok=True)

        if any_applied:
            return all_touched
        # git apply refused every block (its 'corrupt patch' / strict
        # context matching is the single biggest failure source in the
        # logs). Before giving up on the diff, retry with the tolerant,
        # line-number-independent content matcher.
        print("   ↳ git apply rejected all blocks; retrying with content-based matcher...")
        fuzzy_touched = []
        for diff_text in diff_blocks:
            fuzzy_touched.extend(
                _apply_diff_fuzzy(
                    repo_dir, diff_text.rstrip("\n \t"), allowed_paths, relevant_files
                )
            )
        if fuzzy_touched:
            return fuzzy_touched
        print(
            "   ↳ No diff block applied cleanly, trying full-file fallback in the same response..."
        )

    # --- Fallback: full-file overwrite, with the data-loss safety check ---
    strict = re.compile(r"FILE:\s*(.+?)\n<<<CONTENT>>>\n(.*?)\n<<<END>>>", re.DOTALL)
    matches = strict.findall(fix_text)
    if not matches:
        # Models occasionally hit their output limit mid-block and the final
        # FILE block never gets its closing <<<END>>>. Match those too (the
        # block runs to end-of-text or the next FILE marker), so a truncated
        # response is either applied or rejected by the data-loss check that
        # follows rather than simply unparseable.
        permissive = re.compile(r"FILE:\s*(.+?)\n<<<CONTENT>>>\n(.*?)(?=\nFILE:|\Z)", re.DOTALL)
        matches = permissive.findall(fix_text)
        if matches:
            print(
                "   ↳ Response had no complete <<<END>>> markers "
                "(likely truncated) -- applying with truncation-tolerant "
                "parser."
            )
    if not matches:
        # Last-resort safety net: some model responses ignore our wrapper
        # syntax entirely and just dump raw code. If there's exactly one
        # candidate file, and the response doesn't look like a refusal or
        # explanation (no wrapper markers found, but also no "I can't" /
        # apology language), treat the whole response as that file's
        # content -- same data-loss safety check still applies below.
        looks_like_refusal = any(
            phrase in fix_text[:300].lower()
            for phrase in ("i cannot", "i can't", "i'm unable", "sorry", "as an ai")
        )
        if (
            allowed_paths
            and len(allowed_paths) == 1
            and not looks_like_refusal
            and fix_text.strip()
        ):
            only_path = next(iter(allowed_paths))
            print(
                f"   ↳ Response had no format markers, but only one file "
                f"({only_path}) was in scope -- treating raw response as "
                f"its full content."
            )
            matches = [(only_path, fix_text.strip())]
        else:
            print(f"   ↳ Raw model response (for debugging):\n{fix_text[:4000]}")
            if diff_blocks:
                raise ValueError(
                    "Your diff was found but failed to apply cleanly (likely "
                    "incorrect context lines or line numbers). Either fix the "
                    "diff so its context lines exactly match the real file "
                    "content shown to you, or use the FILE:/<<<CONTENT>>>/"
                    "<<<END>>> fallback format instead for this file."
                )
            raise ValueError(
                "Could not parse a diff or any FILE blocks from the model's "
                "response. You MUST wrap your answer starting with either "
                "'FILE: <path>' or a ```diff code block -- do not add any "
                "other text before it."
            )
    changed = []
    skipped = []
    for rel_path, content in matches:
        rel_path = rel_path.strip()
        full_path = repo_dir / rel_path
        if allowed_paths and rel_path not in allowed_paths:
            if full_path.exists():
                # Real file, just outside our initial context -- give the
                # model its actual content instead of letting it guess,
                # and permanently allow-list it so this doesn't repeat.
                allowed_paths.add(rel_path)
                real_content = full_path.read_text(errors="ignore")[:SURFACE_CONTENT_CHARS]
                if relevant_files is not None:
                    relevant_files.append((rel_path, real_content))
                raise ValueError(
                    f"You tried to rewrite '{rel_path}', which exists but "
                    f"wasn't in your context, so you were guessing at its "
                    f"content. Here is its ACTUAL current content -- redo "
                    f"the fix using this:\n\n{real_content}"
                )
            # Doesn't exist and wasn't given -- this is an invented path,
            # same class of bug as the .github/skills/ duplicates before.
            # The one exception: a NEW test file, which is precisely what the
            # regression-test requirement asks the model to create.
            if is_permitted_new_test_path(rel_path):
                allowed_paths.add(rel_path)
            else:
                skipped.append(rel_path)
                continue
        # Safety check: if the file already exists and the model's "full
        # file" response is drastically shorter than the original, it
        # almost certainly only saw a truncated version of the file (we
        # cap context at MAX_FILE_CHARS) and is about to delete the rest.
        # This is what caused the -3,523 line loss on the manja repo.
        if full_path.exists():
            original_size = full_path.stat().st_size
            new_size = len(content.encode("utf-8"))
            if original_size > MAX_FILE_CHARS and new_size < original_size * 0.5:
                raise ValueError(
                    f"Refusing to apply fix to '{rel_path}': original file is "
                    f"{original_size} bytes but the model's replacement is only "
                    f"{new_size} bytes -- likely because the file exceeds the "
                    f"context truncation limit and the model only saw a partial "
                    f"copy. This file is too large for the current full-file-"
                    f"replace approach; skip it or handle it manually."
                )

        # Preserve the original file's line-ending convention (CRLF vs LF)
        # instead of letting Python's default text-write silently pick one
        # -- a mismatch here makes every single line show as changed in
        # the git diff, even for a one-line logical fix.
        original_uses_crlf = False
        if full_path.exists():
            original_uses_crlf = b"\r\n" in full_path.read_bytes()[:2000]
        normalized = content.replace("\r\n", "\n")
        if original_uses_crlf:
            normalized = normalized.replace("\n", "\r\n")

        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8", newline="") as f:
            f.write(normalized)
        changed.append(rel_path)
    if skipped:
        print(f"⚠️  Rejected out-of-scope file writes (not in allowed paths): {skipped}")
    if not changed:
        raise ValueError("All proposed file writes were outside the allowed paths.")
    print(f"✅ Applied changes to: {changed}")
    return changed


# ============================================================
# Step 8: Multi-input test loop with the 120s timeout that was
# missing from the original reference implementation
# ============================================================
CAPABILITY_MAP_PATH = Path("capability_map.json")
# The file ships next to the script. A copy in the working directory still wins
# (so a project can override the language table), but falling back to the
# script's own directory means the agent no longer dies with FileNotFoundError
# just because it was launched from somewhere else.
_BUNDLED_CAPABILITY_MAP = Path(__file__).resolve().parent / "capability_map.json"


def load_capability_map() -> dict:
    """Config-driven language support: adding a new language means editing
    this JSON file, never touching code. This replaces the previous
    if/elif chain, which is exactly the kind of hardcoding that caused
    'works on Python repos, breaks on others' compatibility bugs."""
    path = CAPABILITY_MAP_PATH if CAPABILITY_MAP_PATH.exists() else _BUNDLED_CAPABILITY_MAP
    if not path.exists():
        raise FileNotFoundError(
            f"capability_map.json not found at {CAPABILITY_MAP_PATH.resolve()} "
            f"or {_BUNDLED_CAPABILITY_MAP}. This file defines language support "
            f"and is required."
        )
    return json.loads(path.read_text())


def detect_language_and_commands(repo_dir: Path) -> dict:
    """Auto-detect the repo's language/build system by scoring against the
    Capability Map -- marker files are strong signals, extension frequency
    is a weaker tiebreaker. No hardcoded per-language logic here at all."""
    capability_map = load_capability_map()
    scores = {}
    for lang, config in capability_map.items():
        score = 0
        for marker in config["markers"]:
            if (repo_dir / marker).exists():
                score += 2
        for ext in config["extensions"]:
            match_count = sum(1 for _ in repo_dir.rglob(f"*{ext}"))
            score += min(match_count, 10) * 0.1
        if score > 0:
            scores[lang] = score

    if not scores:
        # Explicit, visible failure -- NOT a silent skip. The SRS flags
        # this exact case (unknown language) as a known risk area.
        print(
            "⚠️  Could not detect a known language/build system from "
            "capability_map.json. Proceeding without install/test "
            "automation -- you'll need to run tests manually, or add "
            "an entry to capability_map.json for this repo's stack."
        )
        return {
            "language": "unknown",
            "install": None,
            "test": "pytest",
            "lint": None,
            "extensions": list(CODE_EXTENSIONS),
        }

    detected = max(scores, key=scores.get)
    config = capability_map[detected]
    return {
        "language": detected,
        "install": config.get("install_cmd"),
        "test": config.get("test_cmd", "pytest"),
        "lint": config.get("lint_cmd"),
        "extensions": config.get("extensions", list(CODE_EXTENSIONS)),
    }


def run_external_command(cmd_list, cwd=None, timeout=None):
    """Run a non-Python external tool (npm, mvn, gradle, go, cargo, bundle,
    gh, git) in a way that works on both Windows and Unix. On Windows these
    are often .cmd/.bat wrapper scripts that subprocess can't locate without
    shell=True -- this is what caused WinError 2 for npm, and would hit the
    same wall for mvn/gradle/go/cargo/bundle too."""
    use_shell = os.name == "nt"
    cmd = subprocess.list2cmdline(cmd_list) if use_shell else cmd_list
    return subprocess.run(
        cmd,
        cwd=cwd,
        shell=use_shell,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _run_install_step(cmd_list, repo_dir: Path, label: str) -> bool:
    """Run one install command, printing what it does. Returns True on
    success (exit 0), False otherwise. Never raises -- install is
    best-effort, so a missing tool or timeout is reported and swallowed."""
    print(f"   ↳ {label}: {' '.join(cmd_list)}")
    try:
        result = run_external_command(cmd_list, cwd=repo_dir, timeout=300)
        if result.returncode != 0:
            print(f"      ⚠️  exited {result.returncode}: {(result.stderr or '')[-300:]}")
            return False
        return True
    except FileNotFoundError:
        print(f"      ⚠️  '{cmd_list[0]}' not found on this machine -- skipping this step.")
        return False
    except subprocess.TimeoutExpired:
        print("      ⚠️  install step timed out after 5 minutes -- continuing anyway.")
        return False


def _install_python_dependencies(repo_dir: Path) -> None:
    """Python repos declare their real dependencies in several
    incompatible ways -- a bare `pip install -e .` only covers one of
    them and fails outright on others, which is how mongoengine-style
    ModuleNotFoundErrors slip through. Try each mechanism in turn,
    best-effort, so the test env ends up with the deps regardless of how
    the repo chose to declare them."""
    py = sys.executable
    pyproject = repo_dir / "pyproject.toml"
    is_poetry = pyproject.exists() and "[tool.poetry]" in pyproject.read_text(
        encoding="utf-8", errors="replace"
    )
    has_build_system = pyproject.exists() and "[build-system]" in pyproject.read_text(
        encoding="utf-8", errors="replace"
    )

    # 1) requirements*.txt -- the most common and most reliable source of
    #    test-time deps. Install every one we find (root + common dirs).
    req_files = sorted(
        set(list(repo_dir.glob("requirements*.txt")) + list(repo_dir.glob("requirements/*.txt")))
    )
    for req in req_files:
        _run_install_step(
            [py, "-m", "pip", "install", "-r", str(req)], repo_dir, f"requirements file {req.name}"
        )

    # 2) Poetry projects (pyproject with [tool.poetry], no PEP 517
    #    build-system) can't be installed by pip's setuptools fallback --
    #    that's the "package discovery" failure. Prefer `poetry install`;
    #    if poetry isn't available, export the lock/deps to pip.
    if is_poetry and not has_build_system:
        if _run_install_step(
            ["poetry", "install", "--no-interaction", "--no-root"], repo_dir, "poetry install"
        ):
            return
        # Fallback: export poetry deps to a requirements list pip can read.
        exported = repo_dir / ".agent_requirements.txt"
        if _run_install_step(
            ["poetry", "export", "-f", "requirements.txt", "--without-hashes", "-o", str(exported)],
            repo_dir,
            "poetry export",
        ):
            _run_install_step(
                [py, "-m", "pip", "install", "-r", str(exported)],
                repo_dir,
                "install exported poetry deps",
            )
            return
        print(
            "      ⚠️  poetry not usable and no requirements*.txt -- the test "
            "suite may be missing dependencies."
        )
        return

    # 3) Standard packaged project: try editable install, and if that
    #    fails (common on setuptools auto-discovery errors), fall back to
    #    a plain non-editable install which is more forgiving.
    if (repo_dir / "setup.py").exists() or has_build_system or pyproject.exists():
        if _run_install_step(
            [py, "-m", "pip", "install", "-e", "."], repo_dir, "editable install (pip install -e .)"
        ):
            return
        _run_install_step(
            [py, "-m", "pip", "install", "."], repo_dir, "non-editable fallback (pip install .)"
        )


def install_dependencies(repo_dir: Path, detected: dict = None):
    """Best-effort install of the target repo's own dependencies, so its
    modules actually import during tests. Python gets a dedicated,
    multi-strategy path (requirements files, Poetry, editable/non-editable
    fallback); other languages use the single install command from
    capability_map.json."""
    detected = detected or detect_language_and_commands(repo_dir)

    if detected["language"] == "python":
        print("   ↳ Detected python -- running multi-strategy dependency install.")
        _install_python_dependencies(repo_dir)
        return

    if detected["install"]:
        _run_install_step(detected["install"], repo_dir, f"Detected {detected['language']}")
    else:
        print("   ↳ Could not detect a known build system -- skipping dependency install.")


def run_tests(repo_dir: Path, test_command: str):
    # Route "pytest" through the current interpreter (python -m pytest) so
    # it works even when pytest isn't directly on PATH, and add
    # --import-mode=importlib to avoid "import file mismatch" crashes on
    # monorepos with duplicate test file basenames across subfolders
    # (e.g. two different skills both having tests/test_security.py).
    cmd_parts = test_command.split()
    if cmd_parts and cmd_parts[0] == "pytest":
        cmd_parts = [sys.executable, "-m", "pytest", "--import-mode=importlib"] + cmd_parts[1:]
        use_shell = False
    else:
        # npm/mvn/gradle/go/cargo/bundle test commands -- same Windows
        # .cmd-wrapper issue as the dependency installer, same fix.
        use_shell = os.name == "nt"
    try:
        result = subprocess.run(
            " ".join(cmd_parts) if use_shell else cmd_parts,
            cwd=repo_dir,
            shell=use_shell,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TEST_TIMEOUT_SECONDS,
        )
        passed = result.returncode == 0
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        passed = False
        output = f"Test suite exceeded {TEST_TIMEOUT_SECONDS}s timeout -- possible infinite loop."
    except FileNotFoundError as e:
        passed = False
        output = f"Test command not found: {e}. Is the test runner installed?"
    print("✅ Tests passed" if passed else "❌ Tests failed")
    return passed, output


def extract_failing_tests(output: str) -> set:
    """Parse pytest output for FAILED test node ids, so we can tell which
    failures are new (caused by the fix) vs pre-existing in the repo."""
    return set(re.findall(r"^FAILED (\S+)", output, re.MULTILINE))


def blame_changed_files(
    changed_paths: list, failing_nodeids: set, output: str
) -> set:
    """Best-effort mapping from failing test node ids back to the changed
    files they implicate, for per-file staging of hard tasks.

    A FAILED node id is usually `tests/test_<mod>.py::test_...`, so we look
    for the changed file whose module matches that stem (`tests/test_foo.py`
    -> `foo.py` / `src/foo.py`). Falling back to files literally named in the
    failure output (tracebacks cite them). If nothing matches, blame *all*
    changed files -- conservative, identical to a full reset, so we never
    keep a genuinely-broken file "applied" just because the heuristic missed
    it. A false-negative is self-healing anyway: apply_fix re-surfaces any
    kept-but-needed file with its real content and re-allows it."""
    normalized = {str(Path(p)).replace("\\", "/"): p for p in changed_paths}
    blamed = set()

    def _matches(test_file: str, norm_path: str) -> bool:
        stem = Path(test_file).stem
        if stem.startswith("test_"):
            mod = stem[len("test_"):]
            if Path(norm_path).stem == mod:
                return True
            if Path(norm_path).name == f"{mod}.py":
                return True
        return False

    for node in failing_nodeids:
        test_file = node.split("::")[0].replace("\\", "/")
        for norm_path, orig in normalized.items():
            if not blamed and _matches(test_file, norm_path):
                blamed.add(orig)
                break
    if not blamed:
        norm_out = output.replace("\\", "/")
        for orig in changed_paths:
            if orig.replace("\\", "/") in norm_out:
                blamed.add(orig)
    if not blamed:
        blamed = set(changed_paths)
    return blamed


def stage_retry(repo_dir: Path, blamed: set, changed_paths: list,
                allowed_paths: set, relevant_files: list) -> set:
    """Per-file staging for a hard task on test failure (#6): revert ONLY the
    files the failing tests implicate, keep every other changed file applied,
    and narrow allowed_paths/relevant_files so the next attempt focuses on the
    blamed file instead of re-spraying the whole patch. Returns the set of
    paths kept applied and out of scope."""
    if not blamed:
        return set(changed_paths)
    for p in blamed:
        subprocess.run(["git", "checkout", "--", p], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "clean", "-fd", "--", p], cwd=repo_dir, capture_output=True)
    kept = set(changed_paths) - blamed
    if kept:
        allowed_paths.clear()
        allowed_paths.update(blamed)
        relevant_files[:] = [(p, c) for p, c in relevant_files if p in blamed]
        print(
            f"   ↳ Staged retry: reverted ONLY {sorted(blamed)}, "
            f"{len(kept)} good file(s) kept applied and out of scope."
        )
    return kept


def baseline_is_unrunnable(output: str) -> str | None:
    """Detect a test suite that can't even be *collected* -- as opposed to
    tests that run and fail. The usual cause is a missing dependency the
    install step couldn't provide (e.g. `import mongoengine` failing in
    conftest.py), which makes pytest abort at collection with an ERROR,
    not a FAILED. No code fix the agent generates can recover from this,
    so the caller should skip the issue rather than burn all its attempts
    looping on an environment problem.

    Returns a short human-readable reason string if the baseline is
    unrunnable, otherwise None."""
    if not output:
        return None
    signals = {
        r"ModuleNotFoundError: No module named ['\"]?([\w.]+)": "missing module: {0}",
        r"ImportError while loading conftest": "conftest import error (missing/broken dependency)",
        r"^ERROR .*during collection": "pytest collection error",
        r"errors? during collection": "pytest collection error",
        r"ImportError: cannot import name": "unresolved import (broken/missing dependency)",
        r"INTERNALERROR": "pytest internal error before tests ran",
    }
    # Only treat these as "unrunnable" when there's no evidence any test
    # actually executed -- if we see PASSED/FAILED node ids, the suite ran
    # and a stray ImportError is just one broken test, not a dead env.
    ran_something = bool(re.search(r"^(PASSED|FAILED) ", output, re.MULTILINE)) or bool(
        re.search(r"\d+ passed", output)
    )
    if ran_something:
        return None
    for pattern, template in signals.items():
        m = re.search(pattern, output, re.MULTILINE)
        if m:
            return template.format(*m.groups()) if m.groups() else template
    return None


# ============================================================
# Reproduction-test gate -- "did this patch actually ship a test
# that would fail without it?"
#
# A patch that turns an already-green suite green again proves
# nothing: the bug may still be there, untested. Maintainers ask for
# a regression test anyway, so requiring one up front both raises the
# odds the fix is real and makes the PR reviewable.
#
# Deliberately free: no extra LLM call (the test is requested in the
# same generate_fix prompt) and no extra test run (the evidence comes
# from the diff plus the suite run that already happens). The strict
# red-phase proof below is opt-in because it *does* cost one run.
# ============================================================
_COLLECTED_RE = re.compile(r"collected\s+(\d+)\s+item")

# Anchored on the basename or a whole path segment. A substring search would
# call src/contest/, src/protester.py and app/latest_release.py "tests".
_TEST_BASENAME_RES = [
    re.compile(r"^test_.+\.py$"),
    re.compile(r"^.+_test\.py$"),
    re.compile(r"^.+_test\.go$"),
    re.compile(r"^.+\.test\.[jt]sx?$"),
    re.compile(r"^.+\.spec\.[jt]sx?$"),
    re.compile(r"^.+Tests?\.java$"),
    re.compile(r"^.+_spec\.rb$"),
    re.compile(r"^.+_test\.rs$"),
    re.compile(r"^conftest\.py$"),
]
_TEST_DIR_SEGMENTS = {"test", "tests", "spec", "specs", "__tests__", "testing"}

# A *new case*, not just an edit inside a test file.
_ADDED_TEST_RES = [
    re.compile(r"^\+\s*(?:async\s+)?def\s+test\w*", re.M),        # python
    re.compile(r"^\+\s*func\s+Test\w+", re.M),                     # go
    re.compile(r"^\+\s*(?:it|test)\s*[.(]", re.M),                 # js/ts
    re.compile(r"^\+\s*describe\s*\(", re.M),                      # js/ts suite
    re.compile(r"^\+\s*@Test\b", re.M),                            # java
    re.compile(r"^\+\s*(?:it|describe|context)\s+['\"]", re.M),     # ruby rspec
    re.compile(r"^\+\s*#\[test\]", re.M),                          # rust
]


def collected_count(output: str):
    """How many tests the runner collected, or None if it isn't pytest."""
    if not output:
        return None
    hits = _COLLECTED_RE.findall(output)
    return int(hits[-1]) if hits else None


def _norm_rel_path(p: str) -> str:
    return (p or "").replace("\\", "/").strip().strip("/")


def looks_like_test_path(path: str) -> bool:
    """Pattern/segment match, never a bare substring search."""
    p = _norm_rel_path(path)
    if not p:
        return False
    segments = p.split("/")
    if any(s.lower() in _TEST_DIR_SEGMENTS for s in segments[:-1]):
        return True
    return any(r.match(segments[-1]) for r in _TEST_BASENAME_RES)


def _is_docs_only_path(path: str) -> bool:
    """True for documentation/content files: docs prose, markdown, restructured
    text, plain-text guides, and non-executable docs assets. A patch confined
    to such files can't break a code path, so it is exempt from the regression
    test gate."""
    p = _norm_rel_path(path).lower()
    if not p:
        return False
    if p.endswith((".md", ".markdown", ".rst", ".txt", ".adoc",
                   ".asciidoc", ".tex", ".pdf", ".svg", ".drawio")):
        return True
    segments = p.split("/")
    if any(s in ("docs", "doc", "documentation") for s in segments[:-1]):
        return True
    return p.endswith(("readme", "license", "changelog"))


def added_test_evidence(added_lines: str) -> bool:
    """True if the added lines define a new test case in any supported language."""
    return bool(added_lines) and any(r.search(added_lines) for r in _ADDED_TEST_RES)


def repo_has_no_test_infra(baseline_output: str) -> bool:
    """A repo whose suite collects nothing can't be asked for a regression test
    -- blocking there would leave the agent unable to fix anything at all."""
    if not baseline_output:
        return False
    low = baseline_output.lower()
    if "no tests ran" in low or "collected 0 items" in low:
        return True
    return "no test" in low and "found" in low


def is_permitted_new_test_path(rel_path: str) -> bool:
    """apply_fix rejects paths it never showed the model, because invented paths
    were a real failure mode (mirrored `.github/skills/` copies). A regression
    test is the one file we explicitly ask it to create, so test paths are
    exempt -- and only test paths."""
    return REQUIRE_REGRESSION_TEST and looks_like_test_path(rel_path)


def describe_test_layout(repo_dir: Path, limit: int = 8) -> str:
    """A few real test paths from this repo, so the model puts its regression
    test where the runner will actually find it. Pure filesystem walk -- free."""
    found = []
    try:
        for path in sorted(repo_dir.rglob("*")):
            if len(found) >= limit:
                break
            if not path.is_file():
                continue
            rel = path.relative_to(repo_dir).as_posix()
            if any(part in SKIP_DIRS for part in rel.split("/")):
                continue
            if looks_like_test_path(rel) and path.suffix in CODE_EXTENSIONS:
                found.append(rel)
    except Exception:
        return ""
    return "\n".join(f"- {p}" for p in found)


def added_lines_of_diff(repo_dir: Path) -> str:
    try:
        subprocess.run(["git", "add", "-A", "-N"], cwd=repo_dir, capture_output=True)
        result = subprocess.run(
            ["git", "diff", "--unified=0"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return ""
    return "\n".join(
        l for l in (result.stdout or "").splitlines() if l.startswith("+") and not l.startswith("+++")
    )


def regression_test_gate(
    changed_paths: list,
    added_lines: str,
    language: str = "unknown",
    baseline_output: str = "",
    after_output: str = None,
) -> tuple[bool, str]:
    """(ok, reason). ok=False means reject the attempt and retry with `reason`
    as guidance. `after_output=None` skips the collected-count corroboration,
    so this can run BEFORE the suite (rejecting there saves a whole test run).

    Chosen over three weaker designs: a substring heuristic (calls
    src/protester.py a test), that plus a count delta (still misses inline
    tests and conftest-only edits), and a filename-only variant (rejects
    Rust's in-module #[cfg(test)] tests). 21/21 on the behaviour matrix.
    """
    paths = [_norm_rel_path(p) for p in (changed_paths or []) if _norm_rel_path(p)]
    if not paths:
        return False, "the patch changed nothing"

    test_paths = [p for p in paths if looks_like_test_path(p)]
    code_paths = [p for p in paths if not looks_like_test_path(p)]
    has_new_case = added_test_evidence(added_lines)

    if not code_paths:
        return False, "the patch only touches tests -- a fix has to change the code that is broken"

    # Docs-only patch exemption: a change confined to documentation
    # (markdown/rst/docs assets) cannot break code, so demanding a regression
    # test for it is wrong and only burns free-tier retries. Such patches go
    # straight to the (typically empty/no-op) test run below instead. This is
    # the difference between landing a real docs PR and never landing one.
    if all(_is_docs_only_path(p) for p in code_paths):
        return True, "docs-only patch -- no regression test required"

    if repo_has_no_test_infra(baseline_output):
        return True, "repo has no runnable test suite -- regression test not required here"

    # A test may live in its own file OR inline in the changed module (normal in
    # Rust, legal in Python/Go), so either kind of evidence is enough.
    if not test_paths and not has_new_case:
        return False, (
            "no regression test: the patch changes only "
            f"{', '.join(code_paths[:3])} and adds no test that would fail without it"
        )
    if not has_new_case:
        return False, (
            f"{test_paths[0]} was edited but no new test case was added "
            "(no new test function/case among the added lines)"
        )

    before, after = collected_count(baseline_output), collected_count(after_output)
    if before is not None and after is not None and after <= before:
        return False, (
            f"the new test was not collected: the suite still runs {after} test(s) "
            f"(was {before}) -- check that the file name and location match this "
            "repo's test layout"
        )
    return True, f"regression test added in {(test_paths or code_paths)[0]}"


def red_phase_check(repo_dir: Path, changed_paths: list, test_command: str) -> tuple[bool, str]:
    """Strict proof that the new test is RED without the code fix: revert the
    non-test files, re-run, restore. Costs one extra suite run, which is why
    VERIFY_TEST_FAILS_WITHOUT_FIX defaults to off.

    Returns (proved_red, detail).
    """
    code_paths = [
        p for p in (_norm_rel_path(x) for x in (changed_paths or [])) if p and not looks_like_test_path(p)
    ]
    if not code_paths:
        return False, "nothing to revert -- no non-test file changed"
    saved = {}
    for rel in code_paths:
        f = repo_dir / rel
        saved[rel] = f.read_bytes() if f.exists() else None
    tracked = []
    for rel in code_paths:
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{rel}"], cwd=repo_dir, capture_output=True
        )
        if probe.returncode == 0:
            tracked.append(rel)
    try:
        if tracked:
            subprocess.run(["git", "checkout", "HEAD", "--"] + tracked, cwd=repo_dir, capture_output=True)
        for rel in code_paths:
            if rel not in tracked:
                (repo_dir / rel).unlink(missing_ok=True)
        red_passed, red_output = run_tests(repo_dir, test_command)
    finally:
        for rel, blob in saved.items():
            f = repo_dir / rel
            if blob is None:
                f.unlink(missing_ok=True)
            else:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(blob)
    if red_passed:
        return False, "the suite still passes without the code fix -- the new test does not reproduce the bug"
    return True, "the new test fails without the fix and passes with it"


# ============================================================
# Supervisor Check -- a judgment-based gate that runs BEFORE the
# human gate. Uses the model's reasoning (not fixed thresholds) to
# ask: "does this diff make sense for this issue?" This is advisory,
# not a replacement for your own review -- it can be wrong. Its job
# is to flag things for you to look at more closely, not to decide
# on its own.
# ============================================================
def supervisor_review(issue, repo_dir: Path, test_output: str) -> tuple[bool, str]:
    diff_result = subprocess.run(
        ["git", "diff"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    diff = (diff_result.stdout or "")[:6000]
    stat_result = subprocess.run(
        ["git", "diff", "--shortstat"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    shortstat = (stat_result.stdout or "").strip()

    prompt = f"""You are a careful senior engineer doing a pre-review of an
automated code fix, before a human sees it. Judge this holistically --
don't just check if tests passed.

ISSUE BEING FIXED:
{issue.title}
{issue.body}

DIFF STATS: {shortstat}

FULL DIFF:
{diff}

TESTS: {"passed" if "failed" not in test_output.lower() else "see output"}

Answer in EXACTLY this format, nothing else:
VERDICT: PROCEED or VERDICT: FLAG
REASON: one or two sentences explaining your judgment

Flag it if: the change is disproportionate to the issue (e.g. deleting
far more than expected, editing unrelated code), it looks like it
could break something not covered by tests, or it touches sensitive
files (CI config, secrets, auth) unnecessarily. Otherwise, proceed.

Also flag it if the diff includes a test that would still pass on the
UNFIXED code -- for example a test that only checks a function is
callable, asserts something already true before the change, or was
weakened to make the suite green. A test that cannot fail is worse
than no test, because it hides the bug.
"""
    try:
        response = call_model(prompt, max_tokens=200)
        verdict_line = next((l for l in response.splitlines() if "VERDICT" in l.upper()), "")
        reason_line = next((l for l in response.splitlines() if "REASON" in l.upper()), "")
        proceed = "PROCEED" in verdict_line.upper()
        return proceed, reason_line.replace("REASON:", "").strip() or response[:200]
    except Exception as e:
        # If the supervisor call itself fails, don't block the pipeline --
        # just tell the human it couldn't be pre-checked, so they know to
        # look more carefully themselves.
        return True, f"(Supervisor check unavailable: {e} -- review the diff yourself.)"


def sde2_review(issue, repo_dir: Path, test_output: str) -> tuple[str, str]:
    """SDE-2 self-review gate run right before a PR is created. Returns
    (status, report) where status is one of READY / READY WITH NOTES /
    NOT READY. Unlike supervisor_review (a light PROCEED/FLAG sanity veto),
    this asks for the full review report the SDE2_SYSTEM_PROMPT describes,
    and the caller BLOCKS PR creation on a NOT READY verdict.

    Fail-open on a call outage (same philosophy as supervisor_review): a
    model that can't answer must not silently strand a validated fix -- the
    human gate still applies afterwards."""
    diff_result = subprocess.run(
        ["git", "diff"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    diff = (diff_result.stdout or "")[:8000]
    stat_result = subprocess.run(
        ["git", "diff", "--shortstat"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    shortstat = (stat_result.stdout or "").strip()
    prompt = f"""Review this change as the SDE-2 engineer described in your
instructions: full diff + affected paths, correctness, edge cases, errors,
regressions, compatibility, security, performance, maintainability, tests.

ISSUE BEING FIXED:
{issue.title}
{issue.body}

DIFF STATS: {shortstat}

FULL DIFF:
{diff}

TESTS: {"passed" if "failed" not in test_output.lower() else "see output"}

Output ONLY the final report in this exact shape:
ISSUES FIXED: ...
CHECKS RUN: ... (tests/lint/build actually performed + results)
REMAINING CONCERNS: ... (or "none")
FINAL STATUS: READY | READY WITH NOTES | NOT READY
"""
    try:
        response = call_model(prompt, max_tokens=700)
    except Exception as e:
        return "READY", f"(SDE-2 review unavailable: {e} -- falls through to the human gate.)"
    status = "READY"
    for line in response.splitlines():
        if "STATUS" in line.upper():
            up = line.upper()
            if "NOT READY" in up:
                status = "NOT READY"
            elif "WITH NOTES" in up or "READY" in up:
                status = "READY WITH NOTES" if "WITH NOTES" in up else "READY"
            break
    return status, response.strip()[:3000]


# ============================================================
# Cloud gate handling (GATE_ASYNC)
#
# The three human gates are interactive `input()` prompts, which is fine on a
# terminal but fatal (EOFError "input(): lost sys.stdin") when run unattended
# on a hosted runner. With GATE_ASYNC=1 and no TTY they instead PARK: write a
# .pending file under GATE_DIR, notify Telegram with Approve/Decline buttons,
# and return None ("nothing decided yet -- come back later"). A separate gate
# poller (an operator tapping a button) writes a .decree file; a later run
# applies that decree and records the resolution in a .outcome file so whoever
# drives this can tell "approved" from "still parked". With GATE_ASYNC unset
# the gates remain the exact interactive prompts they always were.
# ============================================================
GATE_DIR = Path(os.getenv("GATE_DIR", str(AGENT_HOME / "gates")))
GATE_ASYNC = os.getenv("GATE_ASYNC", "").strip().lower() in ("1", "true", "yes", "on")
# Autonomous mode: quality gates ("human" draft-PR submit, "final" sign-off)
# are self-approved when the caller's own validation already passed, instead of
# waiting on a Telegram button tap. The destructive "close" gate is NEVER
# auto-approved -- closing someone else's PR stays an explicit operator act.
GATE_AUTO = os.getenv("GATE_AUTO", "").strip().lower() in ("1", "true", "yes", "on")
GATE_AUTO_CLOSE = os.getenv("GATE_AUTO_CLOSE", "").strip().lower() in ("1", "true", "yes", "on")
TELEGRAM_BOT_TOKEN = os.getenv("MANUAL_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
if not TELEGRAM_CHAT_ID:
    _first_id = os.getenv("TELEGRAM_ALLOWED_IDS", "").strip()
    if _first_id:
        TELEGRAM_CHAT_ID = _first_id.split(",")[0].strip()
# Fixed API IPv4s used only when DNS cannot resolve api.telegram.org (broken
# ISP resolvers / NAT64-only networks); Telegram's API endpoints never move.
_TG_API_IPS = ("149.154.167.220", "149.154.167.99", "149.154.175.100", "149.154.166.110")


def _gate_files(repo_name: str, issue_number: int, gate: str):
    """The three sidecar files (pending/decree/outcome) for one gate decision.

    Keys are filesystem-safe (owner_repo_issueN_gate) so any orchestrator can
    list GATE_DIR to see everything waiting on a human."""
    key = f"{store.safe_repo_dir(repo_name)}_issue{int(issue_number)}_{gate}"
    return (
        key,
        GATE_DIR / f"{key}.pending",
        GATE_DIR / f"{key}.decree",
        GATE_DIR / f"{key}.outcome",
    )


def _tg_send(method: str, payload: dict) -> bool:
    """Best-effort Telegram REST call (stdlib only). Falls back to fixed API
    IPs when DNS is broken so a parked gate still reaches the operator."""
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return False
    from urllib import request as _urlrequest

    body = json.dumps({"chat_id": TELEGRAM_CHAT_ID, **payload}).encode("utf-8")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"

    def _post(base: str) -> bool:
        req = _urlrequest.Request(
            base, data=body, headers={"Content-Type": "application/json"},
        )
        try:
            with _urlrequest.urlopen(req, timeout=20) as resp:
                return resp.status == 200
        except Exception:
            return False

    if _post(url):
        return True
    for ip in _TG_API_IPS:
        req = _urlrequest.Request(
            url.replace("api.telegram.org", ip),
            data=body,
            headers={"Content-Type": "application/json", "Host": "api.telegram.org"},
        )
        try:
            with _urlrequest.urlopen(req, timeout=20) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


def _notify_gate(key: str, repo_name: str, issue_number: int, gate: str,
                 review_text: str) -> None:
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        print(f"   (no Telegram config; gate '{gate}' parked without a notification)")
        return
    text = (
        f"Human gate: {gate}\n"
        f"{repo_name}#{issue_number}\n"
        f"{review_text[:900]}"
    )
    _tg_send("sendMessage", {
        "text": text,
        "reply_markup": json.dumps({
            "inline_keyboard": [[
                {"text": "Approve", "callback_data": f"decree:{key}:1"},
                {"text": "Decline", "callback_data": f"decree:{key}:0"},
            ]],
        }),
    })


def _human_gate(repo_name: str, issue_number: int, gate: str, prompt: str,
                review_text: str):
    """One decision point shared by all three gates.

    Returns True (approved), False (declined), or None (deferred -- cloud mode
    with no decision written yet, so nothing was decided and whoever called
    this should stop and come back later)."""
    key, pending, decree, outcome = _gate_files(repo_name, issue_number, gate)

    if not (GATE_ASYNC and not sys.stdin.isatty()):
        answer = input(prompt)
        return answer.strip().lower() == "y"

    GATE_DIR.mkdir(parents=True, exist_ok=True)

    # Autonomous mode: approve quality gates immediately (a record of the
    # implied decree is still written so the outcome file describes how this
    # gate was satisfied). The close gate demands GATE_AUTO_CLOSE explicitly.
    auto_ok = GATE_AUTO and (gate != "close" or GATE_AUTO_CLOSE)
    if auto_ok and not decree.exists() and not outcome.exists():
        try:
            decree.write_text(
                json.dumps({
                    "repo": repo_name, "issue": int(issue_number), "gate": gate,
                    "decision": True, "auto": True,
                    "time": datetime.now().isoformat(),
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[{gate}] auto-approved (GATE_AUTO) -- decree written")
        except OSError:
            print(f"[{gate}] GATE_AUTO enable but could not write decree: parking")

    if decree.exists():
        decision = False
        try:
            decision = bool(json.loads(decree.read_text(encoding="utf-8")).get("decision"))
        except (OSError, ValueError):
            pass
        written = False
        try:
            decree.unlink()  # a decree is consumed exactly once
            outcome.write_text(
                json.dumps({
                    "repo": repo_name, "issue": int(issue_number), "gate": gate,
                    "decision": decision, "resolved": datetime.now().isoformat(),
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            written = True
        except OSError:
            pass
        print(f"[{gate}] decree consumed: {'APPROVE' if decision else 'DECLINE'}"
              f" (outcome{' written' if written else ' NOT written -- check GATE_DIR'})")
        return decision

    if not pending.exists():
        try:
            pending.write_text(
                json.dumps({
                    "repo": repo_name, "issue": int(issue_number), "gate": gate,
                    "key": key, "time": datetime.now().isoformat(),
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            print("   (could not write .pending under GATE_DIR; gate still parks)")
        _notify_gate(key, repo_name, issue_number, gate, review_text)
    return None


# ============================================================
# Step 9: Human validation gate (synchronous -- this is why
# there's no cron scheduler in this version, see guide)
# ============================================================
def human_gate(
    repo_dir: Path, test_output: str, supervisor_verdict: bool = None, supervisor_reason: str = "",
    repo_name: str = "", issue_number: int = 0,
) -> bool:
    result = subprocess.run(
        ["git", "diff"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    diff = result.stdout or "(no diff output captured)"

    stat_result = subprocess.run(
        ["git", "diff", "--shortstat"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    shortstat = (stat_result.stdout or "").strip()
    deletions_match = re.search(r"(\d+) deletion", shortstat)
    deletions = int(deletions_match.group(1)) if deletions_match else 0

    print("\n" + "=" * 60)
    print("PROPOSED CHANGES")
    print("=" * 60)
    print(f"Diff stats: {shortstat or 'unavailable'}")
    if deletions > 100:
        print(
            f"🚨 WARNING: {deletions} lines deleted -- unusually large for a "
            f"typical bug fix. Double-check this isn't accidental data loss "
            f"(e.g. a truncated file getting overwritten) before approving."
        )
    if supervisor_verdict is not None:
        icon = "✅" if supervisor_verdict else "🚩"
        label = "PROCEED" if supervisor_verdict else "FLAGGED"
        print(f"{icon} Supervisor pre-review: {label} -- {supervisor_reason}")
    print(diff[:2000])
    print("=" * 60)
    print(f"Test result summary: {test_output[-500:]}")
    return _human_gate(
        repo_name, issue_number, "human",
        "\n[Human Gate] Approve and submit Draft PR? (y/n): ",
        f"Diff stats: {shortstat or 'unavailable'}\n{diff[:2000]}",
    )


# ============================================================
# Step 10: Draft PR dispatch, with PR-limit enforcement
# ============================================================
def _signed_commit(repo_dir, message, check=False):
    """Run `git commit` with a DCO sign-off.

    `-s` makes git append a `Signed-off-by: <name> <email>` trailer derived
    from the commit identity -- this is the Developer Certificate of Origin
    line that DCO bots require on *every* commit in a PR (a commit-level
    trailer, so it covers all files staged in that commit). When
    SIGNOFF_NAME/SIGNOFF_EMAIL are set we pass them via `-c` so the commit
    AUTHOR matches the sign-off identity (strict DCO checks compare the two);
    otherwise git uses the repo's own configured user.name/user.email, which
    is the user's real identity -- we never invent one.

    Returns the CompletedProcess. Raises CalledProcessError when check=True
    and the commit fails.
    """
    name, email = os.getenv("SIGNOFF_NAME"), os.getenv("SIGNOFF_EMAIL")
    cmd = ["git"]
    if name and email:
        cmd += ["-c", f"user.name={name}", "-c", f"user.email={email}"]
    cmd += ["commit", "-s", "-m", message]
    cp = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cmd, cp.stdout, cp.stderr)
    return cp


def _ensure_dco_sign_off(repo_dir, check=False):
    """Verify the latest commit's message ends with a DCO `Signed-off-by:`
    trailer, amending it in if missing. GitHub's DCO check rejects a push when
    the tip commit lacks the sign-off line, so this runs right before each
    push -- a no-op on already-signed commits, and never invents an identity
    (same SIGNOFF_* override as _signed_commit). Returns the CompletedProcess
    of the final command; raises CalledProcessError when check=True."""
    probe = subprocess.run(
        ["git", "log", "-1", "--format=%B"], cwd=repo_dir, capture_output=True, text=True
    )
    message = (probe.stdout or "").rstrip("\n")
    if probe.returncode != 0 or not message:
        if check:
            raise subprocess.CalledProcessError(
                probe.returncode, ["git", "log", "-1"], probe.stdout, probe.stderr
            )
        return probe
    if any(line.strip().startswith("Signed-off-by:") for line in message.splitlines()[-3:]):
        return probe
    name, email = os.getenv("SIGNOFF_NAME"), os.getenv("SIGNOFF_EMAIL")
    cmd = ["git"]
    if name and email:
        cmd += ["-c", f"user.name={name}", "-c", f"user.email={email}"]
    cmd += ["commit", "--amend", "-s", "--no-edit"]
    cp = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cmd, cp.stdout, cp.stderr)
    return cp


def find_existing_pr(repo_name: str, branch_name: str, repo=None):
    """READ-ONLY: the open PR already pushed from `branch_name`, or None.

    Exists to make a second PR for the same workflow impossible. Without it, a
    resumed run whose record lost its `pr_number` (crash, corrupt file, manual
    edit) would happily open PR #2 for work maintainers are already reviewing in
    PR #1 -- noisy for them and confusing for everyone.

    Tries `gh` first, falls back to the REST API, and returns None if neither
    can answer. None means "unknown", so callers must treat it as a soft signal:
    the record remains the primary source of truth."""
    fork_owner = ""
    try:
        fork_owner = gh.get_user().login
    except Exception:
        pass
    heads = [f"{fork_owner}:{branch_name}", branch_name] if fork_owner else [branch_name]
    for head in heads:
        try:
            res = subprocess.run(
                ["gh", "pr", "list", "--repo", repo_name, "--head", head,
                 "--state", "open", "--json", "number,url", "--limit", "5"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=60,
            )
            if res.returncode == 0 and (res.stdout or "").strip():
                found = json.loads(res.stdout)
                if found:
                    return int(found[0]["number"]), found[0].get("url", "")
        except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError):
            pass
    try:
        target = repo if repo is not None else gh.get_repo(repo_name)
        for pull in target.get_pulls(state="open"):
            if pull.head.ref == branch_name:
                return int(pull.number), pull.html_url
    except Exception as exc:
        print(f"   ↳ Could not check for an existing PR ({exc}); relying on the local record.")
        return None
    return None


def submit_draft_pr(repo_dir, repo_name, branch_name, issue, state, record=None):
    fork_owner = gh.get_user().login
    # Never open a second PR for work that already has one. The record wins if
    # it knows; otherwise ask GitHub before creating anything.
    known = (record or {}).get("pr_number")
    if known:
        print(f"↩️  PR #{known} already tracks this workflow -- updating it instead of "
              f"creating another.\n   Use: python oss_agent_v2.py --repo {repo_name} "
              f"--issue {issue.number} -conversation")
        return int(known), (record or {}).get("pr_url", "")
    existing = find_existing_pr(repo_name, branch_name)
    if existing:
        number, url = existing
        print(f"↩️  Branch {branch_name} already has open PR #{number} -- adopting it "
              f"instead of opening a duplicate.\n"
              f"   Nothing was pushed. Your new work is still in {repo_dir};\n"
              f"   the next conversation round commits and pushes it to PR #{number}:\n"
              f"      python oss_agent_v2.py --repo {repo_name} "
              f"--issue {issue.number} -conversation")
        _transcript(f"duplicate-PR guard: adopted existing PR #{number} for {branch_name}")
        return number, url
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    _signed_commit(repo_dir, f"fix: resolve #{issue.number} {issue.title}", check=True)
    _ensure_dco_sign_off(repo_dir, check=True)
    subprocess.run(["git", *_git_auth_args(), "push", "-u", "origin", branch_name], cwd=repo_dir, check=True)
    # TOCTOU guard: the startup check ran many minutes ago (clone, tests, AI
    # generation). Re-verify against the freshest state right before we create
    # the PR, so a limit reached in the meantime still aborts us.
    check_pr_limits(state, repo_name)
    create = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--draft",
            "--fill",
            "--repo",
            repo_name,
            "--head",
            f"{fork_owner}:{branch_name}",
            "--body",
            f"Closes #{issue.number}.\n\n"
            f"Fix implemented and verified locally.",
        ],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if create.returncode != 0:
        # gh surfaces the real rejection reason ONLY on stderr; a bare
        # CalledProcessError would leave the operator staring at "exit status 1"
        # with no idea what went wrong (the workflow used to die exactly like
        # this on NewsGraph#6: the fork had been created before upstream
        # rewrote its root commit, so the branches shared no common ancestor).
        gh_err = (create.stderr or "").strip()
        gh_out = (create.stdout or "").strip()
        detail = gh_err or gh_out or "no output from gh"
        hint = ""
        if "no history in common" in detail:
            hint = (
                "\n"
                "   The fork and the upstream base branch share NO common history —\n"
                "   typically the fork was created before upstream force-pushed/amended\n"
                "   its root commit. A PR can never be opened from that branch as-is.\n"
                "   Fix: rebuild the branch on the current upstream base, e.g.\n"
                "       git fetch https://github.com/<owner>/<repo>.git <base>:refs/remotes/upstream/<base>\n"
                "       git checkout -B <branch> upstream/<base>\n"
                "       <re-apply the fix>\n"
                "       git push --force origin <branch>\n"
                "   then re-run this workflow."
            )
        if "workflow" in detail and "scope" in detail:
            hint = (
                "\n"
                "   gh rejected the push/PR because this token lacks the GitHub\n"
                "   'workflow' scope needed to touch .github/workflows. Re-auth with a\n"
                "   token that includes 'workflow' (gh auth refresh --scopes workflow)."
            )
        raise RuntimeError(
            f"gh pr create failed for {repo_name}.\n"
            f"   stderr: {detail}\n"
            f"   stdout: {gh_out}\n\n"
            f"   Branch '{branch_name}' is already pushed to {fork_owner}; nothing on\n"
            f"   GitHub was otherwise changed. You can fix the branch and re-create the\n"
            f"   PR by hand, or fix the issue above and re-run the workflow.{hint}"
        )
    # `gh pr create` prints the PR URL on success; capture it so the workflow
    # record can address the PR later (comments, updates, finalize).
    pr_url = ""
    m = re.search(r"https?://\S+/pull/(\d+)", (create.stdout or "") + (create.stderr or ""))
    pr_number = int(m.group(1)) if m else None
    if m:
        pr_url = m.group(0)
    # Atomic, lock-guarded increment (no lost updates across concurrent runs).
    updated = record_pr_created(repo_name)
    state.clear()
    state.update(updated)
    print(f"✅ Draft PR created{f' ({pr_url})' if pr_url else ''}.")
    return pr_number, pr_url


# ============================================================
# Iterative review loop -- multi-prompt agents (A-F) + feedback intake.
# A draft PR moves the workflow to WAITING_FOR_FEEDBACK; each subsequent
# invocation resumes here, processes new feedback, and updates the PR.
# ============================================================
def _pr_diff_text(repo_dir: Path, base_branch: str) -> str:
    """The PR's current diff (base..HEAD) -- what a reviewer is looking at."""
    res = subprocess.run(
        ["git", "diff", f"{base_branch}...HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (res.stdout or "")[:6000]


def fetch_new_feedback(issue, pr=None) -> list:
    """Collect maintainer/user communication attached to the issue and PR:
    issue comments, PR issue-comments, PR review bodies, and inline review
    comments. Returns normalised dicts; dedup/own-comment filtering happens in
    filter_unprocessed(). Best-effort -- a failure on one channel doesn't lose
    the others."""
    items = []

    def _add(obj, kind):
        try:
            items.append(
                {
                    "id": getattr(obj, "id", None),
                    "author": getattr(getattr(obj, "user", None), "login", "") or "",
                    "body": getattr(obj, "body", "") or "",
                    "kind": kind,
                    # Review verdict (APPROVED / CHANGES_REQUESTED / COMMENTED).
                    # Carried through so signal_for_item() can read an approval
                    # that has no body text at all.
                    "state": getattr(obj, "state", "") or "",
                    "created_at": str(getattr(obj, "created_at", "")),
                }
            )
        except Exception:
            pass

    try:
        for c in issue.get_comments():
            _add(c, "issue_comment")
    except Exception as e:
        print(f"⚠️  Could not read issue comments ({e}).")
    if pr is not None:
        for getter, kind in (
            ("get_issue_comments", "pr_comment"),
            ("get_review_comments", "review_comment"),
            ("get_reviews", "review"),
        ):
            try:
                for obj in getattr(pr, getter)():
                    _add(obj, kind)
            except Exception:
                pass
    return items


def analyze_requirements(
    issue, pr_diff_text: str, feedback_text: str, memory_context: str = "",
    grounding: str = "",
) -> str:
    """Prompt A -- Requirement Analyzer. Turns the original task + current PR +
    latest communication into a structured, actionable implementation plan.
    `memory_context` carries this workflow's own prior-round transcript so the
    plan builds on (instead of re-deriving) what was already decided.
    `grounding` carries real codebase facts (see collect_codebase_facts) so a
    vague issue ("add resultpage in that thing") resolves its referents
    against actual files instead of model guesses."""
    prompt = f"""You are a requirement analyst on an ongoing pull request.
Read the original issue, the CURRENT PR diff, and the LATEST maintainer/user
communication, then produce a concise, structured implementation plan.

ORIGINAL ISSUE: {issue.title}
{issue.body or ""}
{grounding or ""}

CURRENT PR DIFF (may be empty on first pass):
{pr_diff_text or "(none yet)"}

LATEST COMMUNICATION:
{feedback_text or "(none)"}

CONTEXT FROM EARLIER ROUNDS (what was already tried/agreed):
{memory_context or "(first round)"}

Plan the NEXT step only; do not re-plan already-finished work.

Output these sections, nothing else:
ACTIONABLE REQUIREMENTS: numbered, each a concrete change.
ACCEPTANCE CRITERIA: how we know each is satisfied.
REQUIRED TESTS: tests to add/update (or "none").
AMBIGUITIES: anything unclear the maintainer must clarify (or "none").
"""
    try:
        return call_model(prompt, max_tokens=700)
    except Exception as e:
        return f"(Requirement analysis unavailable: {e})"


def analyze_review_feedback(pr_diff_text: str, feedback_items: list) -> list:
    """Prompt B -- Review/Guidance Analyzer. Maps each comment to an action and
    returns a checklist. Also asked to flag conflicting/duplicate/stale
    feedback so downstream steps don't act on contradictions."""
    if not feedback_items:
        return []
    joined = "\n\n".join(
        f"[{it.get('kind')}] {it.get('author')}: {it.get('body')}" for it in feedback_items
    )
    prompt = f"""You are reviewing maintainer feedback on a pull request. For
each comment, decide what it requires: CODE, TEST, DOCS, REFACTOR, or
EXPLANATION-ONLY. Flag any comments that CONFLICT with each other, are
DUPLICATES, or look STALE (already addressed by the current diff).

CURRENT PR DIFF:
{pr_diff_text or "(none)"}

FEEDBACK:
{joined}

Output one checklist item per line, each starting with the category in
brackets, e.g. "[CODE] handle the null case in parse()". Put "[SKIP] ..."
for stale/duplicate/explanation-only items with a one-line reason. Nothing
else."""
    try:
        resp = call_model(prompt, max_tokens=700)
    except Exception as e:
        return [f"[CODE] (review analysis unavailable: {e}) -- address feedback manually"]
    items = [ln.strip() for ln in resp.splitlines() if ln.strip().startswith("[")]
    return items


def actionable_checklist(checklist: list) -> list:
    """Items that require real work (everything except [SKIP]/explanation-only)."""
    return [c for c in checklist if not c.upper().startswith("[SKIP]")]


def _gh_pr(args: list, repo_name: str, repo_dir: Path):
    """Run a `gh pr ...` subcommand, tolerating failure (returns bool ok)."""
    res = subprocess.run(
        ["gh", "pr", *args, "--repo", repo_name],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        print(f"⚠️  `gh pr {args[0]}` failed: {(res.stderr or '').strip()[:200]}")
    return res.returncode == 0, (res.stdout or "")


def update_pr(record: dict, repo_dir: Path, branch: str, repo_name: str, summary: str) -> bool:
    """Prompt F -- PR Update Agent. Commit + push the iteration's changes to the
    existing PR branch and post a structured summary comment. Deliberately does
    NOT close the PR: this is a checkpoint, not completion."""
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True
    )
    if (status.stdout or "").strip():
        _signed_commit(
            repo_dir, f"iterate: address review feedback (#{record['issue']})"
        )
    _ensure_dco_sign_off(repo_dir, check=True)
    push = subprocess.run(
        ["git", *_git_auth_args(), "push", "origin", branch], cwd=repo_dir, capture_output=True, text=True
    )
    if push.returncode != 0:
        print(f"⚠️  Push failed: {(push.stderr or '').strip()[:200]}")
        return False
    pr_no = record.get("pr_number")
    if pr_no:
        _gh_pr(["comment", str(pr_no), "--body", summary], repo_name, repo_dir)
    print("✅ PR updated with the latest iteration.")
    return True


def build_iteration_summary(iteration: dict, tests_result: str) -> str:
    """The comment posted to the PR after each iteration (auditable trail)."""
    lines = []
    if iteration.get("action_taken"):
        lines.append(f"Changed: {iteration['action_taken']}")
    if iteration.get("plan_summary") and "review feedback" not in str(iteration.get("plan_summary", "")):
        lines.append(f"Why: {iteration['plan_summary']}")
    if tests_result:
        lines.append(f"Tests: {tests_result}")
    if iteration.get("unresolved"):
        lines.append("Still open: " + "; ".join(iteration["unresolved"][:6]))
    if not lines:
        lines.append("No code changes this round.")
    return "\n".join(lines)


def final_approval_gate(record: dict, repo_dir: Path, base_branch: str) -> bool:
    """Mandatory human approval before finalization. Shows the final review
    packet and requires an explicit 'y'. A user completion signal is NOT enough
    on its own -- a human still confirms here."""
    diff = _pr_diff_text(repo_dir, base_branch)
    stat = subprocess.run(
        ["git", "diff", "--shortstat", f"{base_branch}...HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    files = subprocess.run(
        ["git", "diff", "--name-only", f"{base_branch}...HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print("\n" + "=" * 60)
    print("FINAL REVIEW -- explicit approval required before finalizing")
    print("=" * 60)
    print(f"PR: {record.get('pr_url') or record.get('pr_number')}")
    print(f"Iterations completed: {len(record.get('iterations', []))}")
    print(f"Files changed:\n{(files.stdout or '').strip() or '(none)'}")
    print(f"Diff stats: {(stat.stdout or '').strip() or 'unavailable'}")
    unresolved = record.get("unresolved_feedback", [])
    print(f"Unresolved feedback: {'; '.join(unresolved) if unresolved else 'none'}")
    print("-" * 60)
    print(diff[:2000])
    print("=" * 60)
    print(
        "All requested changes have been implemented and validated. "
        "Please approve the final changes to proceed."
    )
    return _human_gate(
        record.get("repo", ""), record.get("issue", 0), "final",
        "\n[Final Approval] Finalize and sign off this PR? (y/n): ",
        f"Files changed:\n{(files.stdout or '').strip() or '(none)'}\n"
        f"Diff stats: {(stat.stdout or '').strip() or 'unavailable'}",
    )


def finalize_workflow(record, repo_dir, branch, repo_name, base_branch, test_command) -> bool:
    """Post-approval finalization: verify tree, re-run tests once, commit any
    residue, push, post a final summary, optionally sign off with an authorized
    identity, then mark COMPLETED. No further edits after this."""
    wf_advance(record, WF.FINAL_VALIDATION, "human approved")
    passed, output = run_tests(repo_dir, test_command)
    if not passed:
        new_fail = extract_failing_tests(output)
        if new_fail:
            print("❌ Final validation failed -- not finalizing. Returning to iteration.")
            wf_advance(record, WF.WAITING_FOR_FEEDBACK, "final validation failed")
            return False
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True
    )
    wf_advance(record, WF.COMMIT, "final validation passed")
    if (status.stdout or "").strip():
        # DCO sign-off is applied by _signed_commit (git -s), consistent with
        # the draft-PR and iteration commits so every commit in the PR is
        # signed off.
        _signed_commit(repo_dir, f"finalize: resolve #{record['issue']}")
    _ensure_dco_sign_off(repo_dir, check=True)
    subprocess.run(["git", *_git_auth_args(), "push", "origin", branch], cwd=repo_dir, capture_output=True)
    wf_advance(record, WF.SIGN_OFF, "pushed final")
    if record.get("pr_number"):
        summary = (
            "### ✅ Finalized\n\n"
            f"Completed after {len(record.get('iterations', []))} iteration(s). "
            "All requested changes implemented and validated; final tests pass."
        )
        _gh_pr(["comment", str(record["pr_number"]), "--body", summary], repo_name, repo_dir)
    wf_advance(record, WF.COMPLETED, "signed off")
    print("🎉 Workflow COMPLETED. No further edits unless a new review cycle is started.")
    return True


# ============================================================
# Read-only investigation flow (--analyze)
#
# The agent's ORIGINAL job is "solve a small issue -> PR". A general-purpose
# engineering capability needs a second, fundamentally different output shape:
# an answer. This flow profiles the repo, pulls the files an issue/topic
# touches, checks who depends on them, runs (best-effort) baseline tests, and
# writes a structured findings report to .agent_data/reports/ -- WITHOUT a fork,
# a branch, a commit, a PR, or an issue claim. It is what makes "understand
# this repo", "review this area", "trace this data flow", and "why is this
# broken" load-bearing instead of just bug-fix attempts.
# ============================================================

def _analysis_clone(repo_name: str) -> Path:
    """Clone the ORIGINAL repo (never a fork, never an auth URL) into the
    reports area for read-only inspection. Reuses an existing clone so repeated
    investigations don't re-download. Returns the repo_dir Path."""
    repo_dir = REPORTS_DIR / store.safe_repo_dir(repo_name) / "src"
    if (repo_dir / ".git").exists():
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"📥 Cloning {repo_name} (read-only analysis) ...")
    clone_url = gh.get_repo(repo_name).clone_url
    proc = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(repo_dir)],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Could not clone {repo_name} for analysis: {proc.stderr[-400:]}"
        )
    return repo_dir


def investigate_report(
    repo_name: str,
    issue_number=None,
    test_command: str = "pytest",
    scope_text: str = "",
    task_override: str = "",
) -> str:
    """--analyze entry point. Produces a findings REPORT (saved to
    ``.agent_data/reports/<owner-repo>/issue-N.md`` or ``repo.md``); returns
    its path. Never changes anything: no fork, no branch, no commit, no PR, no
    claim, no comment."""
    repo_dir = _analysis_clone(repo_name)

    issue = None
    title, body = "", ""
    if issue_number:
        try:
            issue = gh.get_repo(repo_name).get_issue(number=issue_number)
            title, body = issue.title, issue.body or ""
        except Exception as e:
            print(f"   ↳ Could not load issue #{issue_number}: {e}; analyzing the repo root only.")
            title, body = repo_name, scope_text or ""
    if not title:
        title = f"Investigation of {repo_name}"
        body = scope_text or "Survey the repository and report on its structure and health."

    detected = detect_language_and_commands(repo_dir)
    eff_test = test_command
    if eff_test == "pytest" and detected["test"] != "pytest":
        eff_test = detected["test"]
    labels = extract_labels(issue) if issue is not None else []
    # Investigation gets the HARD context budget -- understanding ordering
    # matters more than saving tokens here.
    relevant_files = find_relevant_files(
        repo_dir, title, body,
        labels=labels,
        max_files=HARD_CONTEXT_FILES,
        max_chars=HARD_FILE_CHARS,
    )
    allowed = [p for p, _ in relevant_files]
    impact = change_impact(repo_dir, allowed)
    grounding = collect_codebase_facts(repo_dir, issue, relevant_files)
    layout = describe_test_layout(repo_dir)

    print("Step 8a: baseline tests (best-effort, read-only)...")
    test_status = "not run"
    try:
        _, baseline_out = run_tests(repo_dir, eff_test)
        fails = extract_failing_tests(baseline_out)
        passed = not fails and "error" not in baseline_out.lower()[:200]
        test_status = "PASS" if passed else (f"FAIL ({len(fails)} failing)" if fails else "ERRORED (collection/dependency)")
    except Exception as e:
        test_status = f"could not run ({e})"

    task_type = task_override.upper()
    if not task_type and issue is not None:
        try:
            task_type = classify_issue(issue).get("task_type", "")
        except Exception:
            task_type = ""

    task_note = f"TASK TYPE: {task_type}" if task_type else ""

    context = "\n\n".join(f"--- {p} ---\n{c}" for p, c in relevant_files) or "(no code files matched)"
    prompt = f"""You are a senior software engineer doing a READ-ONLY investigation
of the repository {repo_name}. Produce a findings REPORT. You MUST NOT change
code and MUST NOT propose changes as if they were already made -- every claim
must cite a real file/line from GROUNDING FACTS or RELEVANT FILES, or be
explicitly labelled as a hypothesis.

REPO: {repo_name}
ISSUE TITLE: {title}
ISSUE BODY: {body[:4000]}
{task_note}

TEST STATUS (best-effort): {test_status}
TEST LAYOUT: {layout}
{grounding}
{impact}

RELEVANT FILES:
{context[:30000]}

Write the report as plain markdown with EXACTLY these sections:

## Summary
(2-4 sentences: what the subject is and the headline finding/hypothesis.)

## Evidence
- bullet list of concrete observations, each tied to a real file path
  (and line number when you can give one). Distinguish CONFIRMED (directly
  visible in the files above) from SUSPECTED (inferred but not proven) from
  NEEDS-VERIFICATION (would need a runtime test).

## Analysis
- what the code does, the data/call flow, how the pieces relate. Explain the
  mechanism that would cause the reported problem if it is a bug.

## Risks & Open Questions
- anything missing, fragile, or unverifiable from static reading alone.

## Recommendations
- concrete next steps (a command to run, a test to write, a design note).
  Do NOT ship code here; this is an investigation, not an implementation.

Be precise, cite real files only, and do not pad.
"""
    try:
        report = call_model(prompt, max_tokens=4000)
    except Exception as e:
        report = (
            "## Summary\n\nModel unavailable during investigation; report skeleton "
            f"saved anyway.\n\n## Evidence\n\n- CONFIRMED: repository cloned, "
            f"language={detected['language']}, test status: {test_status}\n\n"
            f"## Notes\n\n{scope_text or '(no issue/task text supplied)'}\n\n"
            f"Investigation error: {e}"
        )

    slug = f"issue-{issue_number}" if issue_number else "repo-analysis"
    out = REPORTS_DIR / store.safe_repo_dir(repo_name) / f"{slug}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Investigation: {repo_name} {('#' + str(issue_number)) if issue_number else ''}\n\n", encoding="utf-8")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\n📄 Analysis report saved to:\n   {out}")
    return str(out)


# ============================================================
# Orchestrator
# ============================================================
def main(repo_name: str, issue_number: int, test_command: str, force_workspace: bool = False):
    state = load_state()
    check_pr_limits(state, repo_name)

    repo = gh.get_repo(repo_name)
    issue = repo.get_issue(number=issue_number)
    print(f"Step 1-2: checking #{issue.number} - {issue.title}")

    if issue.state != "open":
        print(f"❌ Issue is already {issue.state} (likely resolved by someone else). Skipping.")
        log_experience(
            repo_name,
            issue,
            "unknown",
            "skipped",
            error_category="B",
            notes="Already closed/not open.",
        )
        return
    if issue.assignees:
        print(f"❌ Issue already assigned to {[a.login for a in issue.assignees]}. Skipping.")
        log_experience(
            repo_name, issue, "unknown", "skipped", error_category="B", notes="Already assigned."
        )
        return
    if issue.locked:
        print("❌ Issue is locked -- no one outside maintainers can comment. Skipping.")
        log_experience(
            repo_name, issue, "unknown", "skipped", error_category="B", notes="Issue locked."
        )
        return

    classification = classify_issue(issue)
    # Analysis-shaped tasks (investigate/review/security/perf) deliver a findings
    # report, not a code change -- route them to --analyze before the fix box
    # gets a chance to reject them or half-solve them as code.
    if is_report_only_task(classification.get("task_type")):
        print(
            f"   ↳ Report-only task ({classification['task_type']}) -- switching to "
            "read-only investigation flow."
        )
        path = investigate_report(
            repo_name, issue_number, test_command, task_override=classification["task_type"],
        )
        print(f"\n🛑 Analysis saved to {path}. Nothing on GitHub was changed.")
        return "analysis"
    if not classification["accepted"]:
        print(
            f"❌ Not a workable issue ({classification['kind']}/{classification['scope']}). "
            f"Skipping. {classification['reason']}"
        )
        log_experience(
            repo_name,
            issue,
            "unknown",
            "skipped",
            notes=f"Classified as {classification['kind']}/{classification['scope']}: "
                  f"{classification['reason']}",
        )
        return

    # Vague/underspecified reports ("add resultpage in that thing") are the
    # classic hallucination trigger: the model has to guess what/where/which.
    # Detect them cheaply and treat them as HARD -- plan-first + wider context
    # grounded in the real repo, instead of hoping the flat budget survives
    # attempts full of invented files.
    vague = is_vague_issue(issue)
    effective_difficulty = (
        "hard" if classification["difficulty"] == "hard" or vague
        else classification["difficulty"]
    )

    # Anti-collision guard only -- we do NOT claim yet. We defer the claim
    # until we actually have a validated, optimal fix in hand (below), so we
    # never post a claim on an issue we can't solve.
    if not check_not_already_claimed(issue):
        log_experience(
            repo_name, issue, "unknown", "skipped", notes="Already claimed by someone else."
        )
        return

    # Start (or resume) the persisted, resumable workflow record. From here on
    # the run is tracked through the state machine; a draft PR will NOT end it.
    record = get_or_create_workflow(repo_name, issue_number, issue.title)
    record["issue_url"] = issue.html_url
    record["issue_state"] = issue.state
    record["base_branch"] = repo.default_branch
    # Difficulty signal survives into the record (labels -> conversation rounds
    # keep the escalation context, and --status shows why a run is using a
    # wider budget than usual).
    record["labels"] = classification.get("labels") or []
    record["difficulty"] = effective_difficulty
    record["vague"] = vague
    record["domain"] = classification.get("domain") or ""
    record["classification"] = {
        "kind": classification["kind"],
        "scope": classification["scope"],
        "difficulty": classification["difficulty"],
        "domain": classification["domain"],
        "task_type": classification.get("task_type") or _default_task_type(classification["kind"]),
        "reason": classification["reason"],
    }
    try:
        record["bot_login"] = gh.get_user().login
    except Exception:
        pass
    # Durable session identity: from here every model turn and every state change
    # is appended to this workflow's own transcript, so an interrupted run leaves
    # evidence behind instead of nothing.
    attach_conversation(record, provider="omniroute", model=OMNIROUTE_MODEL,
                        purpose="the one-shot command")
    resume_workflow_record(record, "resumed by the one-shot command")
    save_workflow(record)
    if record["state"] == WF.TASK_RECEIVED:
        wf_advance(record, WF.ANALYZING, "issue verified + unclaimed")

    repo_dir, branch_name = fork_clone_and_branch(
        repo, issue_number, force_workspace=force_workspace
    )
    record["branch"] = branch_name
    record["workspace"] = store.rel_to_home(repo_dir)
    save_workflow(record)
    detected = detect_language_and_commands(repo_dir)
    print(f"   ↳ Detected language/build system: {detected['language']}")
    install_dependencies(repo_dir, detected)
    if test_command == "pytest" and detected["test"] != "pytest":
        print(f"   ↳ Overriding default test command with detected one: {detected['test']}")
        test_command = detected["test"]

    # Difficulty escalation: this one issue gets its own budget. Hard / scoped-
    # enhancement work (and any VAGUE issue, which gets escalated above) is
    # allowed more attempts, wider context and a mandatory plan-first step
    # instead of the flat 5-attempt box.
    budget = escalation_budget(classification["kind"], effective_difficulty)
    labels = classification.get("labels") or []
    relevant_files = find_relevant_files(
        repo_dir, issue.title, issue.body,
        labels=labels,
        max_files=budget["context_files"],
        max_chars=budget["file_chars"],
    )
    allowed_paths = {p for p, _ in relevant_files}
    # Reverse-dependency surface: which repo files depend on the ones selected
    # for editing. Fed to the model so a correct multi-file change stops
    # surprising callers it never saw (and the impact block is evaluated once,
    # before any fix attempt, not recomputed every retry).
    impact_block = change_impact(repo_dir, list(allowed_paths))

    print("Step 8a: running baseline tests (before any changes)...")
    _, baseline_output = run_tests(repo_dir, test_command)

    # If the suite can't even be collected -- almost always a dependency
    # that install couldn't provide (needs a live DB, an uninstallable
    # package, a system lib) -- no generated code fix can turn it green.
    # Skip cleanly and log it as an environment issue instead of burning
    # all MAX_ATTEMPTS looping on the same ImportError.
    unrunnable = baseline_is_unrunnable(baseline_output)
    if unrunnable:
        print(f"🛑 Test suite is unrunnable on a clean checkout ({unrunnable}).")
        print(
            "   ↳ This is an environment/dependency problem, not a fixable code "
            "bug -- skipping this issue so the agent doesn't loop on it."
        )
        print(f"   ↳ Detail: {baseline_output[-500:]}")
        log_experience(
            repo_name,
            issue,
            detected["language"],
            "skipped",
            error_category="ENV",
            notes=f"Unrunnable baseline: {unrunnable}",
        )
        return

    baseline_failures = extract_failing_tests(baseline_output)
    if baseline_failures:
        print(
            f"   ↳ {len(baseline_failures)} test(s) already fail on a clean checkout "
            f"-- these will be ignored as pre-existing, not caused by this fix."
        )
    subprocess.run(["git", "reset", "--hard"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)

    # Plan-first for hard tasks: draft the implementation plan while still in
    # ANALYZING, then hand it to every fix attempt as guidance so the model
    # implements a plan instead of improvising. State advances to IMPLEMENTING
    # only after the plan exists (keeps the ANALYZING->IMPLEMENTING edge legal).
    # For vague/conceptual issues the plan is grounded in REAL codebase facts
    # (actual symbols, entry points, closest-matching files) so "resultpage in
    # that thing" resolves to real files instead of hallucinated ones.
    grounding = ""
    if budget["plan_first"] or vague:
        grounding = collect_codebase_facts(repo_dir, issue, relevant_files)
    # Reverse-dependency facts ride in with the grounding so EVERY fix attempt
    # sees who depends on the files it is about to touch.
    grounding = "\n".join([g for g in (grounding, impact_block) if g]).strip()
    plan = ""
    if budget["plan_first"]:
        print(f"🗺️  {'Hard' if effective_difficulty == 'hard' else 'Scoped'} task -- "
              "drafting an implementation plan before writing code.")
        _pr_diff_text_base = _pr_diff_text(repo_dir, record["base_branch"])
        plan = analyze_requirements(
            issue, _pr_diff_text_base, "", memory_context="", grounding=grounding,
        )
        plan = plan.strip()
        if plan and not plan.startswith("(Requirement analysis unavailable"):
            wf_advance(record, WF.ANALYZING, "implementation plan drafted")
    # Enter IMPLEMENTING before the first generated fix. Skipping this is what
    # made the later DRAFT_PR_CREATED transition illegal (ANALYZING has no edge
    # to it), which killed the Posnic/POS#52 run seconds after its PR went up.
    wf_walk(record, [WF.IMPLEMENTING], "generating fix")

    error = None
    previous_error_signature = None
    failure_history = []
    test_layout = describe_test_layout(repo_dir)
    attempts = budget["attempts"]
    for attempt in range(1, attempts + 1):
        print(f"\n--- Attempt {attempt}/{attempts} ---")
        try:
            # On a RE-RUN of a failed one-shot the transcript already holds the
            # previous attempt's turns, so the model remembers them. The
            # failure_history below additionally summarises THIS run's retries.
            memory_ctx = (
                _active_conversation.recent_context(max_chars=6000, n=10)
                if _active_conversation is not None else ""
            )
            fix_text = generate_fix(
                issue,
                relevant_files,
                error,
                retry_variant=(attempt > 1),
                language=detected["language"],
                test_layout=test_layout,
                failure_history=failure_history,
                memory_context=memory_ctx,
                guidance=plan or None,
                labels=labels,
                difficulty=effective_difficulty,
                domain=classification.get("domain") or None,
                grounding=grounding or None,
            )
            changed_paths = apply_fix(repo_dir, fix_text, allowed_paths, relevant_files)

            # Cheap static pre-verify (no LLM, no test run): syntax-check the
            # changed files. Catches the "broke import/indent" class of error a
            # full suite would also catch, but at ~100ms instead of minutes --
            # keep the expensive test loop for real behavioural failures.
            syntax_issues = syntax_check(repo_dir, changed_paths)
            if syntax_issues:
                print(f"⛔ Syntax check failed in the patch ({len(syntax_issues)}):")
                for rel_path, why in syntax_issues:
                    print(f"   - {rel_path}: {why}")
                error = (
                    "Your previous patch does not parse:\n"
                    + "\n".join(f"- {rel_path}: {why}" for rel_path, why in syntax_issues)
                    + "\nFix the syntax errors and resubmit."
                )
                failure_history.append({
                    "attempt": attempt,
                    "changed": changed_paths,
                    "signal": error[-300:],
                    "error": error,
                })
                subprocess.run(["git", "reset", "--hard"], cwd=repo_dir, capture_output=True)
                subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
                continue

            # Hallucination guard (static, no LLM): the patch must not reference
            # repo-local modules/imports that don't exist. Invented references
            # are fed straight back to the model as a retryable error instead of
            # shipping a PR full of fake symbols.
            hallucinated = scan_for_hallucinated_symbols(repo_dir, changed_paths)
            if hallucinated:
                print(f"🚫 Hallucinated references detected in the patch ({len(hallucinated)}):")
                for h in hallucinated:
                    print(f"   - {h}")
                error = (
                    "Your previous patch referenced symbols/files that do not "
                    f"exist in this repository:\n{' '.join(hallucinated)}\n"
                    "Rewrite using ONLY the real modules/symbols from the "
                    "GROUNDING FACTS and the RELEVANT FILES above -- do not "
                    "invent paths, imports or class names."
                )
                failure_history.append({
                    "attempt": attempt,
                    "changed": changed_paths,
                    "signal": error[-300:],
                    "error": error,
                })
                subprocess.run(["git", "reset", "--hard"], cwd=repo_dir, capture_output=True)
                subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
                continue
        except ValueError as e:
            # Model responded but didn't follow the required format, or every
            # proposed file was out-of-scope -- treat as a retryable failure
            # instead of crashing, and tell it exactly what went wrong.
            print(f"⚠️  {e}")
            error = (
                f"Your previous response could not be applied: {e}\n"
                f"You MUST follow the exact diff or FILE:/<<<CONTENT>>>/<<<END>>> "
                f"format and only use the exact paths listed under "
                f"ALLOWED FILE PATHS."
            )
            failure_history.append({
                "attempt": attempt,
                "changed": [p for p, _ in relevant_files],
                "signal": error[-300:],
                "error": error,
            })
            continue

        # Pre-run half of the reproduction-test gate. Rejecting an untested
        # patch HERE skips the suite run entirely, so this check is cheaper
        # than not having it.
        added_lines = added_lines_of_diff(repo_dir)
        if REQUIRE_REGRESSION_TEST:
            gate_ok, gate_why = regression_test_gate(
                changed_paths, added_lines, detected["language"], baseline_output, None
            )
            if not gate_ok:
                print(f"🧪 Rejecting this patch before testing it: {gate_why}")
                error = (
                    f"Your previous patch was rejected without being run: {gate_why}\n"
                    "Include a regression test IN THE SAME patch. It must fail on the "
                    "current code and pass with your fix."
                )
                failure_history.append({
                    "attempt": attempt,
                    "changed": changed_paths,
                    "signal": error[-300:],
                    "error": error,
                })
                subprocess.run(["git", "reset", "--hard"], cwd=repo_dir, capture_output=True)
                subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
                continue

        passed, output = run_tests(repo_dir, test_command)

        if not passed:
            # Repeated-identical-failure guard (SRS FR-6.3): if the exact
            # same error recurs, the model is stuck in a loop and more
            # retries won't help -- stop early instead of burning all
            # MAX_ATTEMPTS on an unproductive cycle.
            error_signature = output[-300:]
            if error_signature == previous_error_signature:
                print(
                    "🛑 Identical failure repeated -- model appears stuck, "
                    "stopping retries early instead of wasting the rest."
                )
                log_experience(
                    repo_name,
                    issue,
                    detected["language"],
                    "failed",
                    error_category="C",
                    notes="Repeated identical failure, stopped early.",
                )
                print(
                    f"\n❌ Stopped after {attempt} attempts (repeated failure). "
                    f"Check {repo_dir} manually."
                )
                return
            previous_error_signature = error_signature

            new_failures = extract_failing_tests(output) - baseline_failures
            if not new_failures and "error" not in output.lower()[:200]:
                # Only pre-existing failures remain -- the fix itself is fine.
                print("   ↳ Remaining failures are all pre-existing, not caused by this fix.")
                passed = True

        if passed:
            wf_walk(record, [WF.TESTING], "tests green")
            # Post-run half of the gate: a green suite proves nothing unless the
            # new test was actually collected and run.
            if REQUIRE_REGRESSION_TEST:
                gate_ok, gate_why = regression_test_gate(
                    changed_paths, added_lines, detected["language"], baseline_output, output
                )
                if not gate_ok and attempt < attempts:
                    print(f"🧪 Tests are green but the gate rejected the patch: {gate_why}")
                    error = (
                        f"Your last patch passed the suite but was rejected: {gate_why}\n"
                        "Add a regression test that is actually collected by this "
                        "repo's test runner."
                    )
                    failure_history.append({
                        "attempt": attempt,
                        "changed": changed_paths,
                        "signal": error[-300:],
                        "error": error,
                    })
                    subprocess.run(["git", "reset", "--hard"], cwd=repo_dir, capture_output=True)
                    subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
                    continue
                print(f"🧪 {gate_why}")
                if VERIFY_TEST_FAILS_WITHOUT_FIX:
                    proved, detail = red_phase_check(repo_dir, changed_paths, test_command)
                    print(f"🔴 Red-phase check: {detail}")
                    if not proved and attempt < attempts:
                        error = (
                            f"Your regression test does not reproduce the bug: {detail}\n"
                            "Write a test that fails on the unfixed code."
                        )
                        failure_history.append({
                            "attempt": attempt,
                            "changed": changed_paths,
                            "signal": error[-300:],
                            "error": error,
                        })
                        subprocess.run(
                            ["git", "reset", "--hard"], cwd=repo_dir, capture_output=True
                        )
                        subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
                        continue
            supervisor_ok, supervisor_reason = supervisor_review(issue, repo_dir, output)
            if not supervisor_ok and attempt < attempts:
                # Supervisor flagged something -- give it one more chance to
                # fix itself with that specific concern as feedback, rather
                # than immediately forcing a human decision on a fix the
                # model itself might recognize as flawed if told why.
                print(f"🚩 Supervisor flagged this fix: {supervisor_reason}")
                print("   ↳ Retrying with that feedback instead of proceeding blindly.")
                error = f"A senior review flagged your last fix: {supervisor_reason}\nAddress this concern."
                failure_history.append({
                    "attempt": attempt,
                    "changed": changed_paths,
                    "signal": error[-300:],
                    "error": error,
                })
                continue
            # SDE-2 review gate before the PR exists. When the self-review says
            # NOT READY and attempts remain, the concern is fed back as a
            # retryable error -- the PR is never created against a fix the
            # model itself reports as not ready.
            sde2_status, sde2_report = sde2_review(issue, repo_dir, output)
            if sde2_status == "NOT READY" and attempt < attempts:
                print(f"🚫 SDE-2 self-review is NOT READY: {sde2_report[:400]}")
                print("   ↳ Retrying with those concerns instead of opening a PR.")
                error = (
                    "Your own SDE-2 review marked the change NOT READY:\n"
                    f"{sde2_report[:1200]}\nAddress every listed concern, re-run "
                    "your checks, and only then resubmit."
                )
                failure_history.append({
                    "attempt": attempt,
                    "changed": changed_paths,
                    "signal": error[-300:],
                    "error": error,
                })
                continue
            if sde2_status == "NOT READY":
                print("🚫 SDE-2 self-review is NOT READY on the final attempt -- no PR opened.")
                log_experience(
                    repo_name,
                    issue,
                    detected["language"],
                    "failed",
                    error_category="C",
                    notes=f"SDE-2 review never reached READY: {sde2_report[:300]}",
                )
                return
            if sde2_status == "READY WITH NOTES":
                print(f"⚠️  SDE-2 self-review: READY WITH NOTES -- {sde2_report[:400]}")
            # We now have an optimal, validated fix -- ONLY NOW do we claim the
            # issue, then proceed with the normal cycle (human gate -> PR).
            if not post_claim_comment(issue):
                log_experience(
                    repo_name,
                    issue,
                    detected["language"],
                    "skipped",
                    notes="Solved but could not claim (taken meanwhile or token lacks scope).",
                )
                print("   ↳ Not opening a PR since we couldn't claim the issue.")
                return
            _gate_vote = human_gate(
                repo_dir, output, supervisor_ok, supervisor_reason, repo_name, issue.number,
            )
            if _gate_vote is None:
                print("\n🛑 Human gate parked in cloud mode -- waiting for a decree.")
                return "gate_deferred"
            if _gate_vote:
                pr_number, pr_url = submit_draft_pr(
                    repo_dir, repo_name, branch_name, issue, state, record=record
                )
                log_experience(
                    repo_name,
                    issue,
                    detected["language"],
                    "success",
                    notes=f"Passed on attempt {attempt}, supervisor: {supervisor_reason}",
                )
                # A draft PR is the START of the maintainer conversation, not
                # the end. Persist the PR handle FIRST (before any state
                # transition can fail) so the record can never end up owning an
                # untracked PR, then park the workflow for `-conversation`.
                record["pr_number"] = pr_number
                record["pr_url"] = pr_url
                save_workflow(record)
                append_iteration(
                    record,
                    {
                        "trigger": "initial",
                        "action_taken": f"Initial fix passed on attempt {attempt}.",
                        "plan": supervisor_reason,
                        "tests_result": "pass",
                        "pr_state": WF.DRAFT_PR_CREATED,
                    },
                )
                # Tolerant walk + repair: the PR is already live on GitHub, so
                # bookkeeping here must not raise.
                wf_walk(
                    record,
                    [WF.TESTING, WF.DRAFT_PR_CREATED, WF.WAITING_FOR_FEEDBACK],
                    "draft PR opened",
                )
                if record["state"] != WF.WAITING_FOR_FEEDBACK:
                    wf_repair_for_iteration(record)
                print(
                    f"\n🔄 Draft PR #{pr_number} is parked in the review loop "
                    f"(state: {record['state']}). It will NOT auto-close.\n"
                    f"   Talk to the maintainer / process feedback with:\n"
                    f"      python oss_agent_v2.py --repo {repo_name} "
                    f"--issue {issue.number} -conversation\n"
                    "   That loop finalizes only after an explicit "
                    "'done'/'approved' signal AND a human approval gate."
                )
                return "draft_pr"
            else:
                print("Cancelled by user.")
                log_experience(
                    repo_name,
                    issue,
                    detected["language"],
                    "skipped",
                    notes="Human declined at review gate despite tests passing.",
                )
            return "cancelled"
        print(f"   ↳ Failure detail: {output[-800:]}")
        error = output
        # Per-file staging for hard tasks: revert ONLY the files the failing
        # tests actually implicate, so the good files already applied survive
        # into the next attempt instead of being wiped by a full reset.
        if effective_difficulty == "hard":
            blamed = blame_changed_files(changed_paths, new_failures, output)
            stage_retry(repo_dir, blamed, changed_paths, allowed_paths, relevant_files)
        failure_history.append({
            "attempt": attempt,
            "changed": changed_paths,
            "signal": output[-300:],
            "error": output,
        })

    log_experience(
        repo_name,
        issue,
        detected["language"],
        "failed",
        error_category="E",
        notes=f"Exhausted {attempts} attempts, last error: {(error or '')[:200]}",
    )
    print(f"\n❌ Failed after {attempts} attempts. Check {repo_dir} manually.")


def sync_pr_branch(repo, branch_name: str, issue_number=None,
                   force_workspace: bool = False) -> Path:
    """Prepare the local clone for an ITERATION run. Unlike fork_clone_and_branch
    (which resets the fix branch back to the default branch for a fresh start),
    this preserves the PR's existing commits: it clones if needed, fetches, and
    hard-resets the local branch to match the remote PR head (origin/branch).

    `issue_number` selects this workflow's isolated clone. It stays optional so
    older callers keep working; without it the shared legacy directory is used
    and a warning explains the risk."""
    fork = repo.create_fork() if repo.owner.login != gh.get_user().login else repo
    if issue_number is None:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        repo_dir = WORKSPACE / repo.name
        print(
            f"⚠️  sync_pr_branch called without an issue number: using the shared "
            f"{repo_dir} instead of an isolated per-issue clone."
        )
    else:
        repo_dir = workflow_workspace(repo.full_name, issue_number, force=force_workspace)
    if not (repo_dir / ".git").exists():
        res = subprocess.run(
            ["git", *_git_auth_args(),
                "-c", "http.postBuffer=524288000", "clone", fork.clone_url, str(repo_dir)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode != 0:
            raise RuntimeError(f"Clone failed: {(res.stderr or '')[-500:]}")
    if issue_number is not None:
        # Everything below rewrites the working tree, so ownership + dirtiness
        # are checked first. A resumed workflow may legitimately discard its own
        # leftovers, but only with --force-workspace.
        guard_workspace(
            repo_dir, repo.full_name, issue_number,
            f"reset {branch_name} to origin", force=force_workspace,
        )
    subprocess.run(["git", *_git_auth_args(), "fetch", "origin"], cwd=repo_dir, capture_output=True)
    res = subprocess.run(
        ["git", "checkout", branch_name], cwd=repo_dir, capture_output=True, text=True
    )
    if res.returncode != 0:
        subprocess.run(
            ["git", "checkout", "-b", branch_name, f"origin/{branch_name}"],
            cwd=repo_dir,
            capture_output=True,
        )
    # Match the remote PR head exactly so we iterate on the reviewed state.
    subprocess.run(["git", "reset", "--hard", f"origin/{branch_name}"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
    return repo_dir


def _iteration_solve_loop(
    issue, repo_dir, relevant_files, allowed_paths, test_command, baseline_failures,
    language, guidance, memory_context=None, attempts=None, labels=None,
    difficulty=None, domain=None, grounding=None,
):
    """Compact Analyze->Fix->Test->(supervisor) loop for an iteration, with the
    same safeguards as the initial solve: retry limit + repeated-identical-
    failure guard. Returns (success, output, reason). `memory_context` is the
    PRIOR CONVERSATION block from this workflow's own transcript, so a round
    does not re-lose context accumulated across previous rounds. `attempts` /
    `labels` / `difficulty` / `domain` carry the record's escalation context
    so hard/labelled issues keep their bigger budget in later rounds too."""

    def _paths():
        return [p for p, _ in relevant_files]

    attempts = attempts or MAX_ATTEMPTS
    error, prev_sig = None, None
    failure_history = []
    test_layout = describe_test_layout(repo_dir)
    for attempt in range(1, attempts + 1):
        print(f"\n--- Iteration attempt {attempt}/{attempts} ---")
        try:
            fix_text = generate_fix(
                issue, relevant_files, error, retry_variant=(attempt > 1),
                language=language, guidance=guidance,
                test_layout=test_layout, require_test=False,
                failure_history=failure_history, memory_context=memory_context,
                labels=labels, difficulty=difficulty, domain=domain,
                grounding=grounding,
            )
            changed_paths = apply_fix(repo_dir, fix_text, allowed_paths, relevant_files)
            # Cheap static pre-verify before the full suite (same as the
            # one-shot loop): a syntax error would fail every test run anyway,
            # so catch it without paying for the test runner.
            syntax_issues = syntax_check(repo_dir, changed_paths)
            if syntax_issues:
                print(f"⛔ Syntax check failed in the patch ({len(syntax_issues)}):")
                for rel_path, why in syntax_issues:
                    print(f"   - {rel_path}: {why}")
                error = (
                    "Your previous patch does not parse:\n"
                    + "\n".join(f"- {rel_path}: {why}" for rel_path, why in syntax_issues)
                    + "\nFix the syntax errors and resubmit."
                )
                failure_history.append({
                    "attempt": attempt, "changed": changed_paths,
                    "signal": error[-300:], "error": error,
                })
                subprocess.run(["git", "reset", "--hard"], cwd=repo_dir, capture_output=True)
                subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
                continue
            # Same hallucination guard as the one-shot loop: fake repo-local
            # imports/symbols are retried against real code instead of shipping.
            hallucinated = scan_for_hallucinated_symbols(repo_dir, changed_paths)
            if hallucinated:
                print(f"🚫 Hallucinated references detected in the patch ({len(hallucinated)}):")
                for h in hallucinated:
                    print(f"   - {h}")
                error = (
                    "Your previous patch referenced symbols/files that do not "
                    f"exist in this repository:\n{' '.join(hallucinated)}\n"
                    "Rewrite using ONLY the real modules/symbols from the "
                    "GROUNDING FACTS and the RELEVANT FILES above -- do not "
                    "invent paths, imports or class names."
                )
                failure_history.append({
                    "attempt": attempt, "changed": changed_paths,
                    "signal": error[-300:], "error": error,
                })
                subprocess.run(["git", "reset", "--hard"], cwd=repo_dir, capture_output=True)
                subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
                continue
        except ValueError as e:
            error = f"Your previous response could not be applied: {e}"
            failure_history.append({
                "attempt": attempt, "changed": _paths(),
                "signal": error[-300:], "error": error,
            })
            continue
        passed, output = run_tests(repo_dir, test_command)
        if not passed:
            sig = output[-300:]
            if sig == prev_sig:
                return False, output, "Repeated identical failure -- stopped early."
            prev_sig = sig
            new_fail = extract_failing_tests(output) - baseline_failures
            if not new_fail and "error" not in output.lower()[:200]:
                passed = True
            else:
                # Per-file staging for hard tasks during a review round too:
                # only the implicated file is reverted, good files survive.
                if difficulty == "hard":
                    blamed = blame_changed_files(_paths(), new_fail, output)
                    stage_retry(repo_dir, blamed, _paths(), allowed_paths, relevant_files)
                failure_history.append({
                    "attempt": attempt, "changed": _paths(),
                    "signal": sig, "error": output,
                })
        if passed:
            ok, reason = supervisor_review(issue, repo_dir, output)  # Prompt E
            if not ok and attempt < attempts:
                print(f"🚩 Supervisor flagged: {reason} -- retrying.")
                error = f"A senior review flagged your last fix: {reason}\nAddress this concern."
                failure_history.append({
                    "attempt": attempt, "changed": _paths(),
                    "signal": error[-300:], "error": error,
                })
                continue
            return True, output, reason
        error = output
    return False, error or "", f"Exhausted {attempts} attempts."


def run_conversation(
    repo_name: str, issue_number: int, test_command: str, force_finalize: bool = False,
    force_workspace: bool = False,
):
    """ONE round of the maintainer conversation -- the `-conversation` command.

    A round is: read the maintainer's new comments/reviews -> plan -> change the
    code -> retest -> remake -> retest -> update the PR -> park again. When an
    explicit done/approved signal is in play AND nothing is unresolved, it goes
    through the mandatory human approval gate and only then commits and pushes.

    One round per invocation, on purpose: the agent never sits in a background
    poll loop (Zero Unattended Operations). Re-run the same command whenever the
    maintainer replies -- the workflow record makes it resumable. Returns a
    status string for the caller/banner.
    """
    record = load_workflow(repo_name, issue_number)
    plain_cmd = f"python oss_agent_v2.py --repo {repo_name} --issue {issue_number}"
    if record is None:
        print(
            f"No conversation to continue for {repo_name}#{issue_number} -- no "
            f"workflow exists yet, so there is no draft PR to talk about.\n"
            f"   Start one first (it opens the draft PR, then shuts down):\n"
            f"      {plain_cmd}"
        )
        return "no_workflow"
    if record["state"] == WF.COMPLETED:
        print(f"✅ {repo_name}#{issue_number} is already COMPLETED. Nothing to do.")
        return "completed"
    if record["state"] == WF.ABANDONED:
        print(f"⚠️  {repo_name}#{issue_number} was ABANDONED. Not resuming.")
        return "abandoned"
    if not record.get("pr_number"):
        print(
            f"⚠️  {repo_name}#{issue_number} has a workflow (state: {record['state']}) "
            f"but no PR was ever opened, so there is nothing to discuss yet.\n"
            f"   Run the initial pass first:\n      {plain_cmd}"
        )
        return "no_pr"

    # Reopen this workflow's durable transcript, then un-pause if `--leave` (or a
    # crash handler) parked it. Both are local-only, so a resumed round starts
    # with its history restored before a single GitHub call is made.
    attach_conversation(
        record,
        provider=record.get("provider") or "omniroute",
        model=record.get("model") or OMNIROUTE_MODEL,
        purpose="a conversation round",
    )
    resume_workflow_record(record, "resumed by -conversation")
    memory_context = (
        _active_conversation.recent_context(max_chars=6000, n=10)
        if _active_conversation is not None else ""
    )

    # Heal a state left behind by an earlier crash between "PR created" and the
    # state advance -- otherwise every round below would be an illegal jump.
    wf_repair_for_iteration(record)
    if record["state"] != WF.WAITING_FOR_FEEDBACK:
        print(
            f"⚠️  Workflow is in {record['state']}, which is not a resumable "
            f"conversation point. Inspect "
            f"{_workflow_path(repo_name, issue_number)} before continuing."
        )
        return "not_resumable"

    print("=" * 60)
    print(f"💬 CONVERSATION -- {repo_name}#{issue_number}")
    print(f"   PR: {record.get('pr_url') or '#' + str(record['pr_number'])}")
    print(f"   Rounds so far: {len(record.get('iterations', []))} | state: {record['state']}")
    print("=" * 60)

    repo = gh.get_repo(repo_name)
    issue = repo.get_issue(number=issue_number)
    base_branch = repo.default_branch
    branch = record.get("branch") or f"fix-issue-{issue_number}"
    pr = None
    if record.get("pr_number"):
        try:
            pr = repo.get_pull(record["pr_number"])
        except Exception as e:
            print(f"⚠️  Could not load PR #{record['pr_number']} ({e}).")

    # A merged PR means the cycle is over.
    if pr is not None and getattr(pr, "merged", False):
        print("🎉 PR was merged upstream -- marking workflow COMPLETED.")
        # A merge is authoritative, so this is the one place allowed to walk
        # through the gated states (nothing is left to approve).
        wf_walk(
            record,
            [
                WF.READY_FOR_HUMAN_APPROVAL, WF.HUMAN_APPROVED, WF.FINAL_VALIDATION,
                WF.COMMIT, WF.SIGN_OFF, WF.COMPLETED,
            ],
            "pr merged upstream",
            allow_gated=True,
        )
        if record["state"] != WF.COMPLETED:
            record["state"] = WF.COMPLETED
            save_workflow(record)
        return "merged"

    # Closed without merging: stop talking to it. State is left intact (not
    # auto-abandoned) so the decision to give up stays a human one.
    if pr is not None and getattr(pr, "state", "") == "closed":
        print(
            f"🚪 PR #{record['pr_number']} is closed without being merged. "
            f"Stopping the conversation -- reopen the PR upstream to keep "
            f"talking, or retire the workflow and free the slot with:\n"
            f"      python oss_agent_v2.py --repo {repo_name} "
            f"--issue {issue_number} -close"
        )
        return "pr_closed"

    raw = fetch_new_feedback(issue, pr)
    fresh = filter_unprocessed(record, raw)
    signals = {signal_for_item(it) for it in fresh}
    if "approval" in signals:
        arm_finalize(record, "maintainer approval in new feedback")
    elif "completion" in signals:
        arm_finalize(record, "completion signal in new feedback")
    if force_finalize:
        arm_finalize(record, "--finalize requested by the operator")
    wants_finalize = finalize_armed(record)

    if not fresh and not wants_finalize:
        print(
            f"No new feedback on {repo_name}#{issue_number} "
            f"(state: {record['state']}). Still waiting -- re-run this same "
            f"command after the maintainer replies."
        )
        return "waiting"
    if fresh:
        authors = sorted({it["author"] for it in fresh if it.get("author")})
        print(f"📥 {len(fresh)} new comment(s)/review(s) from: {', '.join(authors) or 'unknown'}")
    if wants_finalize:
        print(f"🟢 Finalize is armed -- {record.get('pending_finalize_reason')}.")

    repo_dir = sync_pr_branch(
        repo, branch, issue_number=issue_number, force_workspace=force_workspace
    )
    record["workspace"] = store.rel_to_home(repo_dir)
    record["base_branch"] = base_branch
    record["issue_url"] = getattr(issue, "html_url", record.get("issue_url", ""))
    record["issue_state"] = getattr(issue, "state", record.get("issue_state", ""))
    save_workflow(record)
    detected = detect_language_and_commands(repo_dir)
    if test_command == "pytest" and detected["test"] != "pytest":
        test_command = detected["test"]

    tests_result, action_taken, unresolved = "n/a", "", []
    todo, checklist, plan = [], [], ""
    if fresh:
        # Phase 1 -- retest the reviewed PR head first, so failures that were
        # already there don't get blamed on this round's changes.
        print("🔁 Phase 1/3 -- baseline retest of the reviewed PR head")
        _, baseline_output = run_tests(repo_dir, test_command)
        baseline_failures = extract_failing_tests(baseline_output)

        # --- Process the actionable feedback (Prompts A, B, C, D, E, F) ---
        pr_diff = _pr_diff_text(repo_dir, base_branch)
        feedback_text = "\n\n".join(f"{it['author']}: {it['body']}" for it in fresh)
        plan = analyze_requirements(
            issue, pr_diff, feedback_text, memory_context=memory_context,
        )  # Prompt A
        checklist = analyze_review_feedback(pr_diff, fresh)  # Prompt B
        todo = actionable_checklist(checklist)
        print(f"   ↳ {len(todo)} actionable checklist item(s).")
    if todo:
        print("🔁 Phase 2/3 -- remake: apply the maintainer's changes, then retest")
        wf_advance(record, WF.PROCESS_FEEDBACK, "new feedback")
        wf_advance(record, WF.IMPLEMENTING, "acting on checklist")
        # Reuse the record's escalation context (labels/difficulty/domain
        # captured when the workflow started) so a hard, labelled issue keeps
        # its wider budget and domain steering in every conversation round.
        rec_labels = record.get("labels") or []
        rec_difficulty = record.get("difficulty", "medium")
        rec_domain = record.get("domain") or ""
        relevant_files = find_relevant_files(
            repo_dir, issue.title, issue.body,
            labels=rec_labels,
            max_files=HARD_CONTEXT_FILES if rec_difficulty == "hard" else MAX_CONTEXT_FILES,
        )
        allowed_paths = {p for p, _ in relevant_files}
        # Reverse-dependency surface: which repo files depend on the ones
        # selected for editing. Fed to the model so a correct multi-file change
        # stops surprising callers it never saw (and the impact block is
        # evaluated once, before any fix attempt, not recomputed every retry).
        impact_block = change_impact(repo_dir, list(allowed_paths))
        # Re-anchor vague/conceptual work on real code (entry points, symbols)
        # each round, so "resultpage in that thing" never relies on guesses.
        grounding = collect_codebase_facts(repo_dir, issue, relevant_files)
        grounding = "\n".join([g for g in (grounding, impact_block) if g]).strip()
        guidance = plan + "\n\nCHECKLIST:\n" + "\n".join(todo)
        success, output, reason = _iteration_solve_loop(
            issue, repo_dir, relevant_files, allowed_paths, test_command,
            baseline_failures, detected["language"], guidance,
            memory_context=memory_context,
            attempts=MAX_HARD_ATTEMPTS if rec_difficulty == "hard" else MAX_ATTEMPTS,
            labels=rec_labels, difficulty=rec_difficulty, domain=rec_domain,
            grounding=grounding,
        )
        if success:
            wf_advance(record, WF.TESTING, "fix validated")
            wf_advance(record, WF.OPTIMIZING, reason)
            wf_advance(record, WF.UPDATE_PR, "selected best solution")
            action_taken = "; ".join(todo)[:400]
            tests_result = "pass"
            record["last_test_status"] = "pass"
            iteration = {
                "trigger": "feedback",
                "feedback": fresh,
                "plan": plan[:1000],
                "plan_summary": reason,
                "checklist": checklist,
                "action_taken": action_taken,
                "tests_result": tests_result,
                "fixed": reason,
                "unresolved": [],
                "pr_state": WF.UPDATE_PR,
            }
            print("🔁 Phase 3/3 -- retest green after remake; updating the PR")
            update_pr(
                record, repo_dir, branch, repo_name,
                build_iteration_summary(iteration, tests_result),
            )
            append_iteration(record, iteration)
            wf_advance(record, WF.WAITING_FOR_FEEDBACK, "pr updated")
        else:
            tests_result = f"fail ({reason})"
            record["last_test_status"] = "failed"
            unresolved = todo
            print(f"❌ Could not satisfy this feedback automatically: {reason}")
            if record.get("pr_number"):
                _gh_pr(
                    ["comment", str(record["pr_number"]), "--body",
                     f"⚠️ I couldn't automatically resolve the latest feedback: {reason}. "
                     "Leaving the PR as-is and awaiting guidance."],
                    repo_name, repo_dir,
                )
            append_iteration(record, {
                "trigger": "feedback", "feedback": fresh, "plan": plan[:1000],
                "checklist": checklist, "action_taken": "attempted, unresolved",
                "tests_result": tests_result, "unresolved": unresolved,
                "pr_state": record["state"],
            })
            wf_advance(record, WF.WAITING_FOR_FEEDBACK, "iteration unresolved, awaiting guidance")
    elif fresh:
        print("   ↳ Feedback is explanation-only/stale/duplicate; nothing to implement.")
        append_iteration(record, {
            "trigger": "feedback", "feedback": fresh, "plan": plan[:1000],
            "checklist": checklist, "action_taken": "no code change required",
            "tests_result": "n/a", "unresolved": [], "pr_state": record["state"],
        })

    mark_feedback_processed(record, fresh)

    # --- Explicit completion/approval -> human gate -> commit ---
    if not wants_finalize:
        print(
            f"💤 Round finished (state: {record['state']}). Re-run the same "
            f"command when the maintainer replies; add --finalize when you want "
            f"the commit gate yourself."
        )
        return "round_done" if fresh else "waiting"

    blocking = record.get("unresolved_feedback", [])
    if blocking:
        print(
            "⚠️  Finalize is armed but unresolved feedback remains -- staying in "
            "the conversation until it is addressed:"
        )
        for item in blocking[:5]:
            print(f"     - {item}")
        return "unresolved"

    wf_advance(record, WF.READY_FOR_HUMAN_APPROVAL, "explicit finalize signal")
    _gate_vote = final_approval_gate(record, repo_dir, base_branch)
    if _gate_vote is None:
        print("\n🛑 Final gate parked in cloud mode -- waiting for a decree.")
        return "gate_deferred"
    if _gate_vote:
        wf_advance(record, WF.HUMAN_APPROVED, "human approved at gate")
        if finalize_workflow(record, repo_dir, branch, repo_name, base_branch, test_command):
            disarm_finalize(record, "finalized")
            return "finalized"
        # finalize_workflow already re-parked the record. The arm stays set so a
        # later round retries the gate once validation is green again.
        return "final_validation_failed"
    print("Human declined the commit gate -- staying in the conversation.")
    disarm_finalize(record, "human declined at the commit gate")
    wf_advance(record, WF.WAITING_FOR_FEEDBACK, "human declined finalize")
    return "declined"


# Back-compat alias: `--iterate` used to call this name.
run_iteration = run_conversation


# ============================================================
# The `-close` command: retire the draft PR this workflow owns.
#
# Closing is a public action in a repository we do not own, so it goes through
# a human gate like the other two. It never passes --delete-branch: keeping the
# branch is what makes the PR reopenable, which is what keeps this reversible.
# ============================================================
def release_pr_slot(repo_name: str) -> dict:
    """Give the repo's active-PR slot back, AFTER the PR is actually closed.

    Runs under the same lock as `record_pr_created` so a concurrent run cannot
    lose the decrement. It deliberately does NOT refund `prs_today`: the daily
    budget counts PRs *opened*, so refunding it would make open -> close ->
    open an unlimited loop around MAX_PRS_PER_DAY.
    """
    with _state_lock():
        fresh = load_state()
        _apply_daily_rollover(fresh)
        left = fresh["active_prs_by_repo"].get(repo_name, 0) - 1
        if left > 0:
            fresh["active_prs_by_repo"][repo_name] = left
        else:
            fresh["active_prs_by_repo"].pop(repo_name, None)
        save_state(fresh)
        return fresh


def build_close_comment(record: dict, reason: str = "") -> str:
    """The comment posted when the agent retires its own PR. A maintainer should
    never have to guess why a PR vanished from their review queue."""
    lines = [
        "Closing this pull request.",
        "",
        reason.strip() or "This draft PR is no longer being worked on.",
        "",
        f"Nothing here was merged, so issue #{record['issue']} is free for "
        "anyone else to pick up.",
    ]
    rounds = len(record.get("iterations", []))
    if rounds:
        lines.append(f"Review rounds completed before closing: {rounds}.")
    unresolved = record.get("unresolved_feedback") or []
    if unresolved:
        lines.append("Feedback still open at close time: " + "; ".join(unresolved[:6]) + ".")
    lines += [
        "",
        "(The branch is kept, so the PR can be reopened if needed.)",
    ]
    return "\n".join(lines)


def release_issue_claim(issue, reason: str = "") -> bool:
    """Un-claim the issue after closing our PR. The agent claimed it publicly
    before opening the PR, so walking away silently would leave the issue
    looking taken. Best-effort: a token that cannot comment is not a reason to
    fail the close, which already happened."""
    from github import GithubException

    body = (
        "I've closed my draft PR for this"
        + (f" ({reason.strip()})" if reason.strip() else "")
        + ". I'm no longer working on this issue -- it's open for anyone else."
    )
    try:
        issue.create_comment(body)
    except GithubException as e:
        print(f"⚠️  Could not post the un-claim comment ({e.status}); close stands.")
        return False
    except Exception as e:
        print(f"⚠️  Could not post the un-claim comment ({e}); close stands.")
        return False
    print("   ↳ Issue un-claimed.")
    return True


def close_pr_gate(record: dict, pr_state: str, reason: str = "") -> bool:
    """Mandatory human confirmation before closing the PR. There is no flag to
    skip this -- the whole point of the gate is that a public action on someone
    else's repository is never taken unattended."""
    print("\n" + "=" * 60)
    print("CLOSE PULL REQUEST -- explicit approval required")
    print("=" * 60)
    print(f"Repo/issue: {record['repo']}#{record['issue']}")
    print(f"PR: {record.get('pr_url') or '#' + str(record.get('pr_number'))}  (state: {pr_state})")
    print(f"Workflow state: {record.get('state')}")
    print(f"Review rounds so far: {len(record.get('iterations', []))}")
    unresolved = record.get("unresolved_feedback") or []
    print(f"Unresolved feedback: {'; '.join(unresolved) if unresolved else 'none'}")
    print(f"Reason to be posted: {reason.strip() or '(default message)'}")
    print("-" * 60)
    print(
        "This closes the PR on GitHub, un-claims the issue, and marks the "
        "workflow ABANDONED. The branch is kept, so the PR can be reopened."
    )
    return _human_gate(
        record.get("repo", ""), record.get("issue", 0), "close",
        "\n[Close Gate] Close this PR? (y/n): ",
        f"PR state: {pr_state}\nReason to be posted: {reason.strip() or '(default message)'}",
    )


def find_open_agent_prs(repo_name: str) -> list:
    """Read-only: this account's open PRs on `repo_name`, as (number, url, title).

    Used only to be helpful when there is no workflow record to close -- the
    agent will not guess which untracked PR the operator meant, so this reports
    and lets them decide. Returns [] on any failure; it is never load-bearing.
    """
    try:
        login = gh.get_user().login
        hits = list(gh.search_issues(f"repo:{repo_name} is:pr is:open author:{login}")[:10])
    except Exception as e:
        print(f"⚠️  Could not list open PRs on {repo_name} ({e}).")
        return []
    return [(h.number, getattr(h, "html_url", ""), getattr(h, "title", "")) for h in hits]


def _report_untracked_prs(repo_name: str) -> None:
    """Print the operator's own open PRs plus the exact command to close one."""
    found = find_open_agent_prs(repo_name)
    if not found:
        return
    print(f"   Your open PR(s) on {repo_name}:")
    for number, url, title in found:
        print(f"      #{number}  {title[:60]}  {url}")
    print(f"   Close one yourself with:  gh pr close <n> --repo {repo_name}")


def close_active_pr(repo_name: str, issue_number: int, reason: str = "") -> str:
    """The `-close` command: close the draft PR this workflow owns, then park the
    record in ABANDONED so nothing tries to talk to it again.

    Guard order, and why each one exists:
      no record / no PR   -> nothing of ours to close; report untracked PRs.
      COMPLETED           -> refuse. That PR was finalized and signed off; if it
                             really must go, that is a decision for GitHub.
      merged upstream     -> refuse, and record COMPLETED. Closing a merged PR
                             is meaningless and the merge is authoritative.
      already closed      -> reconcile locally instead of closing twice, and
                             release the slot only if we never released it.
      otherwise           -> human gate -> close -> un-claim -> ABANDONED.

    Returns a status string for the caller/banner.
    """
    record = load_workflow(repo_name, issue_number)
    plain_cmd = f"python oss_agent_v2.py --repo {repo_name} --issue {issue_number}"
    if record is None:
        print(
            f"Nothing to close for {repo_name}#{issue_number} -- no workflow "
            f"record exists, so this agent never opened a PR for it.\n"
            f"   Start one with:  {plain_cmd}"
        )
        _report_untracked_prs(repo_name)
        return "no_workflow"
    if not record.get("pr_number"):
        print(
            f"⚠️  {repo_name}#{issue_number} has a workflow (state: "
            f"{record['state']}) but no PR was ever opened, so there is nothing "
            f"to close."
        )
        _report_untracked_prs(repo_name)
        return "no_pr"
    if record["state"] == WF.COMPLETED:
        print(
            f"✅ {repo_name}#{issue_number} is COMPLETED (PR "
            f"#{record['pr_number']} was finalized and signed off). Refusing to "
            f"close it from here -- do that on GitHub if you really mean it."
        )
        return "completed"

    pr_number = record["pr_number"]
    try:
        repo = gh.get_repo(repo_name)
        issue = repo.get_issue(number=issue_number)
        pr = repo.get_pull(pr_number)
    except Exception as e:
        print(
            f"⚠️  Could not load PR #{pr_number} on {repo_name} ({e}). Not "
            f"changing anything -- re-run once GitHub is reachable."
        )
        return "pr_unavailable"

    if getattr(pr, "merged", False):
        print(f"🎉 PR #{pr_number} was merged upstream -- nothing to close.")
        wf_walk(
            record,
            [
                WF.READY_FOR_HUMAN_APPROVAL, WF.HUMAN_APPROVED, WF.FINAL_VALIDATION,
                WF.COMMIT, WF.SIGN_OFF, WF.COMPLETED,
            ],
            "pr merged upstream (seen while closing)",
            allow_gated=True,
        )
        if record["state"] != WF.COMPLETED:
            record["state"] = WF.COMPLETED
            save_workflow(record)
        return "merged"

    if getattr(pr, "state", "") == "closed":
        print(f"🚪 PR #{pr_number} is already closed on GitHub. Reconciling local state.")
        _finish_close(record, repo_name, reason, already_closed=True)
        return "already_closed"

    _gate_vote = close_pr_gate(record, getattr(pr, "state", "open") or "open", reason)
    if _gate_vote is None:
        print("🛑 Close gate parked in cloud mode -- waiting for a decree.")
        return "gate_deferred"
    if not _gate_vote:
        print("Cancelled -- the PR is untouched.")
        return "declined"

    comment = build_close_comment(record, reason)
    # `gh` opened this PR, so `gh` is the most likely to be permitted to close
    # it; the REST API is the fallback for a machine where gh isn't installed.
    # No --delete-branch, on purpose: the branch is what makes this reversible.
    ok, _ = _gh_pr(["close", str(pr_number), "--comment", comment], repo_name, Path.cwd())
    if not ok:
        try:
            pr.create_issue_comment(comment)
            pr.edit(state="closed")
            ok = True
            print("   ↳ Closed via the REST API (gh was unavailable).")
        except Exception as e:
            print(
                f"❌ Could not close PR #{pr_number} ({e}). Nothing was changed "
                f"locally, so you can safely re-run this command."
            )
            return "close_failed"

    print(f"🚪 PR #{pr_number} closed.")
    release_issue_claim(issue, reason)
    _finish_close(record, repo_name, reason, already_closed=False)
    return "closed"


def _finish_close(record: dict, repo_name: str, reason: str, already_closed: bool) -> None:
    """Local bookkeeping after the PR is closed (by us or by someone else).

    Runs after the irreversible part, so like the post-PR bookkeeping it must
    not raise. `pr_closed_at` makes the slot release idempotent -- running
    `-close` twice must not decrement the counter twice.
    """
    if not record.get("pr_closed_at"):
        record["pr_closed_at"] = datetime.now().isoformat(timespec="seconds")
        record["close_reason"] = reason.strip() or (
            "closed outside the agent" if already_closed else "closed by operator"
        )
        release_pr_slot(repo_name)
    record.setdefault("iterations", [])  # tolerate a record written by an older build
    append_iteration(
        record,
        {
            "trigger": "close",
            "action_taken": f"Closed PR #{record.get('pr_number')}"
            + (" (was already closed on GitHub)" if already_closed else ""),
            "plan": record.get("close_reason", ""),
            "pr_state": WF.ABANDONED,
        },
    )
    wf_advance(record, WF.ABANDONED, "pr closed")
    print(
        f"📕 Workflow for {record['repo']}#{record['issue']} is ABANDONED. "
        f"The active-PR slot for {repo_name} is free again."
    )


class Tee:
    """Mirrors everything printed to the terminal into a log file too,
    so each run's full transcript is saved under .agent_data/logs/
    instead of being lost once the terminal scrolls away."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except UnicodeEncodeError:
                enc = getattr(s, "encoding", None) or "utf-8"
                s.write(data.encode(enc, errors="replace").decode(enc, errors="replace"))
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ============================================================
# Session commands: --leave / --resume / --status / --list-workflows /
# --conversation-info / --new-conversation / --finish / --review-feedback
#
# Read this section with one rule in mind: everything here except
# --review-feedback is PURELY LOCAL. Pausing a conversation is a statement about
# this machine, never about the repository -- so nothing below closes, merges,
# edits or deletes anything on GitHub, and nothing pushes. --review-feedback is
# the single exception and it is read-only.
# ============================================================
def _resolve_target(issue_ref, repo=None):
    """(repo, issue) from `--leave 52`, `--leave` (newest), or an explicit --repo.

    Raises store.UnknownWorkflow / store.AmbiguousIssue with recovery commands
    already in the message, so callers just print `exc` and exit."""
    _sync_store_home()
    if issue_ref in (None, True, "", "latest"):
        entries = store.index_entries()
        live = [e for e in entries if str(e.get("status")) not in
                ("completed", "abandoned", "finished")]
        pick = (live or entries)
        if not pick:
            raise store.UnknownWorkflow(
                "No workflows are being tracked yet, so there is nothing to act on.\n"
                "   Start one:  python oss_agent_v2.py --repo owner/repo --issue 42"
            )
        if repo:
            for entry in pick:
                if str(entry.get("repo")) == repo:
                    return repo, int(entry["issue"])
            raise store.UnknownWorkflow(
                f"No tracked workflow for {repo}.\n"
                f"   See what exists: python oss_agent_v2.py --list-workflows"
            )
        newest = pick[0]
        print(f"ℹ️  No issue given -- using the most recently updated workflow: "
              f"{newest.get('repo')}#{newest.get('issue')}")
        return str(newest["repo"]), int(newest["issue"])
    entry = store.resolve_issue(int(issue_ref), repo_name=repo)
    return str(entry["repo"]), int(entry["issue"])


def _require_record(repo_name, issue_number):
    """The saved record, or None with an explanation. `--resume` must fail loudly
    here rather than silently starting a new workflow."""
    record = load_workflow(repo_name, issue_number)
    if record is None:
        print(
            f"❌ No saved state for {repo_name}#{issue_number}.\n"
            f"   Nothing was created -- a resume never invents a workflow.\n"
            f"   Start one:      python oss_agent_v2.py --repo {repo_name} "
            f"--issue {issue_number}\n"
            f"   See what exists: python oss_agent_v2.py --list-workflows"
        )
    return record


def _print_resume_instructions(record):
    for line in store.resume_instructions(record):
        print(f"      {line}")


def _record_workspace(record: dict) -> dict:
    """describe_workspace() for a record, tolerating a record with no workspace.
    Without this, an empty path resolves to .agent_data itself and --status would
    cheerfully report the agent's own data directory as the workspace."""
    rel = record.get("workspace")
    if not rel:
        return {"path": "none yet", "exists": False, "is_git": False,
                "branch": None, "dirty": False, "dirty_files": []}
    _sync_store_home()
    return store.describe_workspace(store.home_to_abs(rel))


def leave_workflow(repo_name: str, issue_number: int, reason: str = "") -> str:
    """`--leave`: save everything and step away. LOCAL ONLY.

    What it does NOT do, by design and by test: it does not close, merge, edit or
    comment on the issue or the PR; it does not push; it does not delete the
    branch or the workspace; it does not mark the workflow completed. Walking away
    from a conversation is not a statement about the contribution."""
    record = _require_record(repo_name, issue_number)
    if record is None:
        return "no_workflow"
    if record["state"] in TERMINAL_STATES:
        print(
            f"ℹ️  {repo_name}#{issue_number} is already {record['state']} -- nothing to "
            f"pause. Its state file is untouched."
        )
        return "terminal"
    attach_conversation(
        record,
        provider=record.get("provider") or "omniroute",
        model=record.get("model") or OMNIROUTE_MODEL,
        purpose="--leave",
    )
    if not pause_workflow(record, reason=reason, trigger="--leave"):
        print(f"ℹ️  {repo_name}#{issue_number} could not be paused (state "
              f"{record['state']}); nothing was changed.")
        return "terminal"
    ws = _record_workspace(record)
    print("=" * 60)
    print(f"⏸️  PAUSED {repo_name}#{issue_number} -- {record.get('issue_title', '')[:60]}")
    print("=" * 60)
    print(f"   status:        {record.get('status')}  (was {record.get('paused_from')})")
    print(f"   reason:        {record.get('paused_reason')}")
    print(f"   PR:            {record.get('pr_url') or record.get('pr_number') or 'none yet'}"
          f"   (left open, untouched)")
    print(f"   branch:        {record.get('branch') or 'none'} on base "
          f"{record.get('base_branch') or 'unknown'}")
    print(f"   workspace:     {ws['path']}"
          f"{'  (uncommitted changes preserved)' if ws['dirty'] else ''}")
    print(f"   conversation:  {record.get('conversation_id')} -> {record.get('conversation_dir')}")
    print(f"   record:        {_workflow_path(repo_name, issue_number)}")
    print(f"   paused {record.get('pause_count')}x, resumed {record.get('resume_count')}x")
    print("\n   Nothing on GitHub was changed: the issue and PR are exactly as they were.")
    print("   Come back with:")
    _print_resume_instructions(record)
    detach_conversation()
    return "paused"


def resume_workflow(repo_name: str, issue_number: int, test_command: str,
                    force_workspace: bool = False, run_round: bool = True) -> str:
    """`--resume`: restore a saved workflow and continue it.

    Restores the record, reopens the transcript, re-creates the workspace if it
    went missing, checks out the EXISTING PR branch, then hands off to one
    conversation round (which is what pulls in maintainer feedback and updates
    the existing PR -- never a second one)."""
    record = _require_record(repo_name, issue_number)
    if record is None:
        return "no_workflow"
    if record["state"] in TERMINAL_STATES:
        print(f"ℹ️  {repo_name}#{issue_number} is {record['state']} -- nothing left to resume.")
        show_status(repo_name, issue_number)
        return "terminal"

    conv = store.Conversation.load(repo_name, issue_number)
    print("=" * 60)
    print(f"▶️  RESUMING {repo_name}#{issue_number} -- {record.get('issue_title', '')[:60]}")
    print("=" * 60)
    print(f"   saved status:  {store.derive_status(record)} (state {record['state']})")
    if record.get("paused_at"):
        print(f"   paused at:     {record['paused_at']} -- {record.get('paused_reason')}")
    if conv is None:
        print("   transcript:    none found; a fresh conversation will be opened.")
    else:
        tail = conv.transcript_tail(6)
        print(f"   conversation:  {conv.id} ({conv.meta.get('message_count', 0)} turns)")
        for line in tail:
            print(f"      | {line}")
    attach_conversation(
        record,
        provider=record.get("provider") or "omniroute",
        model=record.get("model") or OMNIROUTE_MODEL,
        purpose="--resume",
    )
    resume_workflow_record(record, "resumed by --resume")

    ws_path = store.home_to_abs(record.get("workspace") or
                               store.rel_to_home(store.workspace_dir(repo_name, issue_number)))
    ws = store.describe_workspace(ws_path)
    if not ws["is_git"]:
        print(f"   workspace:     {ws_path} is missing or not a clone -- it will be "
              f"re-created from the PR branch.")
    else:
        # A clone with no commits yet (or a detached HEAD) has no branch name to
        # report; say so rather than printing "on None".
        on = f" on {ws['branch']}" if ws.get("branch") else " (no branch checked out yet)"
        print(f"   workspace:     {ws_path}{on}"
              f"{' (uncommitted changes preserved)' if ws['dirty'] else ''}")

    if not record.get("pr_number"):
        print(
            "\n⚠️  This workflow never opened a PR, so there is no maintainer "
            "conversation to continue.\n"
            f"   Run the initial pass:  python oss_agent_v2.py --repo {repo_name} "
            f"--issue {issue_number}"
        )
        return "no_pr"
    if not run_round:
        return "restored"
    print("\n   Continuing the maintainer conversation from the saved state...\n")
    return run_conversation(
        repo_name, issue_number, test_command, force_workspace=force_workspace
    )


def open_conversation(repo_name: str, issue_number: int, test_command: str,
                      force_finalize: bool = False, force_workspace: bool = False) -> str:
    """`--conversation N`: restore an existing conversation, or create one.

    Restore-or-create is the only ambiguity allowed in the CLI, and only here:
    with a saved record this is exactly `-conversation` (one round of maintainer
    feedback on the existing PR); with none it starts the one-shot pass that
    creates the workflow and its draft PR. `--resume` deliberately refuses to do
    the second half, so "continue" and "start" stay distinguishable."""
    record = load_workflow(repo_name, issue_number)
    if record is None:
        print(
            f"ℹ️  No conversation exists for {repo_name}#{issue_number} yet -- starting "
            f"one now (solve -> draft PR -> park for feedback)."
        )
        return main(repo_name, issue_number, test_command, force_workspace=force_workspace) or "started"
    if record["state"] in TERMINAL_STATES:
        print(f"ℹ️  {repo_name}#{issue_number} is {record['state']}; not reopening it.")
        show_status(repo_name, issue_number)
        return "terminal"
    if not record.get("pr_number"):
        print(
            f"ℹ️  {repo_name}#{issue_number} has a saved workflow (state "
            f"{record['state']}) but no PR yet -- resuming the initial pass."
        )
        return main(repo_name, issue_number, test_command, force_workspace=force_workspace) or "started"
    return run_conversation(
        repo_name, issue_number, test_command,
        force_finalize=force_finalize, force_workspace=force_workspace,
    )


def new_conversation(repo_name: str, issue_number: int, force: bool = False,
                     reason: str = "") -> str:
    """`--new-conversation`: archive the current transcript and start a fresh one
    for the SAME issue. Never deletes: the old conversation is moved under
    `conversations/<repo>/issue-<N>/archive/<old-id>/`, and the workflow record
    (PR number, branch, iteration history) is kept as-is."""
    record = _require_record(repo_name, issue_number)
    if record is None:
        return "no_workflow"
    _sync_store_home()
    existing = store.Conversation.load(repo_name, issue_number)
    if existing is None:
        conv, _ = store.Conversation.open_or_create(
            repo_name, issue_number,
            provider=record.get("provider", ""), model=record.get("model", ""),
        )
        record["conversation_id"] = conv.id
        record["conversation_dir"] = store.rel_to_home(conv.dir)
        save_workflow(record)
        print(f"✅ Opened the first conversation for {repo_name}#{issue_number}: {conv.id}")
        return "created"
    meta = existing.meta
    print(
        f"⚠️  {repo_name}#{issue_number} already has conversation {existing.id}\n"
        f"      opened  {meta.get('created_at')}, {meta.get('message_count', 0)} turn(s), "
        f"{len(record.get('iterations', []))} round(s)\n"
        f"      files   {store.rel_to_home(existing.dir)}\n"
        f"   Starting a new one ARCHIVES that transcript (moved, not deleted) and the "
        f"AI loses its memory of the discussion.\n"
        f"   The PR (#{record.get('pr_number')}) and the workflow record are kept."
    )
    if not force:
        if input("   Archive it and start fresh? (y/n): ").strip().lower() != "y":
            print("Cancelled -- the existing conversation is untouched.")
            return "declined"
    try:
        fresh = existing.archive_and_restart(reason or "operator ran --new-conversation")
    except store.SessionStoreError as exc:
        print(f"❌ {exc}")
        return "archive_failed"
    record["conversation_id"] = fresh.id
    record["conversation_dir"] = store.rel_to_home(fresh.dir)
    record.setdefault("archived_conversation_ids", []).append(existing.id)
    # A fresh transcript means the processed-feedback memory is the only thing
    # left that remembers the old discussion -- keep it, so a new conversation
    # does not re-litigate comments that were already handled.
    save_workflow(record)
    print(f"✅ New conversation {fresh.id}; previous one archived under "
          f"{store.rel_to_home(existing.archive_dir)}")
    return "restarted"


def show_status(repo_name: str, issue_number: int) -> str:
    """`--status`: everything known about one workflow, read from disk only."""
    record = _require_record(repo_name, issue_number)
    if record is None:
        return "no_workflow"
    _sync_store_home()
    ws = _record_workspace(record)
    conv = store.Conversation.load(repo_name, issue_number)
    print("=" * 60)
    print(f"📋 {repo_name}#{issue_number} -- {record.get('issue_title', '')[:60]}")
    print("=" * 60)
    print(f"   status:        {store.derive_status(record)}")
    print(f"   state:         {record.get('state')}")
    if record.get("paused_at"):
        print(f"   paused:        {record['paused_at']} ({record.get('paused_reason')}) "
              f"from {record.get('paused_from')}")
    print(f"   issue:         {record.get('issue_url') or 'unknown'} "
          f"[{record.get('issue_state') or '?'}]")
    print(f"   PR:            {record.get('pr_url') or record.get('pr_number') or 'none'}")
    print(f"   branch:        {record.get('branch') or 'none'} (base "
          f"{record.get('base_branch') or '?'})")
    if not ws["exists"]:
        ws_note = "[missing]"
    elif ws["dirty"]:
        ws_note = "[dirty]"
    else:
        ws_note = "[clean]"
    if ws.get("branch"):
        ws_note += " on " + str(ws["branch"])
    print(f"   workspace:     {ws['path']} {ws_note}")
    if ws["dirty"]:
        for line in ws["dirty_files"][:8]:
            print(f"      {line}")
    print(f"   rounds:        {len(record.get('iterations', []))}  "
          f"| paused {record.get('pause_count', 0)}x | resumed {record.get('resume_count', 0)}x")
    print(f"   last tests:    {record.get('last_test_status') or 'unknown'}")
    unresolved = record.get("unresolved_feedback") or []
    if unresolved:
        print(f"   unresolved:    {len(unresolved)} item(s)")
        for item in unresolved[:5]:
            print(f"      - {item}")
    if record.get("pending_finalize"):
        print(f"   finalize:      ARMED ({record.get('pending_finalize_reason')})")
    if conv is not None:
        print(f"   conversation:  {conv.id} ({conv.meta.get('message_count', 0)} turns) "
              f"-> {store.rel_to_home(conv.dir)}")
    print(f"   record:        {_workflow_path(repo_name, issue_number)}")
    print("   next:")
    _print_resume_instructions(record)
    return store.derive_status(record)


def list_workflows() -> int:
    """`--list-workflows`: every tracked workflow, newest first. Rebuilds the
    index from the records on disk if it is empty, so old flat-layout records
    from before the session store show up without a migration step."""
    _sync_store_home()
    entries = store.index_entries()
    if not entries:
        print("No workflows tracked yet.\n"
              "   Start one:  python oss_agent_v2.py --repo owner/repo --issue 42")
        return 0
    print(f"{'STATUS':<18} {'REPO#ISSUE':<34} {'PR':>6}  {'ROUNDS':>6}  UPDATED")
    print("-" * 92)
    for entry in entries:
        repo, issue = entry.get("repo"), entry.get("issue")
        record = load_workflow(repo, issue) if repo is not None else None
        status = store.derive_status(record) if record else (entry.get("status") or "unknown")
        rounds = len((record or {}).get("iterations", []))
        pr = record.get("pr_number") if record else entry.get("pr_number")
        flag = " *" if entry.get("legacy_layout") else ""
        print(f"{status:<18} {f'{repo}#{issue}':<34} {str(pr or '-'):>6}  {rounds:>6}  "
              f"{str(entry.get('updated_at') or '')[:19]}{flag}")
    if any(e.get("legacy_layout") for e in entries):
        print("\n  * record still in the old flat layout; it is migrated automatically "
              "the next time that workflow is saved.")
    legacy = store.legacy_workspace_dirs()
    if legacy:
        print(f"\n⚠️  {len(legacy)} shared clone(s) from the old layout are still on disk "
              f"(kept, never deleted -- they may hold real work):")
        for path in legacy[:5]:
            print(f"      {path}")
    print(f"\n{len(entries)} workflow(s). Details: python oss_agent_v2.py --status <issue>")
    return len(entries)


def conversation_info(repo_name: str, issue_number: int, lines: int = 30) -> str:
    """`--conversation-info`: the transcript and round history for one workflow."""
    record = _require_record(repo_name, issue_number)
    if record is None:
        return "no_workflow"
    _sync_store_home()
    conv = store.Conversation.load(repo_name, issue_number)
    print("=" * 60)
    print(f"💬 conversation -- {repo_name}#{issue_number}")
    print("=" * 60)
    if conv is None:
        print("   No conversation has been opened for this workflow yet.\n"
              f"   It is created on the next run: python oss_agent_v2.py --repo "
              f"{repo_name} --issue {issue_number} -conversation")
        return "none"
    summary = conv.summary()
    print(f"   id:            {summary.get('conversation_id')}")
    print(f"   opened:        {summary.get('created_at')}   updated: {summary.get('updated_at')}")
    print(f"   provider:      {summary.get('provider')} / {summary.get('model')}")
    print(f"   turns:         {summary.get('message_count_on_disk')} on disk")
    print(f"   directory:     {summary.get('conversation_dir')}")
    if summary.get("archived"):
        print(f"   archived:      {', '.join(summary['archived'])}")
    if summary.get("restarted_from"):
        print(f"   restarted from {summary['restarted_from']} "
              f"({summary.get('restart_reason')})")
    for item in record.get("iterations", [])[-5:]:
        print(f"\n   round {item.get('n')} [{item.get('trigger')}] {item.get('timestamp')}")
        print(f"      did:    {(item.get('action_taken') or '-')[:100]}")
        print(f"      tests:  {item.get('tests_result') or '-'}")
        if item.get("unresolved"):
            print(f"      open:   {len(item['unresolved'])} unresolved item(s)")
    tail = conv.transcript_tail(lines)
    if tail:
        print(f"\n   --- last {len(tail)} transcript line(s) ---")
        for line in tail:
            print(f"   {line}")
    print(f"\n   full transcript: {summary.get('transcript')}")
    return summary.get("conversation_id") or "none"


def finish_workflow(repo_name: str, issue_number: int, force: bool = False,
                    reason: str = "") -> str:
    """`--finish`: mark a workflow finished LOCALLY and stop tracking it as live.

    Explicitly not a GitHub action: the PR stays open and the issue stays open. It
    is the "I'm done driving this from my machine" button, for when a maintainer
    has taken over or the PR is simply waiting indefinitely. Use `-close` when you
    actually want the PR closed."""
    record = _require_record(repo_name, issue_number)
    if record is None:
        return "no_workflow"
    if record.get("finished_locally"):
        print(f"ℹ️  {repo_name}#{issue_number} was already finished locally on "
              f"{record['finished_locally']}.")
        return "already_finished"
    if record["state"] in TERMINAL_STATES:
        print(f"ℹ️  {repo_name}#{issue_number} is already {record['state']}.")
        return "terminal"
    if record.get("pr_number"):
        print(
            f"ℹ️  PR #{record['pr_number']} will be LEFT OPEN and the issue LEFT AS IS "
            f"-- --finish only stops local tracking.\n"
            f"   To actually close the PR:  python oss_agent_v2.py --repo {repo_name} "
            f"--issue {issue_number} -close"
        )
    if not force:
        if input("   Mark this workflow finished locally? (y/n): ").strip().lower() != "y":
            print("Cancelled -- nothing changed.")
            return "declined"
    attach_conversation(
        record,
        provider=record.get("provider") or "omniroute",
        model=record.get("model") or OMNIROUTE_MODEL,
        purpose="--finish",
    )
    record["finished_locally"] = store.now_iso()
    record["finished_reason"] = reason.strip() or "operator ran --finish"
    _transcript(f"finished locally: {record['finished_reason']} (GitHub untouched)")
    # Parked, not COMPLETED: COMPLETED means "we signed off and pushed", which is
    # a claim about the contribution. This is only a claim about local tracking.
    pause_workflow(record, reason=record["finished_reason"], trigger="--finish")
    print(f"✅ {repo_name}#{issue_number} marked finished locally "
          f"(status: {store.derive_status(record)}). GitHub was not touched.")
    detach_conversation()
    return "finished"


def review_feedback(repo_name: str, issue_number: int) -> str:
    """`--review-feedback`: READ-ONLY dump of maintainer comments and reviews.

    Makes GitHub reads only -- no comment, no push, no state change, and the
    feedback is NOT marked processed, so the next conversation round still acts
    on it. This is the "what are they asking for?" command."""
    record = _require_record(repo_name, issue_number)
    if record is None:
        return "no_workflow"
    repo = gh.get_repo(repo_name)
    issue = repo.get_issue(number=issue_number)
    pr = None
    if record.get("pr_number"):
        try:
            pr = repo.get_pull(record["pr_number"])
        except Exception as exc:
            print(f"⚠️  Could not load PR #{record['pr_number']} ({exc}); showing issue "
                  f"comments only.")
    items = fetch_new_feedback(issue, pr)
    fresh = filter_unprocessed(record, items)
    print("=" * 60)
    print(f"📥 feedback -- {repo_name}#{issue_number} "
          f"(PR #{record.get('pr_number') or '-'})")
    print("=" * 60)
    if not items:
        print("   Nothing yet from anyone.")
        return "none"
    seen_ids = set(record.get("processed_comment_ids") or [])
    for item in items:
        mark = "NEW" if item.get("id") not in seen_ids else "   "
        signal_name = signal_for_item(item)
        verdict = f" ({item['state']})" if item.get("state") else ""
        tag = f" [{signal_name}]" if signal_name else ""
        print(f"\n   {mark} {item.get('kind')} by {item.get('author')}{verdict}{tag}")
        body = (item.get("body") or "").strip().splitlines()
        for line in body[:10]:
            print(f"        {line[:110]}")
        if len(body) > 10:
            print(f"        ... {len(body) - 10} more line(s)")
    print(f"\n   {len(fresh)} of {len(items)} item(s) are new (not yet acted on).")
    print("   Nothing was modified, and none of this was marked processed.")
    if fresh:
        print(f"   Act on it:  python oss_agent_v2.py --repo {repo_name} "
              f"--issue {issue_number} -conversation")
    return f"{len(fresh)}/{len(items)}"


# --- crash safety ---------------------------------------------------------
# Ctrl+C, a provider outage, a dropped network, an unhandled bug: all of them
# used to lose the run. Now the active workflow is parked in PAUSED on the way
# out, so the next --resume picks up from the last stable state instead of
# starting over. Best-effort by construction -- an emergency pause that itself
# raises would replace a recoverable failure with an unrecoverable one.
def emergency_pause(reason: str, trigger: str) -> None:
    record = _active_workflow
    if not isinstance(record, dict) or record.get("state") in TERMINAL_STATES:
        return
    try:
        if pause_workflow(record, reason=reason, trigger=trigger):
            print(f"\n💾 Saved state for {record['repo']}#{record['issue']} before exit "
                  f"(status: {store.derive_status(record)}). Nothing on GitHub changed.")
            print("   Resume with:")
            _print_resume_instructions(record)
    except Exception as exc:                     # noqa: BLE001 - last line of defence
        print(f"\n⚠️  Could not save state before exit: {exc}")
        print(f"   The record on disk is still valid: "
              f"{_workflow_path(record.get('repo', '?'), record.get('issue', 0))}")


def _install_crash_handlers() -> None:
    """Ctrl+C and unhandled exceptions both pause instead of vanishing."""
    def _on_sigint(signum, frame):
        print("\n\n⏸️  Interrupted (Ctrl+C) -- pausing this workflow instead of dropping it.")
        emergency_pause("interrupted with Ctrl+C", "SIGINT")
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError, AttributeError):
        pass          # not the main thread, or a platform without SIGINT

    previous_hook = sys.excepthook

    def _on_exception(exc_type, exc, tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            emergency_pause(f"crashed: {exc_type.__name__}: {exc}", "crash")
        previous_hook(exc_type, exc, tb)

    sys.excepthook = _on_exception


_CLI_EPILOG = """\
three commands, deliberately separate:

  python oss_agent_v2.py --repo owner/repo --issue 52
      One-shot. Solve the issue, open the DRAFT PR, then shut down.

  python oss_agent_v2.py --repo owner/repo --issue 52 -conversation
      One round of the maintainer conversation: read new feedback, change the
      code, retest -> remake -> retest, update the PR. Re-run it each time the
      maintainer replies. Commits only after an explicit done/approved signal
      AND your approval at the final gate.

  python oss_agent_v2.py --repo owner/repo --issue 52 -close
      Close that draft PR, un-claim the issue, free the repo's PR slot, and
      mark the workflow ABANDONED. Asks first; keeps the branch so the PR can
      be reopened on GitHub.

session commands -- walk away mid-issue and pick it back up later:

  --leave [N]            park the workflow and exit. LOCAL ONLY: it never
                         closes, merges, edits or comments on the issue or the
                         PR, never pushes, and never deletes the branch or the
                         workspace.
  --resume N             restore a saved workflow and continue it. Fails
                         loudly if nothing is saved -- it never invents one.
  --conversation N       resume the conversation for issue N, or start one if
                         none exists yet.
  --new-conversation N   archive this issue's transcript and open a fresh one.
                         Keeps the PR, the branch and the workflow record.
  --status N             everything known about one workflow.
  --list-workflows       every tracked workflow, newest first.
  --conversation-info N  transcript + round history for one workflow.
  --finish N             stop tracking it locally. Leaves the PR and the issue
                         open -- use -close if you want the PR closed.
  --review-feedback N    read-only dump of maintainer comments and reviews.

N may be a bare issue number: `--resume 52` finds the repo through
.agent_data/state/index.json. Add `--repo owner/repo` when the same number
exists in more than one tracked repo, or to start new work. N is optional on
--leave, which then acts on the most recently updated workflow.

Everything above except --review-feedback is purely local; --review-feedback
only reads. None of them can change anything on GitHub.
"""


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface. Kept in a function so the flags can be unit-tested."""
    parser = argparse.ArgumentParser(
        description="Autonomous OSS contribution agent: draft PR first, then a "
        "maintainer conversation that stays under human control.",
        epilog=_CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", help="owner/repo (use this OR --repos-file, not both)")
    parser.add_argument("--issue", type=int, help="issue number (required if using --repo)")
    parser.add_argument(
        "--repos-file",
        help="path to a text file, one owner/repo per line, to auto-discover a valid issue from",
    )
    parser.add_argument(
        "--label", default="good first issue",
        help="issue label(s) to search for in --repos-file mode. Comma-separate "
        "to hunt several (e.g. --label \"bug,enhancement,area:frontend\") -- "
        "each is tried in order per repo.",
    )
    parser.add_argument("--test-command", default="pytest")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="REPORT-ONLY investigation: profile the repo, find relevant files, "
        "trace who depends on them, run best-effort baseline tests, and write a "
        "structured findings report to .agent_data/reports/ -- never forks, "
        "claims, commits, opens PRs or comments. Use with --repo (+optional "
        "--issue, or --scope for a repo-wide topic).",
    )
    parser.add_argument(
        "--scope",
        default="",
        help="with --analyze and no --issue: a free-text topic to investigate "
        "across the whole repo (e.g. 'rate-limiting path', 'auth flow').",
    )
    parser.add_argument(
        "--task-type",
        default="",
        help="with --analyze: force the investigation task type (INVESTIGATION, "
        "CODE_REVIEW, SECURITY, PERFORMANCE, FEATURE, ...) instead of deriving "
        "it from classification.",
    )
    parser.add_argument(
        "-conversation",
        "--conversation",
        "--iterate",
        dest="conversation",
        nargs="?",
        const=True,
        default=False,
        metavar="ISSUE",
        help="continue the maintainer conversation on an existing draft PR: "
        "process new feedback, retest/remake/retest, update the PR. One round "
        "per run. Bare (`--repo X --issue N -conversation`) is the original "
        "form and is unchanged; `--conversation N` resolves the repo from the "
        "saved index and starts a conversation if none exists yet. "
        "(--iterate is the old name and still works.)",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="with -conversation: arm finalization yourself instead of waiting for "
        "a maintainer 'done'/'approved'. The human approval gate and a green "
        "final test run are still required before anything is committed.",
    )
    parser.add_argument(
        "-close",
        "--close",
        dest="close",
        action="store_true",
        help="close the draft PR this issue's workflow owns, un-claim the issue "
        "and mark the workflow ABANDONED. Requires --repo + --issue, asks for "
        "confirmation, and keeps the branch so the PR can be reopened.",
    )
    parser.add_argument(
        "--close-reason",
        default="",
        help="with -close: the explanation posted on the PR and the issue "
        "(default: a neutral 'no longer being worked on' message).",
    )
    parser.add_argument(
        "--gate-sync",
        action="store_true",
        help="cloud mode: apply a pre-written gate decree from GATE_DIR for "
        "this repo/issue instead of prompting at the human gates. With a decree "
        "waiting it is consumed exactly once; without one (and GATE_ASYNC=1, "
        "non-TTY stdin) the gates park and notify Telegram instead of crashing.",
    )
    _add_session_arguments(parser)
    return parser


def _add_session_arguments(parser: argparse.ArgumentParser) -> None:
    """The pause/resume surface, in its own group so `--help` reads clearly.

    Every flag here takes an OPTIONAL issue number so three spellings work:
    `--resume 52` (bare number, repo resolved through the index),
    `--repo o/r --resume 52` (exact), and `--resume` on its own where that is
    meaningful. Kept separate from build_parser() only for readability -- the
    arguments land on the same parser."""
    session = parser.add_argument_group(
        "session commands",
        "Pause an issue and come back to it later. Everything here is LOCAL "
        "except --review-feedback, which only reads GitHub. None of it can "
        "close, merge, edit, comment on or push to anything.",
    )
    session.add_argument(
        "--leave",
        nargs="?",
        const=True,
        metavar="ISSUE",
        help="park this workflow and exit, saving everything needed to resume "
        "(state, branch, workspace, PR, transcript). Does NOT touch the issue "
        "or the PR on GitHub, does not push, does not delete the branch or the "
        "workspace, and does not mark anything completed. Without ISSUE it "
        "uses --issue, or else the most recently updated workflow.",
    )
    session.add_argument(
        "--resume",
        nargs="?",
        const=True,
        metavar="ISSUE",
        help="restore a saved workflow and continue it: reopen the transcript, "
        "re-create the workspace if it went missing, check out the EXISTING PR "
        "branch, then run one conversation round. Errors out if no state was "
        "saved -- unlike --conversation it never starts fresh work.",
    )
    session.add_argument(
        "--new-conversation",
        dest="new_conversation",
        nargs="?",
        const=True,
        metavar="ISSUE",
        help="archive this issue's transcript (moved, never deleted) and open a "
        "fresh conversation for the same issue. Asks first unless --force. The "
        "PR, the branch and the workflow record are kept as they are.",
    )
    session.add_argument(
        "--status",
        nargs="?",
        const=True,
        metavar="ISSUE",
        help="read-only report for one workflow: issue + PR state, workflow and "
        "conversation status, branch, workspace, last test result, unresolved "
        "feedback, and the next command to run.",
    )
    session.add_argument(
        "--list-workflows",
        dest="list_workflows",
        action="store_true",
        help="table of every tracked workflow (status, repo#issue, PR, rounds, "
        "last update), newest first. Rebuilds the index from the records on "
        "disk if needed, so pre-session-store workflows show up too.",
    )
    session.add_argument(
        "--conversation-info",
        dest="conversation_info",
        nargs="?",
        const=True,
        metavar="ISSUE",
        help="conversation id, created/last-activity timestamps, provider/model, "
        "round history and the tail of the transcript for one workflow.",
    )
    session.add_argument(
        "--finish",
        nargs="?",
        const=True,
        metavar="ISSUE",
        help="mark the workflow finished LOCALLY and stop tracking it as live. "
        "Does not close or merge the PR, does not close the issue, does not "
        "delete the branch. Asks first unless --force.",
    )
    session.add_argument(
        "--review-feedback",
        dest="review_feedback",
        nargs="?",
        const=True,
        metavar="ISSUE",
        help="read-only dump of the maintainer comments and reviews on this "
        "workflow's PR. Changes nothing and does not mark anything processed, "
        "so the next -conversation round still acts on it.",
    )
    session.add_argument(
        "--reason",
        default="",
        help="free-text note stored with --leave / --finish / --new-conversation, "
        "so --status and the transcript say why you stepped away.",
    )
    session.add_argument(
        "--force",
        action="store_true",
        help="skip the confirmation prompt on --new-conversation / --finish. Never "
        "makes either of them touch GitHub.",
    )
    session.add_argument(
        "--force-workspace",
        dest="force_workspace",
        action="store_true",
        help="allow a run to discard ITS OWN uncommitted workspace changes. A "
        "workspace owned by a different workflow is still refused -- that guard "
        "is not overridable.",
    )
    session.add_argument(
        "--transcript-lines",
        dest="transcript_lines",
        type=int,
        default=30,
        help="with --conversation-info: how many transcript lines to show "
        "(default: 30).",
    )


# ============================================================
# Dispatch
#
# Split into small functions so the whole CLI is unit-testable: build_parser()
# for the flag surface, cli_main(argv) for the routing. Nothing here does any
# work itself -- it validates the invocation, decides which of the handlers
# above owns it, and translates their return values into an exit code.
# ============================================================
# Commands that only READ. They get no log file (a transcript of "I printed a
# table" is noise) and they are safe to run against a workflow another process
# is using, because they take no locks and write nothing.
_READ_ONLY_COMMANDS = frozenset(
    {"--status", "--list-workflows", "--conversation-info", "--review-feedback"}
)

# (previous stdout, previous stderr, open log handle) while a run log is active.
_LOG_STATE = None


def _start_run_log(label: str) -> Path:
    """Mirror this run's output into .agent_data/logs/<label>_<ts>.log.

    Same behaviour the one-shot command always had, extracted so the session
    commands get transcripts too instead of a second copy of the plumbing. Tees
    onto the CURRENT streams (not sys.__stdout__) and remembers them, so
    _stop_run_log() can put them back -- which is what lets cli_main() be called
    more than once in one process, e.g. from the test suite."""
    global _LOG_STATE
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"{store.safe_repo_dir(label)}_{datetime.now():%Y%m%d_%H%M%S}.log"
    handle = open(path, "w", encoding="utf-8")
    _LOG_STATE = (sys.stdout, sys.stderr, handle)
    sys.stdout = Tee(sys.stdout, handle)
    sys.stderr = Tee(sys.stderr, handle)
    print(f"Full transcript being saved to: {path}")
    return path


def _stop_run_log() -> None:
    """Restore the streams and close the log. Safe to call when no log is open."""
    global _LOG_STATE
    if _LOG_STATE is None:
        return
    out, err, handle = _LOG_STATE
    _LOG_STATE = None
    sys.stdout, sys.stderr = out, err
    try:
        handle.close()
    except OSError:
        pass


def _issue_ref(parser, flag: str, value, fallback_issue=None):
    """Normalise a session flag's optional value into an issue reference.

    `--resume 52`               -> 52
    `--issue 52 --resume`       -> 52   (an explicit --issue is not ignored)
    `--resume`                  -> True (caller resolves "most recent")
    Anything non-numeric is a usage error, never a silent guess."""
    if value is True:
        return fallback_issue if fallback_issue is not None else True
    text = str(value).strip().lstrip("#")
    if not text.isdigit():
        parser.error(
            f"{flag} takes an issue NUMBER, not {value!r}.\n"
            f"  e.g.  python oss_agent_v2.py {flag} 52\n"
            f"        python oss_agent_v2.py --repo owner/repo {flag} 52"
        )
    return int(text)


def _selected_session_command(args) -> list:
    """Which session flags were used, as they were spelled on the command line.

    `--conversation` is deliberately dual-purpose: bare it is the original
    `-conversation` round (which needs --repo/--issue and is NOT a session
    command), with a number it is the resume-or-start session command. Keeping
    both on one flag is what preserves the old invocation exactly."""
    selected = [
        flag
        for flag, value in (
            ("--leave", args.leave),
            ("--resume", args.resume),
            ("--new-conversation", args.new_conversation),
            ("--status", args.status),
            ("--conversation-info", args.conversation_info),
            ("--finish", args.finish),
            ("--review-feedback", args.review_feedback),
        )
        if value is not None
    ]
    if args.list_workflows:
        selected.append("--list-workflows")
    # A string is the only thing argparse can produce here that means "the
    # operator typed a number": absent is False (unchanged from store_true) and
    # bare is the const True.
    if isinstance(args.conversation, str):
        selected.append("--conversation")
    return selected


def _validate_flags(parser, args, selected: list) -> None:
    """Reject impossible invocations BEFORE anything expensive happens.

    Auto-discovery scans GitHub and a conversation round calls a model; there is
    no point paying for either only to then refuse the command."""
    if len(selected) > 1:
        parser.error(
            f"{', '.join(selected)} do different things; run one at a time.\n"
            f"  Start with:  python oss_agent_v2.py --status <issue>"
        )
    if selected and args.close:
        parser.error(
            f"{selected[0]} is local-only; -close changes GitHub. Run them "
            f"separately so it is always obvious which one you meant."
        )
    if selected and args.repos_file:
        parser.error(
            f"{selected[0]} acts on a workflow that already exists, but "
            f"--repos-file searches for NEW work. Pass the issue instead: "
            f"python oss_agent_v2.py {selected[0]} 52"
        )
    if args.close and args.conversation:
        parser.error("-close and -conversation do different things; run one at a time")
    if args.close and args.finalize:
        parser.error("--finalize belongs to -conversation, not -close")
    if args.close_reason and not args.close:
        parser.error("--close-reason only makes sense together with -close")
    if args.close and args.repos_file:
        parser.error(
            "-close needs an explicit --repo and --issue -- auto-discovery looks "
            "for NEW work, so it can't tell you which PR you meant to close"
        )
    if args.finalize and not args.conversation:
        parser.error("--finalize only makes sense together with -conversation")
    if args.reason and not (selected and selected[0] in
                            ("--leave", "--finish", "--new-conversation")):
        parser.error(
            "--reason is stored by --leave, --finish and --new-conversation. "
            "For -close use --close-reason (which is posted to GitHub)."
        )
    if args.force and not (selected and selected[0] in ("--finish", "--new-conversation")):
        parser.error(
            "--force skips the confirmation on --finish and --new-conversation. "
            "To let a run discard its own workspace changes use --force-workspace."
        )
    # --force* are local-only escape hatches and must not look like a way past
    # the close confirmation: -close is the one command that changes GitHub, and
    # neither flag is accepted anywhere near it.
    if args.force_workspace and args.close:
        parser.error(
            "--force-workspace lets a RUN discard its own uncommitted workspace "
            "changes; -close never touches the workspace. Nothing skips the "
            "close confirmation -- answer y or don't close the PR."
        )
    if args.force_workspace and selected and selected[0] not in ("--resume", "--conversation"):
        parser.error(
            f"--force-workspace applies to commands that check out code "
            f"(--resume, --conversation, or a plain --repo/--issue run), not to "
            f"{selected[0]}."
        )
    if args.transcript_lines != 30 and selected != ["--conversation-info"]:
        parser.error("--transcript-lines only applies to --conversation-info")
    if args.transcript_lines < 1:
        parser.error("--transcript-lines must be at least 1")

    # --analyze is read-only and standalone: it must never be combined with a
    # GitHub-writing or state-mutating command, and it always targets a real
    # repo (never searches for new issues).
    if args.analyze:
        if selected:
            parser.error(
                f"--analyze is read-only and standalone; it cannot be combined "
                f"with {selected[0]}."
            )
        if args.close or args.conversation or args.finalize:
            parser.error("--analyze is read-only; it cannot be combined with -close or -conversation")
        if args.repos_file:
            parser.error(
                "--analyze targets a specific --repo; it cannot be combined with "
                "--repos-file (auto-discovery looks for NEW work)."
            )
        if not args.repo:
            parser.error("--analyze needs an explicit --repo (optionally with --issue)")
        if args.task_type and args.task_type.upper() not in _TASK_TYPES:
            parser.error(
                f"--task-type must be one of {', '.join(sorted(_TASK_TYPES))}"
            )
        if args.finalize:
            parser.error("--analyze is read-only; --finalize does not apply")
    elif args.task_type:
        parser.error("--task-type only applies to --analyze")


def _resolve_or_start(issue_ref, repo=None):
    """Resolution for --conversation, the one command allowed to start new work.

    With an explicit --repo the pair is taken at face value even when the index
    has never heard of it, because "no conversation exists yet -> start one" is
    that command's documented behaviour. Without --repo it still has to be
    resolvable, since guessing a repository would be unforgivable."""
    if repo and issue_ref not in (None, True, "", "latest"):
        return repo, int(issue_ref)
    return _resolve_target(issue_ref, repo)


# Outcomes that mean "the command declined to act". Reported as a non-zero exit
# so a wrapper script can tell "paused" from "there was nothing to pause".
_INEFFECTIVE = frozenset({"no_workflow", "declined", "archive_failed"})


def _dispatch_session_command(parser, args, command: str) -> int:
    """Run the one selected session command. Returns a process exit code."""
    _sync_store_home()
    if command == "--list-workflows":
        list_workflows()
        return 0

    value, resolver = {
        "--leave": (args.leave, _resolve_target),
        "--resume": (args.resume, _resolve_target),
        "--conversation": (args.conversation, _resolve_or_start),
        "--new-conversation": (args.new_conversation, _resolve_target),
        "--status": (args.status, _resolve_target),
        "--conversation-info": (args.conversation_info, _resolve_target),
        "--finish": (args.finish, _resolve_target),
        "--review-feedback": (args.review_feedback, _resolve_target),
    }[command]
    ref = _issue_ref(parser, command, value, fallback_issue=args.issue)
    if ref is True and command != "--leave":
        parser.error(
            f"{command} needs an issue number: python oss_agent_v2.py {command} 52\n"
            f"  (only --leave defaults to the most recently updated workflow)"
        )
    repo_name, issue_num = resolver(ref, args.repo)

    if command not in _READ_ONLY_COMMANDS:
        _start_run_log(f"{repo_name.replace('/', '-')}_issue{issue_num}_{command.strip('-')}")

    if command == "--leave":
        outcome = leave_workflow(repo_name, issue_num, reason=args.reason)
    elif command == "--resume":
        outcome = resume_workflow(
            repo_name, issue_num, args.test_command, force_workspace=args.force_workspace
        )
    elif command == "--conversation":
        outcome = open_conversation(
            repo_name, issue_num, args.test_command,
            force_finalize=args.finalize, force_workspace=args.force_workspace,
        )
    elif command == "--new-conversation":
        outcome = new_conversation(repo_name, issue_num, force=args.force, reason=args.reason)
    elif command == "--status":
        outcome = show_status(repo_name, issue_num)
    elif command == "--conversation-info":
        outcome = conversation_info(repo_name, issue_num, lines=args.transcript_lines)
    elif command == "--finish":
        outcome = finish_workflow(repo_name, issue_num, force=args.force, reason=args.reason)
    else:                                   # --review-feedback
        outcome = review_feedback(repo_name, issue_num)

    if command not in _READ_ONLY_COMMANDS:
        print(f"\n🛑 {command} finished ({outcome}). Shutting down.")
    return 1 if outcome in _INEFFECTIVE else 0


def _dispatch_workflow_command(parser, args) -> int:
    """The original three commands, unchanged in behaviour.

    Deliberately still driven by --repo/--issue (or --repos-file) rather than by
    the index: this is the path that creates work, and it must keep doing exactly
    what it did before the session store existed."""
    if args.analyze:
        # Read-only investigation: --repo is enough (repo-wide), --issue narrows
        # it to one issue; --scope gives a topic when no issue is given.
        _start_run_log(f"{args.repo.replace('/', '-')}_analyze")
        path = investigate_report(
            args.repo,
            issue_number=args.issue,
            test_command=args.test_command,
            scope_text=args.scope,
            task_override=args.task_type,
        )
        print("\n🛑 Analysis complete. Nothing on GitHub or in the workflow store was changed.")
        return 0

    if args.repos_file:
        repo_list = Path(args.repos_file).read_text(encoding="utf-8").splitlines()
        repo, issue_num = discover_valid_issue(repo_list, args.label)
        if repo is None:
            print("Nothing to do -- no valid issue found in the given repo list.")
            return 0
        if input(f"\nProceed with {repo}#{issue_num}? (y/n): ").strip().lower() != "y":
            print("Cancelled.")
            return 0
    elif args.repo and args.issue:
        repo, issue_num = args.repo, args.issue
    else:
        parser.error(
            "Provide either --repo + --issue, or --repos-file.\n"
            "  To act on work already in progress, use a session command:\n"
            "    python oss_agent_v2.py --list-workflows\n"
            "    python oss_agent_v2.py --status 52"
        )

    _start_run_log(f"{repo.replace('/', '-')}_issue{issue_num}")

    if args.close:
        outcome = close_active_pr(repo, issue_num, args.close_reason)
        print(f"\n🛑 Close command finished ({outcome}). Shutting down.")
        return 0

    if args.conversation:
        outcome = run_conversation(
            repo, issue_num, args.test_command,
            force_finalize=args.finalize, force_workspace=args.force_workspace,
        )
        print(f"\n🛑 Conversation round finished ({outcome}). Shutting down.")
        return 0

    # Plain mode is a ONE-SHOT: solve -> draft PR -> shut down. It no longer
    # slides into the feedback loop on its own; the same command typed twice used
    # to do two completely different things.
    existing = load_workflow(repo, issue_num)
    if (
        existing is not None
        and existing.get("pr_number")
        and existing.get("state") not in (WF.COMPLETED, WF.ABANDONED)
    ):
        print(
            f"⚠️  {repo}#{issue_num} already has a live workflow "
            f"(state: {existing['state']}, PR #{existing['pr_number']}).\n"
            f"   Not starting over -- that would redo the work and risk a second PR.\n"
            f"   Continue the maintainer conversation instead:\n"
            f"      python oss_agent_v2.py --repo {repo} --issue {issue_num} -conversation\n"
            f"   See where it stands:\n"
            f"      python oss_agent_v2.py --status {issue_num} --repo {repo}\n"
            f"   Or retire that PR and free the slot:\n"
            f"      python oss_agent_v2.py --repo {repo} --issue {issue_num} -close"
        )
        return 0

    outcome = main(repo, issue_num, args.test_command, force_workspace=args.force_workspace)
    if outcome == "draft_pr":
        print("\n🛑 Draft PR is up -- shutting down here by design (nothing runs unattended).")
    return 0


def cli_main(argv=None, install_handlers: bool = True) -> int:
    """Parse, validate, route. Returns an exit code instead of calling sys.exit
    so the whole CLI can be driven from the test suite.

    Exit codes: 0 acted, 1 declined to act (nothing saved, or you said no), 2 the
    invocation could not be resolved to a workflow, 3 the workspace belongs to a
    different workflow, 130 interrupted. Failures print recovery commands rather
    than a bare traceback."""
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # not a TextIOWrapper (e.g. a fake in the tests) -- leave it
    parser = build_parser()
    args = parser.parse_args(argv)
    selected = _selected_session_command(args)
    _validate_flags(parser, args, selected)

    # Ctrl+C, an unhandled exception or a provider outage parks the active
    # workflow in PAUSED on the way out instead of losing the run. Installed for
    # every command: the read-only ones simply never have an active workflow, so
    # there the handler is a no-op.
    if install_handlers:
        _install_crash_handlers()

    try:
        if selected:
            return _dispatch_session_command(parser, args, selected[0])
        return _dispatch_workflow_command(parser, args)
    except store.WorkspaceConflict as exc:
        print(f"\n❌ {exc}")
        return 3
    except store.SessionStoreError as exc:
        # UnknownWorkflow / AmbiguousIssue: the message already carries the
        # commands that fix it, so print it plainly rather than as a crash.
        print(f"\n❌ {exc}")
        return 2
    except KeyboardInterrupt:
        # _install_crash_handlers() has already paused and printed the resume
        # instructions; this just keeps a traceback off the screen.
        return 130
    finally:
        _stop_run_log()


if __name__ == "__main__":
    sys.exit(cli_main())

