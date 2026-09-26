# oss-agent — Workflow Architecture

End-to-end structure of the automated GitHub-contribution workflow: every
trigger, component, API, account, token and secret and how they fit together
from "schedule fires" to "PR is raised". This document names account roles,
secret IDs and endpoints; it deliberately never contains secret values.

---

## 1. End-to-end flow (architectural view)

```
                    GitHub Actions  (repo: zakirHacks56/dc-sandbox)
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ .github/workflows/controller.yml                                         │
  │  cron "*/10" x2  +  "0 17 * * *" (digest)  +  workflow_dispatch          │
  │                                                                          │
  │   env  (mapped from repo Secrets)                                        │
  │   ├─ GITHUB_TOKEN / GH_TOKEN      <- PR_PAT                              │
  │   ├─ LLM_API_KEY                   <- LLM_API_KEY                        │
  │   ├─ OMNIROUTE_API_KEY / BASE_URL  <- omniRoute gateway + tunnel         │
  │   ├─ GEMINI / GROQ / OPENROUTER / MISTRAL / HETZNER / LLM7 / COPILOT     │
  │   │   _API_KEY                              (new direct-provider tiers)  │
  │   ├─ OMNIROUTE_MODEL(+fallbacks, fast)     concrete model overrides      │
  │   ├─ OMNIROUTE_FALLBACK_*  (6)             hosted outage fallback        │
  │   ├─ SIGNOFF_NAME / SIGNOFF_EMAIL                                         │
  │   └─ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID                               │
  │                                                                          │
  │   tick job:                                                              │
  │    1. checkout (GH_PAT, depth 0)                                         │
  │    2. setup-python 3.12 + pip cache       (W13: cached deps)             │
  │    3. pip install -r fixer/requirements.txt                              │
  │    4. beacon/gate_poller.py   <- Telegram Approve/Decline -> .decree     │
  │    5. beacon/tick.py          <- ONE unit of work (see §3)               │
  │    6. commit state  (oss-relay[bot], pull -X ours, push x5)              │
  │    7. notify on failure  (Telegram)        (W17: tick-CRASH ping only)   │
  │   digest job (17:00 UTC): beacon/digest.py --send                       │
  │    1. reads data/failures.jsonl + data/metrics.jsonl                     │
  │    2. one Telegram daily digest          (W17: per-issue noise, batched) │
  └───────────────┬──────────────────────────────┬───────────────────────────┘
                  │ tick.py spawns fixer        │ gate_poller polls
                  v                             v
   ┌───────────────────────────┐   Telegram Bot API  (offset polling)
   │  fixer/oss_agent_v2.py    │   GET /getUpdates -> data/gates/*.decree
   │  ~/.agent_data record     │
   └──────────────┬────────────┘
                  │ LLM ROUTING (per run)  -> fixer/llm_router.py (W1/W14)
   ┌──────────────▼──────────────┬────────────────────────────────────────────
   │ TIER ORDER (healthy+budget) │  complete(): watchdog (W11) + breaker (W12) │
   │  1 omniroute (gateway, auto/*)  reachable via localhost / tunnel         │
   │  2 gemini / groq / openrouter_free / mistral   (direct cloud, opt-in)    │
   │  3 omniroute_fallback (hosted, outage only)    (OMNIROUTE_FALLBACK_*)    │
   │  4 copilot (esc. tier, HARD issues only)       (COPILOT_API_KEY, W9)     │
   │  5 hetzner / llm7 (opportunistic ONLY, W19)                              │
   │                                                                          │
   │  data/provider_usage.json  (W18)  tokens/calls/fails per provider,       │
   │                                  checked BEFORE each call + daily cap    │
   └─────────────────────────────┬────────────────────────────────────────────
     per-issue guards (fixer/guards.py): budget overrun -> abandon (W15),
     exception -> data/failures.jsonl + metrics.jsonl (W16/W22),
     checkpoint after every sub-step (W10), repo-map grounding (W6)

   RESULT of a tick (committed back by step 6):                               │
     data/board.json            lanes, prs_today, attempted, gate queue      │
     data/gates/*.pending|decree|outcome                                      │
     data/failures.jsonl + data/metrics.jsonl + data/provider_usage.json     │
     fixer/.agent_data/**       workflow records, transcripts, checkpoints    │
     data/log*.txt              beacon + controller logs                      │
```

## 2. Components

| Component | File / host | Role |
|---|---|---|
| Controller job | `.github/workflows/controller.yml` | Triggers (2 crons + dispatch + daily digest cron), env wiring, checkout, poll, tick, commit-back, notify |
| Gate poller | `beacon/gate_poller.py` | Pumps Telegram bot; turns button presses into `data/gates/*.decree` (stdlib only) |
| Tick | `beacon/tick.py` | One bounded unit of work per invocation; spawns the fixer; budget guards; stale-draft sweep |
| Beacon util | `beacon/beacon_util.py` | Glue: paths, JSON, TG transport (multi-IP), gate conventions |
| Daily digest | `beacon/digest.py` | W17: one-per-day summary of failures/metrics from the committed JSONL |
| Fixer | `fixer/oss_agent_v2.py` | Hunt → classify → plan-first → repo-map grounding → implement → test-loop → gate → draft PR |
| LLM router | `fixer/llm_router.py` | W1/W9/W11/W12/W14/W18/W19: tiered provider pool, watchdog, breaker, budget, escalation |
| Guards | `fixer/guards.py` | W15/W16/W22/W10: issue budget, failure isolation, metrics/failures JSONL, checkpoints |
| Repo map | `fixer/repo_map.py` | W6: ast/regex structural map + `read_file()` tool, fed into grounding |
| Patches | `fixer/patches.py` | W7: SEARCH/REPLACE + full-file + unified-diff applator (never-guessing match) |
| Repo policy | `fixer/repo_policy.py` | W4: etiquette preflight (docs/cached refusal) before any repo is targeted |
| Validation | `fixer/validate_language.py` + `capability_map.json` | W5: proves a language on synthetic fixtures before it is trusted on real repos |
| Token audit | `scripts/audit_tokens.py` | W3: masked credential inventory + rotation reminders |
| Manual bot | `beacon/manual_bot.py` | Telegram operator CLI for local ops |

## 3. `tick.py` decision order (exactly one unit per tick)

1. **Gate decree first** — any `.decree` (operator tapped Approve/Decline) → run fixer `--gate-sync`, consume once.
2. **Hunt + solve** — only if under budget: `max_pending_gates` (3), `max_prs_per_day` (2), per-repo `max_attempts_per_repo_day` (4), open-PR guard, pending-cap 14 days. Fixer runs etiquette preflight (W4) before spending tokens.
3. **Housekeeping** — close stale own-drafts older than 3 days via `--force -close`.

## 4. Fixed paths & data layout

| Path (repo root) | Purpose |
|---|---|
| `config/targets.json` | target repos + labels + caps: `yunaremaia/sandbox-ffi-layers`, `Rekin226/aquascope`, `taranis-ai/taranis-ai`, `izzywdev/FuzeFront` |
| `data/board.json` | lanes, `prs_today`, `attempted`, gate queue |
| `data/gates/` | `.pending` / `.decree` / `.outcome` gate files |
| `data/failures.jsonl` | W16/W22: append-only per-failure rows (never overwrites) |
| `data/metrics.jsonl` | W22: one row per attempt (repo, issue, language, outcome, tokens) |
| `data/provider_usage.json` | W18: per-provider tokens/calls/fails, daily rollover |
| `data/offset.json` | Telegram getUpdates offset |
| `fixer/.agent_data/` | workflows, conversations, transcripts, logs, reports, checkpoints |
| `manual.env` | local secrets (git-ignored); see §6 |

## 5. APIs in play

| API | Endpoint(s) | Used by | Needs |
|---|---|---|---|
| GitHub REST | `/repos`, `/issues`, `/pulls`, `/contents`, `/commits`, `/search/issues` | tick → fixer | `PR_PAT` (repo scope) |
| GitHub Actions | `/actions/secrets`, `/actions/workflows/*/dispatches`, `/actions/runs` | ops / wiring | admin or `workflow` scope |
| Telegram Bot API | `getUpdates`, `sendMessage` | gate_poller, manual_bot, digest | `TELEGRAM_BOT_TOKEN` |
| omniRoute gateway | `GET /v1/models`, `POST /v1/chat/completions` | router tier 1 | `OMNIROUTE_API_KEY` |
| Gemini / Groq / OpenRouter / Mistral (OpenAI-compatible) | `POST /v1/chat/completions` | router tiers 2 | respective keys |
| Hosted fallback | `POST /v1/chat/completions` | router tier 3 | `OMNIROUTE_FALLBACK_*` |
| Copilot (escalation) | `https://api.githubcopilot.com/chat/completions` | router tier 4 | `COPILOT_API_KEY` |
| cloudflared | quick/named tunnel (no API) | routes gateway | none |

## 6. Accounts, tokens, secrets — roles & scopes (never values)

### GitHub accounts
| Account | Role in the machine | Where used |
|---|---|---|
| `zakirHacks56` | Owner of `dc-sandbox`. One stored PAT (Windows Credential Manager `git:https://zakirHacks56@github.com`, 40 chars, `workflow` scope) for local push incl. `controller.yml`; `manual.env` `GITHUB_TOKEN` (classic `repo` scope) by the local fixer. | git push, fixer API |
| `IbrahimCodes347` | gh CLI login (scopes `gist, read:org, repo, workflow`). Author of fork PRs (sandbox-ffi-layers #33/#34/#38). No write access to `dc-sandbox`. | gh CLI, fork PRs |
| `ansarifahad7577-sketch` | legacy name of `IbrahimCodes347` (keyring label) | history only |
| `izzywdev` | Owner of `FuzeFront` (target). PR policy `collaborators_only` → refused; now caught by W4 caching too. | target repo (blocked) |
| `yunaremaia`, `Rekin226`, `taranis-ai`, `Posnic` | Owners of target repos. | target repos |

### GitHub secrets on `zakirHacks56/dc-sandbox`
| Secret | Env var | Purpose |
|---|---|---|
| `GH_PAT` | — | checkout token |
| `PR_PAT` | `GITHUB_TOKEN`/`GH_TOKEN` | fixer API calls |
| `LLM_API_KEY` | `LLM_API_KEY` | legacy/fallback key |
| `OMNIROUTE_API_KEY` | `OMNIROUTE_API_KEY` | bearer for primary gateway |
| `OMNIROUTE_BASE_URL` | `OMNIROUTE_BASE_URL` | primary gateway URL, must end `/v1` |
| `OMNIROUTE_MODEL*/OMNIROUTE_FALLBACK_*` (10) | same | concrete model overrides; empty wins over stale |
| `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`, `HETZNER_API_KEY`, `LLM7_API_KEY` | same | NEW W1/W14 direct-provider tiers (opt-in; absent → skipped) |
| `COPILOT_API_KEY` (+ `COPILOT_BASE_URL`, `COPILOT_MODEL`) | same | NEW W9 escalation tier (opt-in) |
| `GLOBAL_DAILY_TOKEN_BUDGET` | `GLOBAL_DAILY_TOKEN_BUDGET` | NEW W18 global ceiling override |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | same | gates + failure ping + daily digest |
| `SIGNOFF_NAME` / `SIGNOFF_EMAIL` | same | signoff identity |

### Local `manual.env` (git-ignored)
`MANUAL_BOT_TOKEN`, `TELEGRAM_ALLOWED_IDS`, `LLM_API_KEY`, `OMNIROUTE_API_KEY`,
`GITHUB_TOKEN`, model overrides — mirrors for local runs; audited masked by
`scripts/audit_tokens.py` (W3) with a rotate-after-85-days reminder.

## 7. LLM routing rules (the free path)

- Client is built as `llm_router.CompletionProxy` (`_make_ai_client`, oss_agent_v2.py:410) — every existing `ai_client.chat.completions.create(...)` call goes through the pool.
- Tier order (per §1): omniRoute gateway → direct cloud providers (opt-in keys) → hosted fallback → copilot (HARD only) → opportunistic last.
- `complete()` applies: per-provider circuit breaker (3 fails → 300s cooldown), watchdog thread cap (default 120s), budget pre-check (W18: skip provider if its daily tokens are spent), and usage recording after every call.
- `auto/*` combos are omniRoute-virtual; `resolve_model()` substitutes concrete defaults on plain OpenAI-compatible endpoints (empty-beats-stale everywhere).
- Escalation (`set_escalation`, W9) routes HARD-flagged issues to the copilot tier when `COPILOT_API_KEY` is set; otherwise the primary pool serves.

## 8. Hard limits (kept from the solution plan)

- Free-tier model reasoning cannot be engineered past a wall — tiering only raises reliability/efficiency, never capability. Keep the Telegram human gate permanent.
- The omniRoute gateway + tunnel remain optional after W1: with direct cloud keys set, the runner calls providers directly and needs no PC. Without them, the gateway stays the primary path.

## 9. Solution-plan implementation status (W1–W22)

| # | Fix | Where | Status |
|---|---|---|---|
| W1 | drop single-gateway dependency | `fixer/llm_router.py` providers | implemented, live-tested vs gateway |
| W2 | test-and-iterate loop | built-in attempt loop + `--test-command` | existed, re-verified |
| W3 | token rotation + audit | `scripts/audit_tokens.py` (masked, 85d reminder) | implemented + tested |
| W4 | repo etiquette preflight | `fixer/repo_policy.py` wired into discovery + `main()` | implemented + tested (fake-GH suite) |
| W5 | language validation harness | `fixer/validate_language.py` + `capability_map.json` | implemented; python **proven** E2E |
| W6 | repo-map instead of dumps | `fixer/repo_map.py` fed into grounding | implemented + tested |
| W7 | diffs, not full rewrites | `fixer/patches.py` SEARCH/REPLACE + unified + full-file | implemented + tested |
| W8 | plan before code | existing plan-first (escalation_budget) | existed, re-verified |
| W9 | selective escalation | copilot tier + `set_escalation(difficulty==hard)` | implemented |
| W10 | checkpoint persistence | `guards.checkpoint()` per attempt | implemented + tested |
| W11 | per-call watchdog | `call_with_watchdog` thread cap | implemented + tested |
| W12 | circuit breaker | per-provider trip + cooldown + recovery | implemented + tested |
| W13 | shallow clone / dep cache | setup-python `cache: pip` (kept depth-0 clone) | implemented |
| W14 | multi-provider failover | `complete()` tier chain | implemented, live-tested |
| W15 | per-issue token budget | `guards.IssueBudget` around attempt loop | implemented + tested |
| W16 | failure isolation | `_guarded_call` + `failure_isolation` | implemented + tested |
| W17 | alert split | `beacon/digest.py` + controller digest job | implemented + tested |
| W18 | usage counters before call | `data/provider_usage.json` pre-check | implemented + tested |
| W19 | demote volatile providers | hetzner/llm7 → opportunistic-only tier | implemented + tested |
| W20 | Aider/SWE-agent eval | See below (spike recommendation) | documented, not spun up |
| W21 | Copilot Student | `COPILOT_API_KEY` opt-in tier | wired (needs the user's key) |
| W22 | metrics log | `data/metrics.jsonl` per attempt | implemented + tested |

**W20 note (evaluation):** before hand-rolling more of the fixer, run one
spike: point `Aider` (or `SWE-agent`) at the sandbox-ffi-layers fixtures with a
free-tier model and compare PR-draft success per token against the current
`generate_fix`/`apply_fix` path. The repo-map + patch engine in this repo
mirror their design deliberately, so adopting them is a swap, not a rewrite —
do it only if the spike beats the in-house path on the same issues.

## 10. Invariants / failure modes

- One unit per tick; everything committed back; empty secret beats stale.
- Tunnel is **ephemeral** — a restart gives a new URL; without direct cloud keys configured, a dead tunnel = fat fall to the hosted `OMNIROUTE_FALLBACK_*` tier. Optionally set `GEMINI_API_KEY`/`GROQ_API_KEY` to make the free path independent of the PC (W1).
- Tokens scoped per account: `workflow` scope required to push `.github/workflows/*`; `IbrahimCodes347` lacks `dc-sandbox` access; push via the stored `zakirHacks56` credential.
- New GH secrets must be added for the direct-provider tiers for them to take effect; absent secrets are skipped, so adding them is non-breaking.
- New `fixer/tests/` (46 tests) guard every module; `pytest.ini` scopes collection to `fixer/tests`.