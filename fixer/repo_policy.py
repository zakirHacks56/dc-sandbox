"""
repo_policy.py
Repository-etiquette preflight check (W4).

Before the agent targets a repo it checks the repo's own signals for whether
external/automated contributions are welcome. It is deliberately heuristic --
GitHub exposes no API for a repo's PR-contribution policy, so we read the
signals a maintainer actually uses:

  * CONTRIBUTING.md / README wording (bots, spam-automation, "no external PRs")
  * archived / disabled repos
  * absence of issues or the search label the agent hunts
  * known policy trap: GitHub's `pull_request_creation_policy: collaborators_only`
    (the izzywdev/FuzeFront case) is invisible to the API, so a repo that has
    already refused us is remembered in the skip-list beside this module.

The check is cheap (2 file fetches max) and runs during discovery, so a repo
that fails it is skipped before any issue is claimed or any tokens spent.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable, Optional

_SKIP_FILE = Path(__file__).with_name("repo_skip_list.json")

# Phrases that mark a repo as not wanting external automation.
_BANNED_PATTERNS = [
    r"no\s+(?:external|outside|automated|bot|spam)\s+(?:prs?|contributions?|pull requests?)",
    r"don'?t?\s+(?:submit|open|send)\s+(?:prs?|pull requests?)",
    r"no\s+spam",
    r"bots?\s+(?:not|aren'?t|are\s+not)\s+welcome",
    r"(?:maintainers?\s+only|collaborators?\s+only)",
    r"no\s+automated\s+(?:prs?|contributions?|work)",
    r"do\s+not\s+automate",
]
_BANNED_RE = [re.compile(p, re.IGNORECASE) for p in _BANNED_PATTERNS]


class PolicyRefusal(Exception):
    """Raised when etiquette preflight rejects a repo."""


def _read_skip_list() -> set:
    try:
        data = json.loads(_SKIP_FILE.read_text(encoding="utf-8"))
        return set(data.get("refused", []))
    except (OSError, ValueError):
        return set()


def _write_skip_list(repos: set) -> None:
    try:
        _SKIP_FILE.write_text(
            json.dumps({"refused": sorted(repos)}, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def remember_refused(repo_name: str, reason: str) -> None:
    """Persist a refusal (e.g. a rejected PR attempt) so future runs skip it."""
    repos = _read_skip_list()
    repos.add(repo_name)
    _write_skip_list(repos)


def is_remembered_refused(repo_name: str) -> bool:
    return repo_name in _read_skip_list()


def _repo_says_no(repo, issue_enabled: bool, default_branch: Optional[str]) -> str:
    """Inspect repo metadata + docs; returns a reason string or '' if OK."""
    if getattr(repo, "archived", False):
        return "archived"
    if getattr(repo, "disabled", False):
        return "disabled"
    if not issue_enabled:
        return "issues disabled"
    docs = _fetch_docs(repo, default_branch)
    for text in docs:
        for pat in _BANNED_RE:
            if pat.search(text):
                return f"contributing guidance refuses automation: {pat.pattern[:60]}"
    return ""


def _fetch_docs(repo, default_branch: Optional[str]) -> list[str]:
    """Fetch CONTRIBUTING.md and README raw text (2 requests max)."""
    out: list[str] = []
    for name in ("CONTRIBUTING.md", "CONTRIBUTING.rst", "README.md", "README.rst"):
        try:
            content = repo.get_contents(name, ref=default_branch)
            text = content.decoded_content.decode("utf-8", errors="replace")
            out.append(text)
            if name.startswith("CONTRIBUTING"):
                out.append(text)  # contributing text weighs double
        except Exception:  # noqa: BLE001
            continue
    return out


def etiquette_ok(repo_name: str, gh) -> str:
    """Preflight one repo. Returns '' when OK, or the refusal reason.

    `gh` is a PyGithub instance (or test fake). Lightweight: 1 get_repo + up
    to 4 get_contents. Refusals are cached in the skip-list so discovery does
    not re-ask GitHub on every tick."""
    if is_remembered_refused(repo_name):
        return "already refused us (skip-list)"
    try:
        repo = gh.get_repo(repo_name)
    except Exception:  # noqa: BLE001 - bad name, no access
        return "unable to read repo metadata"
    reason = _repo_says_no(repo, getattr(repo, "has_issues", True), repo.default_branch)
    if reason:
        remember_refused(repo_name, reason)
        return reason
    return ""


def filter_ok_repos(repo_names: list, gh) -> list[str]:
    """W4: drop every repo that fails etiquette, return the survivors in order."""
    ok = []
    for name in repo_names:
        reason = etiquette_ok(name, gh)
        if not reason:
            ok.append(name)
        else:
            print(f"  ↳ skipping {name} (etiquette: {reason})")
    return ok


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="repo_policy",
                                     description="W4 etiquette preflight")
    parser.add_argument("--repo", action="append", help="owner/repo to check")
    parser.add_argument("--list-skipped", action="store_true")
    parser.add_argument("--forget", metavar="REPO",
                        help="remove REPO from the skip-list")
    args = parser.parse_args(argv)
    if args.list_skipped:
        print("\n".join(sorted(_read_skip_list())) or "(none)")
    if args.forget:
        repos = _read_skip_list()
        repos.discard(args.forget)
        _write_skip_list(repos)
        print(f"forgot {args.forget}")
    if args.repo:
        from github import Auth, Github
        import os
        gh = Github(auth=Auth.Token(os.getenv("GITHUB_TOKEN", "")))
        for name in args.repo:
            reason = etiquette_ok(name, gh)
            print(f"{name}: {'OK' if not reason else f'REFUSED ({reason})'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())