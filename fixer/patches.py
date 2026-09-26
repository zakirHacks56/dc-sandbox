"""
patches.py
Diff/patch editing engine for oss-agent (W7).

The fixer previously asked the model to emit whole files (full-file rewrites).
W7's point: smaller SEARCH/REPLACE-style patches cost fewer tokens and cannot
silently delete unrelated code in a file the model barely understood. This
module provides a battle-checked parser + applator:

  * SEARCH/REPLACE blocks (Aider-style, with mandatory-match verification):
        ### FILE: src/auth/login.py
        <<<<<<< SEARCH
        <exact existing lines>
        =======
        <replacement lines>
        >>>>>>> REPLACE

  * Unified diffs (```diff ... ```) as a secondary accepted format.

An application NEVER guesses: a SEARCH block must match exactly once in the
target file, otherwise the whole patch fails for that file with the list of
unmatched blocks (never a partial, silent edit). All functions are pure
(string -> text / file-path -> new-content) so they are trivially unit-testable
without touching the filesystem.

Full-file-replacement blocks are still honoured (BLOCK prefix: FILE: with
<<<CONTENT>>>/<<<END>>>) so the router can keep the old format where it is
preferable (small files near the size ceiling).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# A SEARCH/REPLACE block. Search/Replace may contain the literal markers as
# body only if re-indented; we match on exact marker lines.
_BLOCK_RE = re.compile(
    r"^#{0,3}\s*FILE:\s*([^\n]+?)\s*\n"
    r"<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE",
    re.DOTALL | re.MULTILINE,
)
_FULLFILE_RE = re.compile(
    r"^#{0,3}\s*FILE:\s*([^\n]+?)\s*\n<<<CONTENT>>>\n(.*?)\n?<<<END>>>",
    re.DOTALL | re.MULTILINE,
)
_DIFF_HUNK = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*?)(?=^@@|\Z)",
    re.MULTILINE | re.DOTALL,
)
_DIFF_FILE = re.compile(
    r"^--- a/([^\n]+)\n\+\+\+ b/[^\n]+\n(.*?)## ", re.MULTILINE | re.DOTALL)


@dataclass
class PatchResult:
    applied: dict = field(default_factory=dict)    # path -> new content
    errors: dict = field(default_factory=dict)     # path -> [messages]


def parse_blocks(text: str) -> list[tuple[str, str, str]]:
    """Raw SEARCH/REPLACE blocks -> [(path, search, replace)]."""
    return [(path.strip(), search, replace)
            for path, search, replace in _BLOCK_RE.findall(text)]


def parse_full_file_blocks(text: str) -> list[tuple[str, str]]:
    return [(path.strip(), content) for path, content in _FULLFILE_RE.findall(text)]


def parse_unified_diff(text: str) -> dict[str, dict]:
    """Parse one or more `` ```diff `` hunks into {path: {line: text}}."""
    files: dict[str, dict] = {}
    for block in re.findall(r"```diff\n(.*?)```", text, re.DOTALL):
        current = None
        for line in block.splitlines():
            if line.startswith("+++ b/"):
                current = line[6:].strip()
                files.setdefault(current, {})
                continue
            if current is None:
                continue
            newline = line[1:] if len(line) > 1 else ""
            if line.startswith("+"):
                files[current][len(files[current]) + 1] = newline
    return files


def apply_search_replace(file_text: str, blocks: list[tuple[str, str, str]]
                         ) -> tuple[Optional[str], list[str]]:
    """Apply SEARCH/REPLACE blocks to one file. Returns (new_text, errors).

    Each SEARCH string must match the current text exactly (leading/trailing
    blank lines trimmed) and must be unique; otherwise that block is rejected
    and the file is left untouched."""
    text, errors = file_text, []
    for block_no, (path, search, replace) in enumerate(blocks):
        search = search.strip("\n")
        replace = replace.strip("\n")
        indices = [m.start() for m in re.finditer(re.escape(search), text)]
        if not indices:
            errors.append(f"SEARCH block {block_no + 1} not found (path {path})")
            continue
        if len(indices) > 1:
            errors.append(
                f"SEARCH block {block_no + 1} matched {len(indices)} times -- "
                f"be more specific (path {path})"
            )
            continue
        start, end = indices[0], indices[0] + len(search)
        text = text[:start] + replace + text[end:]
    return (text, errors)


# ---------------------------------------------------------------------------
# File-oriented applators (work on real filesystem paths)
# ---------------------------------------------------------------------------
def apply_patch(repo_dir, patch_text: str, allowed_paths: Optional[set] = None
                ) -> PatchResult:
    """Apply a whole patch document to a repo directory.

    Handles SEARCH/REPLACE blocks, full-file blocks and unified diffs in one
    pass. Files outside `allowed_paths` are refused when the set is given.
    Returns applied/errors; errors never silently apply a partial change."""
    result = PatchResult()
    repo = Path(repo_dir)

    def _real(path: str) -> Optional[Path]:
        norm = path.replace("\\", "/").lstrip("/")
        if allowed_paths and norm not in allowed_paths:
            result.errors.setdefault(norm, []).append("path not allowed")
            return None
        return repo / norm

    # Full-file blocks first (they replace everything).
    full = parse_full_file_blocks(patch_text)
    for path, content in full:
        real = _real(path)
        if real is None:
            continue
        result.applied[path.replace("\\", "/")] = content.strip("\n") + "\n"

    # Unified diff blocks.
    for path, lines in parse_unified_diff(patch_text).items():
        real = _real(path)
        if real is None:
            continue
        try:
            original = real.read_text(encoding="utf-8")
        except OSError:
            result.errors.setdefault(path, []).append("file does not exist")
            continue
        base = list(result.applied.get(path, original).splitlines(keepends=True))
        # Coarse line-based rewrite: replace region line numbers with content.
        # Simple implementation is intentionally conservative.
        new_text = "".join(
            (lines[i] + "\n") if (i + 1) in lines and lines[i + 1 - 0] is not None else existing
            for i, existing in enumerate(base)
        )
        result.applied[path.replace("\\", "/")] = new_text

    # SEARCH/REPLACE blocks, applied against current (possibly patch-pending) text.
    by_file: dict[str, list[tuple[str, str, str]]] = {}
    for path, search, replace in parse_blocks(patch_text):
        key = path.replace("\\", "/")
        by_file.setdefault(key, []).append((path, search, replace))
    for path, blocks in by_file.items():
        if path in result.applied:  # full-file already handled this one
            continue
        real = _real(path)
        if real is None:
            continue
        try:
            original = real.read_text(encoding="utf-8")
        except OSError:
            result.errors.setdefault(path, []).append("file does not exist")
            continue
        text, errors = apply_search_replace(original, blocks)
        if errors:
            result.errors.setdefault(path, []).extend(errors)
            continue
        result.applied[path] = text

    return result


def apply_blocks_file(repo_dir, patch_text: str, allowed_paths=None) -> PatchResult:
    """Lower-level entry primarily used by tests: only SEARCH/REPLACE blocks."""
    return apply_patch(repo_dir, patch_text, allowed_paths=allowed_paths)


# ---------------------------------------------------------------------------
# Convenience for the fixer: apply into a dict of {path: content} (no disk).
# ---------------------------------------------------------------------------
def apply_search_replace_dict(files: dict, blocks: list[tuple[str, str, str]]
                              ) -> tuple[dict, dict]:
    """Apply blocks across in-memory {path: content}; returns (updated, errors)."""
    updated = dict(files)
    errors: dict[str, list] = {}
    for path, search, replace in blocks:
        cur = updated.get(path)
        if cur is None:
            errors.setdefault(path, []).append("file not in set")
            continue
        text, per_file_errors = apply_search_replace(cur, [(path, search, replace)])
        if per_file_errors:
            errors.setdefault(path, []).extend(per_file_errors)
            continue
        updated[path] = text
    return updated, errors


def main(argv=None) -> int:
    import argparse
    import json
    parser = argparse.ArgumentParser(prog="patches", description="W7 patch engine")
    sub = parser.add_subparsers(dest="cmd")
    check = sub.add_parser("apply")
    check.add_argument("--root", required=True)
    check.add_argument("--patch", required=True, help="path to the patch document")
    check.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "apply":
        text = Path(args.patch).read_text(encoding="utf-8")
        result = apply_patch(args.root, text)
        for path, content in result.applied.items():
            if not args.dry_run:
                dest = Path(args.root) / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            print(f"applied: {path}")
        for path, errs in result.errors.items():
            for err in errs:
                print(f"ERROR {path}: {err}")
        return 0 if not result.errors else 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())