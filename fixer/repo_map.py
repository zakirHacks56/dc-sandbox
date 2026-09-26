"""
repo_map.py
Lightweight structural map of a repository (W6).

Builds a compact map -- files, function/class signatures, import graph -- so a
small-context free model can reason about a large repo without being handed
raw dumps of every file. The model is fed the MAP plus a `read_file(path)`
tool (RepoContext) it uses to pull in only the files it actually needs.

Fulfils solution-plan weakness W6 (repo-map instead of raw dumps), replacing
the old never-enforced "dump everything" approach. Uses stdlib `ast` for
Python; a conservative regex fallback covers other languages so the map is
never empty on a non-Python kid. Requires zero third-party deps.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__",
             "dist", "build", "coverage", ".mypy_cache", ".pytest_cache",
             ".agent_data", "target", "vendor", "site-packages"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb",
                   ".rs", ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".swift",
                   ".kt", ".scala", ".vue", ".svelte", ".sh", ".dart"}
MAX_MAP_FILES = int(os.getenv("REPO_MAP_MAX_FILES", "200"))
MAX_MAP_CHARS = int(os.getenv("REPO_MAP_MAX_CHARS", "8000"))
READ_CHARS = int(os.getenv("REPO_MAP_READ_CHARS", "16000"))

# Conservative signature patterns for non-Python languages (best effort; the
# ast path is authoritative for Python).
_ARG_PATTERNS = {
    ".js":    [r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", r"^const\s+(\w+)\s*=\s*(?:async\s*)?\(", r"^class\s+(\w+)"],
    ".ts":    [r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", r"^const\s+(\w+)\s*=\s*(?:async\s*)?\(", r"^class\s+(\w+)"],
    ".tsx":   [r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", r"^export\s+(?:default\s+)?function\s+(\w+)\s*\(", r"^class\s+(\w+)"],
    ".java":  [r"(?:public|private|protected)\s+[\w<>,?\s\[\]]+\s+(\w+)\s*\(", r"^class\s+(\w+)"],
    ".go":    [r"^func\s+\(?[\w\*\[\]\s]+\)?\s*(\w+)\s*\(", r"^func\s+(\w+)\s*\(", r"^type\s+(\w+)\s+struct"],
    ".rb":    [r"^\s*def\s+(self\.)?(\w+)", r"^\s*class\s+(\w+)"],
    ".rs":    [r"^\s*fn\s+(\w+)\s*\(.*\)", r"^\s*(?:pub\s+)?struct\s+(\w+)", r"^\s*(?:pub\s+)?trait\s+(\w+)"],
    ".c":     [r"^[A-Za-z_][\w\s\*]*\b(\w+)\s*\("],
    ".cpp":   [r"[A-Za-z_][\w\s\*<>:&]*\b(\w+)\s*\(", r"^class\s+(\w+)"],
    ".cs":    [r"(?:public|private|internal|protected)\s+[\w<>,?\s\[\]]+\s+(\w+)\s*\(", r"^class\s+(\w+)"],
    ".dart":  [r"^\s*(?:Future|void|int|String|bool|double|var|final|const|List\S*|Map\S*)\s*(\w+)\s*\("],
    ".sh":    [r"^\s*(\w+)\s*\(\)\s*\{"],
}
_IMPORT_PAT = re.compile(
    r"^\s*(?:from\s+[\w.]+|import\s+[\w.]+|import\s+\{[\w,\s]+\}\s+from\s+['\"][\w./]+['\"]|"
    r"require\s*\(\s*['\"][\w./]+['\"]\s*\)|using\s+[\w.]+;)"
)


@dataclass
class RepoFile:
    path: str
    language: str
    functions: list = field(default_factory=list)   # [(name, line)]
    classes: list = field(default_factory=list)     # [(name, line)]
    imports: list = field(default_factory=list)

    def render(self) -> str:
        parts = [self.path]
        if self.functions:
            sigs = ", ".join(f"{n}@{ln}" for n, ln in self.functions[:25])
            parts.append(f"  funcs: {sigs}")
        if self.classes:
            sigs = ", ".join(f"{n}@{ln}" for n, ln in self.classes[:25])
            parts.append(f"  classes: {sigs}")
        if self.imports:
            parts.append(f"  imports: {', '.join(self.imports[:20])}")
        return "\n".join(parts)


def _walk_files(repo_dir: Path, max_files: int = MAX_MAP_FILES) -> Iterable[Path]:
    count = 0
    for root, dirs, names in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            path = Path(root) / name
            if path.suffix not in CODE_EXTENSIONS:
                continue
            yield path
            count += 1
            if count >= max_files:
                return


def _parse_python(source: str) -> tuple[list, list, list]:
    functions, classes, imports = [], [], []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return functions, classes, imports
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append((node.name, node.lineno))
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append((f"async {node.name}", node.lineno))
        elif isinstance(node, ast.ClassDef):
            classes.append((node.name, node.lineno))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imports.append(alias.asname or alias.name.split(".")[0])
    return functions, classes, imports


def _parse_generic(source: str, ext: str) -> tuple[list, list, list]:
    functions, classes, imports = [], [], []
    patterns = _ARG_PATTERNS.get(ext, [])
    for line_no, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#", "/*", "*", "//!", "// ")):
            continue
        if _IMPORT_PAT.match(line):
            imports.append(stripped.split()[1] if "from " in stripped
                           else stripped.replace("`", "").strip())
            continue
        for pat in patterns:
            m = re.match(pat, line)
            if m:
                groups = m.groups()
                name = groups[-1] if groups else "?"
                (classes if "class " in line or "struct " in line or "trait " in line
                 else functions).append((name, line_no))
                break
    return functions, classes, imports


@dataclass
class RepoMap:
    root: str
    files: list = field(default_factory=list)

    def render(self, max_chars: int = MAX_MAP_CHARS) -> str:
        lines = [f"# repository map: {self.root} ({len(self.files)} files)"]
        budget = max_chars
        for f in self.files:
            block = f.render()
            if len(block) + len("\n".join(lines)) > budget and budget != max_chars == MAX_MAP_CHARS:
                pass
            if len("\n".join(lines)) + len(block) + 2 > budget:
                lines.append("# ... map truncated ...")
                break
            lines.append(block)
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "root": self.root,
            "file_count": len(self.files),
            "files": [
                {"path": f.path, "language": f.language,
                 "functions": f.functions, "classes": f.classes,
                 "imports": f.imports}
                for f in self.files
            ],
        }


def build_repo_map(repo_dir, max_files: int = MAX_MAP_FILES) -> RepoMap:
    """W6: build the structural map for a repo directory (cached-feel: cheap;
    callers decide caching). Non-Python files use regex signatures."""
    repo_dir = Path(repo_dir)
    files = []
    for path in _walk_files(repo_dir, max_files=max_files):
        rel = path.relative_to(repo_dir).as_posix()
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ext = path.suffix.lower()
        if ext == ".py":
            functions, classes, imports = _parse_python(source)
        else:
            functions, classes, imports = _parse_generic(source, ext)
        files.append(RepoFile(path=rel, language=ext.lstrip(".") or "text",
                              functions=functions, classes=classes,
                              imports=imports))
    files.sort(key=lambda f: f.path)
    return RepoMap(root=str(repo_dir), files=files)


def render_compact(repo_dir, max_chars: int = MAX_MAP_CHARS) -> str:
    return build_repo_map(repo_dir).render(max_chars=max_chars)


# ---------------------------------------------------------------------------
# The read_file() tool the model can call (W6)
# ---------------------------------------------------------------------------
class RepoContext:
    """Grounds one fix in exactly the files it needs.

    `read(path)` returns the file's REAL content with line numbers, truncated
    to READ_CHARS, and refuses paths outside the allowed set -- so a model
    cannot be fed an enormous file by accident."""

    def __init__(self, repo_dir, allowed_paths: Optional[set] = None,
                 max_chars: int = READ_CHARS):
        self.repo_dir = Path(repo_dir)
        self.allowed_paths = {p.replace("\\", "/") for p in (allowed_paths or set())}
        self.max_chars = max_chars

    def read(self, path: str) -> str:
        norm = path.replace("\\", "/").lstrip("/")
        if self.allowed_paths and norm not in self.allowed_paths:
            return f"ERROR: {path} is not in the allowed file set"
        full = self.repo_dir / norm
        try:
            lines = full.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return f"ERROR: no such file: {path}"
        out, used = [], 0
        for i, line in enumerate(lines, start=1):
            block = f"{i:6} | {line}"
            out.append(block)
            used += len(block) + 1
            if used >= self.max_chars:
                out.append(f"... ({len(lines) - i} more lines omitted, re-read with a range)")
                break
        return "\n".join(out)


def mapping_tools_hint(rel_paths: list) -> str:
    """Instruction line reminding the model it may call read_file()."""
    files = ", ".join(sorted(rel_paths)[:12])
    return ("You have the repository MAP above, not the files. Before writing "
            "any change, call read_file() on the specific files you need "
            f"({' ... ' if len(files) > 400 else ''}{files} ...). Only edit files "
            "whose exact content you have seen.") if files else ""


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="repo_map", description="W6 repo map")
    parser.add_argument("--map", metavar="DIR", help="build + print the map")
    parser.add_argument("--json", action="store_true", help="dump as JSON")
    parser.add_argument("--read", metavar="PATH", help="read one file via RepoContext")
    args = parser.parse_args(argv)
    if args.map:
        m = build_repo_map(args.map)
        if args.json:
            print(json.dumps(m.to_json(), indent=2))
        else:
            print(m.render())
        if args.read:
            ctx = RepoContext(args.map, {p for f in m.files for p in [f.path]})
            print(ctx.read(args.read))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())