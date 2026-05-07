#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEPENDENCY_FILE_PATTERNS = (
    "requirements*.txt",
    "environment*.yml",
    "environment*.yaml",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "Pipfile",
)

REQ_LINE = re.compile(r"^\s*([a-zA-Z0-9_.-]+)\s*([<>=!~].+)?\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe dependency/version hints from repository files.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_dep_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in DEPENDENCY_FILE_PATTERNS:
        for path in repo_root.rglob(pattern):
            if not path.is_file() or path in seen:
                continue
            rel = str(path.resolve().relative_to(repo_root.resolve()))
            if rel.startswith("temp/") or rel.startswith(".git/"):
                continue
            seen.add(path)
            files.append(path)
    return sorted(files)


def parse_constraints(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, str]] = []
    for ln, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = REQ_LINE.match(stripped)
        if not m:
            continue
        pkg = m.group(1).strip()
        spec = (m.group(2) or "").strip()
        rows.append({"package": pkg, "specifier": spec, "line": str(ln), "text": stripped})
    return rows


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()

    files = iter_dep_files(repo_root)
    constraints: dict[str, list[dict[str, str]]] = {}
    records: list[dict[str, Any]] = []

    for dep_file in files:
        parsed = parse_constraints(dep_file)
        rel = str(dep_file.resolve().relative_to(repo_root))
        records.append({"path": rel, "constraint_count": len(parsed), "constraints": parsed[:200]})
        for row in parsed:
            key = row["package"].lower().replace("_", "-")
            constraints.setdefault(key, []).append({"path": rel, "specifier": row["specifier"], "line": row["line"]})

    payload = {
        "repo_root": str(repo_root),
        "dependency_files": records,
        "constraint_index": constraints,
        "file_count": len(files),
    }
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
