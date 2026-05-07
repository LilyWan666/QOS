#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

import run_reproduce as rr  # noqa: E402
from repro_toolkit import (  # noqa: E402
    assert_action_allowed,
    append_history,
    blocked_action_payload,
    ensure_isolated_workspace_recipe,
    ensure_run_root,
    get_fsm_state,
    load_state,
    next_actions,
    repo_root_from_here,
    resolve_recipe_path,
    set_fsm_state,
    write_state,
)

EXTERNAL_RISK_PATTERNS = {
    "qpu_backend": ["IBMProvider", "IBMQ", "QiskitRuntimeService", "get_backend", "backend.run", "qiskit_ibm_runtime"],
    "credential": ["token", "credential", "secret", "api_key", "IBM_TOKEN", "QISKIT_IBM_TOKEN"],
    "network": ["requests.", "urllib.request", "http.client", "socket.", "grpc"],
    "cloud_api": ["openai", "boto3", "google.cloud", "azure"],
    "process": ["subprocess.", "os.system", "Popen("],
    "tracking": ["wandb", "mlflow"],
}
MISSING_MODULE_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
FRAMEWORK_PREFIXES = ("evaluation/agent_framework/reproduce/",)
INVALID_RUNTIME_ENV_PATTERNS = (
    "invalid environment",
    "qiskit is installed in an invalid environment",
    "cannot import name 'provider' from 'qiskit.providers'",
)


def is_framework_path(rel_path: str) -> bool:
    return any(rel_path.startswith(prefix) for prefix in FRAMEWORK_PREFIXES)


IMPORT_ERROR_NAME_RE = re.compile(r"cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--output-root",
        default="temp/agent_framework/reproduce/tools/runs",
        help="Root directory for probe artifacts.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _recipe_candidate_strings(recipe: dict[str, Any]) -> list[str]:
    out: list[str] = []
    entry = recipe.get("entry") or {}
    for item in _as_list(entry.get("command")):
        if isinstance(item, str):
            out.append(item)
    authoring = recipe.get("recipe_authoring") or {}
    for key in ("allowed_entrypoints", "blocked_entrypoints"):
        for item in _as_list(authoring.get(key)):
            if isinstance(item, str):
                out.append(item)
    preflight = recipe.get("preflight") or {}
    for item in _as_list(preflight.get("paths_exist")):
        if isinstance(item, str):
            out.append(item)
    return out


def _path_from_recipe_string(workspace_root: Path, raw: str) -> Path | None:
    if not raw or raw.startswith("{"):
        return None
    if raw.startswith("-"):
        return None
    if not raw.endswith(".py") and ".py" not in raw and "/" not in raw:
        return None
    cleaned = raw.split("{", 1)[0].strip()
    if not cleaned:
        return None
    path = Path(cleaned)
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


def _focus_term_candidates(workspace_root: Path, recipe: dict[str, Any], limit: int = 20) -> list[Path]:
    paper = recipe.get("paper") or {}
    terms = [str(term).lower() for term in _as_list(paper.get("focus_terms")) if str(term).strip()]
    if not terms:
        return []
    scored: list[tuple[int, Path]] = []
    for root_name in ("qos", "Baseline_Multiprogramming", "evaluation"):
        root = workspace_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in {"__pycache__", ".git", ".venv", "venv"} for part in path.parts):
                continue
            rel = str(path.relative_to(workspace_root)).lower()
            try:
                text = path.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                continue
            score = sum(1 for term in terms if term in rel or term in text)
            if score:
                scored.append((score, path.resolve()))
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return [path for _, path in scored[:limit]]


def collect_candidate_paths(workspace_root: Path, recipe: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for raw in _recipe_candidate_strings(recipe):
        path = _path_from_recipe_string(workspace_root, raw)
        if path is None:
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*.py"))[:50]:
                candidates.append(child.resolve())
        elif path.exists() and path.suffix == ".py":
            candidates.append(path.resolve())
    candidates.extend(_focus_term_candidates(workspace_root, recipe))
    # Include shallow top-level packages as generic anchors.
    for child in sorted(workspace_root.iterdir() if workspace_root.exists() else []):
        if child.is_dir() and (child / "__init__.py").exists():
            for py in sorted(child.glob("*.py"))[:5]:
                candidates.append(py.resolve())
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        try:
            path.relative_to(workspace_root.resolve())
        except ValueError:
            continue
        if path in seen or not path.exists() or path.suffix != ".py":
            continue
        seen.add(path)
        out.append(path)
    return out[:80]


def module_name_for_path(workspace_root: Path, path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        return None
    if rel.suffix != ".py":
        return None
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = rel.stem
    if not parts:
        return None
    return ".".join(parts)


def static_scan(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {
        "syntax_ok": False,
        "classes": [],
        "functions": [],
        "imports": [],
        "external_risks": [],
        "errors": [],
    }
    try:
        tree = ast.parse(text, filename=str(path))
        result["syntax_ok"] = True
    except SyntaxError as exc:
        result["errors"].append(f"syntax_error: {exc}")
        tree = None
    if tree is not None:
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                result["classes"].append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result["functions"].append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        result["imports"] = sorted(set(imports))
    risks: list[dict[str, str]] = []
    lower_text = text.lower()
    for category, patterns in EXTERNAL_RISK_PATTERNS.items():
        for pattern in patterns:
            haystack = lower_text if pattern.islower() else text
            if pattern in haystack:
                risks.append({"category": category, "pattern": pattern})
    result["external_risks"] = risks
    return result


def import_probe(workspace_root: Path, module: str, python_executable: str, timeout: int = 20) -> dict[str, Any]:
    code = (
        "import importlib, json, traceback\n"
        f"module = {module!r}\n"
        "try:\n"
        "    imported = importlib.import_module(module)\n"
        "    payload = {'ok': True, 'module': module, 'file': getattr(imported, '__file__', None)}\n"
        "except Exception as exc:\n"
        "    payload = {'ok': False, 'module': module, 'error': repr(exc), 'traceback_tail': traceback.format_exc()[-4000:]}\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workspace_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("REPRO_SIMULATION_ONLY", "1")
    proc = subprocess.run(
        [python_executable, "-B", "-c", code],
        cwd=workspace_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
    except Exception:
        payload = {"ok": False, "error": "invalid_probe_stdout", "raw_stdout": proc.stdout[-1000:]}
    payload["returncode"] = proc.returncode
    payload["stderr_tail"] = proc.stderr[-2000:]
    if not payload.get("ok"):
        text = str(payload.get("traceback_tail", "")) + "\n" + str(payload.get("error", ""))
        missing = MISSING_MODULE_RE.findall(text)
        imports = IMPORT_ERROR_NAME_RE.findall(text)
        if missing:
            payload["missing_module"] = missing[-1]
        if imports:
            symbol, source = imports[-1]
            payload["missing_symbol"] = symbol
            payload["missing_symbol_source"] = source
    return payload


def recommended_actions(scan: dict[str, Any], probe: dict[str, Any], simulation_only: bool) -> list[str]:
    actions: list[str] = []
    error_text = (
        str(probe.get("traceback_tail", ""))
        + "\n"
        + str(probe.get("error", ""))
        + "\n"
        + str(probe.get("stderr_tail", ""))
    ).lower()
    if any(pattern in error_text for pattern in INVALID_RUNTIME_ENV_PATTERNS):
        actions.append("runtime_env_select")
    risks = scan.get("external_risks") or []
    if simulation_only and any(item.get("category") in {"qpu_backend", "credential"} for item in risks):
        actions.append("simulation_backend_fix")
    missing = str(probe.get("missing_module", ""))
    if missing:
        root = missing.split(".", 1)[0]
        if root in {"qos", "qvm", "Baseline_Multiprogramming", "settings", "data", "src"}:
            actions.append("source_path_fix")
        else:
            actions.append("dependency_runtime_fix")
    if probe.get("missing_symbol"):
        actions.append("runtime_trace_fix")
    if not actions and probe.get("ok"):
        actions.append("build_original_runner")
    if not actions:
        actions.append("source_fix")
    out: list[str] = []
    for action in actions:
        if action not in out:
            out.append(action)
    return out


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, "repo_path_probe")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_repo_path_probe",
            action="repo_path_probe",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_repo_path_probe", payload)
        state["last_step"] = "repro_repo_path_probe"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    recipe_path = resolve_recipe_path(args.recipe, repo_root)
    recipe = rr.load_json(recipe_path)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    run_root = ensure_run_root(str(recipe.get("name") or recipe_path.stem), state, output_root)
    workspace_recipe_path, workspace_recipe = ensure_isolated_workspace_recipe(
        state=state,
        run_root=run_root,
        repo_root=repo_root,
        recipe_path=recipe_path,
        recipe=recipe,
    )
    workspace_root = rr.resolve_workspace_root(repo_root, str(workspace_recipe.get("workspace_root", ".")))
    simulation_only = bool((workspace_recipe.get("recovery") or {}).get("simulation_only"))

    candidates = collect_candidate_paths(workspace_root, workspace_recipe)
    entries: list[dict[str, Any]] = []
    aggregate_actions: list[str] = []
    for path in candidates:
        module = module_name_for_path(workspace_root, path)
        rel = str(path.relative_to(workspace_root))
        framework_path = is_framework_path(rel)
        scan = static_scan(path)
        probe = import_probe(workspace_root, module, args.python_executable) if module and scan.get("syntax_ok") else {
            "ok": False,
            "module": module,
            "error": "module_name_or_syntax_unavailable",
        }
        actions = recommended_actions(scan, probe, simulation_only)
        is_substantive = bool(scan.get("classes") or scan.get("functions"))
        for action in actions:
            if action == "build_original_runner" and (not is_substantive or rel.endswith("/__init__.py")):
                continue
            if not framework_path and action not in aggregate_actions:
                aggregate_actions.append(action)
        entries.append(
            {
                "path": rel,
                "module": module,
                "path_role": "reproduce_framework" if framework_path else "candidate_original_repo_path",
                "syntax_ok": scan.get("syntax_ok"),
                "classes": scan.get("classes", [])[:30],
                "functions": scan.get("functions", [])[:50],
                "imports": scan.get("imports", [])[:80],
                "external_risks": scan.get("external_risks", []),
                "import_probe": probe,
                "recommended_next_actions": actions,
            }
        )

    importable = [item for item in entries if item.get("import_probe", {}).get("ok")]
    blocked = [item for item in entries if not item.get("import_probe", {}).get("ok")]
    risky = [item for item in entries if item.get("external_risks")]
    original_entries = [item for item in entries if item.get("path_role") == "candidate_original_repo_path"]
    substantive_original_entries = [
        item
        for item in original_entries
        if (item.get("classes") or item.get("functions")) and not str(item.get("path") or "").endswith("/__init__.py")
    ]
    best_paths = sorted(
        substantive_original_entries or original_entries,
        key=lambda item: (
            -(len(item.get("classes") or []) + len(item.get("functions") or [])),
            not bool(item.get("import_probe", {}).get("ok")),
            item.get("path") or "",
        ),
    )[:10]

    status = "success" if entries else "repo_path_probe_empty"
    set_fsm_state(state, "CLASSIFY_FAILURE")
    state["last_status"] = "repo_path_probe_completed" if entries else status
    payload = {
        "tool": "repro_repo_path_probe",
        "status": status,
        "workspace_root": str(workspace_root),
        "workspace_recipe_path": str(workspace_recipe_path),
        "simulation_only": simulation_only,
        "candidate_count": len(entries),
        "candidates": entries,
        "summary": {
            "importable_modules": len(importable),
            "blocked_modules": len(blocked),
            "external_risk_count": len(risky),
            "original_candidate_count": len(original_entries),
            "best_original_code_paths": [item.get("path") for item in best_paths],
            "recommended_next_actions": aggregate_actions,
        },
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
    }
    state["last_step"] = "repro_repo_path_probe"
    state["last_repo_path_probe"] = payload
    append_history(state, "repro_repo_path_probe", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
