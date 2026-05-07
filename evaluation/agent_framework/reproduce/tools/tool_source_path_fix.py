#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

MISSING_MODULE_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
CONFIG_MODULE_KEYWORDS = {"token", "credential", "credentials", "secret", "secrets", "key", "keys"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_missing_modules(state: dict[str, Any]) -> list[str]:
    artifacts = ((state.get("last_manifest") or {}).get("artifacts")) or {}
    metrics_raw = artifacts.get("metrics")
    if not metrics_raw:
        return []
    metrics_path = Path(str(metrics_raw)).resolve()
    if not metrics_path.exists():
        return []
    try:
        payload = _load_json(metrics_path)
    except Exception:
        return []
    missing: list[str] = []
    for item in payload.get("checked_modules", []):
        if not isinstance(item, dict):
            continue
        err = str(item.get("error", ""))
        for module in MISSING_MODULE_RE.findall(err):
            if module and module not in missing:
                missing.append(module)
    return missing


def _collect_missing_module_candidates(state: dict[str, Any]) -> dict[str, list[str]]:
    attempt_payload = state.get("last_attempt_payload") or {}
    diagnosis = attempt_payload.get("diagnosis_payload") or {}
    classification = diagnosis.get("classification") or {}
    evidence = classification.get("evidence") or []
    if not isinstance(evidence, list):
        return {}
    out: dict[str, list[str]] = {}
    for item in evidence:
        if not isinstance(item, dict):
            continue
        missing_module = str(item.get("missing_module", "")).strip()
        candidates = item.get("module_candidates")
        if not missing_module or not isinstance(candidates, list):
            continue
        cleaned = [str(path).strip() for path in candidates if str(path).strip()]
        if cleaned:
            out[missing_module] = cleaned
    return out


def _path_to_module(workspace_root: Path, file_path: Path) -> str | None:
    try:
        rel = file_path.resolve().relative_to(workspace_root.resolve())
    except Exception:
        return None
    if rel.suffix != ".py":
        return None
    parts = list(rel.parts)
    if not parts:
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = Path(parts[-1]).stem
    if not parts:
        return None
    return ".".join(parts)


def _module_exists(workspace_root: Path, module: str) -> bool:
    module_path = Path(*module.split("."))
    package_init = workspace_root / module_path / "__init__.py"
    module_file = workspace_root / module_path.with_suffix(".py")
    return package_init.exists() or module_file.exists()


def _common_prefix_len(left: list[str], right: list[str]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _discover_module_candidates(workspace_root: Path, missing_module: str) -> list[str]:
    """Find same-basename modules near the missing module namespace."""
    parts = [part for part in missing_module.split(".") if part]
    if len(parts) < 2:
        return []
    namespace = parts[0]
    basename = parts[-1]
    namespace_root = workspace_root / namespace
    if not namespace_root.is_dir():
        return []

    candidates: list[tuple[int, str]] = []
    for path in namespace_root.rglob(f"{basename}.py"):
        if not path.is_file():
            continue
        module = _path_to_module(workspace_root, path)
        if not module or module == missing_module:
            continue
        module_parts = module.split(".")
        score = 0
        if module_parts[0] == namespace:
            score += 100
        if module_parts[-1] == basename:
            score += 50
        score += _common_prefix_len(parts, module_parts) * 10
        score -= abs(len(module_parts) - len(parts))
        candidates.append((score, module))

    for path in namespace_root.rglob("__init__.py"):
        if path.parent.name != basename:
            continue
        module = _path_to_module(workspace_root, path)
        if not module or module == missing_module:
            continue
        module_parts = module.split(".")
        score = 90 + _common_prefix_len(parts, module_parts) * 10
        candidates.append((score, module))

    ordered: list[str] = []
    for _, module in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if module not in ordered:
            ordered.append(module)
    return ordered


def _select_module_remap(
    *,
    workspace_root: Path,
    missing_modules: set[str],
    missing_candidates: dict[str, list[str]],
) -> dict[str, str]:
    remap: dict[str, str] = {}
    for missing_module, candidates in missing_candidates.items():
        if not missing_module:
            continue
        for candidate in candidates:
            candidate_path = Path(candidate)
            if not candidate_path.is_absolute():
                candidate_path = (workspace_root / candidate_path).resolve()
            if not candidate_path.exists():
                continue
            module_name = _path_to_module(workspace_root, candidate_path)
            if not module_name or module_name == missing_module:
                continue
            remap[missing_module] = module_name
            break

    legacy_defaults = {
        "qos.backends.types": "qos.types.types",
        "qos.dag": "qvm.compiler.dag",
        "qvm.qvm": "qvm",
    }
    for missing_module in missing_modules:
        if missing_module in remap:
            continue
        default = legacy_defaults.get(missing_module)
        if default and _module_exists(workspace_root, default):
            remap[missing_module] = default
            continue
        if missing_module.startswith("qvm.qvm."):
            candidate = "qvm." + missing_module[len("qvm.qvm.") :]
            if _module_exists(workspace_root, candidate):
                remap[missing_module] = candidate
                continue
        discovered = _discover_module_candidates(workspace_root, missing_module)
        if discovered:
            remap[missing_module] = discovered[0]
    return remap


def _rewrite_text(
    content: str,
    *,
    module_remap: dict[str, str],
    current_module: str | None,
) -> str:
    rewritten = content
    for old, new in sorted(module_remap.items(), key=lambda item: len(item[0]), reverse=True):
        # Only skip when rewriting would point a module path back to itself.
        # Do not skip entire package trees (e.g., new="qvm"), otherwise
        # qvm.* files never get legacy imports rewritten.
        if current_module and (current_module == new or current_module == old):
            continue
        escaped_old = re.escape(old)
        rewritten = re.sub(rf"(\bfrom\s+){escaped_old}(\s+import\b)", rf"\1{new}\2", rewritten)
        rewritten = re.sub(rf"(\bfrom\s+){escaped_old}(\.)", rf"\1{new}\2", rewritten)
        rewritten = re.sub(rf"(\bimport\s+){escaped_old}(\b)", rf"\1{new}\2", rewritten)
        rewritten = re.sub(rf"(\bimport\s+){escaped_old}(\.)", rf"\1{new}\2", rewritten)
    return rewritten


def _rewrite_workspace_imports(
    *,
    workspace_root: Path,
    module_remap: dict[str, str],
) -> list[Path]:
    changed_files: list[Path] = []
    if not module_remap:
        return changed_files
    for path in workspace_root.rglob("*.py"):
        if not path.is_file():
            continue
        current_module = _path_to_module(workspace_root, path)
        try:
            original = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if not any(source in original for source in module_remap):
            continue
        rewritten = _rewrite_text(
            original,
            module_remap=module_remap,
            current_module=current_module,
        )
        if rewritten == original:
            continue
        path.write_text(rewritten, encoding="utf-8")
        changed_files.append(path.resolve())
    return changed_files


def _ensure_module_alias_files(
    *,
    workspace_root: Path,
    module_remap: dict[str, str],
) -> list[Path]:
    """Create missing legacy module files that re-export from new module paths."""
    created: list[Path] = []
    for old, new in sorted(module_remap.items(), key=lambda item: len(item[0]), reverse=True):
        old_parts = old.split(".")
        if not old_parts:
            continue
        old_module_file = workspace_root / Path(*old_parts).with_suffix(".py")
        if old_module_file.exists():
            continue

        new_parts = new.split(".")
        new_module_file = workspace_root / Path(*new_parts).with_suffix(".py")
        new_pkg_init = workspace_root / Path(*new_parts) / "__init__.py"
        if not (new_module_file.exists() or new_pkg_init.exists()):
            continue

        old_module_file.parent.mkdir(parents=True, exist_ok=True)
        alias_body = (
            f'"""Auto-generated compatibility alias for `{old}`."""\n'
            f"from {new} import *  # noqa: F401,F403\n"
        )
        old_module_file.write_text(alias_body, encoding="utf-8")
        created.append(old_module_file.resolve())
    return created


def _is_config_like_missing_module(module: str) -> bool:
    parts = [part.lower() for part in module.split(".") if part]
    tokens: set[str] = set()
    for part in parts:
        tokens.add(part)
        tokens.update(piece for piece in re.split(r"[_-]+", part) if piece)
    return bool(tokens & CONFIG_MODULE_KEYWORDS)


def _target_file_for_missing_module(
    *,
    workspace_root: Path,
    missing_module: str,
    candidates: list[str],
) -> Path:
    for candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = (workspace_root / candidate_path).resolve()
        try:
            candidate_path.relative_to(workspace_root.resolve())
        except Exception:
            continue
        if candidate_path.suffix == ".py":
            return candidate_path
    return workspace_root / Path(*missing_module.split(".")).with_suffix(".py")


def _ensure_parent_packages(workspace_root: Path, module_file: Path) -> list[Path]:
    created: list[Path] = []
    try:
        rel_parent = module_file.parent.resolve().relative_to(workspace_root.resolve())
    except Exception:
        return created
    current = workspace_root
    for part in rel_parent.parts:
        current = current / part
        init_file = current / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Workspace package for reproduce-run fixes."""\n', encoding="utf-8")
            created.append(init_file.resolve())
    return created


def _config_constant_name(module: str) -> str:
    basename = module.split(".")[-1]
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", basename).strip("_").upper()
    return cleaned or "REPRO_CONFIG_VALUE"


def _env_fallbacks_for_config_module(module: str, constant: str) -> list[str]:
    names = [constant]
    if "ibm" in module.lower() and "token" in module.lower():
        names.extend(["QISKIT_IBM_TOKEN", "IBM_QUANTUM_TOKEN"])
    names.append(f"REPRO_{constant}")
    out: list[str] = []
    for name in names:
        if name and name not in out:
            out.append(name)
    return out


def _ensure_config_placeholder_modules(
    *,
    workspace_root: Path,
    missing_candidates: dict[str, list[str]],
) -> list[Path]:
    """Create env-backed local config modules for missing credentials only."""
    created: list[Path] = []
    for missing_module, candidates in missing_candidates.items():
        if not _is_config_like_missing_module(missing_module):
            continue
        target = _target_file_for_missing_module(
            workspace_root=workspace_root,
            missing_module=missing_module,
            candidates=candidates,
        )
        try:
            target.resolve().relative_to(workspace_root.resolve())
        except Exception:
            continue
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        created.extend(_ensure_parent_packages(workspace_root, target))
        constant = _config_constant_name(missing_module)
        env_expr = " or ".join(
            f'os.environ.get("{name}")' for name in _env_fallbacks_for_config_module(missing_module, constant)
        )
        body = (
            '"""Workspace-local credential config for reproduce runs.\n\n'
            "The real project source is left unchanged; values are read from\n"
            "environment variables when available.\n"
            '"""\n'
            "import os\n\n"
            f"{constant} = {env_expr} or \"\"\n"
        )
        target.write_text(body, encoding="utf-8")
        created.append(target.resolve())
    return created


def _apply_qos_backends_types_compat(workspace_root: Path) -> list[Path]:
    """Patch legacy qos.backends.types import in isolated workspace only."""
    edited: list[Path] = []
    target = workspace_root / "qos" / "types" / "types.py"
    if not target.is_file():
        return []
    try:
        original = target.read_text(encoding="utf-8")
    except Exception:
        return []

    changed = False
    rewritten = original
    legacy_line = "from qos.backends.types import QPU"
    if legacy_line in rewritten:
        rewritten = rewritten.replace(legacy_line + "\n", "")
        changed = True

    if "class QPU:" not in rewritten:
        anchor = "class Transformations:"
        qpu_block = (
            "\n\nclass QPU:\n"
            "    \"\"\"Shared backend descriptor used across qos modules.\"\"\"\n\n"
            "    def __init__(self) -> None:\n"
            "        self.id: int = -1\n"
            "        self.provider: str = \"\"\n"
            "        self.name: str = \"\"\n"
            "        self.alias: str = \"\"\n"
            "        self.args: Dict[str, Any] = {}\n"
            "        self.local_queue: List[tuple] = []\n"
        )
        if anchor in rewritten:
            rewritten = rewritten.replace(anchor, qpu_block + "\n\n" + anchor, 1)
            changed = True

    if not changed or rewritten == original:
        pass
    else:
        target.write_text(rewritten, encoding="utf-8")
        edited.append(target.resolve())

    init_target = workspace_root / "qos" / "types" / "__init__.py"
    init_body = "from .types import *  # noqa: F401,F403\n"
    if not init_target.exists() or init_target.read_text(encoding="utf-8") != init_body:
        init_target.write_text(init_body, encoding="utf-8")
        edited.append(init_target.resolve())
    return edited


def _validate_python_files(python_executable: str, files: list[Path], cwd: Path) -> dict[str, Any]:
    if not files:
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}
    cmd = [python_executable, "-m", "py_compile", *[str(path) for path in files]]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "command": cmd,
    }


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)

    is_allowed, allowed_actions = assert_action_allowed(state, "source_path_fix")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_source_path_fix",
            action="source_path_fix",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_source_path_fix", payload)
        state["last_step"] = "repro_source_path_fix"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    recipe_path = resolve_recipe_path(args.recipe, repo_root)
    recipe = rr.load_json(recipe_path)
    run_root = ensure_run_root(
        recipe["name"],
        state,
        (repo_root / "temp/agent_framework/reproduce/tools/runs").resolve(),
    )
    recipe_path, recipe = ensure_isolated_workspace_recipe(
        state=state,
        run_root=run_root,
        repo_root=repo_root,
        recipe_path=recipe_path,
        recipe=recipe,
    )
    workspace_root = rr.resolve_workspace_root(repo_root, recipe.get("workspace_root", "."))
    missing_modules = _collect_missing_modules(state)
    missing_candidates = _collect_missing_module_candidates(state)
    if not missing_modules and not missing_candidates:
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = "source_path_fix_skipped"
        payload = {
            "tool": "repro_source_path_fix",
            "status": "skipped",
            "reason": "no src-style missing-module signal in latest metrics",
            "missing_modules": missing_modules,
            "fsm_state": get_fsm_state(state),
            "next_allowed_actions": next_actions(state),
        }
        state["last_step"] = "repro_source_path_fix"
        append_history(state, "repro_source_path_fix", payload)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    rewrite_targets = set(missing_modules) | set(missing_candidates.keys())
    module_remap = _select_module_remap(
        workspace_root=workspace_root,
        missing_modules=rewrite_targets,
        missing_candidates=missing_candidates,
    )
    config_paths = _ensure_config_placeholder_modules(
        workspace_root=workspace_root,
        missing_candidates=missing_candidates,
    )
    if not module_remap:
        if config_paths:
            validation = _validate_python_files(args.python_executable, config_paths, workspace_root)
            applied = bool(validation.get("ok"))
            if applied:
                set_fsm_state(state, "APPLY_FIX")
                state["last_status"] = "source_path_fix_applied"
                status = "success"
                rc = 0
            else:
                set_fsm_state(state, "CLASSIFY_FAILURE")
                state["last_status"] = "source_path_fix_failed"
                status = "failed"
                rc = 1
            payload = {
                "tool": "repro_source_path_fix",
                "status": status,
                "reason": "created env-backed config module for missing local credential",
                "missing_modules": missing_modules,
                "missing_candidates": missing_candidates,
                "module_remap": module_remap,
                "edits": [str(path) for path in config_paths],
                "validate": validation,
                "fsm_state": get_fsm_state(state),
                "next_allowed_actions": next_actions(state),
            }
            state["last_step"] = "repro_source_path_fix"
            append_history(state, "repro_source_path_fix", payload)
            write_state(state_path, state)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return rc
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = "source_path_fix_skipped"
        payload = {
            "tool": "repro_source_path_fix",
            "status": "skipped",
            "reason": "no valid module remap candidates resolved to existing modules",
            "missing_modules": missing_modules,
            "missing_candidates": missing_candidates,
            "fsm_state": get_fsm_state(state),
            "next_allowed_actions": next_actions(state),
        }
        state["last_step"] = "repro_source_path_fix"
        append_history(state, "repro_source_path_fix", payload)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    rewritten_paths = _rewrite_workspace_imports(
        workspace_root=workspace_root,
        module_remap=module_remap,
    )
    alias_paths = _ensure_module_alias_files(
        workspace_root=workspace_root,
        module_remap=module_remap,
    )
    compat_paths = _apply_qos_backends_types_compat(workspace_root)
    edited_files: list[str] = []
    touched_python: list[Path] = []
    edited_files.extend(str(path) for path in rewritten_paths)
    touched_python.extend(rewritten_paths)
    edited_files.extend(str(path) for path in alias_paths)
    touched_python.extend(alias_paths)
    edited_files.extend(str(path) for path in config_paths)
    touched_python.extend(config_paths)
    edited_files.extend(str(path) for path in compat_paths)
    touched_python.extend(compat_paths)

    validation = _validate_python_files(args.python_executable, touched_python, workspace_root)
    applied = bool(edited_files) and bool(validation.get("ok"))
    if applied:
        set_fsm_state(state, "APPLY_FIX")
        state["last_status"] = "source_path_fix_applied"
        status = "success"
        rc = 0
    else:
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = "source_path_fix_failed"
        status = "failed"
        rc = 1

    payload = {
        "tool": "repro_source_path_fix",
        "status": status,
        "missing_modules": missing_modules,
        "module_remap": module_remap,
        "edits": edited_files,
        "validate": validation,
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
    }
    state["last_step"] = "repro_source_path_fix"
    append_history(state, "repro_source_path_fix", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
