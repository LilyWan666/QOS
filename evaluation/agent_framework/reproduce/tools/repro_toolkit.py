#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

import run_reproduce as rr
import state_machine as sm

WORKSPACE_IGNORE_NAMES = {
    ".git",
    "temp",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "target",
    "build",
    "dist",
    ".venv",
    "venv",
    ".tox",
    "node_modules",
}


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[4]


def ensure_run_root(recipe_name: str, state: dict[str, Any], output_root: Path) -> Path:
    existing = state.get("run_root")
    if isinstance(existing, str) and existing:
        return Path(existing).resolve()
    run_root = output_root / f"{recipe_name}_toolchain_{rr.utc_stamp()}"
    run_root.mkdir(parents=True, exist_ok=True)
    state["run_root"] = str(run_root)
    return run_root


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_recipe_path(raw_recipe: str, repo_root: Path) -> Path:
    recipe_path = Path(raw_recipe)
    if recipe_path.is_absolute():
        return recipe_path
    return (repo_root / recipe_path).resolve()


def _copy_isolated_workspace(source_repo_root: Path, target_repo_root: Path) -> None:
    if target_repo_root.exists():
        shutil.rmtree(target_repo_root)
    target_repo_root.mkdir(parents=True, exist_ok=True)

    source_root = source_repo_root.resolve()
    for root, dirs, files in os.walk(source_root, topdown=True):
        root_path = Path(root)
        is_source_root = root_path == source_root

        filtered_dirs: list[str] = []
        for dirname in dirs:
            if dirname in WORKSPACE_IGNORE_NAMES:
                continue
            if is_source_root and dirname == ".git":
                continue
            filtered_dirs.append(dirname)
        dirs[:] = filtered_dirs

        rel = root_path.relative_to(source_root)
        dst_root = (target_repo_root / rel).resolve()
        dst_root.mkdir(parents=True, exist_ok=True)
        os.chmod(dst_root, dst_root.stat().st_mode | 0o700)

        for filename in files:
            if filename in WORKSPACE_IGNORE_NAMES:
                continue
            src_file = root_path / filename
            dst_file = dst_root / filename
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_file, dst_file)


def _resolve_isolated_workspace_root(
    *,
    repo_root: Path,
    isolated_repo_root: Path,
    raw_workspace_root: str,
) -> Path:
    source_workspace_root = rr.resolve_workspace_root(repo_root, raw_workspace_root)
    try:
        rel = source_workspace_root.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return source_workspace_root
    return (isolated_repo_root / rel).resolve()


def ensure_isolated_workspace_recipe(
    *,
    state: dict[str, Any],
    run_root: Path,
    repo_root: Path,
    recipe_path: Path,
    recipe: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    existing = str(state.get("workspace_recipe_path", "")).strip()
    if existing:
        existing_path = Path(existing).resolve()
        if existing_path.exists():
            return existing_path, rr.load_json(existing_path)

    workspace_repo_root = run_root / "workspace"
    _copy_isolated_workspace(repo_root, workspace_repo_root)

    workspace_recipe = dict(recipe)
    workspace_root = _resolve_isolated_workspace_root(
        repo_root=repo_root,
        isolated_repo_root=workspace_repo_root,
        raw_workspace_root=str(recipe.get("workspace_root", ".")),
    )
    workspace_recipe["workspace_root"] = str(workspace_root)

    workspace_recipe_path = run_root / "recipe.generated.snapshot.json"
    rr.write_json(workspace_recipe_path, workspace_recipe)
    state["workspace_repo_root"] = str(workspace_repo_root)
    state["workspace_recipe_path"] = str(workspace_recipe_path)
    state["source_recipe_path"] = str(recipe_path)
    state["workspace_root"] = str(workspace_root)
    return workspace_recipe_path, workspace_recipe


def append_history(state: dict[str, Any], step: str, payload: dict[str, Any]) -> None:
    history = state.setdefault("history", [])
    if not isinstance(history, list):
        history = []
        state["history"] = history
    history.append({"step": step, "payload": payload})


def next_attempt_ordinal(state: dict[str, Any]) -> int:
    ordinal = int(state.get("attempt_ordinal", 0) or 0) + 1
    state["attempt_ordinal"] = ordinal
    return ordinal


def ensure_fix_budget(state: dict[str, Any], recipe: dict[str, Any], default: int = 2) -> int:
    current = state.get("fix_budget")
    if isinstance(current, int):
        return current
    configured = recipe.get("recovery", {}).get("fix_budget")
    if isinstance(configured, int):
        state["fix_budget"] = configured
        return configured
    state["fix_budget"] = default
    return default


def get_fsm_state(state: dict[str, Any]) -> str:
    raw = state.get("fsm_state")
    if isinstance(raw, str) and raw:
        return raw
    return sm.INIT


def set_fsm_state(state: dict[str, Any], fsm_state: str) -> None:
    state["fsm_state"] = fsm_state


def _recoverable_from_state(state: dict[str, Any]) -> bool:
    classification = (state.get("last_classification") or {}).get("recoverable")
    if isinstance(classification, bool):
        return classification
    attempt_payload = state.get("last_attempt_payload") or {}
    diagnosis = attempt_payload.get("diagnosis_payload") or {}
    cls = diagnosis.get("classification") or {}
    return bool(cls.get("recoverable", True))


def _last_failure_category(state: dict[str, Any]) -> str:
    attempt_payload = state.get("last_attempt_payload") or {}
    diagnosis = attempt_payload.get("diagnosis_payload") or {}
    classification = diagnosis.get("classification") or {}
    raw = classification.get("category")
    return str(raw or "").strip()


def _last_failure_evidence_text(state: dict[str, Any]) -> str:
    attempt_payload = state.get("last_attempt_payload") or {}
    diagnosis = attempt_payload.get("diagnosis_payload") or {}
    classification = diagnosis.get("classification") or {}
    evidence = classification.get("evidence") or []
    try:
        return json.dumps(evidence, sort_keys=True)
    except TypeError:
        return str(evidence)


def _simulation_only_recipe_enabled(state: dict[str, Any]) -> bool:
    for key in ("workspace_recipe_path", "source_recipe_path", "recipe_path"):
        raw = str(state.get(key, "")).strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.exists():
            continue
        try:
            recipe = rr.load_json(path)
        except Exception:
            continue
        recovery = recipe.get("recovery") or {}
        if bool(recovery.get("simulation_only")):
            return True
    return False


def next_actions(state: dict[str, Any], *, tests_required: bool = False) -> list[str]:
    fsm_state = get_fsm_state(state)
    fix_budget = int(state.get("fix_budget", 1) or 0)
    recoverable = _recoverable_from_state(state)
    actions = sm.next_allowed_actions(
        fsm_state,
        recoverable=recoverable,
        fix_budget=fix_budget,
        tests_required=tests_required,
    )
    # When run_once fails on source-compat missing-module signatures, ensure
    # the next decision can apply a source-path repair before another run_once.
    if fsm_state == sm.RUN_FAILED:
        category = _last_failure_category(state)
        evidence_text = _last_failure_evidence_text(state)
        if category == "original_code_path_not_exercised":
            prioritized = []
            for candidate in ("repo_path_probe", "build_original_runner", "classify_failure"):
                if candidate in actions and candidate not in prioritized:
                    prioritized.append(candidate)
            for action in actions:
                if action not in prioritized:
                    prioritized.append(action)
            return prioritized
        if _simulation_only_recipe_enabled(state) and (
            category == "runtime_import_surface_mismatch"
            or "qiskit_ibm_runtime" in evidence_text
            or "IBMProvider" in evidence_text
            or "ibm_token" in evidence_text
        ):
            prioritized = []
            if "simulation_backend_fix" not in actions:
                prioritized.append("simulation_backend_fix")
            for candidate in ("classify_failure", "runtime_trace_fix", "source_path_fix", "env_fix"):
                if candidate in actions and candidate not in prioritized:
                    prioritized.append(candidate)
            for action in actions:
                if action not in prioritized:
                    prioritized.append(action)
            return prioritized
        if category in {
            "source_compat_missing_module_file",
            "source_compat_missing_symbol_import",
            "source_compat_missing_symbol_reference",
        }:
            prioritized: list[str] = []
            if "source_path_fix" not in actions:
                prioritized.append("source_path_fix")
            for candidate in ("classify_failure", "source_fix", "constraint_resolve", "env_fix"):
                if candidate in actions and candidate not in prioritized:
                    prioritized.append(candidate)
            for action in actions:
                if action not in prioritized:
                    prioritized.append(action)
            return prioritized
    return actions


def assert_action_allowed(
    state: dict[str, Any],
    action: str,
    *,
    tests_required: bool = False,
) -> tuple[bool, list[str]]:
    allowed = next_actions(state, tests_required=tests_required)
    return (action in allowed, allowed)


def blocked_action_payload(
    *,
    tool: str,
    action: str,
    state: dict[str, Any],
    next_allowed_actions: list[str],
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "tool": tool,
        "status": "blocked",
        "reason": reason or f"action {action} is not allowed in current state",
        "action": action,
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_allowed_actions,
    }


def _runtime_selector_candidate_dir(path: Path) -> Path | None:
    parts = path.absolute().parts
    if "runtime_python_selector" not in parts:
        return None
    idx = parts.index("runtime_python_selector")
    if idx + 1 >= len(parts):
        return None
    candidate = parts[idx + 1]
    if not candidate.startswith("cand_"):
        return None
    return Path(*parts[: idx + 2])


def _stable_runtime_venv_dir(*, state: dict[str, Any], repo_root: Path) -> Path:
    run_root = str(state.get("run_root", "")).strip()
    run_name = Path(run_root).name if run_root else "runtime_default"
    return (repo_root / "temp/agent_framework/reproduce/runtime_venvs" / run_name).resolve()


def ensure_runtime_env_selected(
    *,
    state: dict[str, Any],
    repo_root: Path,
    bootstrap_python_executable: str,
) -> tuple[Path, Path]:
    runtime_python_raw = str(state.get("runtime_python_executable", "")).strip()
    if runtime_python_raw:
        runtime_python = Path(runtime_python_raw)
        # runtime_python_select creates disposable candidate venvs for probing.
        # Follow-up repair tools mutate the environment, so they must use the
        # stable per-run runtime venv instead of installing into a probe slot.
        if runtime_python.exists() and _runtime_selector_candidate_dir(runtime_python) is None:
            runtime_venv_dir = runtime_python.parent.parent
            state["runtime_venv_dir"] = str(runtime_venv_dir)
            state["runtime_python_executable"] = str(runtime_python)
            return runtime_venv_dir, runtime_python

    existing = str(state.get("runtime_venv_dir", "")).strip()
    if existing and _runtime_selector_candidate_dir(Path(existing)) is None:
        venv_dir = Path(existing).resolve()
    else:
        venv_dir = _stable_runtime_venv_dir(state=state, repo_root=repo_root)
        state["runtime_venv_dir"] = str(venv_dir)

    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [bootstrap_python_executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "failed to create runtime venv: "
                f"python={bootstrap_python_executable} venv_dir={venv_dir} "
                f"stderr={proc.stderr[-500:]}"
            )
    state["runtime_venv_dir"] = str(venv_dir)
    state["runtime_python_executable"] = str(venv_python)
    return venv_dir, venv_python
