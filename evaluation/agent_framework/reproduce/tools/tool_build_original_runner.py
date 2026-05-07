#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    get_fsm_state,
    load_state,
    next_actions,
    repo_root_from_here,
    resolve_recipe_path,
    set_fsm_state,
    write_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _module_for_path(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    raw = path[:-3]
    if raw.endswith("/__init__"):
        raw = raw[: -len("/__init__")]
    parts = [part for part in raw.split("/") if part and part != "."]
    if not parts:
        return None
    if not all(part.replace("_", "a").isalnum() for part in parts):
        return None
    return ".".join(parts)


def _requirements(recipe: dict[str, Any]) -> dict[str, Any]:
    reqs = recipe.get("reproduction_requirements") or {}
    original = reqs.get("original_code_path") or {}
    return {
        "min_original_modules": int(original.get("min_original_modules", 1)),
        "required_modules": [str(x).strip() for x in _as_list(original.get("required_modules")) if str(x).strip()],
        "preferred_path_prefixes": [str(x).strip() for x in _as_list(original.get("preferred_path_prefixes")) if str(x).strip()],
    }


def _candidate_modules(state: dict[str, Any], recipe: dict[str, Any]) -> list[dict[str, str]]:
    req = _requirements(recipe)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for module in req["required_modules"]:
        if module not in seen:
            seen.add(module)
            out.append({"module": module, "source": "recipe_required_modules"})
    probe = state.get("last_repo_path_probe") or {}
    candidates = probe.get("candidates") or []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if item.get("path_role") == "reproduce_framework":
            continue
        module = str(item.get("module") or "").strip() or _module_for_path(str(item.get("path") or ""))
        if not module or module in seen:
            continue
        if not (item.get("classes") or item.get("functions")):
            continue
        seen.add(module)
        out.append({"module": module, "path": str(item.get("path") or ""), "source": "repo_path_probe"})
    return out[:24]


def _runner_source(candidate_modules: list[dict[str, str]], req: dict[str, Any]) -> str:
    return f'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

CANDIDATE_MODULES = {json.dumps(candidate_modules, indent=2, sort_keys=True)}
REQUIREMENTS = {json.dumps(req, indent=2, sort_keys=True)}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    workspace_root = Path(os.environ.get("REPRO_WORKSPACE_ROOT") or os.getcwd()).resolve()
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    checked = []
    imported = []
    failed = []
    files = []
    for item in CANDIDATE_MODULES:
        module = item.get("module", "")
        result = dict(item)
        result["module"] = module
        try:
            imported_module = importlib.import_module(module)
            result["ok"] = True
            result["file"] = getattr(imported_module, "__file__", None)
            if result["file"]:
                try:
                    result["relative_file"] = str(Path(result["file"]).resolve().relative_to(workspace_root))
                except Exception:
                    result["relative_file"] = str(result["file"])
                files.append(result["relative_file"])
            imported.append(module)
        except Exception as exc:
            result["ok"] = False
            result["error"] = repr(exc)
            result["traceback"] = traceback.format_exc()[-4000:]
            failed.append(result)
        checked.append(result)

    preferred_hits = []
    evidence_blob = "\\n".join(imported + files)
    for prefix in REQUIREMENTS.get("preferred_path_prefixes", []):
        if prefix and prefix in evidence_blob:
            preferred_hits.append(prefix)
    module_hits = [m for m in REQUIREMENTS.get("required_modules", []) if m in imported]
    evidence_count = len(set(preferred_hits + module_hits))
    min_modules = int(REQUIREMENTS.get("min_original_modules", 1) or 1)
    success = evidence_count >= min_modules

    metrics = {{
        "success": success,
        "figure_id": "original_code_path_probe",
        "threshold_count": 0,
        "methods_count": 0,
        "summary": {{
            "original_modules_imported": len(imported),
            "original_modules_failed": len(failed),
            "original_code_path_evidence_count": evidence_count,
        }},
        "output_files": {{}},
        "checked_modules": checked,
        "original_code_path_evidence": {{
            "required": True,
            "ok": success,
            "modules_imported": imported,
            "files_touched": files,
            "preferred_hits": preferred_hits,
            "module_hits": module_hits,
            "evidence_count": evidence_count,
            "min_original_modules": min_modules,
        }},
    }}
    write_json(Path(args.metrics_path), metrics)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _entry_script_from_recipe(recipe: dict[str, Any], workspace_root: Path) -> str | None:
    entry = recipe.get("entry") or {}
    command = entry.get("command") or []
    if not isinstance(command, list):
        return None
    for raw in command:
        value = str(raw)
        if value.endswith(".py") and "{" not in value:
            path = Path(value)
            if not path.is_absolute():
                path = workspace_root / path
            if path.exists():
                try:
                    return path.resolve().relative_to(workspace_root.resolve()).as_posix()
                except Exception:
                    return str(path.resolve())
    return None


def _metrics_wrapper_source(
    candidate_modules: list[dict[str, str]],
    req: dict[str, Any],
    harness_rel_path: str,
) -> str:
    return f'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

CANDIDATE_MODULES = {json.dumps(candidate_modules, indent=2, sort_keys=True)}
REQUIREMENTS = {json.dumps(req, indent=2, sort_keys=True)}
HARNESS_REL_PATH = {json.dumps(harness_rel_path)}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def collect_original_evidence(workspace_root: Path) -> tuple[list[dict], dict]:
    checked = []
    imported = []
    failed = []
    files = []
    for item in CANDIDATE_MODULES:
        module = item.get("module", "")
        result = dict(item)
        result["module"] = module
        try:
            imported_module = importlib.import_module(module)
            result["ok"] = True
            result["file"] = getattr(imported_module, "__file__", None)
            if result["file"]:
                try:
                    result["relative_file"] = str(Path(result["file"]).resolve().relative_to(workspace_root))
                except Exception:
                    result["relative_file"] = str(result["file"])
                files.append(result["relative_file"])
            imported.append(module)
        except Exception as exc:
            result["ok"] = False
            result["error"] = repr(exc)
            result["traceback"] = traceback.format_exc()[-4000:]
            failed.append(result)
        checked.append(result)

    preferred_hits = []
    evidence_blob = "\\n".join(imported + files)
    for prefix in REQUIREMENTS.get("preferred_path_prefixes", []):
        if prefix and prefix in evidence_blob:
            preferred_hits.append(prefix)
    module_hits = [m for m in REQUIREMENTS.get("required_modules", []) if m in imported]
    evidence_count = len(set(preferred_hits + module_hits))
    min_modules = int(REQUIREMENTS.get("min_original_modules", 1) or 1)
    evidence = {{
        "required": True,
        "ok": evidence_count >= min_modules,
        "modules_imported": imported,
        "modules_failed": [item.get("module") for item in failed],
        "files_touched": files,
        "preferred_hits": preferred_hits,
        "module_hits": module_hits,
        "evidence_count": evidence_count,
        "min_original_modules": min_modules,
    }}
    return checked, evidence


def load_harness(workspace_root: Path):
    harness_path = (workspace_root / HARNESS_REL_PATH).resolve()
    spec = importlib.util.spec_from_file_location("repro_generated_seed_harness", harness_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load harness: {{harness_path}}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    workspace_root = Path(os.environ.get("REPRO_WORKSPACE_ROOT") or os.getcwd()).resolve()
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    checked, evidence = collect_original_evidence(workspace_root)
    output_dir = Path(args.output_dir).resolve()
    harness = load_harness(workspace_root)
    if not hasattr(harness, "build_metrics"):
        raise RuntimeError("seed harness does not expose build_metrics(repo_root, output_dir)")
    metrics = harness.build_metrics(workspace_root, output_dir)
    metrics["checked_modules"] = checked
    metrics["original_code_path_evidence"] = evidence
    metrics.setdefault("summary", {{}})
    metrics["summary"]["original_code_path_evidence_count"] = evidence["evidence_count"]
    metrics["summary"]["original_modules_imported"] = len(evidence["modules_imported"])
    metrics["summary"]["original_modules_failed"] = len(evidence["modules_failed"])
    metrics["success"] = bool(metrics.get("success")) and bool(evidence.get("ok"))
    write_json(Path(args.metrics_path), metrics)
    return 0 if metrics["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, "build_original_runner")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_build_original_runner",
            action="build_original_runner",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_build_original_runner", payload)
        state["last_step"] = "repro_build_original_runner"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    recipe_path = Path(str(state.get("workspace_recipe_path") or "")).resolve()
    if not recipe_path.exists():
        recipe_path = resolve_recipe_path(args.recipe, repo_root)
    recipe = rr.load_json(recipe_path)
    workspace_root = rr.resolve_workspace_root(repo_root, str(recipe.get("workspace_root", ".")))
    req = _requirements(recipe)
    candidates = _candidate_modules(state, recipe)
    full_metric_keys = set(str(x) for x in _as_list((recipe.get("parsing") or {}).get("required_metric_keys")))
    source_entry_script = _entry_script_from_recipe(recipe, workspace_root)
    if not source_entry_script or source_entry_script.startswith(".repro_generated/"):
        source_recipe_raw = str(state.get("source_recipe_path") or "").strip()
        if source_recipe_raw and Path(source_recipe_raw).exists():
            source_recipe = rr.load_json(Path(source_recipe_raw))
            source_entry_script = _entry_script_from_recipe(source_recipe, workspace_root)
            full_metric_keys.update(
                str(x) for x in _as_list((source_recipe.get("parsing") or {}).get("required_metric_keys"))
            )

    if not candidates:
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_step"] = "repro_build_original_runner"
        state["last_status"] = "build_original_runner_failed"
        payload = {
            "tool": "repro_build_original_runner",
            "status": "failed",
            "reason": "no_original_candidate_modules",
            "fsm_state": get_fsm_state(state),
            "next_allowed_actions": next_actions(state),
        }
        append_history(state, "repro_build_original_runner", payload)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    generated_dir = workspace_root / ".repro_generated"
    runner_path = generated_dir / (
        "original_metrics_runner.py"
        if source_entry_script and {"threshold_count", "methods_count"} & full_metric_keys
        else "original_code_runner.py"
    )
    generated_dir.mkdir(parents=True, exist_ok=True)
    if source_entry_script and {"threshold_count", "methods_count"} & full_metric_keys:
        runner_path.write_text(
            _metrics_wrapper_source(candidates, req, source_entry_script),
            encoding="utf-8",
        )
    else:
        runner_path.write_text(_runner_source(candidates, req), encoding="utf-8")
    runner_path.chmod(0o755)

    rel_runner = runner_path.relative_to(workspace_root).as_posix()
    recipe["entry"] = {
        "kind": "command",
        "command": [
            "{python_executable}",
            rel_runner,
            "--metrics-path",
            "{metrics_path}",
            "--output-dir",
            "{run_dir}/figures",
        ],
        "timeout_seconds": int((recipe.get("entry") or {}).get("timeout_seconds", 300)),
    }
    parsing = recipe.setdefault("parsing", {})
    parsing["kind"] = "json_file"
    parsing["path"] = "{metrics_path}"
    if source_entry_script and {"threshold_count", "methods_count"} & full_metric_keys:
        parsing["required_metric_keys"] = [
            "success",
            "figure_id",
            "threshold_count",
            "methods_count",
            "summary",
            "output_files",
            "checked_modules",
            "original_code_path_evidence",
        ]
    else:
        parsing["required_metric_keys"] = ["success", "figure_id", "summary", "output_files", "checked_modules"]
    verification = recipe.setdefault("verification", {})
    verification["enabled"] = False
    success_criteria = recipe.setdefault("success_criteria", {})
    success_criteria["require_verification_success"] = False
    rr.write_json(recipe_path, recipe)

    set_fsm_state(state, "APPLY_FIX")
    state["last_step"] = "repro_build_original_runner"
    state["last_status"] = "build_original_runner_applied"
    state["original_runner_path"] = str(runner_path)
    state["workspace_recipe_path"] = str(recipe_path)
    payload = {
        "tool": "repro_build_original_runner",
        "status": "success",
        "runner_path": str(runner_path),
        "workspace_recipe_path": str(recipe_path),
        "candidate_modules": candidates,
        "entry_command": recipe["entry"]["command"],
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
    }
    append_history(state, "repro_build_original_runner", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
