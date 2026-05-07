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

from repro_toolkit import (  # noqa: E402
    assert_action_allowed,
    append_history,
    blocked_action_payload,
    ensure_runtime_env_selected,
    get_fsm_state,
    load_state,
    next_actions,
    repo_root_from_here,
    set_fsm_state,
    write_state,
)
from runtime_profiles import (  # noqa: E402
    RUNTIME_PROFILES,
    apply_runtime_profile,
    detect_profile_for_modules,
)

MISSING_MODULE_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
LOCAL_MODULE_SKIP = {"src", "test", "tests", "qos", "qvm", "Baseline_Multiprogramming"}
PIP_PACKAGE_CANDIDATES: dict[str, list[str]] = {
    "qiskit_ibm_runtime": ["qiskit-ibm-runtime"],
    "qiskit_ibm_provider": ["qiskit-ibm-provider"],
    "qiskit_algorithms": ["qiskit-algorithms==0.3.1", "qiskit-algorithms"],
    "qiskit_optimization": ["qiskit-optimization==0.6.1", "qiskit-optimization"],
    "qiskit_aer": ["qiskit-aer==0.14.2", "qiskit-aer"],
    "pkg_resources": ["setuptools<81"],
    "pymoo": ["pymoo==0.6.1.3", "pymoo==0.6.0.1", "pymoo"],
    "yaml": ["PyYAML"],
    "jsonpickle": ["jsonpickle"],
    "mapomatic": ["mapomatic==0.10.0", "mapomatic==0.9.0", "mapomatic"],
    "mqt": ["mqt.predictor==1.2.2", "mqt.predictor", "mqt.qmap", "mqt"],
}
LEGACY_QISKIT_STACK = ["qiskit==0.46.3", "qiskit-ibm-provider==0.10.0", "qiskit-aer==0.14.2"]
LEGACY_QISKIT_TRIGGER_MODULES = {
    "qiskit",
    "qiskit_aer",
    "qiskit_ibm_provider",
    "qiskit_ibm_runtime",
    "mapomatic",
    "mqt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_missing_modules_from_state(state: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    artifacts = ((state.get("last_manifest") or {}).get("artifacts")) or {}
    metrics_raw = artifacts.get("metrics")
    if metrics_raw:
        metrics_path = Path(str(metrics_raw)).resolve()
        if metrics_path.exists():
            try:
                payload = _load_json(metrics_path)
            except Exception:
                payload = {}
            for item in payload.get("checked_modules", []):
                if not isinstance(item, dict):
                    continue
                err = str(item.get("error", ""))
                for module in MISSING_MODULE_RE.findall(err):
                    if module and module not in missing:
                        missing.append(module)
    attempt_payload = state.get("last_attempt_payload") or {}
    diagnosis = attempt_payload.get("diagnosis_payload") or {}
    evidence = ((diagnosis.get("classification") or {}).get("evidence")) or []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        module = str(item.get("missing_module", "")).strip()
        if module and module not in missing:
            missing.append(module)
        for raw in item.get("missing_imports", []) or []:
            mod = str(raw or "").strip()
            if mod and mod not in missing:
                missing.append(mod)
    probe = state.get("last_repo_path_probe") or {}
    for candidate in probe.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("path_role") == "reproduce_framework":
            continue
        actions = candidate.get("recommended_next_actions") or []
        if "dependency_runtime_fix" not in actions:
            continue
        import_probe = candidate.get("import_probe") or {}
        module = str(import_probe.get("missing_module", "")).strip()
        if module and module not in missing:
            missing.append(module)
    return missing


def _module_to_package_candidates(module: str) -> list[str]:
    root = module.split(".", 1)[0].strip()
    if not root:
        return []
    mapped = PIP_PACKAGE_CANDIDATES.get(root, [root])
    out: list[str] = []
    for name in mapped:
        if name not in out:
            out.append(name)
    return out


def _workspace_has_module_root(state: dict[str, Any], module_root: str) -> bool:
    raw_workspace = str(state.get("workspace_root") or "").strip()
    if not raw_workspace or not module_root:
        return False
    workspace_root = Path(raw_workspace).resolve()
    if not workspace_root.exists():
        return False
    return (workspace_root / module_root).is_dir() or (workspace_root / f"{module_root}.py").is_file()


def _pip_install(python_executable: str, requirement: str) -> dict[str, Any]:
    cmd = [python_executable, "-m", "pip", "install", requirement]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "requirement": requirement,
        "command": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def _read_qiskit_major(python_executable: str) -> int | None:
    cmd = [
        python_executable,
        "-c",
        (
            "from importlib import metadata as m; "
            "v=m.version('qiskit'); "
            "print(int(str(v).split('.',1)[0]))"
        ),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    if not text.isdigit():
        return None
    return int(text)


def _enforce_legacy_qiskit_stack_if_needed(
    python_executable: str,
    runtime_modules: list[str],
    state: dict[str, Any],
) -> dict[str, Any]:
    last_status = str(state.get("last_status", ""))
    should_enforce = bool(set(runtime_modules) & LEGACY_QISKIT_TRIGGER_MODULES) or (
        "qiskit" in last_status
    )
    if not should_enforce:
        return {"attempted": False, "ok": True, "reason": "not_required", "install_results": []}

    major = _read_qiskit_major(python_executable)
    if major is not None and major < 1:
        return {
            "attempted": False,
            "ok": True,
            "reason": "already_legacy_compatible",
            "qiskit_major": major,
            "install_results": [],
        }

    installs: list[dict[str, Any]] = []
    ok = True
    for requirement in LEGACY_QISKIT_STACK:
        res = _pip_install(python_executable, requirement)
        res["module"] = "__qiskit_legacy_stack__"
        installs.append(res)
        ok = ok and bool(res.get("ok"))

    final_major = _read_qiskit_major(python_executable)
    if final_major is None or final_major >= 1:
        ok = False
    return {
        "attempted": True,
        "ok": ok,
        "reason": "enforced",
        "qiskit_major_before": major,
        "qiskit_major_after": final_major,
        "install_results": installs,
    }


def _verify_import(python_executable: str, module_name: str) -> dict[str, Any]:
    cmd = [python_executable, "-c", f"import {module_name}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "module": module_name,
        "command": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout[-1000:],
        "stderr": proc.stderr[-2000:],
    }


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)

    is_allowed, allowed_actions = assert_action_allowed(state, "dependency_runtime_fix")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_dependency_runtime_fix",
            action="dependency_runtime_fix",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_dependency_runtime_fix", payload)
        state["last_step"] = "repro_dependency_runtime_fix"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    missing_modules = _collect_missing_modules_from_state(state)
    runtime_modules: list[str] = []
    local_workspace_modules: list[str] = []
    for module in missing_modules:
        root = module.split(".", 1)[0].strip()
        if not root:
            continue
        if root in LOCAL_MODULE_SKIP or _workspace_has_module_root(state, root):
            if module not in local_workspace_modules:
                local_workspace_modules.append(module)
            continue
        if root not in runtime_modules:
            runtime_modules.append(root)

    venv_dir, venv_python = ensure_runtime_env_selected(
        state=state,
        repo_root=repo_root,
        bootstrap_python_executable=args.python_executable,
    )
    profile_name = detect_profile_for_modules(runtime_modules)
    profile_result: dict[str, Any] | None = None
    profile_handled_modules: set[str] = set()
    if profile_name:
        triggers = (((RUNTIME_PROFILES.get(profile_name) or {}).get("triggers") or {}).get("modules")) or []
        profile_handled_modules = {str(t).strip() for t in triggers if str(t).strip()}
        profile_result = apply_runtime_profile(
            str(venv_python),
            profile_name,
        )
    install_results: list[dict[str, Any]] = []
    import_verify_results: list[dict[str, Any]] = []
    resolved: list[str] = []
    unresolved: list[dict[str, str]] = []
    verify_failures = state.setdefault("runtime_import_verify_failures", {})
    if not isinstance(verify_failures, dict):
        verify_failures = {}
        state["runtime_import_verify_failures"] = verify_failures
    blocked_repeated_modules: list[str] = []
    for module in runtime_modules:
        root = module.split(".", 1)[0].strip()
        if module in profile_handled_modules or root in profile_handled_modules:
            continue
        previous_failures = int(verify_failures.get(module, 0) or 0)
        if previous_failures >= 3:
            blocked_repeated_modules.append(module)
            unresolved.append(
                {
                    "module": module,
                    "reason": "blocked_after_repeated_import_verify_failure",
                }
            )
            install_results.append(
                {
                    "module": module,
                    "ok": False,
                    "blocked": True,
                    "reason": "blocked_after_repeated_import_verify_failure",
                    "previous_verify_failures": previous_failures,
                }
            )
            continue
        candidates = _module_to_package_candidates(module)
        if not candidates:
            unresolved.append({"module": module, "reason": "no_package_candidates"})
            continue
        installed = False
        for candidate in candidates:
            result = _pip_install(str(venv_python), candidate)
            result["module"] = module
            install_results.append(result)
            if not result.get("ok"):
                continue
            verify = _verify_import(str(venv_python), module)
            import_verify_results.append(verify)
            if verify.get("ok"):
                verify_failures[module] = 0
                resolved.append(module)
                installed = True
                break
            verify_failures[module] = previous_failures + 1
            result["ok"] = False
            result["reason"] = "install_succeeded_but_import_verify_failed"
            result["import_verify"] = verify
        if not installed:
            unresolved.append({"module": module, "reason": "all_candidate_installs_failed"})

    if profile_result and profile_result.get("install_results"):
        install_results.extend(profile_result.get("install_results", []))
    if profile_result and profile_result.get("ok"):
        for module in runtime_modules:
            root = module.split(".", 1)[0].strip()
            if module in profile_handled_modules or root in profile_handled_modules:
                if module not in resolved:
                    resolved.append(module)

    qiskit_guard = _enforce_legacy_qiskit_stack_if_needed(str(venv_python), runtime_modules, state)
    install_results.extend(qiskit_guard.get("install_results", []))

    if profile_result and not profile_result.get("ok"):
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = "dependency_runtime_fix_incompatible_or_broken_env"
        status = "failed"
        rc = 1
    elif qiskit_guard.get("attempted") and not qiskit_guard.get("ok"):
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = "dependency_runtime_fix_failed"
        status = "failed"
        rc = 1
    elif resolved:
        set_fsm_state(state, "APPLY_FIX")
        state["last_status"] = "dependency_runtime_fix_applied"
        state["runtime_python_executable"] = str(venv_python)
        status = "success"
        rc = 0
    else:
        set_fsm_state(state, "CLASSIFY_FAILURE")
        incompatible_or_broken_env = bool(blocked_repeated_modules) or any(
            result.get("reason") == "install_succeeded_but_import_verify_failed"
            for result in install_results
        )
        state["last_status"] = (
            "dependency_runtime_fix_incompatible_or_broken_env"
            if incompatible_or_broken_env
            else "dependency_runtime_fix_failed"
        )
        status = "failed"
        rc = 1

    payload = {
        "tool": "repro_dependency_runtime_fix",
        "status": status,
        "missing_modules": missing_modules,
        "runtime_modules": runtime_modules,
        "local_workspace_modules": local_workspace_modules,
        "resolved_modules": resolved,
        "runtime_profile": {
            "selected": profile_name,
            "result": profile_result,
        },
        "install_results": install_results,
        "import_verify_results": import_verify_results,
        "blocked_repeated_modules": blocked_repeated_modules,
        "unresolved": unresolved,
        "qiskit_guard": qiskit_guard,
        "isolation": {
            "type": "venv",
            "venv_dir": str(venv_dir),
            "python_executable": str(venv_python),
        },
        "python_executable": str(venv_python),
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
    }
    state["last_step"] = "repro_dependency_runtime_fix"
    append_history(state, "repro_dependency_runtime_fix", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
