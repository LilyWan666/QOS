#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

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
    set_fsm_state,
    write_state,
)
from runtime_profiles import (  # noqa: E402
    RUNTIME_PROFILES,
    apply_runtime_profile,
    detect_profile_for_modules,
)


MISSING_MODULE_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
IMPORT_NAME_RE = re.compile(r"cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]")
PIP_NAME_MAP = {
    "qiskit_ibm_runtime": "qiskit-ibm-runtime",
    "qiskit_ibm_provider": "qiskit-ibm-provider",
    "pkg_resources": "setuptools<81",
}
DEPENDENCY_FILE_PATTERNS = (
    "requirements*.txt",
    "environment*.yml",
    "environment*.yaml",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "Pipfile",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_missing_modules_from_metrics(metrics_path: Path) -> list[str]:
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
        error = str(item.get("error", ""))
        for match in MISSING_MODULE_RE.findall(error):
            if match and match not in missing:
                missing.append(match)
    return missing


def _collect_import_name_errors_from_metrics(metrics_path: Path) -> list[dict]:
    if not metrics_path.exists():
        return []
    try:
        payload = _load_json(metrics_path)
    except Exception:
        return []
    errors: list[dict] = []
    for item in payload.get("checked_modules", []):
        if not isinstance(item, dict):
            continue
        error = str(item.get("error", ""))
        for symbol, from_module in IMPORT_NAME_RE.findall(error):
            pkg_root = from_module.split(".", 1)[0].strip()
            if not pkg_root:
                continue
            rec = {
                "symbol": symbol.strip(),
                "from_module": from_module.strip(),
                "package_root": pkg_root,
            }
            if rec not in errors:
                errors.append(rec)
    return errors


def _collect_missing_modules_from_stderr(stderr_path: Path) -> list[str]:
    if not stderr_path.exists():
        return []
    text = stderr_path.read_text(encoding="utf-8", errors="replace")
    missing: list[str] = []
    for match in MISSING_MODULE_RE.findall(text):
        if match and match not in missing:
            missing.append(match)
    return missing


def _collect_missing_modules_from_preflight(preflight_path: Path) -> list[str]:
    if not preflight_path.exists():
        return []
    try:
        payload = _load_json(preflight_path)
    except Exception:
        return []
    missing: list[str] = []
    checks = (payload.get("checks") or {}).get("python_imports") or []
    for item in checks:
        if not isinstance(item, dict) or item.get("ok"):
            continue
        stderr = str(item.get("stderr", "") or "")
        for match in MISSING_MODULE_RE.findall(stderr):
            if match and match not in missing:
                missing.append(match)
    return missing


def _collect_missing_modules_from_diagnosis(state: dict) -> list[str]:
    missing: list[str] = []
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
    return missing


def _collect_candidates(state: dict) -> tuple[list[str], list[dict]]:
    artifacts = ((state.get("last_manifest") or {}).get("artifacts")) or {}
    missing: list[str] = []
    import_name_errors: list[dict] = []
    for path_key, collector in (
        ("metrics", _collect_missing_modules_from_metrics),
        ("preflight", _collect_missing_modules_from_preflight),
        ("stderr", _collect_missing_modules_from_stderr),
    ):
        raw = artifacts.get(path_key)
        if not raw:
            continue
        for module in collector(Path(str(raw)).resolve()):
            if module not in missing:
                missing.append(module)
    raw_metrics = artifacts.get("metrics")
    if raw_metrics:
        for item in _collect_import_name_errors_from_metrics(Path(str(raw_metrics)).resolve()):
            if item not in import_name_errors:
                import_name_errors.append(item)
    for module in _collect_missing_modules_from_diagnosis(state):
        if module not in missing:
            missing.append(module)
    return missing, import_name_errors


def _iter_dependency_files(repo_root: Path) -> Iterable[Path]:
    # Avoid full-repo recursive scans on large datasets. Dependency manifests
    # are expected at repo root or one directory below.
    seen: set[Path] = set()
    for pattern in DEPENDENCY_FILE_PATTERNS:
        for path in repo_root.glob(pattern):
            resolved = path.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            yield resolved
        for path in repo_root.glob(f"*/{pattern}"):
            resolved = path.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            yield resolved


def _extract_constraint_for_package(repo_root: Path, package: str) -> str | None:
    pkg = package.replace("-", "_")
    pattern = re.compile(
        rf"^\s*{re.escape(package)}\s*([<>=!~].+)?\s*$|^\s*{re.escape(pkg)}\s*([<>=!~].+)?\s*$",
        re.IGNORECASE,
    )
    for dep_file in _iter_dependency_files(repo_root):
        text = dep_file.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = pattern.match(stripped)
            if not m:
                continue
            raw = (m.group(1) or m.group(2) or "").strip()
            return raw or ""
    return None


def _pip_install(python_executable: str, module_name: str, version_constraint: str | None = None) -> dict:
    pip_name = PIP_NAME_MAP.get(module_name, module_name)
    requirement = f"{pip_name}{version_constraint}" if version_constraint else pip_name
    cmd = [python_executable, "-m", "pip", "install", requirement]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "module": module_name,
        "pip_name": pip_name,
        "version_constraint": version_constraint,
        "requirement": requirement,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "ok": proc.returncode == 0,
    }


def _resolve_constraints_file(repo_root: Path, recipe_path: Path) -> Path | None:
    try:
        recipe_payload = _load_json(recipe_path)
    except Exception:
        return None
    runtime_cfg = recipe_payload.get("runtime") or {}
    raw = str(runtime_cfg.get("constraints_file", "")).strip()
    if not raw:
        return None
    candidate = Path(raw)
    resolved = candidate if candidate.is_absolute() else (repo_root / candidate).resolve()
    if not resolved.exists() or not resolved.is_file():
        return None
    return resolved


def _pip_install_with_constraints(
    python_executable: str,
    module_name: str,
    version_constraint: str | None,
    constraints_file: Path | None,
) -> dict:
    pip_name = PIP_NAME_MAP.get(module_name, module_name)
    requirement = f"{pip_name}{version_constraint}" if version_constraint else pip_name
    cmd = [python_executable, "-m", "pip", "install"]
    if constraints_file is not None:
        cmd.extend(["-c", str(constraints_file)])
    cmd.append(requirement)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "module": module_name,
        "pip_name": pip_name,
        "version_constraint": version_constraint,
        "requirement": requirement,
        "constraints_file": str(constraints_file) if constraints_file else None,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "ok": proc.returncode == 0,
    }


def _verify_import(python_executable: str, module_name: str) -> dict:
    cmd = [python_executable, "-c", f"import {module_name}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "module": module_name,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-1000:],
        "stderr": proc.stderr[-2000:],
        "ok": proc.returncode == 0,
    }


def main() -> int:
    args = parse_args()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)

    is_allowed, allowed_actions = assert_action_allowed(state, "env_fix")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_env_fix",
            action="env_fix",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_env_fix", payload)
        state["last_step"] = "repro_env_fix"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    repo_root = Path(__file__).resolve().parents[4]
    recipe_path = Path(args.recipe).resolve()
    constraints_file = _resolve_constraints_file(repo_root, recipe_path)
    venv_dir, venv_python = ensure_runtime_env_selected(
        state=state,
        repo_root=repo_root,
        bootstrap_python_executable=args.python_executable,
    )
    missing, import_name_errors = _collect_candidates(state)
    local_modules = [m for m in missing if m.startswith("qos.")]
    pip_modules = [m for m in missing if not m.startswith("qos.")]
    version_mismatch_packages = sorted({e["package_root"] for e in import_name_errors})

    profile_name = detect_profile_for_modules(missing)
    profile_result: dict | None = None
    profile_handled_modules: set[str] = set()
    if profile_name:
        triggers = (((RUNTIME_PROFILES.get(profile_name) or {}).get("triggers") or {}).get("modules")) or []
        profile_handled_modules = {str(t).strip() for t in triggers if str(t).strip()}
        profile_result = apply_runtime_profile(
            str(venv_python),
            profile_name,
            constraints_file=constraints_file,
        )

    install_plan: list[dict] = []
    for module in pip_modules:
        root = module.split(".", 1)[0].strip()
        if module in profile_handled_modules or root in profile_handled_modules:
            continue
        install_plan.append(
            {
                "module": module,
                "kind": "missing_module",
                "version_constraint": _extract_constraint_for_package(
                    repo_root, PIP_NAME_MAP.get(module, module)
                ),
            }
        )
    unresolved_version_mismatch: list[dict] = []
    for pkg in version_mismatch_packages:
        normalized = PIP_NAME_MAP.get(pkg, pkg)
        if any(x["module"] == pkg for x in install_plan):
            continue
        constraint = _extract_constraint_for_package(repo_root, normalized)
        if constraint is None:
            unresolved_version_mismatch.append(
                {"package": pkg, "reason": "no_version_constraint_found_in_repo"}
            )
            continue
        install_plan.append(
            {
                "module": pkg,
                "kind": "import_name_mismatch",
                "version_constraint": constraint,
            }
        )

    verify_failures = state.setdefault("runtime_import_verify_failures", {})
    if not isinstance(verify_failures, dict):
        verify_failures = {}
        state["runtime_import_verify_failures"] = verify_failures
    install_results: list[dict] = []
    import_verify_results: list[dict] = []
    blocked_repeated_modules: list[str] = []
    for plan in install_plan:
        module = str(plan["module"])
        previous_failures = int(verify_failures.get(module, 0) or 0)
        if previous_failures >= 2:
            blocked_repeated_modules.append(module)
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
        result = _pip_install_with_constraints(
            str(venv_python),
            module,
            plan.get("version_constraint"),
            constraints_file,
        )
        install_results.append(result)
        if not result.get("ok"):
            continue
        verify = _verify_import(str(venv_python), module)
        import_verify_results.append(verify)
        if verify.get("ok"):
            verify_failures[module] = 0
        else:
            verify_failures[module] = previous_failures + 1
            result["ok"] = False
            result["reason"] = "install_succeeded_but_import_verify_failed"
            result["import_verify"] = verify
    installed = [r["module"] for r in install_results if r.get("ok")]
    failed = [r["module"] for r in install_results if not r.get("ok")]
    if profile_result and profile_result.get("ok"):
        for module in ("qiskit", "qiskit_ibm_runtime", "qiskit_ibm_provider", "qiskit_aer"):
            if module in missing and module not in installed:
                installed.append(module)
    incompatible_or_broken_env = bool(blocked_repeated_modules) or any(
        r.get("reason") == "install_succeeded_but_import_verify_failed" for r in install_results
    ) or bool(profile_result and not profile_result.get("ok"))

    if installed:
        set_fsm_state(state, "APPLY_FIX")
        state["last_status"] = "env_fix_applied"
        state["runtime_python_executable"] = str(venv_python)
        status = "success"
        rc = 0
    else:
        # If nothing installable exists (or installs fail), route to classify path
        # to avoid repeated env_fix loops.
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = (
            "env_fix_incompatible_or_broken_env" if incompatible_or_broken_env else "env_fix_failed"
        )
        status = "failed"
        rc = 1

    state["last_step"] = "repro_env_fix"
    payload = {
        "tool": "repro_env_fix",
        "status": status,
        "python_executable": str(venv_python),
        "missing_modules_detected": missing,
        "import_name_mismatch_detected": import_name_errors,
        "local_modules_not_auto_installed": local_modules,
        "pip_install_plan": install_plan,
        "pip_install_attempted": [p["module"] for p in install_plan],
        "pip_install_installed": installed,
        "pip_install_failed": failed,
        "import_verify_results": import_verify_results,
        "blocked_repeated_modules": blocked_repeated_modules,
        "runtime_profile": {
            "selected": profile_name,
            "result": profile_result,
        },
        "version_mismatch_unresolved": unresolved_version_mismatch,
        "install_results": install_results,
        "constraints_file": str(constraints_file) if constraints_file else None,
        "isolation": {
            "type": "venv",
            "venv_dir": str(venv_dir),
            "python_executable": str(venv_python),
            "bootstrap_python_executable": args.python_executable,
        },
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
        "policy": "no_shim_first",
    }
    append_history(state, "repro_env_fix", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
