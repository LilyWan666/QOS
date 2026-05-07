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

IMPORT_NAME_RE = re.compile(r"cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]")
INVALID_QISKIT_ENV_RE = re.compile(r"Qiskit.*invalid environment", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_last_error_text(state: dict[str, Any]) -> str:
    artifacts = ((state.get("last_manifest") or {}).get("artifacts")) or {}
    stderr_path = Path(str(artifacts.get("stderr", ""))).resolve() if artifacts.get("stderr") else None
    stdout_path = Path(str(artifacts.get("stdout", ""))).resolve() if artifacts.get("stdout") else None
    parts: list[str] = []
    for path in (stderr_path, stdout_path):
        if path and path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    metrics_raw = artifacts.get("metrics")
    if metrics_raw:
        metrics_path = Path(str(metrics_raw)).resolve()
        if metrics_path.exists():
            try:
                metrics = _load_json(metrics_path)
            except Exception:
                metrics = {}
            for item in metrics.get("checked_modules", []):
                if isinstance(item, dict):
                    err = str(item.get("error", "")).strip()
                    tb = str(item.get("traceback", "")).strip()
                    if err:
                        parts.append(err)
                    if tb:
                        parts.append(tb)
    return "\n".join(parts)


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


JOB_IMPORT_OLD = "from qiskit.providers import Job"
JOB_IMPORT_NEW = (
    "try:\n"
    "    from qiskit.providers import Job\n"
    "except Exception:\n"
    "    try:\n"
    "        from qiskit.providers.job import JobV1 as Job\n"
    "    except Exception:\n"
    "        Job = object\n"
)

QREG_IMPORT_OLD = "from qiskit.circuit import QuantumRegister as Fragment"
QREG_IMPORT_NEW = (
    "try:\n"
    "    from qiskit.circuit import QuantumRegister as Fragment\n"
    "except Exception:\n"
    "    from qiskit.circuit.quantumregister import QuantumRegister as Fragment\n"
)

BACKEND_PROPERTIES_IMPORT_OLD = "from qiskit_ibm_runtime.models import BackendProperties"
BACKEND_PROPERTIES_IMPORT_NEW = (
    "try:\n"
    "    from qiskit_ibm_runtime.models import BackendProperties\n"
    "except Exception:\n"
    "    try:\n"
    "        from qiskit.providers.models import BackendProperties\n"
    "    except Exception:\n"
    "        BackendProperties = object\n"
)


def _patch_qiskit_job_imports(repo_root: Path) -> list[str]:
    patched: list[str] = []
    for base in ("qos", "qvm"):
        root = (repo_root / base).resolve()
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if JOB_IMPORT_OLD not in text:
                continue
            updated = text.replace(JOB_IMPORT_OLD, JOB_IMPORT_NEW)
            if updated != text:
                path.write_text(updated, encoding="utf-8")
                patched.append(str(path))
    return patched


def _patch_backend_properties_imports(repo_root: Path) -> list[str]:
    patched: list[str] = []
    for base in ("qos", "qvm"):
        root = (repo_root / base).resolve()
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if BACKEND_PROPERTIES_IMPORT_OLD not in text:
                continue
            updated = text.replace(BACKEND_PROPERTIES_IMPORT_OLD, BACKEND_PROPERTIES_IMPORT_NEW)
            if updated != text:
                path.write_text(updated, encoding="utf-8")
                patched.append(str(path))
    return patched


def _patch_qiskit_quantumregister_imports(repo_root: Path) -> list[str]:
    patched: list[str] = []
    for base in ("qos", "qvm"):
        root = (repo_root / base).resolve()
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if QREG_IMPORT_OLD not in text:
                continue
            updated = text.replace(QREG_IMPORT_OLD, QREG_IMPORT_NEW)
            if updated != text:
                path.write_text(updated, encoding="utf-8")
                patched.append(str(path))
    return patched


def _trace_rules(error_text: str) -> tuple[str | None, list[str]]:
    normalized = error_text.strip()
    if not normalized:
        return None, []

    import_fails = IMPORT_NAME_RE.findall(normalized)
    for symbol, source in import_fails:
        if source == "qiskit.circuit" and symbol == "Store":
            return "qiskit_legacy_store_symbol", ["qiskit==0.46.3", "qiskit-ibm-provider==0.10.0"]
        if source == "qiskit.circuit" and symbol == "QuantumRegister":
            return "qiskit_circuit_quantumregister_surface", []
        if source == "qiskit.providers" and symbol in {"ProviderV1", "Job"}:
            return "qiskit_legacy_provider_v1", ["qiskit==0.46.3", "qiskit-ibm-provider==0.10.0"]
        if source == "qiskit" and symbol == "QuantumCircuit":
            return "qiskit_import_surface_mismatch", ["qiskit==0.46.3", "qiskit-ibm-provider==0.10.0"]

    if "qiskit.providers.fake_provider.fake_backend" in normalized:
        return "qiskit_fake_backend_layout", ["qiskit==0.46.3", "qiskit-ibm-provider==0.10.0"]

    if "qiskit_ibm_runtime.models" in normalized:
        return "qiskit_ibm_runtime_models_surface", []

    if INVALID_QISKIT_ENV_RE.search(normalized):
        return "qiskit_mixed_major_versions", ["qiskit==0.46.3", "qiskit-ibm-provider==0.10.0"]

    return None, []


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)

    is_allowed, allowed_actions = assert_action_allowed(state, "runtime_trace_fix")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_runtime_trace_fix",
            action="runtime_trace_fix",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_runtime_trace_fix", payload)
        state["last_step"] = "repro_runtime_trace_fix"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    error_text = _read_last_error_text(state)
    rule_id, requirements = _trace_rules(error_text)
    venv_dir, venv_python = ensure_runtime_env_selected(
        state=state,
        repo_root=repo_root,
        bootstrap_python_executable=args.python_executable,
    )

    install_results: list[dict[str, Any]] = []
    source_patch_results: list[dict[str, Any]] = []
    if rule_id:
        for requirement in requirements:
            install_results.append(_pip_install(str(venv_python), requirement))
        if rule_id == "qiskit_legacy_provider_v1":
            patched = _patch_qiskit_job_imports(repo_root)
            source_patch_results.append(
                {
                    "rule": "qiskit_job_import_compat",
                    "patched_files": patched,
                    "ok": True,
                }
            )
        if rule_id == "qiskit_circuit_quantumregister_surface":
            patched = _patch_qiskit_quantumregister_imports(repo_root)
            source_patch_results.append(
                {
                    "rule": "qiskit_quantumregister_import_compat",
                    "patched_files": patched,
                    "ok": True,
                }
            )
        if rule_id == "qiskit_ibm_runtime_models_surface":
            patched = _patch_backend_properties_imports(repo_root)
            source_patch_results.append(
                {
                    "rule": "qiskit_backend_properties_import_compat",
                    "patched_files": patched,
                    "ok": True,
                }
            )

    applied = bool(rule_id) and all(result.get("ok") for result in install_results) and all(
        result.get("ok") for result in source_patch_results
    )
    if applied:
        set_fsm_state(state, "APPLY_FIX")
        state["last_status"] = "runtime_trace_fix_applied"
        state["runtime_python_executable"] = str(venv_python)
        status = "runtime_trace_fix_applied"
        rc = 0
    else:
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = "runtime_trace_fix_failed"
        status = "runtime_trace_fix_skipped" if not rule_id else "runtime_trace_fix_failed"
        rc = 1

    payload = {
        "tool": "repro_runtime_trace_fix",
        "status": status,
        "rule_id": rule_id,
        "requirements": requirements,
        "install_results": install_results,
        "source_patch_results": source_patch_results,
        "isolation": {
            "type": "venv",
            "venv_dir": str(venv_dir),
            "python_executable": str(venv_python),
        },
        "python_executable": str(venv_python),
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
    }
    state["last_step"] = "repro_runtime_trace_fix"
    append_history(state, "repro_runtime_trace_fix", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
