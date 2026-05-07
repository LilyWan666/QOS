#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


RUNTIME_PROFILES: dict[str, dict[str, Any]] = {
    "qiskit_legacy_046": {
        "description": "Legacy qiskit stack for older provider/runtime import surfaces.",
        "triggers": {
            "modules": [
                "qiskit",
                "qiskit_aer",
                "qiskit_ibm_provider",
                "qiskit_ibm_runtime",
                "mapomatic",
                "mqt",
            ]
        },
        "packages": [
            "qiskit==0.46.3",
            "qiskit-terra==0.46.3",
            "qiskit-aer==0.14.2",
            "qiskit-ibm-provider==0.10.0",
            "qiskit-ibm-runtime==0.20.0",
            "mapomatic==0.10.0",
            "numpy<2",
            "setuptools<81",
        ],
        "probes": [
            "import qiskit",
            "from qiskit import QuantumCircuit",
            "import qiskit_ibm_provider",
            "import qiskit_ibm_runtime",
            "import mapomatic",
        ],
    }
}


def detect_profile_for_modules(missing_modules: list[str]) -> str | None:
    module_roots = {str(m).split(".", 1)[0].strip() for m in missing_modules if str(m).strip()}
    module_roots.update(
        str(m).strip()
        for m in missing_modules
        if str(m).strip() in {"qiskit_ibm_provider", "qiskit_ibm_runtime", "qiskit_aer"}
    )
    for profile_name, profile in RUNTIME_PROFILES.items():
        triggers = ((profile.get("triggers") or {}).get("modules")) or []
        if module_roots.intersection(set(str(x).strip() for x in triggers if str(x).strip())):
            return profile_name
    return None


def _pip_install(
    python_executable: str,
    requirement: str,
    constraints_file: Path | None = None,
) -> dict[str, Any]:
    cmd = [python_executable, "-m", "pip", "install"]
    if constraints_file is not None:
        cmd.extend(["-c", str(constraints_file)])
    cmd.append(requirement)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=90)
    except subprocess.TimeoutExpired as exc:
        return {
            "requirement": requirement,
            "constraints_file": str(constraints_file) if constraints_file else None,
            "command": cmd,
            "returncode": 124,
            "ok": False,
            "stdout": str(exc.stdout or "")[-4000:],
            "stderr": f"timeout: {exc}"[-4000:],
        }
    return {
        "requirement": requirement,
        "constraints_file": str(constraints_file) if constraints_file else None,
        "command": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def _run_probe(python_executable: str, probe: str) -> dict[str, Any]:
    cmd = [python_executable, "-c", probe]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=20)
    except subprocess.TimeoutExpired as exc:
        return {
            "probe": probe,
            "command": cmd,
            "returncode": 124,
            "ok": False,
            "stdout": str(exc.stdout or "")[-1000:],
            "stderr": f"timeout: {exc}"[-2000:],
        }
    return {
        "probe": probe,
        "command": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout[-1000:],
        "stderr": proc.stderr[-2000:],
    }


def apply_runtime_profile(
    python_executable: str,
    profile_name: str,
    *,
    constraints_file: Path | None = None,
) -> dict[str, Any]:
    profile = RUNTIME_PROFILES.get(profile_name)
    if profile is None:
        return {
            "profile": profile_name,
            "ok": False,
            "error": "unknown_profile",
            "install_results": [],
            "probe_results": [],
        }

    install_results = [
        _pip_install(python_executable, req, constraints_file=constraints_file)
        for req in profile.get("packages", [])
    ]
    if not all(result.get("ok") for result in install_results):
        return {
            "profile": profile_name,
            "ok": False,
            "error": "profile_package_install_failed",
            "install_results": install_results,
            "probe_results": [],
        }

    probe_results = [_run_probe(python_executable, probe) for probe in profile.get("probes", [])]
    return {
        "profile": profile_name,
        "ok": all(result.get("ok") for result in probe_results),
        "error": None,
        "install_results": install_results,
        "probe_results": probe_results,
    }
