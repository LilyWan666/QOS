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
    get_fsm_state,
    load_state,
    next_actions,
    set_fsm_state,
    write_state,
)


IMPORT_NAME_RE = re.compile(r"cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]")
MISSING_MODULE_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
DEPENDENCY_FILE_PATTERNS = (
    "requirements*.txt",
    "environment*.yml",
    "environment*.yaml",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "Pipfile",
)
PIP_NAME_MAP = {
    "qiskit_ibm_provider": "qiskit-ibm-provider",
}
KNOWN_COMPAT_CANDIDATES: dict[str, list[str]] = {
    # Keep this list small and conservative; it is a generic fallback when
    # repository constraints are missing and import-level mismatch is detected.
    "qiskit": ["==0.46.3", "==0.45.3", ""],
    "pymoo": ["==0.6.1.3", "==0.6.0.1", ""],
    "qiskit-ibm-provider": ["==0.10.0", "==0.11.0", ""],
}
COMPANION_PACKAGES: dict[str, list[str]] = {
    "qiskit": ["qiskit_ibm_provider"],
}
LOCAL_MODULE_SKIP = {"src", "test", "tests"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_dep_files(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in DEPENDENCY_FILE_PATTERNS:
        for path in repo_root.rglob(pattern):
            if not path.is_file() or path in seen:
                continue
            rel = str(path.resolve().relative_to(repo_root.resolve()))
            if rel.startswith("temp/") or rel.startswith(".git/"):
                continue
            seen.add(path)
            out.append(path)
    return sorted(out)


def _extract_constraint(repo_root: Path, package: str) -> str | None:
    pkg = package.replace("-", "_")
    regex = re.compile(
        rf"^\s*{re.escape(package)}\s*([<>=!~].+)?\s*$|^\s*{re.escape(pkg)}\s*([<>=!~].+)?\s*$",
        re.IGNORECASE,
    )
    for dep_file in _iter_dep_files(repo_root):
        text = dep_file.read_text(encoding="utf-8", errors="replace")
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = regex.match(line)
            if not m:
                continue
            return (m.group(1) or m.group(2) or "").strip()
    return None


def _extract_mismatch_packages(metrics_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not metrics_path.exists():
        return ([], [])
    try:
        payload = _load_json(metrics_path)
    except Exception:
        return ([], [])
    mismatches: list[dict[str, str]] = []
    missing_ext: list[str] = []
    for item in payload.get("checked_modules", []):
        if not isinstance(item, dict):
            continue
        err = str(item.get("error", ""))
        for symbol, from_module in IMPORT_NAME_RE.findall(err):
            pkg = from_module.split(".", 1)[0].strip()
            if pkg and pkg != "qos":
                rec = {"package": pkg, "symbol": symbol.strip(), "from_module": from_module.strip()}
                if rec not in mismatches:
                    mismatches.append(rec)
        for module in MISSING_MODULE_RE.findall(err):
            root = module.split(".", 1)[0].strip()
            if root and root != "qos" and root not in missing_ext:
                missing_ext.append(root)
    return (mismatches, missing_ext)


def _pip_install(python_executable: str, package: str, specifier: str | None) -> dict[str, Any]:
    mapped = PIP_NAME_MAP.get(package, package)
    requirement = f"{mapped}{specifier}" if specifier else mapped
    cmd = [python_executable, "-m", "pip", "install", requirement]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "package": package,
        "pip_name": mapped,
        "specifier": specifier,
        "requirement": requirement,
        "command": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def _candidate_specifiers(package: str, repo_constraint: str | None) -> list[str]:
    if repo_constraint is not None:
        return [repo_constraint]
    mapped = PIP_NAME_MAP.get(package, package)
    candidates = KNOWN_COMPAT_CANDIDATES.get(mapped, [""])
    # De-duplicate while preserving order.
    out: list[str] = []
    for item in candidates:
        if item not in out:
            out.append(item)
    return out


def _looks_like_local_module(package: str, repo_root: Path) -> bool:
    root = package.split(".", 1)[0].strip()
    if not root:
        return False
    if root in LOCAL_MODULE_SKIP:
        return True
    return (repo_root / root).is_dir() or (repo_root / f"{root}.py").is_file()


def main() -> int:
    args = parse_args()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)

    is_allowed, allowed_actions = assert_action_allowed(state, "constraint_resolve")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_constraint_resolve",
            action="constraint_resolve",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_constraint_resolve", payload)
        state["last_step"] = "repro_constraint_resolve"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    repo_root = Path(__file__).resolve().parents[4]
    artifacts = ((state.get("last_manifest") or {}).get("artifacts")) or {}
    metrics_path = Path(str(artifacts.get("metrics", ""))).resolve() if artifacts.get("metrics") else None
    mismatches, missing_ext = (
        _extract_mismatch_packages(metrics_path) if metrics_path else ([], [])
    )
    packages = set({x["package"] for x in mismatches}.union(set(missing_ext)))
    for pkg in list(packages):
        mapped = PIP_NAME_MAP.get(pkg, pkg)
        for companion in COMPANION_PACKAGES.get(mapped, []):
            packages.add(companion)
    packages = sorted(packages)

    install_plan: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []
    for pkg in packages:
        if _looks_like_local_module(pkg, repo_root):
            unresolved.append({"package": pkg, "reason": "local_module_like_skip_pip"})
            continue
        spec = _extract_constraint(repo_root, PIP_NAME_MAP.get(pkg, pkg))
        install_plan.append(
            {
                "package": pkg,
                "repo_constraint": spec,
                "candidate_specifiers": _candidate_specifiers(pkg, spec),
            }
        )

    install_results: list[dict[str, Any]] = []
    resolved_pkgs: set[str] = set()
    for item in install_plan:
        pkg = str(item["package"])
        candidates = list(item.get("candidate_specifiers", [])) or [""]
        pkg_ok = False
        for spec in candidates:
            result = _pip_install(args.python_executable, pkg, spec)
            result["candidate_specifier"] = spec
            install_results.append(result)
            if result.get("ok"):
                resolved_pkgs.add(pkg)
                pkg_ok = True
                break
        if not pkg_ok:
            unresolved.append({"package": pkg, "reason": "all_candidate_installs_failed"})

    applied = bool(resolved_pkgs)
    if applied:
        set_fsm_state(state, "APPLY_FIX")
        state["last_status"] = "constraint_resolve_applied"
        status = "success"
        rc = 0
    else:
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = "constraint_resolve_failed"
        status = "failed"
        rc = 1

    payload = {
        "tool": "repro_constraint_resolve",
        "status": status,
        "mismatch_signals": mismatches,
        "missing_external_modules": missing_ext,
        "install_plan": install_plan,
        "install_results": install_results,
        "unresolved": unresolved,
        "python_executable": args.python_executable,
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
    }
    state["last_step"] = "repro_constraint_resolve"
    append_history(state, "repro_constraint_resolve", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
