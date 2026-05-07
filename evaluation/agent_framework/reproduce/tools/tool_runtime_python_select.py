#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

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
    repo_root_from_here,
    set_fsm_state,
    write_state,
)
from runtime_profiles import apply_runtime_profile  # noqa: E402

PROBES = [
    "import qiskit",
    "from qiskit import QuantumCircuit",
    "import qiskit_ibm_provider",
    "import qiskit_ibm_runtime",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def _candidate_pythons(repo_root: Path, bootstrap_python: str, *, include_hints: bool) -> list[str]:
    candidates: list[str] = []
    env_hints: list[str] = []
    if include_hints:
        # Prefer generic runtime hints; keep legacy hint name for backwards compatibility.
        hint_files = [
            repo_root / "temp/agent_framework/reproduce/python_hints/runtime_python",
            repo_root / "temp/agent_framework/reproduce/python_hints/qos_fig11_python",
        ]
        for hint_file in hint_files:
            if hint_file.is_file():
                hinted = hint_file.read_text(encoding="utf-8", errors="replace").strip()
                if hinted:
                    env_hints.append(hinted)
    bins = [bootstrap_python, "python3.11", "python3.10", "python3.12", "python3.13", "python3"]
    for item in [*env_hints, *bins]:
        val = str(item).strip()
        if not val:
            continue
        resolved = shutil.which(val) if "/" not in val else val
        if resolved and resolved not in candidates:
            candidates.append(resolved)
    return candidates[:4]


def _runtime_python_select_failures(state: dict) -> int:
    history = state.get("history")
    if not isinstance(history, list):
        return 0
    failures = 0
    for entry in history:
        if not isinstance(entry, dict):
            continue
        if entry.get("step") != "repro_runtime_python_select":
            continue
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("status") == "failed":
            failures += 1
    return failures


def _run(cmd: list[str]) -> dict:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return {
            "command": cmd,
            "returncode": 127,
            "ok": False,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout[-1500:],
        "stderr": proc.stderr[-2000:],
    }


def _score_python(python_exe: str, venv_dir: Path) -> dict:
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    mk = _run([python_exe, "-m", "venv", str(venv_dir)])
    vpy = str(venv_dir / "bin" / "python")
    profile_result = None
    probes: list[dict] = []
    passed = 0
    if mk["ok"]:
        profile_result = apply_runtime_profile(vpy, "qiskit_legacy_046")
        if profile_result.get("ok"):
            passed += 2
        for probe in PROBES:
            res = _run([vpy, "-c", probe])
            res["probe"] = probe
            probes.append(res)
            if res["ok"]:
                passed += 1
    return {
        "python": python_exe,
        "venv_dir": str(venv_dir),
        "venv_python": vpy,
        "venv_create": mk,
        "profile_result": profile_result,
        "probe_results": probes,
        "score": passed,
        "ok": mk["ok"] and bool(profile_result and profile_result.get("ok")) and passed >= len(PROBES) + 2,
    }


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)

    is_allowed, allowed_actions = assert_action_allowed(state, "runtime_python_select")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_runtime_python_select",
            action="runtime_python_select",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_runtime_python_select", payload)
        state["last_step"] = "repro_runtime_python_select"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    run_root = Path(str(state.get("run_root", "")).strip() or repo_root / "temp/agent_framework/reproduce/runtime_select").resolve()
    select_root = (run_root / "runtime_python_selector").resolve()
    select_root.mkdir(parents=True, exist_ok=True)

    prior_failures = _runtime_python_select_failures(state)
    include_hints = prior_failures > 0
    attempts = []
    try:
        for idx, py in enumerate(
            _candidate_pythons(repo_root, args.python_executable, include_hints=include_hints),
            start=1,
        ):
            attempts.append(_score_python(py, select_root / f"cand_{idx:02d}"))
    except Exception as exc:
        attempts.append(
            {
                "python": None,
                "venv_dir": None,
                "venv_python": None,
                "venv_create": {"ok": False, "returncode": 1, "stderr": str(exc), "stdout": ""},
                "probe_results": [],
                "score": 0,
                "ok": False,
            }
        )

    best = sorted(attempts, key=lambda x: (x.get("ok", False), x.get("score", 0)), reverse=True)[0] if attempts else None
    if best and best.get("ok"):
        state["runtime_venv_dir"] = best["venv_dir"]
        state["runtime_python_executable"] = best["venv_python"]
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = "runtime_python_selected"
        status = "success"
        rc = 0
    else:
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = "incompatible_python_runtime"
        status = "failed"
        rc = 1

    payload = {
        "tool": "repro_runtime_python_select",
        "status": status,
        "selected": best,
        "attempts": attempts,
        "hint_policy": {
            "mode": "fallback_only",
            "include_hints": include_hints,
            "prior_runtime_python_select_failures": prior_failures,
        },
        "python_executable": (best or {}).get("venv_python"),
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
    }
    state["last_step"] = "repro_runtime_python_select"
    append_history(state, "repro_runtime_python_select", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
