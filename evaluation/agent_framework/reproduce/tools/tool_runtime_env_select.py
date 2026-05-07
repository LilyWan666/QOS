#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)

    is_allowed, allowed_actions = assert_action_allowed(state, "runtime_env_select")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_runtime_env_select",
            action="runtime_env_select",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_runtime_env_select", payload)
        state["last_step"] = "repro_runtime_env_select"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    try:
        venv_dir, venv_python = ensure_runtime_env_selected(
            state=state,
            repo_root=repo_root,
            bootstrap_python_executable=args.python_executable,
        )
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = "runtime_env_selected"
        status = "success"
        rc = 0
    except Exception as exc:
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = "runtime_env_select_failed"
        status = "failed"
        rc = 1
        error = str(exc)
    else:
        error = None

    payload = {
        "tool": "repro_runtime_env_select",
        "status": status,
        "isolation": {
            "type": "venv",
            "venv_dir": str(venv_dir) if status == "success" else None,
            "python_executable": str(venv_python) if status == "success" else None,
            "bootstrap_python_executable": args.python_executable,
        },
        "python_executable": str(venv_python) if status == "success" else None,
        "error": error,
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
    }
    state["last_step"] = "repro_runtime_env_select"
    append_history(state, "repro_runtime_env_select", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

