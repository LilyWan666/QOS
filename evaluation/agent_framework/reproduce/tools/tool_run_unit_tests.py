#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from repro_toolkit import (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    return parser.parse_args()


def run_commands(commands: list[str], cwd: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for command in commands:
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        results.append(
            {
                "command": command,
                "returncode": proc.returncode,
                "ok": proc.returncode == 0,
                "stdout_tail": proc.stdout.strip().splitlines()[-20:],
                "stderr_tail": proc.stderr.strip().splitlines()[-20:],
            }
        )
    return results


def main() -> int:
    args = parse_args()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, "run_unit_tests")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_run_unit_tests",
            action="run_unit_tests",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_run_unit_tests", payload)
        state["last_step"] = "repro_run_unit_tests"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    repo_root = repo_root_from_here()

    plan = state.get("test_plan", {})
    commands = plan.get("unit_tests", [])
    if not isinstance(commands, list) or not commands:
        payload = {
            "tool": "repro_run_unit_tests",
            "status": "failed",
            "reason": "missing unit_tests in state.test_plan",
        }
        append_history(state, "repro_run_unit_tests", payload)
        state["last_step"] = "repro_run_unit_tests"
        state["last_status"] = "unit_test_failed"
        set_fsm_state(state, "UNIT_TEST")
        payload["fsm_state"] = get_fsm_state(state)
        payload["next_allowed_actions"] = next_actions(state)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    results = run_commands([str(item) for item in commands], repo_root)
    ok = all(bool(item.get("ok")) for item in results)
    payload = {
        "tool": "repro_run_unit_tests",
        "status": "success" if ok else "failed",
        "results": results,
    }
    state["last_step"] = "repro_run_unit_tests"
    state["last_status"] = "unit_test_success" if ok else "unit_test_failed"
    state["last_unit_test"] = payload
    set_fsm_state(state, "UNIT_TEST")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, "repro_run_unit_tests", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
