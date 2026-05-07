#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

import run_reproduce as rr
from repro_toolkit import (
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, "test_plan")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_test_plan",
            action="test_plan",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_test_plan", payload)
        state["last_step"] = "repro_test_plan"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    recipe = rr.load_json(resolve_recipe_path(args.recipe, repo_root))

    test_cfg = recipe.get("testing", {})
    unit_tests = test_cfg.get("unit_tests", [])
    regression_tests = test_cfg.get("regression_tests", [])
    if not unit_tests:
        unit_tests = [
            "python -m unittest evaluation/agent_framework/reproduce/tools/tests/test_state_machine.py -v"
        ]
    if not regression_tests:
        regression_tests = [
            "python evaluation/agent_framework/reproduce/tools/tool_run_once.py "
            "--recipe evaluation/agent_framework/reproduce/examples/qos_import_smoke.json "
            f"--state {state_path}"
        ]

    payload = {
        "tool": "repro_test_plan",
        "status": "success",
        "unit_tests": unit_tests,
        "regression_tests": regression_tests,
    }
    state["last_step"] = "repro_test_plan"
    state["last_status"] = "test_plan_ready"
    state["test_plan"] = payload
    set_fsm_state(state, "TEST_PLAN")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, "repro_test_plan", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
