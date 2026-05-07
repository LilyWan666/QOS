#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repro_toolkit import (
    assert_action_allowed,
    append_history,
    blocked_action_payload,
    get_fsm_state,
    load_state,
    next_actions,
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
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, "fix_validate")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_fix_validate",
            action="fix_validate",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_fix_validate", payload)
        state["last_step"] = "repro_fix_validate"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    last_recovery = state.get("last_recovery") or {}
    recovery_payload = last_recovery.get("recovery") if isinstance(last_recovery, dict) else None
    applied = bool((recovery_payload or {}).get("applied"))

    if not applied:
        set_fsm_state(state, "TERMINAL_FAILED")
        payload = {
            "tool": "repro_fix_validate",
            "status": "failed",
            "reason": "last_recovery is missing or not applied",
            "fsm_state": get_fsm_state(state),
            "next_allowed_actions": next_actions(state),
        }
        append_history(state, "repro_fix_validate", payload)
        state["last_step"] = "repro_fix_validate"
        state["last_status"] = "fix_validate_failed"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    set_fsm_state(state, "FIX_VALIDATE")
    payload = {
        "tool": "repro_fix_validate",
        "status": "success",
        "reason": "recovery applied; ready for verify/test or classify loop",
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
    }
    append_history(state, "repro_fix_validate", payload)
    state["last_step"] = "repro_fix_validate"
    state["last_status"] = "fix_validate_ready"
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
