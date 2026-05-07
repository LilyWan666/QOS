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
import state_machine as sm
from repro_toolkit import (
    assert_action_allowed,
    append_history,
    blocked_action_payload,
    ensure_fix_budget,
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


def status_to_state(last_status: str) -> str:
    mapping = {
        "preflight_failed": sm.PREFLIGHT_FAILED,
        "run_failed": sm.RUN_FAILED,
        "parse_failed": sm.RUN_FAILED,
        "metric_failed": sm.RUN_FAILED,
        "verification_failed": sm.RUN_FAILED,
        "success": sm.VERIFY_CLAIM,
        "preflight_only": sm.PREFLIGHT,
    }
    return mapping.get(last_status, sm.CLASSIFY_FAILURE)


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, "classify_failure")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_classify_failure",
            action="classify_failure",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_classify_failure", payload)
        state["last_step"] = "repro_classify_failure"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    recipe = rr.load_json(resolve_recipe_path(args.recipe, repo_root))

    fix_budget = ensure_fix_budget(state, recipe)
    attempt_payload = state.get("last_attempt_payload") or {}
    diagnosis = attempt_payload.get("diagnosis_payload") or {}
    classification = diagnosis.get("classification") or {}
    recovery_plan = diagnosis.get("recovery_plan") or {}

    recoverable = bool(classification.get("recoverable", False))
    state_name = status_to_state(str(state.get("last_status", "")))
    candidate_actions = sm.next_allowed_actions(
        state_name,
        recoverable=recoverable,
        fix_budget=fix_budget,
    )
    if recoverable and fix_budget > 0:
        set_fsm_state(state, "CLASSIFY_FAILURE")
    elif fix_budget <= 0:
        set_fsm_state(state, "BUDGET_EXHAUSTED")
    else:
        set_fsm_state(state, "TERMINAL_FAILED")
    payload = {
        "tool": "repro_classify_failure",
        "status": "success",
        "state": state_name,
        "last_status": state.get("last_status"),
        "failure_category": classification.get("category", "unknown_failure"),
        "recoverable": recoverable,
        "fix_budget": fix_budget,
        "next_allowed_actions": candidate_actions,
        "suggested_recovery_actions": [
            item.get("action")
            for item in recovery_plan.get("actions", [])
            if isinstance(item, dict) and item.get("action")
        ],
        "diagnosis_summary": diagnosis.get("summary", ""),
        "fsm_state": get_fsm_state(state),
    }
    payload["next_allowed_actions"] = next_actions(state)

    state["last_step"] = "repro_classify_failure"
    state["last_classification"] = payload
    append_history(state, "repro_classify_failure", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
