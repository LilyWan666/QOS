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
    ensure_fix_budget,
    ensure_isolated_workspace_recipe,
    ensure_run_root,
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
    parser.add_argument("--recovery-action", default="")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def choose_recovery_action(diagnosis: dict[str, object], requested: str) -> str:
    if requested:
        return requested
    recovery_plan = diagnosis.get("recovery_plan", {})
    actions = recovery_plan.get("actions", []) if isinstance(recovery_plan, dict) else []
    for item in actions:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", ""))
        if not action:
            continue
        if action.startswith("retry_"):
            continue
        return action
    return ""


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, "apply_fix")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_apply_fix",
            action="apply_fix",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_apply_fix", payload)
        state["last_step"] = "repro_apply_fix"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    recipe_path = resolve_recipe_path(args.recipe, repo_root)
    recipe = rr.load_json(recipe_path)
    run_root = ensure_run_root(
        recipe["name"],
        state,
        (repo_root / "temp/agent_framework/reproduce/tools/runs").resolve(),
    )
    recipe_path, recipe = ensure_isolated_workspace_recipe(
        state=state,
        run_root=run_root,
        repo_root=repo_root,
        recipe_path=recipe_path,
        recipe=recipe,
    )

    fix_budget = ensure_fix_budget(state, recipe)
    if fix_budget <= 0:
        payload = {
            "tool": "repro_apply_fix",
            "status": "blocked",
            "reason": "fix_budget exhausted",
            "fix_budget": fix_budget,
        }
        append_history(state, "repro_apply_fix", payload)
        state["last_step"] = "repro_apply_fix"
        state["last_status"] = "budget_exhausted"
        set_fsm_state(state, "BUDGET_EXHAUSTED")
        payload["fsm_state"] = get_fsm_state(state)
        payload["next_allowed_actions"] = next_actions(state)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    attempt_payload = state.get("last_attempt_payload") or {}
    diagnosis = attempt_payload.get("diagnosis_payload") or {}
    action_name = choose_recovery_action(diagnosis, args.recovery_action.strip())
    if not action_name:
        payload = {
            "tool": "repro_apply_fix",
            "status": "failed",
            "reason": "no eligible recovery action found in diagnosis.recovery_plan.actions",
        }
        append_history(state, "repro_apply_fix", payload)
        state["last_step"] = "repro_apply_fix"
        state["last_status"] = "fix_failed"
        set_fsm_state(state, "TERMINAL_FAILED")
        payload["fsm_state"] = get_fsm_state(state)
        payload["next_allowed_actions"] = next_actions(state)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    run_root = Path(str(state.get("run_root", run_root))).resolve()
    workspace_root = rr.resolve_workspace_root(repo_root, recipe.get("workspace_root", "."))
    raw_attempt_dir = str(attempt_payload.get("attempt_dir", state.get("last_attempt_dir", ""))).strip()
    if not raw_attempt_dir:
        payload = {
            "tool": "repro_apply_fix",
            "status": "failed",
            "reason": "missing last attempt directory in state",
        }
        append_history(state, "repro_apply_fix", payload)
        state["last_step"] = "repro_apply_fix"
        state["last_status"] = "fix_failed"
        set_fsm_state(state, "TERMINAL_FAILED")
        payload["fsm_state"] = get_fsm_state(state)
        payload["next_allowed_actions"] = next_actions(state)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    attempt_dir = Path(raw_attempt_dir).resolve()

    minimal_attempt = {
        "attempt_dir": attempt_dir,
        "diagnosis_payload": diagnosis,
    }
    recovery = rr.execute_recovery_handler(
        recipe=recipe,
        repo_root=repo_root,
        workspace_root=workspace_root,
        python_exe=args.python_executable,
        run_dir=run_root,
        attempt=minimal_attempt,
        action_name=action_name,
        ordinal=int(state.get("recovery_ordinal", 0) or 0) + 1,
    )
    state["recovery_ordinal"] = int(state.get("recovery_ordinal", 0) or 0) + 1

    if recovery is None:
        payload = {
            "tool": "repro_apply_fix",
            "status": "failed",
            "reason": f"no handler configured for recovery action: {action_name}",
            "recovery_action": action_name,
        }
        append_history(state, "repro_apply_fix", payload)
        state["last_step"] = "repro_apply_fix"
        state["last_status"] = "fix_failed"
        set_fsm_state(state, "TERMINAL_FAILED")
        payload["fsm_state"] = get_fsm_state(state)
        payload["next_allowed_actions"] = next_actions(state)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    state["fix_budget"] = max(0, fix_budget - 1)
    payload = {
        "tool": "repro_apply_fix",
        "status": "success" if bool(recovery.get("applied")) else "failed",
        "recovery_action": action_name,
        "recovery": recovery,
        "fix_budget_before": fix_budget,
        "fix_budget_after": state["fix_budget"],
    }
    state["last_step"] = "repro_apply_fix"
    state["last_status"] = "fix_applied" if bool(recovery.get("applied")) else "fix_failed"
    state["last_recovery"] = payload
    if bool(recovery.get("applied")):
        set_fsm_state(state, "APPLY_FIX")
    else:
        set_fsm_state(state, "TERMINAL_FAILED")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, "repro_apply_fix", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(recovery.get("applied")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
