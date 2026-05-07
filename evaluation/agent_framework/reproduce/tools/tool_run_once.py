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
    ensure_isolated_workspace_recipe,
    ensure_run_root,
    get_fsm_state,
    load_state,
    next_attempt_ordinal,
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
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--allow-run-on-preflight-failure",
        action="store_true",
        help="Run entry command even when preflight fails.",
    )
    parser.add_argument(
        "--output-root",
        default="temp/agent_framework/reproduce/tools/runs",
        help="Root directory for step artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, "repro_run_once")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_run_once",
            action="repro_run_once",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_run_once", payload)
        state["last_step"] = "repro_run_once"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    recipe_path = resolve_recipe_path(args.recipe, repo_root)
    recipe = rr.load_json(recipe_path)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    run_root = ensure_run_root(recipe["name"], state, output_root)
    recipe_path, recipe = ensure_isolated_workspace_recipe(
        state=state,
        run_root=run_root,
        repo_root=repo_root,
        recipe_path=recipe_path,
        recipe=recipe,
    )
    attempts_root = run_root / "attempts"
    ordinal = next_attempt_ordinal(state)
    attempt_dir = attempts_root / f"attempt_{ordinal:04d}_run_once"

    workspace_root = rr.resolve_workspace_root(repo_root, recipe.get("workspace_root", "."))
    attempt = rr.run_attempt(
        recipe=recipe,
        repo_root=repo_root,
        workspace_root=workspace_root,
        python_exe=args.python_executable,
        attempt_dir=attempt_dir,
        preflight_only=False,
        allow_run_on_preflight_failure=args.allow_run_on_preflight_failure,
    )

    state.update(
        {
            "recipe_name": recipe["name"],
            "recipe_path": str(recipe_path),
            "workspace_root": str(workspace_root),
            "last_step": "repro_run_once",
            "last_attempt_dir": str(attempt_dir),
            "last_status": attempt["status"],
            "last_manifest": attempt["manifest"],
            "last_attempt_payload": {
                "status": attempt["status"],
                "status_payload": attempt["status_payload"],
                "diagnosis_payload": attempt["diagnosis_payload"],
                "artifacts": attempt["artifacts"],
                "attempt_dir": str(attempt_dir),
            },
        }
    )
    classification = attempt["diagnosis_payload"].get("classification", {})
    if classification:
        state["last_classification"] = {
            **classification,
            "failure_category": classification.get("category"),
        }
    if attempt["status"] in {"success", "preflight_only"}:
        set_fsm_state(state, "RUN_ONCE")
    else:
        set_fsm_state(state, "RUN_FAILED")
    current_state = get_fsm_state(state)
    append_history(
        state,
        "repro_run_once",
        {
            "attempt_dir": str(attempt_dir),
            "status": attempt["status"],
            "artifacts": attempt["artifacts"],
            "diagnosis": attempt["diagnosis_payload"],
            "fsm_state": current_state,
            "next_allowed_actions": next_actions(state),
        },
    )
    write_state(state_path, state)

    print(
        json.dumps(
            {
                "tool": "repro_run_once",
                "state_path": str(state_path),
                "attempt_dir": str(attempt_dir),
                "status": attempt["status"],
                "artifacts": attempt["artifacts"],
                "diagnosis": attempt["diagnosis_payload"],
                "fsm_state": current_state,
                "next_allowed_actions": next_actions(state),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if attempt["status"] in {"success", "preflight_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
