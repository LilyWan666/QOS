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
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, "render_artifacts")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_render_artifacts",
            action="render_artifacts",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_render_artifacts", payload)
        state["last_step"] = "repro_render_artifacts"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    run_root = Path(str(state.get("run_root", "."))).resolve()
    artifact_dir = run_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pre_render_state = get_fsm_state(state)
    pre_render_status = str(state.get("last_status", ""))
    render_outcome = "success" if pre_render_state == "SUCCESS" else "failed"

    summary = {
        "recipe_name": state.get("recipe_name"),
        "last_status": pre_render_status,
        "last_step": state.get("last_step"),
        "pre_render_state": pre_render_state,
        "pre_render_status": pre_render_status,
        "render_outcome": render_outcome,
        "fix_budget": state.get("fix_budget"),
        "history_count": len(state.get("history", [])) if isinstance(state.get("history"), list) else 0,
        "last_manifest": state.get("last_manifest"),
        "last_verdict_path": state.get("last_verdict_path"),
    }
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "tool": "repro_render_artifacts",
        "status": "success",
        "artifact_dir": str(artifact_dir),
        "summary_path": str(summary_path),
        "summary": summary,
        "render_outcome": render_outcome,
        "pre_render_state": pre_render_state,
        "pre_render_status": pre_render_status,
    }
    state["last_step"] = "repro_render_artifacts"
    state["last_status"] = "artifacts_rendered"
    state["last_artifact_summary_path"] = str(summary_path)
    state["render_outcome"] = render_outcome
    set_fsm_state(state, "ARTIFACTS_RENDERED")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, "repro_render_artifacts", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
