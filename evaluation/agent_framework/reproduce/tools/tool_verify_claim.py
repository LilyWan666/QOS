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
from figure_claim import evaluate_figure_contract
from repro_toolkit import (
    assert_action_allowed,
    append_history,
    blocked_action_payload,
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
    recipe_path = resolve_recipe_path(args.recipe, repo_root)
    recipe = rr.load_json(recipe_path)
    tests_required = bool((recipe.get("testing") or {}).get("required_before_success", False))
    is_allowed, allowed_actions = assert_action_allowed(
        state,
        "verify_claim",
        tests_required=tests_required,
    )
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_verify_claim",
            action="verify_claim",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_verify_claim", payload)
        state["last_step"] = "repro_verify_claim"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    run_root = ensure_run_root(recipe["name"], state, output_root)

    artifacts = (((state.get("last_manifest") or {}).get("artifacts")) or {})
    metrics_path_raw = artifacts.get("metrics")
    if not metrics_path_raw:
        payload = {
            "tool": "repro_verify_claim",
            "status": "failed",
            "reason": "missing metrics artifact in state.last_manifest.artifacts.metrics",
        }
        append_history(state, "repro_verify_claim", payload)
        state["last_step"] = "repro_verify_claim"
        state["last_status"] = "verification_failed"
        set_fsm_state(state, "CLASSIFY_FAILURE")
        payload["fsm_state"] = get_fsm_state(state)
        payload["next_allowed_actions"] = next_actions(state, tests_required=tests_required)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    metrics_path = Path(metrics_path_raw).resolve()
    if not metrics_path.exists():
        payload = {
            "tool": "repro_verify_claim",
            "status": "failed",
            "reason": f"metrics file not found: {metrics_path}",
        }
        append_history(state, "repro_verify_claim", payload)
        state["last_step"] = "repro_verify_claim"
        state["last_status"] = "verification_failed"
        set_fsm_state(state, "CLASSIFY_FAILURE")
        payload["fsm_state"] = get_fsm_state(state)
        payload["next_allowed_actions"] = next_actions(state, tests_required=tests_required)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    verification_cfg = recipe.get("verification", {})
    if not bool(verification_cfg.get("enabled", False)):
        payload = {
            "tool": "repro_verify_claim",
            "status": "skipped",
            "reason": "recipe verification.enabled=false",
        }
        append_history(state, "repro_verify_claim", payload)
        state["last_step"] = "repro_verify_claim"
        state["last_status"] = "verification_skipped"
        set_fsm_state(state, "VERIFY_CLAIM")
        payload["fsm_state"] = get_fsm_state(state)
        payload["next_allowed_actions"] = next_actions(state, tests_required=tests_required)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    contract_raw = verification_cfg.get("contract_path")
    if not contract_raw:
        payload = {
            "tool": "repro_verify_claim",
            "status": "failed",
            "reason": "verification.contract_path is required when verification is enabled",
        }
        append_history(state, "repro_verify_claim", payload)
        state["last_step"] = "repro_verify_claim"
        state["last_status"] = "verification_failed"
        set_fsm_state(state, "CLASSIFY_FAILURE")
        payload["fsm_state"] = get_fsm_state(state)
        payload["next_allowed_actions"] = next_actions(state, tests_required=tests_required)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    contract_path = Path(str(contract_raw))
    if not contract_path.is_absolute():
        contract_path = (repo_root / contract_path).resolve()
    if not contract_path.exists():
        payload = {
            "tool": "repro_verify_claim",
            "status": "failed",
            "reason": f"contract file not found: {contract_path}",
        }
        append_history(state, "repro_verify_claim", payload)
        state["last_step"] = "repro_verify_claim"
        state["last_status"] = "verification_failed"
        set_fsm_state(state, "CLASSIFY_FAILURE")
        payload["fsm_state"] = get_fsm_state(state)
        payload["next_allowed_actions"] = next_actions(state, tests_required=tests_required)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    metrics_payload = rr.load_json(metrics_path)
    contract_payload = rr.load_json(contract_path)
    verdict = evaluate_figure_contract(contract_payload, metrics_payload)
    verdict["contract_path"] = str(contract_path)
    verdict["metrics_path"] = str(metrics_path)

    verify_dir = run_root / "verification"
    verify_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = verify_dir / "figure_verdict.json"
    rr.write_json(verdict_path, verdict)

    payload = {
        "tool": "repro_verify_claim",
        "status": "success" if bool(verdict.get("success")) else "failed",
        "verdict_path": str(verdict_path),
        "verdict": verdict,
    }
    append_history(state, "repro_verify_claim", payload)
    state["last_step"] = "repro_verify_claim"
    state["last_status"] = "verification_success" if bool(verdict.get("success")) else "verification_failed"
    state["last_verdict_path"] = str(verdict_path)
    if bool(verdict.get("success")):
        set_fsm_state(state, "SUCCESS")
    else:
        set_fsm_state(state, "CLASSIFY_FAILURE")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state, tests_required=tests_required)
    write_state(state_path, state)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(verdict.get("success")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
