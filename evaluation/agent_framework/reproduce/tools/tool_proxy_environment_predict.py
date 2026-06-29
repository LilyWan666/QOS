#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

from figure_claim import evaluate_figure_contract  # noqa: E402
from physical_qpu_proxy_utils import (  # noqa: E402
    artifact_rel,
    load_score_function,
    resolve_existing_path,
    summarize_physical_records,
    write_json,
)
from repro_toolkit import (  # noqa: E402
    append_history,
    assert_action_allowed,
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
from tool_openevolve_proxy_search import _build_initial_program, _selected_qos_target  # noqa: E402
from tool_openevolve_target_probe import build_probe  # noqa: E402


ACTION = "proxy_environment_predict"
TOOL_NAME = "repro_proxy_environment_predict"
CONTRACT_REL_PATH = Path(
    "evaluation/agent_framework/reproduce/examples/figure_contracts/"
    "qos_proxy_environment_predict.contract.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output-root", default="temp/agent_framework/reproduce/tools/runs")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records_path(state: dict[str, Any], repo_root: Path) -> Path | None:
    path = resolve_existing_path(state.get("last_physical_qpu_records_path"), repo_root)
    if path:
        return path
    probe = state.get("last_physical_qpu_env_probe") or {}
    if isinstance(probe, dict):
        return resolve_existing_path(probe.get("physical_records_path"), repo_root)
    return None


def _best_program_from_state(state: dict[str, Any], repo_root: Path) -> Path | None:
    path = resolve_existing_path(state.get("last_openevolve_evolved_program"), repo_root)
    if path:
        return path
    proxy = state.get("last_openevolve_proxy_search") or {}
    metrics_path = resolve_existing_path(proxy.get("metrics_path"), repo_root) if isinstance(proxy, dict) else None
    if metrics_path:
        metrics = _load_json(metrics_path)
        output_dir = resolve_existing_path(((metrics.get("evolution_config") or {}).get("output_dir")), repo_root)
        if output_dir:
            return resolve_existing_path(output_dir / "best" / "best_program.py", repo_root)
    return None


def _baseline_program(state: dict[str, Any], repo_root: Path, artifact_dir: Path) -> Path | None:
    probe = state.get("last_openevolve_target_probe")
    if not isinstance(probe, dict) or not probe.get("success"):
        probe = build_probe(repo_root)
    qos_target = _selected_qos_target(state, probe)
    if not qos_target:
        return None
    path = artifact_dir / "baseline_initial_program.py"
    path.write_text(_build_initial_program(qos_target), encoding="utf-8")
    return path


def build_prediction(repo_root: Path, artifact_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    records_path = _records_path(state, repo_root)
    phase = "post_evolution" if _best_program_from_state(state, repo_root) is not None else "baseline"
    program_path = _best_program_from_state(state, repo_root) or _baseline_program(state, repo_root, artifact_dir)
    payload: dict[str, Any] = {
        "success": False,
        "execution_mode": "proxy_environment_predict",
        "phase": phase,
        "not_fig11_strict_success": True,
        "strict_fig11_simulation_substitute": False,
        "hardware_calibrated": False,
        "prediction": {
            "method": "score_rank_summary_over_physical_qpu_environment_records",
            "physical_metric_source": "hellinger_mean",
            "physical_records_reference": str(records_path) if records_path else None,
            "proxy_program_reference": str(program_path) if program_path else None,
        },
        "warnings": [],
    }
    if records_path is None:
        payload["warnings"].append("missing physical QPU records; run physical_qpu_env_probe first")
        return payload
    if program_path is None:
        payload["warnings"].append("missing baseline/evolved proxy program; run semantic target selection or proxy search first")
        return payload
    records_payload = _load_json(records_path)
    records = [record for record in records_payload.get("records", []) if isinstance(record, dict)]
    try:
        score_fn = load_score_function(program_path)
    except Exception as exc:
        payload["warnings"].append(f"failed to load proxy score function: {type(exc).__name__}: {exc}")
        return payload
    summary = summarize_physical_records(records, score_fn, repo_root=repo_root)
    payload["prediction"].update(summary)
    payload["success"] = bool(records and summary.get("groups"))
    write_json(artifact_dir / f"{phase}_physical_proxy_summary.json", summary)
    return payload


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, ACTION)
    if not is_allowed:
        payload = blocked_action_payload(tool=TOOL_NAME, action=ACTION, state=state, next_allowed_actions=allowed_actions)
        append_history(state, TOOL_NAME, payload)
        state["last_step"] = TOOL_NAME
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    recipe_path = resolve_recipe_path(args.recipe, repo_root)
    run_root = ensure_run_root(recipe_path.stem, state, Path(args.output_root).resolve())
    artifact_dir = run_root / "proxy_environment_predict"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = build_prediction(repo_root, artifact_dir, state)
    payload["tool"] = TOOL_NAME
    payload["action"] = ACTION
    payload["fsm_state_before"] = get_fsm_state(state)
    metrics_path = artifact_dir / f"proxy_environment_predict.{payload.get('phase', 'unknown')}.metrics.json"
    write_json(metrics_path, payload)
    contract_path = repo_root / CONTRACT_REL_PATH
    payload["contract_path"] = artifact_rel(repo_root, contract_path)
    payload["metrics_path"] = str(metrics_path)
    if contract_path.exists():
        verdict = evaluate_figure_contract(_load_json(contract_path), payload)
        payload["contract_verdict"] = verdict
        payload["success"] = bool(payload.get("success") and verdict.get("success"))
    write_json(metrics_path, payload)

    history = state.setdefault("proxy_environment_predictions", [])
    if isinstance(history, list):
        history.append(payload)
    state["last_proxy_environment_predict"] = payload
    state["last_step"] = TOOL_NAME
    state["last_status"] = "proxy_environment_predict_succeeded" if payload.get("success") else "proxy_environment_predict_failed"
    set_fsm_state(state, "APPLY_FIX")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
