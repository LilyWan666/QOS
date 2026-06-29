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

from repro_toolkit import (  # noqa: E402
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


ACTION = "simulation_checkpoint_analyze"
TOOL_NAME = "repro_simulation_checkpoint_analyze"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output-root", default="temp/agent_framework/reproduce/tools/runs")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _artifact_rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def _candidate_checkpoint_paths(run_root: Path) -> list[Path]:
    attempts = run_root / "attempts"
    paths: list[Path] = []
    if attempts.exists():
        paths.extend(sorted(attempts.glob("attempt_*_run_once/partial_simulation_checkpoint.json")))
        paths.extend(sorted(attempts.glob("attempt_*_run_once/metrics.json")))
    return paths


def _selected_pair_record_count(payload: dict[str, Any]) -> int:
    by_threshold = payload.get("selected_pair_simulation_records_by_threshold") or {}
    count = 0
    if not isinstance(by_threshold, dict):
        return 0
    for threshold_payload in by_threshold.values():
        if not isinstance(threshold_payload, dict):
            continue
        for mode in ("baseline", "qos", "fig11c_qos"):
            records = ((threshold_payload.get(mode) or {}).get("records") or [])
            if isinstance(records, list):
                count += len(records)
    return count


def _no_mp_completed_thresholds(payload: dict[str, Any]) -> list[str]:
    records = payload.get("no_mp_target_size_records") or {}
    if not isinstance(records, dict):
        return []
    return sorted(
        str(key)
        for key, value in records.items()
        if (
            isinstance(value, dict)
            and value.get("fidelity") is not None
            and not value.get("deferred_for_proxy_memory")
        )
    )


def _selected_pair_completed_thresholds(payload: dict[str, Any]) -> list[str]:
    by_threshold = payload.get("selected_pair_simulation_records_by_threshold") or {}
    out: list[str] = []
    if not isinstance(by_threshold, dict):
        return out
    for threshold, threshold_payload in by_threshold.items():
        if not isinstance(threshold_payload, dict):
            continue
        qos_records = ((threshold_payload.get("qos") or {}).get("records") or [])
        if qos_records:
            out.append(str(threshold))
    return sorted(out)


def _threshold_norm(value: Any) -> float:
    try:
        raw = float(str(value).strip().rstrip("%"))
    except Exception:
        return -1.0
    return raw / 100.0 if raw > 1.0 else raw


def _target_thresholds(all_thresholds: list[Any], completed: list[str]) -> list[str]:
    completed_norm = {_threshold_norm(item) for item in completed}
    targets: list[str] = []
    for threshold in all_thresholds:
        norm = _threshold_norm(threshold)
        if norm < 0:
            continue
        if not any(abs(norm - seen) < 1e-9 for seen in completed_norm):
            targets.append(str(threshold))
    return targets


def _score_checkpoint(path: Path, payload: dict[str, Any]) -> tuple[int, float]:
    return (
        _selected_pair_record_count(payload) * 1000 + len(_no_mp_completed_thresholds(payload)),
        path.stat().st_mtime if path.exists() else 0.0,
    )


def _best_checkpoint(run_root: Path) -> tuple[Path | None, dict[str, Any]]:
    best_path: Path | None = None
    best_payload: dict[str, Any] = {}
    best_score = (-1, -1.0)
    for path in _candidate_checkpoint_paths(run_root):
        payload = _load_json(path)
        if not payload:
            continue
        score = _score_checkpoint(path, payload)
        if score > best_score:
            best_score = score
            best_path = path
            best_payload = payload
    return best_path, best_payload


def build_memory(repo_root: Path, run_root: Path) -> dict[str, Any]:
    checkpoint_path, checkpoint = _best_checkpoint(run_root)
    selected_pair_thresholds = _selected_pair_completed_thresholds(checkpoint)
    no_mp_thresholds = _no_mp_completed_thresholds(checkpoint)
    all_thresholds = checkpoint.get("thresholds") or []
    completed_for_transfer = selected_pair_thresholds or no_mp_thresholds
    record_count = _selected_pair_record_count(checkpoint)
    memory = {
        "schema": "qos_agent.simulation_memory.v1",
        "success": bool(checkpoint_path and (record_count > 0 or no_mp_thresholds)),
        "execution_mode": "simulation_checkpoint_analyze",
        "not_fig11_strict_success": True,
        "source_checkpoint": _artifact_rel(repo_root, checkpoint_path) if checkpoint_path else None,
        "checkpoint_kind": checkpoint.get("checkpoint_kind"),
        "simulation_shots": checkpoint.get("simulation_shots"),
        "qpu_qubits": checkpoint.get("qpu_qubits"),
        "simulation_noise_model": checkpoint.get("simulation_noise_model") or {},
        "completed_thresholds": completed_for_transfer,
        "completed_no_mp_thresholds": no_mp_thresholds,
        "completed_selected_pair_thresholds": selected_pair_thresholds,
        "timed_out_or_missing_thresholds": _target_thresholds(all_thresholds, completed_for_transfer),
        "selected_pair_record_count": record_count,
        "selected_pair_simulation_records_by_threshold": checkpoint.get("selected_pair_simulation_records_by_threshold") or {},
        "no_mp_target_size_records": checkpoint.get("no_mp_target_size_records") or {},
        "application_pair_metric_records": checkpoint.get("application_pair_metric_records") or {},
        "simulation_timing_by_threshold": checkpoint.get("simulation_timing_by_threshold") or [],
        "simulation_timing_by_threshold_map": checkpoint.get("simulation_timing_by_threshold_map") or {},
        "usable_for_proxy_seed": bool(record_count > 0),
        "usable_as_scale_memory": bool(no_mp_thresholds),
        "seed_transfer": {
            "enabled": True,
            "strategy": "small_to_large",
            "source_thresholds": completed_for_transfer,
            "target_thresholds": _target_thresholds(all_thresholds, completed_for_transfer),
            "uses_simulation_grounded_sources": bool(record_count > 0 or no_mp_thresholds),
            "uses_pair_level_ground_truth": bool(record_count > 0),
            "strict_fig11_success": False,
        },
        "warnings": [],
    }
    if not checkpoint_path:
        memory["warnings"].append("no partial_simulation_checkpoint.json or metrics.json found in attempts")
    if checkpoint_path and record_count == 0:
        memory["warnings"].append("checkpoint contains no selected pair simulation records; proxy ranking verification needs pair-level records")
    return memory


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
    artifact_dir = run_root / "simulation_checkpoint_analyze"
    memory = build_memory(repo_root, run_root)
    memory["tool"] = TOOL_NAME
    memory["action"] = ACTION
    memory["fsm_state_before"] = get_fsm_state(state)
    memory_path = artifact_dir / "simulation_memory.json"
    memory["artifact_path"] = str(memory_path)
    _write_json(memory_path, memory)

    state["last_simulation_memory"] = memory
    state["last_simulation_memory_path"] = str(memory_path)
    state["last_step"] = TOOL_NAME
    state["last_status"] = "simulation_checkpoint_analyze_succeeded" if memory.get("success") else "simulation_checkpoint_analyze_failed"
    set_fsm_state(state, "APPLY_FIX")
    memory["fsm_state"] = get_fsm_state(state)
    memory["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, memory)
    write_state(state_path, state)
    print(json.dumps(memory, indent=2, sort_keys=True))
    return 0 if memory.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
