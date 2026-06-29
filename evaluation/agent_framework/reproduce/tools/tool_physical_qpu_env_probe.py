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
from physical_qpu_proxy_utils import app_from_name, artifact_rel, qubits_from_name, write_json  # noqa: E402
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


ACTION = "physical_qpu_env_probe"
TOOL_NAME = "repro_physical_qpu_env_probe"
CONTRACT_REL_PATH = Path(
    "evaluation/agent_framework/reproduce/examples/figure_contracts/"
    "qos_physical_qpu_env_probe.contract.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output-root", default="temp/agent_framework/reproduce/tools/runs")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_record(raw: dict[str, Any], *, backend: str, util_label: str, source_file: Path, repo_root: Path) -> dict[str, Any] | None:
    hellinger = raw.get("hellinger_mean")
    if hellinger is None:
        return None
    name_1 = str(raw.get("name_1") or "")
    name_2 = str(raw.get("name_2") or "")
    left_qubits = qubits_from_name(name_1)
    right_qubits = qubits_from_name(name_2)
    joint_qubits = left_qubits + right_qubits
    try:
        effective_util = float(raw.get("csv_effective_utilization") or 0.0)
    except Exception:
        effective_util = 0.0
    return {
        "backend": str(raw.get("backend") or backend),
        "util_label": util_label,
        "pair_label": f"{name_1}__{name_2}",
        "name_1": name_1,
        "name_2": name_2,
        "left_application": app_from_name(name_1),
        "right_application": app_from_name(name_2),
        "left_qubits": left_qubits,
        "right_qubits": right_qubits,
        "joint_qubits": joint_qubits,
        "effective_utilization": effective_util,
        "hellinger_mean": float(hellinger),
        "job_id": raw.get("job_id"),
        "shots": int(raw.get("shots") or 0),
        "optimization_level": raw.get("optimization_level"),
        "layout_mode": raw.get("layout_mode"),
        "collected_at_utc": raw.get("collected_at_utc"),
        "source_file": artifact_rel(repo_root, source_file),
        "metric_source": "hellinger_mean",
        "provenance_kind": "physical_qpu_measurement",
    }


def build_probe(repo_root: Path, artifact_dir: Path) -> dict[str, Any]:
    manifest_path = repo_root / "qpu_env/physical_qpu_jobs/manifest.json"
    payload: dict[str, Any] = {
        "success": False,
        "execution_mode": "physical_qpu_env_probe",
        "not_fig11_strict_success": True,
        "strict_fig11_simulation_substitute": False,
        "manifest_path": artifact_rel(repo_root, manifest_path),
        "provenance": {
            "kind": "physical_qpu_measurement",
            "metric_source": "hellinger_mean",
            "strict_fig11_simulation_substitute": False,
        },
        "backend_summaries": [],
        "warnings": [],
    }
    if not manifest_path.exists():
        payload["warnings"].append("missing qpu_env/physical_qpu_jobs/manifest.json")
        return payload
    manifest = _load_json(manifest_path)
    records: list[dict[str, Any]] = []
    missing_or_null = 0
    count_mismatches: list[dict[str, Any]] = []
    for backend, util_map in sorted((manifest.get("records") or {}).items()):
        for util_label, spec in sorted((util_map or {}).items()):
            fidelity_dir = repo_root / str(spec.get("fidelity_dir") or "")
            files = sorted(fidelity_dir.glob("*.json")) if fidelity_dir.exists() else []
            expected = int(spec.get("pair_json_count") or 0)
            if len(files) != expected:
                count_mismatches.append(
                    {
                        "backend": backend,
                        "util_label": util_label,
                        "expected": expected,
                        "actual": len(files),
                    }
                )
            before = len(records)
            for file_path in files:
                try:
                    normalized = _normalize_record(
                        _load_json(file_path),
                        backend=backend,
                        util_label=util_label,
                        source_file=file_path,
                        repo_root=repo_root,
                    )
                except Exception as exc:
                    payload["warnings"].append(f"failed to parse {file_path}: {type(exc).__name__}: {exc}")
                    continue
                if normalized is None:
                    missing_or_null += 1
                    continue
                records.append(normalized)
            payload["backend_summaries"].append(
                {
                    "backend": backend,
                    "util_label": util_label,
                    "expected_pair_json_count": expected,
                    "actual_pair_json_count": len(files),
                    "normalized_record_count": len(records) - before,
                    "fidelity_dir": artifact_rel(repo_root, fidelity_dir),
                }
            )
    records_path = artifact_dir / "physical_qpu_pair_records.json"
    write_json(records_path, {"records": records, "manifest": manifest})
    payload.update(
        {
            "success": bool(records and not count_mismatches),
            "physical_records_path": str(records_path),
            "record_count": len(records),
            "null_hellinger_record_count": missing_or_null,
            "count_mismatches": count_mismatches,
            "shots": manifest.get("shots"),
            "metric_source": manifest.get("metric_source"),
            "backends": sorted((manifest.get("records") or {}).keys()),
        }
    )
    if count_mismatches:
        payload["warnings"].append("manifest pair_json_count did not match files on disk")
    if manifest.get("metric_source") != "hellinger_mean":
        payload["warnings"].append("physical QPU manifest metric_source is not hellinger_mean")
        payload["success"] = False
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
    artifact_dir = run_root / "physical_qpu_env_probe"
    payload = build_probe(repo_root, artifact_dir)
    payload["tool"] = TOOL_NAME
    payload["action"] = ACTION
    payload["fsm_state_before"] = get_fsm_state(state)
    metrics_path = artifact_dir / "physical_qpu_env_probe.metrics.json"
    write_json(metrics_path, payload)
    contract_path = repo_root / CONTRACT_REL_PATH
    payload["contract_path"] = artifact_rel(repo_root, contract_path)
    payload["metrics_path"] = str(metrics_path)
    if contract_path.exists():
        verdict = evaluate_figure_contract(_load_json(contract_path), payload)
        payload["contract_verdict"] = verdict
        payload["success"] = bool(payload.get("success") and verdict.get("success"))
    write_json(metrics_path, payload)

    state["last_physical_qpu_env_probe"] = payload
    state["last_physical_qpu_records_path"] = payload.get("physical_records_path")
    state["last_step"] = TOOL_NAME
    state["last_status"] = "physical_qpu_env_probe_succeeded" if payload.get("success") else "physical_qpu_env_probe_failed"
    set_fsm_state(state, "APPLY_FIX")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
