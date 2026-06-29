#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

from figure_claim import evaluate_figure_contract  # noqa: E402
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


ACTION = "openevolve_proxy_verify"
TOOL_NAME = "repro_openevolve_proxy_verify"
CONTRACT_REL_PATH = Path(
    "evaluation/agent_framework/reproduce/examples/figure_contracts/"
    "qos_openevolve_proxy_verify.contract.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument(
        "--output-root",
        default="temp/agent_framework/reproduce/tools/runs",
        help="Root directory for verification artifacts.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: Any, repo_root: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path if path.exists() else None


def _ground_truth_metrics_path(state: dict[str, Any], repo_root: Path) -> Path | None:
    env_path = _resolve_path(os.getenv("REPRO_OPENEVOLVE_GROUND_TRUTH_METRICS"), repo_root)
    if env_path:
        return env_path
    manifest_path = (((state.get("last_manifest") or {}).get("artifacts") or {}).get("metrics"))
    path = _resolve_path(manifest_path, repo_root)
    if path:
        return path
    raw = str(state.get("last_fig11_metrics_path") or "").strip()
    return _resolve_path(raw, repo_root)


def _proxy_metrics_path(state: dict[str, Any], repo_root: Path) -> Path | None:
    proxy = state.get("last_openevolve_proxy_search") or {}
    return _resolve_path(proxy.get("metrics_path"), repo_root)


def _load_proxy_metrics(state: dict[str, Any], repo_root: Path) -> tuple[Path | None, dict[str, Any]]:
    path = _proxy_metrics_path(state, repo_root)
    if path is None:
        return None, {}
    return path, _load_json(path)


def _proxy_artifact_path(proxy_metrics: dict[str, Any], repo_root: Path, key: str) -> Path | None:
    raw = ((proxy_metrics.get("evolution_config") or {}).get(key))
    return _resolve_path(raw, repo_root)


def _best_program_path(proxy_metrics: dict[str, Any], repo_root: Path, state: dict[str, Any] | None = None) -> Path | None:
    if state is not None:
        collect = state.get("last_openevolve_slurm_collect")
        mutation = (collect or {}).get("mutation_verification") if isinstance(collect, dict) else {}
        collect_success = bool(
            isinstance(collect, dict)
            and collect.get("success")
            and (mutation or {}).get("mutation_applied")
        )
        path = _resolve_path((collect or {}).get("evolved_program") if isinstance(collect, dict) else None, repo_root)
        if collect_success and path is not None:
            return path
        if os.getenv("REPRO_OPENEVOLVE_VERIFY_ALLOW_UNCOLLECTED_BEST", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            path = _resolve_path(state.get("last_openevolve_evolved_program"), repo_root)
            if path is not None:
                return path
    output_dir = _proxy_artifact_path(proxy_metrics, repo_root, "output_dir")
    if output_dir is None:
        return None
    if os.getenv("REPRO_OPENEVOLVE_VERIFY_ALLOW_UNCOLLECTED_BEST", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return _resolve_path(output_dir / "best" / "best_program.py", repo_root)
    return None


class _Backend:
    def __init__(self, num_qubits: Any):
        self.num_qubits = max(float(num_qubits or 1.0), 1.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except Exception:
        val = default
    if not math.isfinite(val):
        val = default
    return val


def _bounded01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


class _Qernel:
    def __init__(self, features: dict[str, Any], side: str):
        self.features = features
        self.side = side

    def get_metadata(self) -> dict[str, float]:
        left_qubits = float(self.features.get("left_qubits", 0.0) or 0.0)
        right_qubits = float(self.features.get("right_qubits", 0.0) or 0.0)
        qubits = left_qubits if self.side == "left" else right_qubits
        suffix = "1" if self.side == "left" else "2"
        depth = float(self.features.get(f"depth_{suffix}") or max(10.0 * qubits, 1.0))
        instr = _safe_float(self.features.get(f"instr_{suffix}"), depth)
        nonlocal_gates = _safe_float(self.features.get(f"nonlocal_{suffix}"), 0.0)
        measurements = _safe_float(self.features.get(f"measure_{suffix}"), qubits)
        cnot_gates = _safe_float(self.features.get(f"cnot_{suffix}"), 0.0)
        return {
            "depth": depth,
            "num_qubits": qubits,
            "num_clbits": qubits,
            "num_nonlocal_gates": nonlocal_gates,
            "num_connected_components": float(self.features.get("connected_components", 1.0) or 1.0),
            "number_instructions": instr,
            "num_measurements": measurements,
            "num_cnot_gates": cnot_gates,
            "program_communication": float(self.features.get("program_communication", 0.0) or 0.0),
            "liveness": float(self.features.get("utilization_pressure", 0.0) or 0.0),
            "parallelism": _bounded01(
                self.features.get(f"parallelism_{suffix}"),
                1.0 - (depth / max(instr, 1.0)),
            ),
            "measurement": _bounded01(
                self.features.get(f"measurement_{suffix}", 0.0),
                measurements / max(depth, 1.0),
            ),
            "entanglement_ratio": _bounded01(
                self.features.get(f"entanglement_ratio_{suffix}"),
                nonlocal_gates / max(instr, 1.0),
            ),
            "critical_depth": float(self.features.get(f"critical_depth_{suffix}", depth) or depth),
        }


class _CandidateSelf:
    def effective_utilization(self, q1, q2, backend):
        return float(q1.features.get("effective_utilization_percent", 0.0) or 0.0)

    def entanglementComparison(self, q1, q2):
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        return _bounded01((1.0 - meta1["entanglement_ratio"]) * (1.0 - meta2["entanglement_ratio"]))

    def measurementComparison(self, q1, q2):
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        return _bounded01((1.0 - meta1["measurement"]) * (1.0 - meta2["measurement"]))

    def parallelismComparison(self, q1, q2):
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        return _bounded01((1.0 - meta1["parallelism"]) * (1.0 - meta2["parallelism"]))

    def depthComparison(self, q1, q2):
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        return _bounded01(math.exp(-0.05 * abs(float(meta1["depth"]) - float(meta2["depth"]))))

    def fidelityComparison(self, q1, q2):
        return max(0.0, min(1.0, float(q1.features.get("fidelity", 0.0) or 0.0)))


def _load_score_function(path: Path):
    spec = importlib.util.spec_from_file_location("qos_agent_evolved_proxy", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import evolved proxy program: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    score_pair_features = getattr(module, "score_pair_features", None)
    if callable(score_pair_features):
        return score_pair_features
    get_matching_score = getattr(module, "get_matching_score", None)
    if callable(get_matching_score):
        def score_from_features(features: dict[str, Any]) -> float:
            q1 = _Qernel(features, "left")
            q2 = _Qernel(features, "right")
            backend = _Backend(features.get("selected_qubits", features.get("joint_qubits", 1.0)))
            return float(get_matching_score(_CandidateSelf(), q1, q2, backend, False, []))
        return score_from_features
    raise ValueError(f"evolved proxy program does not define score_pair_features or get_matching_score: {path}")


def _rank(values: list[tuple[str, float]]) -> dict[str, int]:
    ordered = sorted(values, key=lambda item: (-float(item[1]), item[0]))
    return {label: index + 1 for index, (label, _) in enumerate(ordered)}


def _spearman(left: dict[str, int], right: dict[str, int]) -> float | None:
    labels = sorted(set(left) & set(right))
    n = len(labels)
    if n < 2:
        return None
    d2 = sum((left[label] - right[label]) ** 2 for label in labels)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def _collect_pair_records(metrics: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    by_threshold = metrics.get("selected_pair_simulation_records_by_threshold") or {}
    if not isinstance(by_threshold, dict):
        return out
    for threshold, payload in by_threshold.items():
        if not isinstance(payload, dict):
            continue
        records = ((payload.get("qos") or {}).get("records") or [])
        normalized = []
        for record in records:
            if not isinstance(record, dict):
                continue
            item = dict(record)
            item.setdefault("selected_qubits", payload.get("selected_qubits"))
            normalized.append(item)
        if normalized:
            out[str(threshold)] = normalized
    return out


def _training_pair_records(training_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    thresholds = training_data.get("thresholds") or {}
    if not isinstance(thresholds, dict):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for threshold, records in thresholds.items():
        if isinstance(records, list) and records:
            out[str(threshold)] = [record for record in records if isinstance(record, dict)]
    return out


def _record_ground_truth_score(record: dict[str, Any]) -> float:
    for key in ("fidelity_like_label", "hellinger_mean", "physical_hellinger_mean", "simulation_relative_fidelity", "relative_fidelity"):
        if record.get(key) is not None:
            return float(record.get(key) or 0.0)
    return 0.0


def _record_features(record: dict[str, Any]) -> dict[str, Any]:
    features = record.get("features")
    if isinstance(features, dict):
        return features
    joint_qubits = int(record.get("joint_qubits") or 0)
    selected = int(record.get("selected_qubits") or joint_qubits or 1)
    return {
        "matching_score": float(record.get("matching_score") or record.get("selection_score") or 0.0),
        "selection_score": float(record.get("selection_score") or 0.0),
        "effective_utilization": float(record.get("effective_utilization") or 0.0),
        "effective_utilization_percent": float(record.get("effective_utilization_percent") or 0.0),
        "fidelity": float(record.get("fidelity") or 0.0),
        "solo_fidelity": float(record.get("solo_fidelity") or 0.0),
        "entanglement": float(record.get("entanglement", 0.0) or 0.0),
        "measurement": float(record.get("measurement", 0.0) or 0.0),
        "parallelism": float(record.get("parallelism", 0.0) or 0.0),
        "entanglement_ratio": float(record.get("entanglement_ratio", 0.0) or 0.0),
        "joint_qubits": float(joint_qubits),
        "selected_qubits": float(selected),
        "depth_ratio": float(joint_qubits) / max(float(selected), 1.0),
        "left_qubits": float(record.get("left_qubits") or 0.0),
        "right_qubits": float(record.get("right_qubits") or 0.0),
        "layout_span": 0.0,
    }


def _pair_rank_comparison(pair_records: dict[str, list[dict[str, Any]]], score_fn) -> dict[str, Any] | None:
    if not pair_records:
        return None
    thresholds = []
    correlations = []
    for threshold, records in sorted(pair_records.items()):
        sim_values = []
        proxy_values = []
        for idx, record in enumerate(records):
            label = str(record.get("pair_label") or f"{threshold}:{idx}")
            sim_values.append((label, _record_ground_truth_score(record)))
            try:
                proxy_score = float(score_fn(_record_features(record)))
                if math.isnan(proxy_score) or math.isinf(proxy_score):
                    proxy_score = -1e9
            except Exception:
                proxy_score = -1e9
            proxy_values.append((label, proxy_score))
        sim_rank = _rank(sim_values)
        proxy_rank = _rank(proxy_values)
        corr = _spearman(sim_rank, proxy_rank)
        if corr is not None and not math.isnan(corr):
            correlations.append(corr)
        topk = min(5, len(sim_rank), len(proxy_rank))
        sim_top = {label for label, _ in sorted(sim_values, key=lambda item: (-item[1], item[0]))[:topk]}
        proxy_top = {label for label, _ in sorted(proxy_values, key=lambda item: (-item[1], item[0]))[:topk]}
        thresholds.append(
            {
                "threshold": threshold,
                "pair_count": len(records),
                "spearman": corr,
                "topk": topk,
                "topk_overlap": len(sim_top & proxy_top) / max(float(topk), 1.0),
            }
        )
    return {
        "status": "pair_rank_verified",
        "method": "spearman_and_topk_overlap_between_evolved_proxy_score_and_pair_level_ground_truth_label",
        "thresholds": thresholds,
        "mean_spearman": sum(correlations) / len(correlations) if correlations else None,
    }


def _aggregate_comparison(metrics: dict[str, Any]) -> dict[str, Any] | None:
    rows = metrics.get("relative_fidelity_by_utilization") or []
    if not isinstance(rows, list) or not rows:
        return None
    return {
        "status": "aggregate_only",
        "method": "aggregate_selected_pair_relative_fidelity_by_utilization_no_pair_records_available",
        "thresholds": [
            {
                "threshold": row.get("threshold"),
                "selected_qubits": row.get("selected_qubits"),
                "baseline_relative_fidelity": row.get("baseline_relative_fidelity"),
                "qos_relative_fidelity": row.get("qos_relative_fidelity"),
                "baseline_pair_count": row.get("baseline_pair_count"),
                "qos_pair_count": row.get("qos_pair_count"),
            }
            for row in rows
            if isinstance(row, dict)
        ],
    }


def build_verification(
    *,
    repo_root: Path,
    artifact_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    gt_path = _ground_truth_metrics_path(state, repo_root)
    proxy_path, proxy_metrics = _load_proxy_metrics(state, repo_root)
    training_path = _proxy_artifact_path(proxy_metrics, repo_root, "training_data")
    best_program = _best_program_path(proxy_metrics, repo_root, state)
    collect = state.get("last_openevolve_slurm_collect")
    mutation = (collect or {}).get("mutation_verification") if isinstance(collect, dict) else {}
    ground_truth_reference = gt_path or training_path
    ground_truth_kind = "fig11_metrics" if gt_path else ("training_pair_records" if training_path else "missing")
    payload: dict[str, Any] = {
        "success": False,
        "execution_mode": "openevolve_proxy_verify",
        "not_fig11_strict_success": True,
        "verification": {
            "eval_shots_target": 8192,
            "simulation_ground_truth_reference": str(ground_truth_reference) if ground_truth_reference else None,
            "proxy_ground_truth_label_metric": None,
            "ground_truth_kind": ground_truth_kind,
            "proxy_metrics_reference": str(proxy_path) if proxy_path else None,
            "training_data_reference": str(training_path) if training_path else None,
            "evolved_program_reference": str(best_program) if best_program else None,
            "openevolve_collect_success": bool(isinstance(collect, dict) and collect.get("success")),
            "mutation_applied": bool((mutation or {}).get("mutation_applied")),
            "fig11_strict_contract_satisfied": False,
            "rank_comparison": {
                "status": "missing_ground_truth",
                "method": "unavailable",
            },
        },
        "warnings": [],
    }
    if proxy_path is None:
        payload["warnings"].append("missing OpenEvolve proxy-search metrics; run openevolve_proxy_search first")
    if training_path is None:
        payload["warnings"].append("missing OpenEvolve simulation-pair training data")
    if not isinstance(collect, dict):
        payload["warnings"].append("missing OpenEvolve collect result; run openevolve_slurm_collect before verify")
    elif not collect.get("success"):
        payload["warnings"].append("OpenEvolve collect did not succeed; verifier refuses stale or no-op best_program.py")
    elif not (mutation or {}).get("mutation_applied"):
        payload["warnings"].append("OpenEvolve collect found no code mutation; verifier requires a changed evolved program")
    if best_program is None:
        payload["warnings"].append("missing verified evolved best_program.py from successful OpenEvolve collect")
    if training_path is None or best_program is None:
        return payload

    metrics = _load_json(gt_path) if gt_path is not None else {}
    training_data = _load_json(training_path)
    payload["verification"]["proxy_ground_truth_label_metric"] = (
        training_data.get("expensive_label_metric")
        or training_data.get("label_metric")
    )
    try:
        score_fn = _load_score_function(best_program)
    except Exception as exc:
        payload["warnings"].append(f"failed to load evolved proxy score function: {type(exc).__name__}: {exc}")
        return payload

    training_records = _training_pair_records(training_data)
    rank_comparison = _pair_rank_comparison(training_records, score_fn)
    if rank_comparison is None:
        rank_comparison = _aggregate_comparison(metrics)
        if rank_comparison is not None:
            payload["warnings"].append(
                "pair-level proxy training records are unavailable; verifier used aggregate-only evidence"
            )
    if rank_comparison is None:
        payload["warnings"].append("ground-truth metrics do not contain pair or aggregate relative-fidelity evidence")
        return payload

    payload["verification"]["rank_comparison"] = rank_comparison
    payload["verification"]["ground_truth_success"] = bool(metrics.get("success"))
    payload["verification"]["ground_truth_has_pair_records"] = bool(training_records)
    payload["verification"]["training_data_status"] = training_data.get("status")
    payload["verification"]["training_record_count"] = training_data.get("record_count")
    payload["verification"]["training_thresholds"] = sorted(training_records)
    payload["verification"]["ground_truth_has_aggregate"] = _aggregate_comparison(metrics) is not None
    payload["verification"]["ground_truth_simulation_shots"] = metrics.get("simulation_shots")
    payload["verification"]["ground_truth_metric_provenance_shots"] = (
        ((metrics.get("metric_provenance") or {}).get("relative_fidelity") or {}).get("shots")
    )
    payload["success"] = bool(
        rank_comparison.get("status") in {"pair_rank_verified", "aggregate_only"}
        and (
            bool(metrics.get("success"))
            or training_data.get("status") == "pair_record_ground_truth"
        )
    )
    _write_json(artifact_dir / "ground_truth_metrics.preview.json", {
        "path": str(ground_truth_reference) if ground_truth_reference else None,
        "ground_truth_kind": ground_truth_kind,
        "success": metrics.get("success"),
        "relative_fidelity_utilization_count": len(metrics.get("relative_fidelity_by_utilization") or []),
        "has_pair_records": payload["verification"]["ground_truth_has_pair_records"],
        "training_data_status": training_data.get("status"),
        "training_record_count": training_data.get("record_count"),
    })
    return payload


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, ACTION)
    if not is_allowed:
        payload = blocked_action_payload(
            tool=TOOL_NAME,
            action=ACTION,
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, TOOL_NAME, payload)
        state["last_step"] = TOOL_NAME
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    recipe_path = resolve_recipe_path(args.recipe, repo_root)
    recipe_name = recipe_path.stem
    run_root = ensure_run_root(recipe_name, state, Path(args.output_root).resolve())
    artifact_dir = run_root / "openevolve_proxy_verify"

    payload = build_verification(repo_root=repo_root, artifact_dir=artifact_dir, state=state)
    payload["tool"] = TOOL_NAME
    payload["action"] = ACTION
    payload["fsm_state_before"] = get_fsm_state(state)

    metrics_path = artifact_dir / "openevolve_proxy_verify.metrics.json"
    _write_json(metrics_path, payload)
    contract_path = (repo_root / CONTRACT_REL_PATH).resolve()
    payload["contract_path"] = str(contract_path)
    payload["metrics_path"] = str(metrics_path)
    if contract_path.exists():
        contract = _load_json(contract_path)
        verdict = evaluate_figure_contract(contract, payload)
        verdict["contract_path"] = str(contract_path)
        verdict["metrics_path"] = str(metrics_path)
        payload["contract_verdict"] = verdict
        payload["success"] = bool(payload.get("success") and verdict.get("success"))
    else:
        payload["success"] = False
        payload["contract_verdict"] = {"success": False, "reason": f"contract not found: {contract_path}"}
    _write_json(metrics_path, payload)

    state["last_openevolve_proxy_verify"] = payload
    state["last_step"] = TOOL_NAME
    state["last_status"] = "openevolve_proxy_verify_succeeded" if payload.get("success") else "openevolve_proxy_verify_failed"
    set_fsm_state(state, "APPLY_FIX")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, payload)
    write_state(state_path, state)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
