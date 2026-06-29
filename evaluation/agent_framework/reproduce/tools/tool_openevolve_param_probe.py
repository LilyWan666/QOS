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


ACTION = "openevolve_param_probe"
TOOL_NAME = "repro_openevolve_param_probe"

DEFAULT_PROFILE = {
    "name": "qos_fig11_proxy_multiutil_default",
    "source": "agent_recipe_default",
    "objective": {
        "name": "physical_validated_pair_selection",
        "selection_target": "multiprogramming_pair_selection",
        "ground_truth_metrics": ["effective_utilization", "proxy_estimated_fidelity"],
        "aggregation": "selected_topk_average_pareto_rank",
        "primary_score": "inv_avg_pareto_rank",
        "topk_selection": True,
        "diagnostic_metrics": ["rank_agreement", "top_rank_overlap"],
        "forbid_objective_gating_by_front_overlap": True,
        "forbid_required_rank1_overlap": True,
        "forbid_single_metric_fidelity_only": True,
        "candidate_function": "auto_discovered_qos_target",
    },
    "objective_requirements": {
        "selection_target": "multiprogramming_pair_selection",
        "primary_metric": "effective_utilization",
        "expensive_metric": "ibm_torino.hellinger_mean",
        "direct_metric_source": "physical_qpu_pair_measurement",
        "proxy_policy": "use_proxy_only_when_direct_metric_is_unavailable_or_too_expensive",
        "aggregation": "selected_topk_average_pareto_rank",
        "primary_score": "inv_avg_pareto_rank",
        "topk_selection": True,
        "diagnostic_metrics": ["rank_agreement", "top_rank_overlap"],
        "forbid_objective_gating_by_front_overlap": True,
        "forbid_required_rank1_overlap": True,
        "proxy_role": "pre_evolution_expensive_metric_substitute",
    },
    "settings": {
        "utilization_targets": "30:60:88",
        "multi_util_aggregation": "mean",
        "top_k_ratio": "0.10",
        "eval_shots": "8192",
        "external_pair_csv_allowed": "false",
        "proxy_features": "qubit_imbalance,scale_ratio,layout_span,joint_qubits,left_qubits,right_qubits,depth_ratio,depth_density,cnot_ratio,cnot_density,nonlocal_ratio,nonlocal_density,measure_ratio,measure_density,instr_ratio,instr_density,critical_depth_ratio,critical_depth_density",
        "default_proxy_feature": "qubit_imbalance",
        "seed_transfer": "small_to_large",
        "seed_source_policy": "completed_smaller_thresholds",
        "seed_target_policy": "timed_out_larger_thresholds",
        "physical_backend": "ibm_torino",
    },
    "env": {
        "OE_EVAL_UTILS": "30:60:88",
        "OE_MULTI_UTIL_AGG": "mean",
        "OE_PARETO_SECOND_METRIC": "proxy_estimated_fidelity",
        "OE_PROXY_FEATURE": "qubit_imbalance",
        "OE_PROXY_FEATURES": "qubit_imbalance,scale_ratio,layout_span,joint_qubits,left_qubits,right_qubits,depth_ratio,depth_density,cnot_ratio,cnot_density,nonlocal_ratio,nonlocal_density,measure_ratio,measure_density,instr_ratio,instr_density,critical_depth_ratio,critical_depth_density",
        "OE_TOP_K_RATIO": "0.10",
        "OE_EVAL_SHOTS": "8192",
        "OE_RESTRICT_TO_PAIR_CSV": "0",
        "OE_SEED_TRANSFER": "small_to_large",
        "OE_SEED_SOURCE_POLICY": "completed_smaller_thresholds",
        "OE_SEED_TARGET_POLICY": "timed_out_larger_thresholds",
        "OE_PHYSICAL_BACKEND": "ibm_torino",
    },
    "required_explicit": ["OE_TOP_K_RATIO"],
    "agent_overrides": {
        "OE_EVAL_SHOTS": "8192",
        "OE_RESTRICT_TO_PAIR_CSV": "0",
    },
}

RECOGNIZED_GROUPS = {
    "objective": ["simulation_validated_pair_selection", "physical_validated_pair_selection"],
    "multi_util": ["utilization_targets", "multi_util_aggregation"],
    "selection": ["top_k_ratio", "eval_shots", "external_pair_csv_allowed"],
    "proxy": ["default_proxy_feature", "proxy_features"],
    "seed_transfer": ["seed_transfer", "seed_source_policy", "seed_target_policy"],
    "provider": ["OE_LLM_PROVIDER", "OE_MAX_ITERATIONS", "OE_RUN_TAG"],
}


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
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _merge_profile(recipe: dict[str, Any]) -> dict[str, Any]:
    evolution = recipe.get("evolution") if isinstance(recipe.get("evolution"), dict) else {}
    proxy_search = evolution.get("proxy_search") if isinstance(evolution.get("proxy_search"), dict) else {}
    configured_profile = proxy_search.get("parameter_profile") if isinstance(proxy_search.get("parameter_profile"), dict) else {}

    profile = json.loads(json.dumps(DEFAULT_PROFILE))
    if configured_profile.get("name"):
        profile["name"] = str(configured_profile["name"])
    if configured_profile.get("source"):
        profile["source"] = str(configured_profile["source"])

    configured_objective = configured_profile.get("objective") if isinstance(configured_profile.get("objective"), dict) else {}
    for key, value in configured_objective.items():
        if value is not None:
            if isinstance(value, list):
                profile["objective"][str(key)] = [str(item) for item in value]
                continue
            if isinstance(value, bool):
                profile["objective"][str(key)] = bool(value)
                continue
            profile["objective"][str(key)] = str(value)

    configured_requirements = (
        configured_profile.get("objective_requirements")
        if isinstance(configured_profile.get("objective_requirements"), dict)
        else {}
    )
    for key, value in configured_requirements.items():
        if value is not None:
            if isinstance(value, list):
                profile["objective_requirements"][str(key)] = [str(item) for item in value]
                continue
            if isinstance(value, bool):
                profile["objective_requirements"][str(key)] = bool(value)
                continue
            profile["objective_requirements"][str(key)] = str(value)

    configured_settings = configured_profile.get("settings") if isinstance(configured_profile.get("settings"), dict) else {}
    for key, value in configured_settings.items():
        if value is not None:
            profile["settings"][str(key)] = str(value).lower() if isinstance(value, bool) else str(value)

    configured_env = configured_profile.get("env") if isinstance(configured_profile.get("env"), dict) else {}
    for key, value in configured_env.items():
        if value is not None:
            profile["env"][str(key)] = str(value)

    required = configured_profile.get("required_explicit")
    if isinstance(required, list):
        profile["required_explicit"] = [str(item) for item in required]

    agent_overrides = configured_profile.get("agent_overrides")
    if isinstance(agent_overrides, dict):
        profile["agent_overrides"] = {str(key): str(value) for key, value in agent_overrides.items()}

    for key, value in profile.get("agent_overrides", {}).items():
        profile["env"][key] = str(value)

    settings = profile["settings"]
    profile["env"].update(
        {
            "OE_EVAL_UTILS": str(settings["utilization_targets"]),
            "OE_MULTI_UTIL_AGG": str(settings["multi_util_aggregation"]),
            "OE_PARETO_SECOND_METRIC": "proxy_estimated_fidelity",
            "OE_PROXY_FEATURE": str(settings["default_proxy_feature"]),
            "OE_PROXY_FEATURES": str(settings["proxy_features"]),
            "OE_TOP_K_RATIO": str(settings["top_k_ratio"]),
            "OE_EVAL_SHOTS": str(settings["eval_shots"]),
            "OE_RESTRICT_TO_PAIR_CSV": "0"
            if str(settings["external_pair_csv_allowed"]).lower() in {"false", "0", "no", "off"}
            else "1",
            "OE_SEED_TRANSFER": str(settings.get("seed_transfer", "small_to_large")),
            "OE_SEED_SOURCE_POLICY": str(settings.get("seed_source_policy", "completed_smaller_thresholds")),
            "OE_SEED_TARGET_POLICY": str(settings.get("seed_target_policy", "timed_out_larger_thresholds")),
            "OE_PHYSICAL_BACKEND": str(settings.get("physical_backend", "ibm_torino")),
        }
    )

    profile["resolution_order"] = [
        "current recipe evolution.proxy_search.parameter_profile objective/settings",
        "QOS-Agent built-in Fig11 proxy defaults",
        "OpenEvolve compatibility env generated from agent settings",
    ]
    return profile


def _failure_category(state: dict[str, Any] | None) -> str:
    if not isinstance(state, dict):
        return ""
    candidates = [
        state.get("failure_category"),
        ((state.get("last_classification") or {}).get("category")),
        ((state.get("last_classification") or {}).get("failure_category")),
        ((state.get("last_failure_classification") or {}).get("category")),
        ((state.get("last_classify_failure") or {}).get("classification") or {}).get("category"),
        ((state.get("last_classify_failure") or {}).get("category")),
    ]
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    history = state.get("history") or []
    if isinstance(history, list):
        for item in reversed(history[-12:]):
            if not isinstance(item, dict):
                continue
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else item
            category = (
                payload.get("failure_category")
                or ((payload.get("classification") or {}).get("category") if isinstance(payload.get("classification"), dict) else None)
                or payload.get("category")
            )
            text = str(category or "").strip()
            if text:
                return text
    return ""


def _proxy_on_timeout(recipe: dict[str, Any]) -> bool:
    budget = recipe.get("simulation_budget") if isinstance(recipe.get("simulation_budget"), dict) else {}
    return bool(budget.get("proxy_on_timeout"))


def _resolve_objective(profile: dict[str, Any], recipe: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    requirements = profile.get("objective_requirements") if isinstance(profile.get("objective_requirements"), dict) else {}
    settings = profile.get("settings") if isinstance(profile.get("settings"), dict) else {}
    category = _failure_category(state)
    needs_proxy = category == "simulation_too_expensive" and _proxy_on_timeout(recipe)
    expensive_metric = str(requirements.get("expensive_metric") or "ibm_torino.hellinger_mean")
    direct_source = str(requirements.get("direct_metric_source") or "physical_qpu_pair_measurement")
    if needs_proxy:
        resolved_second_metric = "proxy_estimated_fidelity"
        proxy_feature = str(settings.get("default_proxy_feature") or "qubit_imbalance")
        reason = "strict pair-level simulation was classified as too expensive, so the expensive fidelity-like metric is replaced by a pre-evolution proxy calibrated against available evidence"
        mode = "proxy_substitution"
    else:
        resolved_second_metric = str(requirements.get("expensive_metric") or "ibm_torino.hellinger_mean")
        proxy_feature = None
        reason = "direct pair-level simulation is available or has not been classified as too expensive"
        mode = "direct_simulation"
    return {
        "mode": mode,
        "primary_metric": str(requirements.get("primary_metric") or "effective_utilization"),
        "expensive_metric": expensive_metric,
        "direct_metric_source": direct_source,
        "resolved_second_metric": resolved_second_metric,
        "proxy_feature": proxy_feature,
        "proxy_role": "pre_evolution_expensive_metric_substitute" if needs_proxy else "not_used",
        "proxy_substitutes_metric": expensive_metric if needs_proxy else None,
        "proxy_is_evolved_function": False,
        "calibration_or_validation_label_metric": expensive_metric,
        "trigger": category or "none",
        "proxy_on_timeout": _proxy_on_timeout(recipe),
        "strict_fig11_success": False,
        "reason": reason,
    }


def _apply_objective_resolution(profile: dict[str, Any], resolution: dict[str, Any]) -> None:
    env = profile.get("env") if isinstance(profile.get("env"), dict) else {}
    env["OE_PARETO_SECOND_METRIC"] = str(resolution.get("resolved_second_metric") or "fidelity")
    if resolution.get("proxy_feature"):
        env["OE_PROXY_FEATURE"] = str(resolution["proxy_feature"])
    profile["env"] = env
    profile["objective_resolution"] = resolution


def _training_data_schema(recipe: dict[str, Any]) -> str:
    evolution = recipe.get("evolution") if isinstance(recipe.get("evolution"), dict) else {}
    proxy_search = evolution.get("proxy_search") if isinstance(evolution.get("proxy_search"), dict) else {}
    return str(proxy_search.get("training_data_schema") or "ibm_torino_physical_pair_records")


def _schema_for_profile(profile: dict[str, Any]) -> dict[str, Any]:
    env = profile.get("env") if isinstance(profile.get("env"), dict) else {}
    parameters: dict[str, dict[str, Any]] = {}
    for name, value in sorted(env.items()):
        parameters[str(name)] = {
            "name": str(name),
            "default": str(value),
            "source": profile.get("source", "agent_recipe_default"),
        }
    return {
        "parameter_count": len(parameters),
        "parameters": parameters,
        "recognized_groups": RECOGNIZED_GROUPS,
    }


def build_probe(repo_root: Path, recipe_path: Path | None = None, state: dict[str, Any] | None = None) -> dict[str, Any]:
    recipe = _load_json(recipe_path) if recipe_path else {}
    profile = _merge_profile(recipe)
    objective_resolution = _resolve_objective(profile, recipe, state)
    _apply_objective_resolution(profile, objective_resolution)
    env = profile.get("env") if isinstance(profile.get("env"), dict) else {}
    missing = [
        name for name in profile.get("required_explicit", [])
        if not str(env.get(name, "")).strip()
    ]

    payload = {
        "success": not missing and bool(env),
        "execution_mode": "openevolve_param_probe",
        "not_fig11_strict_success": True,
        "agent_owned_profile": True,
        "manual_reference_used": False,
        "recipe_reference": str(recipe_path) if recipe_path else None,
        "training_data_schema": _training_data_schema(recipe),
        "objective_requirements": profile.get("objective_requirements") or {},
        "objective_resolution": objective_resolution,
        "parameter_schema": _schema_for_profile(profile),
        "parameter_profile": profile,
        "warnings": [],
    }
    if missing:
        payload["warnings"].append(f"missing required agent profile parameters: {', '.join(missing)}")
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
    artifact_dir = run_root / "openevolve_param_probe"
    payload = build_probe(repo_root, recipe_path, state=state)
    payload["tool"] = TOOL_NAME
    payload["action"] = ACTION
    payload["fsm_state_before"] = get_fsm_state(state)
    payload["artifact_path"] = str(artifact_dir / "openevolve_param_probe.json")
    _write_json(artifact_dir / "openevolve_param_probe.json", payload)

    state["last_openevolve_param_probe"] = payload
    state["last_step"] = TOOL_NAME
    state["last_status"] = "openevolve_param_probe_succeeded" if payload.get("success") else "openevolve_param_probe_failed"
    set_fsm_state(state, "APPLY_FIX" if payload.get("success") else "CLASSIFY_FAILURE")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
