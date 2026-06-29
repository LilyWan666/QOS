#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

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
from tool_openevolve_proxy_search import (  # noqa: E402
    _build_training_data,
    _feature_values,
    _ground_truth_metrics_path,
    _parse_csv_list,
    _physical_records_path,
    _profile_env,
    _profile_objective_resolution,
    _rank_for_correlation,
    _simulation_memory_path,
    _spearman_values,
)


ACTION = "proxy_feature_semantic_validate"
TOOL_NAME = "repro_proxy_feature_semantic_validate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output-root", default="temp/agent_framework/reproduce/tools/runs")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except Exception:
        return default
    return val if math.isfinite(val) else default


def _pearson_from_values(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _spearman_values(left, right)


def _label_for_record(record: dict[str, Any]) -> float | None:
    for key in ("hellinger_mean", "fidelity_like_label", "relative_fidelity", "simulation_relative_fidelity"):
        if record.get(key) is not None:
            return _safe_float(record.get(key))
    return None


def _values_from_records(records: list[dict[str, Any]], feature: str) -> tuple[list[float], list[float]]:
    values: list[float] = []
    labels: list[float] = []
    for record in records:
        features = record.get("features") if isinstance(record.get("features"), dict) else {}
        if features.get(feature) is None:
            continue
        label = _label_for_record(record)
        if label is None:
            continue
        try:
            values.append(float(features.get(feature)))
            labels.append(float(label))
        except Exception:
            continue
    return values, labels


def _corr_for_records(records: list[dict[str, Any]], feature: str, direction: str) -> dict[str, Any]:
    values, labels = _values_from_records(records, feature)
    if direction == "inverse":
        values = [-value for value in values]
    corr = _pearson_from_values(values, labels)
    distinct = len({round(float(value), 12) for value in values})
    return {
        "record_count": len(values),
        "distinct_value_count": distinct,
        "spearman": corr,
    }


def _group_records_by_application(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for key in ("left_application", "right_application"):
            app = str(record.get(key) or "").strip()
            if app:
                grouped.setdefault(app, []).append(record)
    return grouped


def _candidate_features(state: dict[str, Any], profile: dict[str, str]) -> tuple[list[str], str, list[str]]:
    warnings: list[str] = []
    proposal = state.get("last_proxy_metric_semantic_proposal")
    features: list[str] = []
    if isinstance(proposal, dict) and proposal.get("success"):
        for item in proposal.get("candidates") or []:
            if isinstance(item, dict) and item.get("materialized_feature_name"):
                features.append(str(item.get("materialized_feature_name")))
    if features:
        return list(dict.fromkeys(features)), "llm_semantic_metric_proposal", warnings
    configured = _parse_csv_list(
        profile.get("OE_PROXY_FEATURES", ""),
        [],
    )
    if configured and os.getenv("REPRO_PROXY_VALIDATE_ALLOW_CONFIG_FALLBACK", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        warnings.append("using configured feature fallback because no LLM semantic proposal was present")
        return list(dict.fromkeys(configured)), "configured_feature_fallback", warnings
    warnings.append("missing LLM semantic proxy metric proposal; run proxy_metric_semantic_propose first")
    return [], "missing_llm_semantic_metric_proposal", warnings


def _score_feature(
    thresholds: dict[str, list[dict[str, Any]]],
    split: dict[str, Any],
    feature: str,
) -> dict[str, Any]:
    train_thresholds = [str(item) for item in split.get("train_thresholds", [])]
    validation_thresholds = [str(item) for item in split.get("validation_thresholds", [])]
    train_values, train_labels = _feature_values(thresholds, train_thresholds, feature)
    direct = _spearman_values(train_values, train_labels) if len(train_values) >= 2 else None
    inverse = _spearman_values([-value for value in train_values], train_labels) if len(train_values) >= 2 else None
    if inverse is not None and direct is not None and inverse > direct:
        direction = "inverse"
        train_corr = inverse
    else:
        direction = "direct"
        train_corr = direct

    validation_records = [
        record
        for threshold in validation_thresholds
        for record in thresholds.get(str(threshold), [])
    ]
    all_records = [
        record
        for records in thresholds.values()
        for record in records
    ]
    validation = _corr_for_records(validation_records, feature, direction)
    per_threshold = {
        str(threshold): _corr_for_records(records, feature, direction)
        for threshold, records in sorted(thresholds.items())
    }
    per_app = {
        app: _corr_for_records(records, feature, direction)
        for app, records in sorted(_group_records_by_application(all_records).items())
    }
    app_corrs = [
        float(item["spearman"])
        for item in per_app.values()
        if item.get("spearman") is not None and item.get("record_count", 0) >= 2
    ]
    threshold_corrs = [
        float(item["spearman"])
        for item in per_threshold.values()
        if item.get("spearman") is not None and item.get("record_count", 0) >= 2
    ]
    all_values, _ = _values_from_records(all_records, feature)
    coverage = len(all_values) / max(len(all_records), 1)
    stability_terms = []
    if validation.get("spearman") is not None:
        stability_terms.append(max(-1.0, min(1.0, float(validation["spearman"]))))
    if app_corrs:
        stability_terms.append(sum(app_corrs) / len(app_corrs))
    if threshold_corrs:
        stability_terms.append(sum(threshold_corrs) / len(threshold_corrs))
    stability_score = sum(stability_terms) / len(stability_terms) if stability_terms else -1.0
    variance_ok = len({round(float(value), 12) for value in all_values}) >= 2
    return {
        "feature": feature,
        "direction": direction,
        "train_spearman": train_corr,
        "validation_spearman": validation.get("spearman"),
        "validation_record_count": validation.get("record_count"),
        "validation_distinct_value_count": validation.get("distinct_value_count"),
        "per_threshold": per_threshold,
        "per_application": per_app,
        "per_application_mean_spearman": sum(app_corrs) / len(app_corrs) if app_corrs else None,
        "per_application_min_spearman": min(app_corrs) if app_corrs else None,
        "per_threshold_mean_spearman": sum(threshold_corrs) / len(threshold_corrs) if threshold_corrs else None,
        "per_threshold_min_spearman": min(threshold_corrs) if threshold_corrs else None,
        "coverage": coverage,
        "distinct_value_count": len({round(float(value), 12) for value in all_values}),
        "variance_ok": variance_ok,
        "semantic_generalization_score": stability_score * coverage if variance_ok else -1.0,
    }


def build_payload(repo_root: Path, state: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    profile = _profile_env(state)
    objective_resolution = _profile_objective_resolution(state)
    physical_backend_filter = str(
        os.getenv(
            "REPRO_OPENEVOLVE_PHYSICAL_BACKEND",
            profile.get("OE_PHYSICAL_BACKEND") or "ibm_torino",
        )
    )
    ground_truth_metrics = _ground_truth_metrics_path(state, repo_root)
    simulation_memory = _simulation_memory_path(state, repo_root)
    physical_records = _physical_records_path(state, repo_root, artifact_dir)
    training_payload = _build_training_data(
        ground_truth_metrics,
        simulation_memory,
        physical_records,
        physical_backend=physical_backend_filter,
        repo_root=repo_root,
    )
    candidate_features, candidate_source, warnings = _candidate_features(state, profile)
    thresholds = training_payload.get("thresholds") if isinstance(training_payload.get("thresholds"), dict) else {}
    split = training_payload.get("split") if isinstance(training_payload.get("split"), dict) else {}
    scored = [_score_feature(thresholds, split, feature) for feature in candidate_features]
    try:
        min_validation = float(os.getenv("REPRO_PROXY_MIN_VALIDATION_SPEARMAN", "0.20"))
    except ValueError:
        min_validation = 0.20
    try:
        min_coverage = float(os.getenv("REPRO_PROXY_MIN_FEATURE_COVERAGE", "0.80"))
    except ValueError:
        min_coverage = 0.80
    try:
        max_bundle_size = int(os.getenv("REPRO_PROXY_FEATURE_BUNDLE_SIZE", "3"))
    except ValueError:
        max_bundle_size = 3
    selectable = [
        item
        for item in scored
        if item.get("variance_ok")
        and float(item.get("coverage") or 0.0) >= min_coverage
        and item.get("validation_spearman") is not None
    ]
    stable = [
        item
        for item in selectable
        if float(item.get("validation_spearman") or 0.0) >= min_validation
    ]
    selected_pool = stable or selectable
    selected_pool = sorted(
        selected_pool,
        key=lambda item: (
            -float(item.get("semantic_generalization_score") or -1.0),
            -float(item.get("validation_spearman") or -1.0),
            -float(item.get("train_spearman") or -1.0),
            str(item.get("feature")),
        ),
    )
    selected_bundle = selected_pool[: max(1, max_bundle_size)]
    selected_features = [str(item["feature"]) for item in selected_bundle]
    rejected = [
        {
            "feature": item.get("feature"),
            "reason": (
                "no_variance"
                if not item.get("variance_ok")
                else "coverage_below_threshold"
                if float(item.get("coverage") or 0.0) < min_coverage
                else "missing_validation_correlation"
                if item.get("validation_spearman") is None
                else f"validation_spearman_below_{min_validation}"
            ),
            "validation_spearman": item.get("validation_spearman"),
            "coverage": item.get("coverage"),
        }
        for item in scored
        if item.get("feature") not in set(selected_features)
    ]
    payload = {
        "success": bool(selected_features and training_payload.get("status") == "pair_record_ground_truth"),
        "execution_mode": ACTION,
        "validation_mode": "semantic_feature_bundle_evidence_gate",
        "manual_proxy_source_used": False,
        "proxy_role": "pre_evolution_expensive_metric_substitute",
        "objective_resolution_mode": objective_resolution.get("mode"),
        "candidate_source": candidate_source,
        "candidate_features": candidate_features,
        "selected_feature": selected_features[0] if selected_features else None,
        "selected_feature_bundle": selected_bundle,
        "selected_feature_names": selected_features,
        "rejected_features": rejected,
        "feature_scores": scored,
        "selection_policy": {
            "mode": "llm_proposal_then_deterministic_cross_app_validation",
            "min_validation_spearman": min_validation,
            "min_feature_coverage": min_coverage,
            "max_bundle_size": max_bundle_size,
            "stable_feature_count": len(stable),
            "selectable_feature_count": len(selectable),
            "fallback_to_best_available_when_below_threshold": bool(not stable and selectable),
        },
        "training_data": {
            "source_kind": training_payload.get("source_kind"),
            "status": training_payload.get("status"),
            "record_count": training_payload.get("record_count"),
            "train_record_count": training_payload.get("train_record_count"),
            "validation_record_count": training_payload.get("validation_record_count"),
            "split": split,
            "physical_backend": training_payload.get("physical_backend"),
            "label_metric": training_payload.get("label_metric"),
            "physical_records_path": str(physical_records) if physical_records else None,
            "ground_truth_metrics_path": str(ground_truth_metrics) if ground_truth_metrics else None,
            "simulation_memory_path": str(simulation_memory) if simulation_memory else None,
        },
        "provenance": {
            "llm_selects_semantic_feature_candidates": candidate_source == "llm_semantic_metric_proposal",
            "tool_materializes_and_validates_features": True,
            "uses_manual_proxy_choice": False,
            "raw_pair_records_not_embedded_in_prompt": True,
        },
        "warnings": warnings,
    }
    if training_payload.get("status") != "pair_record_ground_truth":
        payload["warnings"].append("missing pair-level trusted records for feature validation")
    if not selected_features:
        payload["warnings"].append("no proposed feature passed the semantic validation gate")
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
    artifact_dir = run_root / "proxy_feature_semantic_validate"
    payload = build_payload(repo_root, state, artifact_dir)
    payload["tool"] = TOOL_NAME
    payload["action"] = ACTION
    payload["fsm_state_before"] = get_fsm_state(state)
    metrics_path = artifact_dir / "proxy_feature_semantic_validate.metrics.json"
    _write_json(metrics_path, payload)
    payload["metrics_path"] = str(metrics_path)
    state["last_proxy_feature_semantic_validation"] = payload
    state["last_step"] = TOOL_NAME
    state["last_status"] = "proxy_feature_semantic_validate_succeeded" if payload.get("success") else "proxy_feature_semantic_validate_failed"
    set_fsm_state(state, "APPLY_FIX")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
