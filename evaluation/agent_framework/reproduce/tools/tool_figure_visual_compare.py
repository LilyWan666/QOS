#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import struct
import sys
from pathlib import Path
from typing import Any

REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

import run_reproduce as rr
from figure_claim import _resolve_path
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


def png_info(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {"exists": False, "valid_png": False, "reason": str(exc)}
    if len(data) < 33 or not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR":
        return {"exists": True, "valid_png": False, "size_bytes": len(data)}
    width, height = struct.unpack(">II", data[16:24])
    return {
        "exists": True,
        "valid_png": True,
        "size_bytes": len(data),
        "width": int(width),
        "height": int(height),
    }


def get_path(payload: dict[str, Any], path: str) -> tuple[bool, Any]:
    return _resolve_path(payload, path)


def list_contains_all(actual: Any, expected: list[Any]) -> bool:
    if not isinstance(actual, list):
        return False
    actual_norm = {str(item) for item in actual}
    return all(str(item) in actual_norm for item in expected)


def evaluate_expectation(figure_id: str, spec: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    output_path_expr = str(spec.get("output_path", f"output_files.{figure_id}"))
    exists, output_path_raw = get_path(metrics, output_path_expr)
    output_path = Path(str(output_path_raw)).resolve() if exists and output_path_raw else None
    info = png_info(output_path) if output_path else {"exists": False, "valid_png": False, "reason": "missing output path"}
    checks.append({
        "id": f"{figure_id}_png_valid",
        "type": "png_valid",
        "passed": bool(info.get("exists")) and bool(info.get("valid_png")),
        "expected": "valid PNG artifact",
        "actual": info,
        "severity": "high",
        "recommended_action": "repro_run_once",
    })

    min_size = spec.get("min_png_size_bytes")
    if isinstance(min_size, int):
        size = int(info.get("size_bytes") or 0)
        checks.append({
            "id": f"{figure_id}_png_size",
            "type": "png_size_gte",
            "passed": size >= min_size,
            "expected": min_size,
            "actual": size,
            "severity": "medium",
            "recommended_action": "source_fix",
        })

    for item in spec.get("metric_expectations", []) or []:
        if not isinstance(item, dict):
            continue
        check_id = str(item.get("id", item.get("path", "metric_expectation")))
        path = str(item.get("path", ""))
        exists, actual = get_path(metrics, path)
        expected = item.get("value")
        check_type = str(item.get("type", "equals"))
        passed = False
        reason = ""
        if not exists:
            reason = "metric path not found"
        elif check_type == "equals":
            passed = actual == expected
            reason = "ok" if passed else "value mismatch"
        elif check_type == "bool_equals":
            passed = bool(actual) is bool(expected)
            reason = "ok" if passed else "boolean value mismatch"
        elif check_type == "gte":
            try:
                passed = float(actual) >= float(expected)
                reason = "ok" if passed else "actual < expected"
            except (TypeError, ValueError):
                reason = "non-numeric comparison"
        elif check_type == "lte":
            try:
                passed = float(actual) <= float(expected)
                reason = "ok" if passed else "actual > expected"
            except (TypeError, ValueError):
                reason = "non-numeric comparison"
        elif check_type == "numeric_between":
            min_value = item.get("min")
            max_value = item.get("max")
            expected = {"min": min_value, "max": max_value}
            try:
                passed = float(min_value) <= float(actual) <= float(max_value)
                reason = "ok" if passed else "actual outside [min, max]"
            except (TypeError, ValueError):
                reason = "non-numeric comparison"
        elif check_type == "numeric_abs_diff_gte":
            other_path = str(item.get("other_path", ""))
            min_diff = item.get("min_diff")
            other_exists, other_actual = get_path(metrics, other_path)
            expected = {"other_path": other_path, "min_abs_diff": min_diff}
            if not other_exists:
                reason = "comparison metric path not found"
            else:
                try:
                    diff = abs(float(actual) - float(other_actual))
                    passed = diff >= float(min_diff)
                    reason = "ok" if passed else "absolute difference below minimum"
                    expected["actual_abs_diff"] = diff
                except (TypeError, ValueError):
                    reason = "non-numeric comparison"
        elif check_type == "contains_all":
            values = item.get("values", [])
            passed = isinstance(values, list) and list_contains_all(actual, values)
            reason = "ok" if passed else "missing expected entries"
            expected = values
        elif check_type == "in_set":
            values = item.get("values", [])
            passed = isinstance(values, list) and actual in values
            reason = "ok" if passed else "value not in expected set"
            expected = values
        elif check_type == "not_in_set":
            values = item.get("values", [])
            passed = isinstance(values, list) and actual not in values
            reason = "ok" if passed else "value is explicitly disallowed"
            expected = {"not_in": values}
        elif check_type == "contains_substring":
            expected = str(item.get("value", ""))
            passed = expected in str(actual)
            reason = "ok" if passed else "substring not found"
        elif check_type == "not_contains_substring":
            expected = str(item.get("value", ""))
            passed = expected not in str(actual)
            reason = "ok" if passed else "forbidden substring found"
        else:
            reason = f"unsupported metric expectation type: {check_type}"
        checks.append({
            "id": check_id,
            "type": f"metric_{check_type}",
            "path": path,
            "passed": passed,
            "expected": expected,
            "actual": actual if exists else None,
            "reason": reason,
            "severity": str(item.get("severity", "high")),
            "recommended_action": str(item.get("recommended_action", "source_fix")),
        })

    group_count_path = spec.get("group_count_path")
    total_bars_path = spec.get("total_bars_path")
    bars_per_group = spec.get("bars_per_group")
    if isinstance(group_count_path, str) and isinstance(total_bars_path, str) and isinstance(bars_per_group, int):
        group_exists, group_count = get_path(metrics, group_count_path)
        total_exists, total_bars = get_path(metrics, total_bars_path)
        expected_total = None
        passed = False
        if group_exists and total_exists:
            try:
                expected_total = int(group_count) * bars_per_group
                passed = int(total_bars) == expected_total
            except (TypeError, ValueError):
                passed = False
        checks.append({
            "id": f"{figure_id}_bars_per_group",
            "type": "bars_per_group",
            "passed": passed,
            "expected": {"group_count_path": group_count_path, "bars_per_group": bars_per_group, "total": expected_total},
            "actual": {"group_count": group_count if group_exists else None, "total_bars": total_bars if total_exists else None},
            "severity": "high",
            "recommended_action": str(spec.get("structure_mismatch_action", "build_original_runner")),
        })

    return checks


def evaluate_visual_expectations(contract: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    visual = contract.get("visual_expectations") or {}
    figures = visual.get("figures") if isinstance(visual, dict) else None
    if not isinstance(figures, dict) or not figures:
        return {
            "success": False,
            "skipped": True,
            "reason": "contract has no visual_expectations.figures",
            "figure_results": [],
            "differences": [],
            "recommended_actions": ["verify_claim", "terminal_failed"],
        }

    figure_results: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    recommended: list[str] = []
    for figure_id, raw_spec in figures.items():
        if not isinstance(raw_spec, dict):
            continue
        checks = evaluate_expectation(str(figure_id), raw_spec, metrics)
        failed = [check for check in checks if not check.get("passed")]
        for check in failed:
            action = str(check.get("recommended_action") or "source_fix")
            if action not in recommended:
                recommended.append(action)
            differences.append({
                "figure": str(figure_id),
                "check_id": check.get("id"),
                "severity": check.get("severity", "high"),
                "expected": check.get("expected"),
                "actual": check.get("actual"),
                "reason": check.get("reason", "visual expectation mismatch"),
                "recommended_action": action,
            })
        figure_results.append({
            "figure": str(figure_id),
            "success": not failed,
            "checks": checks,
            "paper_evidence": raw_spec.get("paper_evidence"),
            "expected_structure": raw_spec.get("expected_structure"),
        })

    success = not differences
    if success:
        recommended = ["render_artifacts", "done"]
    return {
        "success": success,
        "skipped": False,
        "figure_results": figure_results,
        "differences": differences,
        "recommended_actions": recommended,
    }


def _constant_string(value: ast.AST) -> str | None:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _constant_bool(value: ast.AST) -> bool | None:
    if isinstance(value, ast.Constant) and isinstance(value.value, bool):
        return value.value
    return None


def _contains_name(node: ast.AST, names: set[str]) -> bool:
    return any(isinstance(child, ast.Name) and child.id in names for child in ast.walk(node))


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for item in target.elts:
            out.update(_target_names(item))
        return out
    return set()


def _dict_key_value_pairs(node: ast.Dict) -> list[tuple[str, ast.AST]]:
    pairs: list[tuple[str, ast.AST]] = []
    for key, value in zip(node.keys, node.values):
        if key is None:
            continue
        key_text = _constant_string(key)
        if key_text:
            pairs.append((key_text, value))
    return pairs


def fitted_metric_static_guard(state: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Reject fitted/synthetic Fig. 11 metric code paths."""
    derivation = (((metrics.get("metric_derivations") or {}).get("figure_11b") or {}).get("effective_utilization") or {})
    if not derivation:
        return {"success": True, "skipped": True, "reason": "no figure_11b effective-utilization derivation"}

    candidates: list[Path] = []
    runner_raw = str(state.get("original_runner_path") or "").strip()
    if runner_raw:
        candidates.append(Path(runner_raw))

    metrics_raw = (((state.get("last_manifest") or {}).get("artifacts")) or {}).get("metrics")
    metrics_path = Path(str(metrics_raw or ""))
    if metrics_path.exists():
        for parent in metrics_path.parents:
            generated = parent / "workspace" / ".repro_generated" / "original_pipeline_runner.py"
            if generated.exists():
                candidates.append(generated)
                break

    paths: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            paths.append(resolved)

    if not paths:
        return {
            "success": False,
            "skipped": False,
            "violations": [{"reason": "original generated runner path not found for static guard"}],
        }

    violations: list[dict[str, Any]] = []
    forbidden_source_types = {"synthetic_curve", "hard_coded_target"}
    forbidden_names = {"qos_lift", "synthetic_qos_lift", "_local_marginal_count_fidelity"}
    utilization_outputs = {"baseline_util", "qos_util", "no_util"}
    fidelity_outputs = {"baseline_fid", "qos_fid", "no_fid"}
    fitted_inputs = {"utilization_pressure", "threshold"}

    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            violations.append({"path": str(path), "reason": "python source parse failed", "error": repr(exc)})
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(
                    {
                        "path": str(path),
                        "lineno": getattr(node, "lineno", None),
                        "reason": f"forbidden synthetic/fitted symbol used: {node.id}",
                    }
                )
            elif isinstance(node, ast.Assign):
                assigned: set[str] = set()
                for target in node.targets:
                    assigned.update(_target_names(target))
                if assigned & utilization_outputs and _contains_name(node.value, fitted_inputs):
                    violations.append(
                        {
                            "path": str(path),
                            "lineno": getattr(node, "lineno", None),
                            "reason": "effective-utilization output is derived from utilization/threshold curve input",
                            "targets": sorted(assigned & utilization_outputs),
                        }
                    )
                if assigned & fidelity_outputs and _contains_name(node.value, {"threshold"}):
                    violations.append(
                        {
                            "path": str(path),
                            "lineno": getattr(node, "lineno", None),
                            "reason": "fidelity output is derived from threshold/fitted curve input",
                            "targets": sorted(assigned & fidelity_outputs),
                        }
                    )
            elif isinstance(node, ast.Dict):
                for key, value in _dict_key_value_pairs(node):
                    text = _constant_string(value)
                    flag = _constant_bool(value)
                    if key == "source_type" and text in forbidden_source_types:
                        violations.append(
                            {
                                "path": str(path),
                                "lineno": getattr(node, "lineno", None),
                                "reason": "forbidden effective-utilization source_type",
                                "key": key,
                                "value": text,
                            }
                        )
                    if key in {"fitted_curve", "synthetic_qos_lift"} and flag is True:
                        violations.append(
                            {
                                "path": str(path),
                                "lineno": getattr(node, "lineno", None),
                                "reason": "forbidden fitted/synthetic provenance flag",
                                "key": key,
                                "value": flag,
                            }
                        )

    return {
        "success": not violations,
        "skipped": False,
        "paths": [str(path) for path in paths],
        "violations": violations,
    }


def fig11b_static_guard(state: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    guard = fitted_metric_static_guard(state, metrics)
    guard["metric"] = "figure_11b.effective_utilization"
    return guard


def fig11a_static_guard(state: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    guard = fitted_metric_static_guard(state, metrics)
    guard["metric"] = "figure_11a.fidelity"
    return guard


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
        "figure_visual_compare",
        tests_required=tests_required,
    )
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_figure_visual_compare",
            action="figure_visual_compare",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_figure_visual_compare", payload)
        state["last_step"] = "repro_figure_visual_compare"
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
            "tool": "repro_figure_visual_compare",
            "status": "failed",
            "reason": "missing metrics artifact in state.last_manifest.artifacts.metrics",
        }
        append_history(state, "repro_figure_visual_compare", payload)
        state["last_step"] = "repro_figure_visual_compare"
        state["last_status"] = "figure_visual_compare_failed"
        set_fsm_state(state, "CLASSIFY_FAILURE")
        payload["fsm_state"] = get_fsm_state(state)
        payload["next_allowed_actions"] = next_actions(state, tests_required=tests_required)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    verification_cfg = recipe.get("verification", {})
    contract_raw = verification_cfg.get("contract_path")
    if not contract_raw:
        payload = {
            "tool": "repro_figure_visual_compare",
            "status": "failed",
            "reason": "verification.contract_path is required for figure visual comparison",
        }
        append_history(state, "repro_figure_visual_compare", payload)
        state["last_step"] = "repro_figure_visual_compare"
        state["last_status"] = "figure_visual_compare_failed"
        set_fsm_state(state, "CLASSIFY_FAILURE")
        payload["fsm_state"] = get_fsm_state(state)
        payload["next_allowed_actions"] = next_actions(state, tests_required=tests_required)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    metrics_path = Path(str(metrics_path_raw)).resolve()
    contract_path = Path(str(contract_raw))
    if not contract_path.is_absolute():
        contract_path = (repo_root / contract_path).resolve()
    metrics = rr.load_json(metrics_path)
    contract = rr.load_json(contract_path)
    comparison = evaluate_visual_expectations(contract, metrics)
    static_guards = {
        "fig11a_fidelity": fig11a_static_guard(state, metrics),
        "fig11b_effective_utilization": fig11b_static_guard(state, metrics),
    }
    comparison["static_guards"] = static_guards
    for guard_name, static_guard in static_guards.items():
        if bool(static_guard.get("success")):
            continue
        figure = "figure_11a" if guard_name.startswith("fig11a") else "figure_11b"
        guard_difference = {
            "figure": figure,
            "check_id": f"{guard_name}_static_guard",
            "severity": "high",
            "expected": "no fitted/synthetic metric code path",
            "actual": static_guard.get("violations", []),
            "reason": f"static guard found forbidden {figure} metric implementation",
            "recommended_action": "build_original_runner",
        }
        comparison.setdefault("differences", []).append(guard_difference)
        recommended_actions = comparison.setdefault("recommended_actions", [])
        if "build_original_runner" not in recommended_actions:
            recommended_actions.insert(0, "build_original_runner")
        comparison["success"] = False
    comparison["contract_path"] = str(contract_path)
    comparison["metrics_path"] = str(metrics_path)

    visual_dir = run_root / "verification"
    visual_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = visual_dir / "figure_visual_comparison.json"
    rr.write_json(comparison_path, comparison)

    payload = {
        "tool": "repro_figure_visual_compare",
        "status": "success" if bool(comparison.get("success")) else "failed",
        "comparison_path": str(comparison_path),
        "visual_comparison": comparison,
    }
    append_history(state, "repro_figure_visual_compare", payload)
    state["last_step"] = "repro_figure_visual_compare"
    state["last_status"] = "figure_visual_compare_success" if bool(comparison.get("success")) else "figure_visual_compare_failed"
    state["last_visual_comparison_path"] = str(comparison_path)
    state["last_visual_comparison"] = comparison
    if bool(comparison.get("success")):
        set_fsm_state(state, "SUCCESS")
    else:
        state["last_classification"] = {
            "category": "figure_visual_mismatch",
            "recoverable": True,
            "evidence": comparison.get("differences", []),
        }
        set_fsm_state(state, "CLASSIFY_FAILURE")
    payload["fsm_state"] = get_fsm_state(state)
    if bool(comparison.get("success")):
        # Avoid repeatedly re-running render/compare after the visual and
        # semantic expectations pass.
        payload["next_allowed_actions"] = ["done"]
    else:
        payload["next_allowed_actions"] = next_actions(state, tests_required=tests_required)
    write_state(state_path, state)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(comparison.get("success")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
