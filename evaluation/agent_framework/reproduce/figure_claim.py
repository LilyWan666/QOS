#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_tokens(path: str) -> list[str]:
    if not path:
        return []
    tokens: list[str] = []
    current = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if current:
                tokens.append(current)
                current = ""
            i += 1
            continue
        if ch == "[":
            if current:
                tokens.append(current)
                current = ""
            end = path.find("]", i + 1)
            if end == -1:
                raise ValueError(f"invalid path token (missing ']'): {path!r}")
            tokens.append(path[i : end + 1])
            i = end + 1
            continue
        current += ch
        i += 1
    if current:
        tokens.append(current)
    return tokens


def _resolve_path(payload: dict[str, Any], path: str) -> tuple[bool, Any]:
    node: Any = payload
    for token in _path_tokens(path):
        if token.startswith("[") and token.endswith("]"):
            if not isinstance(node, list):
                return False, None
            raw_idx = token[1:-1]
            if not raw_idx.isdigit():
                return False, None
            idx = int(raw_idx)
            if idx < 0 or idx >= len(node):
                return False, None
            node = node[idx]
            continue
        if not isinstance(node, dict) or token not in node:
            return False, None
        node = node[token]
    return True, node


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _evaluate_single_check(
    check: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    check_id = str(check.get("id", "unnamed_check"))
    check_type = str(check.get("type", "exists"))
    path = str(check.get("path", ""))

    result: dict[str, Any] = {
        "id": check_id,
        "type": check_type,
        "path": path,
        "passed": False,
        "reason": "",
        "actual": None,
    }

    exists, actual = _resolve_path(metrics, path)
    result["actual"] = actual if exists else None

    if check_type == "exists":
        result["passed"] = exists
        result["reason"] = "ok" if exists else "metric path not found"
        return result

    if not exists:
        result["reason"] = "metric path not found"
        return result

    if check_type in {"equals", "bool_equals"}:
        expected = check.get("value")
        result["expected"] = expected
        result["passed"] = actual == expected
        result["reason"] = "ok" if result["passed"] else "value mismatch"
        return result

    if check_type == "in_set":
        values = check.get("values", [])
        if not isinstance(values, list):
            result["reason"] = "contract error: values must be a list"
            return result
        result["expected"] = values
        result["passed"] = actual in values
        result["reason"] = "ok" if result["passed"] else "value not in expected set"
        return result

    if check_type == "numeric_gte":
        expected = _as_float(check.get("value"))
        actual_num = _as_float(actual)
        result["expected"] = expected
        if expected is None or actual_num is None:
            result["reason"] = "non-numeric comparison"
            return result
        result["passed"] = actual_num >= expected
        result["reason"] = "ok" if result["passed"] else "actual < expected"
        return result

    if check_type == "numeric_lte":
        expected = _as_float(check.get("value"))
        actual_num = _as_float(actual)
        result["expected"] = expected
        if expected is None or actual_num is None:
            result["reason"] = "non-numeric comparison"
            return result
        result["passed"] = actual_num <= expected
        result["reason"] = "ok" if result["passed"] else "actual > expected"
        return result

    if check_type == "numeric_between":
        min_value = _as_float(check.get("min"))
        max_value = _as_float(check.get("max"))
        actual_num = _as_float(actual)
        result["expected"] = {"min": min_value, "max": max_value}
        if min_value is None or max_value is None or actual_num is None:
            result["reason"] = "non-numeric comparison"
            return result
        result["passed"] = min_value <= actual_num <= max_value
        result["reason"] = "ok" if result["passed"] else "actual outside [min, max]"
        return result

    if check_type == "approx_equals":
        expected = _as_float(check.get("value"))
        tol = _as_float(check.get("tolerance"))
        actual_num = _as_float(actual)
        result["expected"] = {"value": expected, "tolerance": tol}
        if expected is None or tol is None or actual_num is None:
            result["reason"] = "non-numeric comparison"
            return result
        result["passed"] = abs(actual_num - expected) <= tol
        result["reason"] = "ok" if result["passed"] else "absolute error above tolerance"
        return result

    if check_type in {"monotonic_non_decreasing", "monotonic_non_increasing"}:
        if not isinstance(actual, list):
            result["reason"] = "actual value is not a list"
            return result
        numbers: list[float] = []
        for item in actual:
            value = _as_float(item)
            if value is None:
                result["reason"] = "list contains non-numeric value"
                return result
            numbers.append(value)
        if len(numbers) <= 1:
            result["passed"] = True
            result["reason"] = "ok"
            return result
        if check_type == "monotonic_non_decreasing":
            result["passed"] = all(numbers[i] <= numbers[i + 1] for i in range(len(numbers) - 1))
            result["reason"] = "ok" if result["passed"] else "sequence is not non-decreasing"
            return result
        result["passed"] = all(numbers[i] >= numbers[i + 1] for i in range(len(numbers) - 1))
        result["reason"] = "ok" if result["passed"] else "sequence is not non-increasing"
        return result

    result["reason"] = f"unsupported check type: {check_type}"
    return result


def evaluate_figure_contract(
    contract: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    checks = contract.get("checks", [])
    if not isinstance(checks, list):
        checks = []

    check_results = [_evaluate_single_check(check, metrics) for check in checks if isinstance(check, dict)]
    total_checks = len(check_results)
    passed_checks = sum(1 for item in check_results if item.get("passed"))

    pass_rule = contract.get("pass_rule", "all")
    required_passes = total_checks
    pass_rule_label = "all"
    if isinstance(pass_rule, str):
        lowered = pass_rule.strip().lower()
        if lowered == "any":
            required_passes = 1 if total_checks > 0 else 0
            pass_rule_label = "any"
        else:
            required_passes = total_checks
            pass_rule_label = "all"
    elif isinstance(pass_rule, dict) and isinstance(pass_rule.get("min_pass"), int):
        required_passes = max(0, min(int(pass_rule["min_pass"]), total_checks))
        pass_rule_label = f"min_pass={required_passes}"

    success = total_checks > 0 and passed_checks >= required_passes

    return {
        "success": success,
        "checked_at": iso_now(),
        "figure_id": contract.get("figure_id"),
        "paper_claim": contract.get("paper_claim"),
        "expected_trend": contract.get("expected_trend"),
        "tolerance": contract.get("tolerance"),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": [item["id"] for item in check_results if not item.get("passed")],
        "pass_rule": pass_rule_label,
        "required_passes": required_passes,
        "check_results": check_results,
    }
