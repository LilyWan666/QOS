#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable

# Orchestration states for tool-first reproduce.
INIT = "INIT"
PREFLIGHT = "PREFLIGHT"
PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
RUN_ONCE = "RUN_ONCE"
RUN_FAILED = "RUN_FAILED"
CLASSIFY_FAILURE = "CLASSIFY_FAILURE"
APPLY_FIX = "APPLY_FIX"
FIX_VALIDATE = "FIX_VALIDATE"
TEST_PLAN = "TEST_PLAN"
UNIT_TEST = "UNIT_TEST"
REGRESSION_TEST = "REGRESSION_TEST"
VERIFY_CLAIM = "VERIFY_CLAIM"
SUCCESS = "SUCCESS"
ARTIFACTS_RENDERED = "ARTIFACTS_RENDERED"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
TERMINAL_FAILED = "TERMINAL_FAILED"

# Action names are intentionally flat strings so OpenClaw/tool runners can
# share one action vocabulary without extra adapters.
STATE_ACTIONS = {
    INIT: ["repro_preflight", "terminal_failed"],
    PREFLIGHT: ["repro_run_once", "test_plan", "classify_failure", "terminal_failed"],
    PREFLIGHT_FAILED: [
        "runtime_python_select",
        "runtime_env_select",
        "constraint_resolve",
        "env_fix",
        "classify_failure",
        "test_plan",
        "terminal_failed",
    ],
    RUN_ONCE: ["verify_claim", "test_plan", "classify_failure", "terminal_failed"],
    RUN_FAILED: [
        "repo_path_probe",
        "build_original_runner",
        "runtime_python_select",
        "runtime_env_select",
        "constraint_resolve",
        "env_fix",
        "classify_failure",
        "test_plan",
        "terminal_failed",
    ],
    CLASSIFY_FAILURE: [
        "repo_path_probe",
        "build_original_runner",
        "runtime_python_select",
        "runtime_env_select",
        "simulation_backend_fix",
        "runtime_trace_fix",
        "source_path_fix",
        "dependency_runtime_fix",
        "source_fix",
        "apply_fix",
        "terminal_failed",
        "escalate_human_review",
    ],
    APPLY_FIX: [
        "build_original_runner",
        "simulation_backend_fix",
        "source_path_fix",
        "source_fix",
        "fix_validate",
        "repro_preflight",
        "repro_run_once",
        "terminal_failed",
    ],
    FIX_VALIDATE: ["repro_run_once", "classify_failure", "test_plan", "terminal_failed"],
    TEST_PLAN: ["repo_path_probe", "run_unit_tests", "run_regression_tests", "terminal_failed"],
    UNIT_TEST: ["verify_claim", "classify_failure", "rollback_last_fix", "terminal_failed"],
    REGRESSION_TEST: ["verify_claim", "classify_failure", "rollback_last_fix", "terminal_failed"],
    VERIFY_CLAIM: ["success", "classify_failure", "terminal_failed"],
    SUCCESS: ["render_artifacts", "done"],
    ARTIFACTS_RENDERED: ["done"],
    BUDGET_EXHAUSTED: ["render_artifacts", "terminal_failed", "done"],
    TERMINAL_FAILED: ["render_artifacts", "done"],
}


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def next_allowed_actions(
    state: str,
    *,
    recoverable: bool = True,
    fix_budget: int = 1,
    max_steps_exceeded: bool = False,
    timeout_exceeded: bool = False,
    tests_required: bool = False,
) -> list[str]:
    """Return allowed next actions for the current state and guard conditions."""
    if max_steps_exceeded or timeout_exceeded:
        return ["terminal_failed"]

    actions = list(STATE_ACTIONS.get(state, ["terminal_failed"]))

    if state == CLASSIFY_FAILURE:
        if not recoverable:
            return ["terminal_failed", "escalate_human_review"]
        if fix_budget <= 0:
            return ["render_artifacts", "terminal_failed"]

    if state == VERIFY_CLAIM and tests_required:
        # Force test gate before allowing success when tests are mandated.
        actions = [action for action in actions if action != "success"]
        actions.append("test_plan")

    return _dedupe(actions)


def is_action_allowed(
    state: str,
    action: str,
    *,
    recoverable: bool = True,
    fix_budget: int = 1,
    max_steps_exceeded: bool = False,
    timeout_exceeded: bool = False,
    tests_required: bool = False,
) -> bool:
    return action in next_allowed_actions(
        state,
        recoverable=recoverable,
        fix_budget=fix_budget,
        max_steps_exceeded=max_steps_exceeded,
        timeout_exceeded=timeout_exceeded,
        tests_required=tests_required,
    )
