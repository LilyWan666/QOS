#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import run_reproduce as rr


def build_registry(available_actions: list[str]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "name": "finish",
            "category": "control",
            "description": "Mark run complete when current attempt already satisfies success criteria.",
            "args_schema": {},
        },
        {
            "name": "stop",
            "category": "control",
            "description": "Stop when no safe progress action exists.",
            "args_schema": {"reason": "short string"},
        },
        {
            "name": "run_env_diagnose",
            "category": "observe",
            "description": "Run environment diagnostics to gather dependency/path/runtime hints.",
            "args_schema": {},
        },
    ]
    if available_actions:
        tools.append(
            {
                "name": "apply_recovery_action",
                "category": "repair",
                "description": "Apply one recovery action from diagnosis/recovery plan.",
                "args_schema": {"action": {"enum": available_actions}},
            }
        )
    return tools


def normalize_tool_call(payload: dict[str, Any], available_actions: list[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("agent output must be a JSON object")

    if "tool" not in payload and "decision" in payload:
        decision = str(payload.get("decision", "stop"))
        recovery_action = str(payload.get("recovery_action", ""))
        mapped_tool = {
            "finish": "finish",
            "run_env_diagnose": "run_env_diagnose",
            "apply_recovery_action": "apply_recovery_action",
            "stop": "stop",
        }.get(decision, "stop")
        payload = {
            "tool": mapped_tool,
            "args": {"action": recovery_action} if mapped_tool == "apply_recovery_action" else {},
            "reason": payload.get("reason", ""),
            "expected_outcome": payload.get("expected_outcome", ""),
        }

    tool = str(payload.get("tool", "")).strip()
    args = payload.get("args", {})
    if not isinstance(args, dict):
        args = {}

    if tool not in {"finish", "stop", "run_env_diagnose", "apply_recovery_action"}:
        raise ValueError(f"unsupported tool: {tool}")

    if tool == "apply_recovery_action":
        action = str(args.get("action", "")).strip()
        if action not in available_actions:
            raise ValueError(
                f"invalid recovery action for apply_recovery_action: {action}; "
                f"available={available_actions}"
            )
        args = {"action": action}
    else:
        args = {}

    return {
        "tool": tool,
        "args": args,
        "reason": str(payload.get("reason", "")),
        "expected_outcome": str(payload.get("expected_outcome", "")),
    }


@dataclass
class AgentLoopState:
    workspace_recipe: dict[str, Any]
    isolated_repo_root: Path
    workspace_recipe_path: Path
    workspace_root: Path
    python_executable: str
    args_python_executable: str | None
    run_dir: Path
    paper_context: dict[str, Any] | None
    api_base: str
    api_key: str
    model: str
    repo_root: Path
    rule_fallback_enabled: bool
    builtin_recovery_enabled: bool
    current_attempt: dict[str, Any]
    env_diagnose_payload: dict[str, Any] | None
    recoveries: list[dict[str, Any]]


def execute_tool_call(
    *,
    state: AgentLoopState,
    tool_call: dict[str, Any],
    available_actions: list[str],
    step_dir: Path,
    run_env_diagnose_fn: Callable[..., dict[str, Any]],
    execute_builtin_recovery_fn: Callable[..., dict[str, Any] | None],
    execute_model_recovery_fn: Callable[..., dict[str, Any]],
    rerun_attempt_fn: Callable[[Path], None],
    write_json_fn: Callable[[Path, dict[str, Any]], None],
    exception_payload_fn: Callable[[Exception, str], dict[str, Any]],
    builtin_only_actions: set[str] | None = None,
) -> str:
    tool_name = tool_call.get("tool", "stop")
    if tool_name in {"finish", "stop"}:
        return "break"

    if tool_name == "run_env_diagnose":
        try:
            state.env_diagnose_payload = run_env_diagnose_fn(
                repo_root=state.isolated_repo_root,
                recipe_path=state.workspace_recipe_path,
                python_executable=state.args_python_executable,
                step_dir=step_dir,
            )
        except Exception as exc:
            error = exception_payload_fn(exc, "step.run_env_diagnose")
            state.env_diagnose_payload = {
                "returncode": None,
                "report": None,
                "error": error,
            }
            write_json_fn(step_dir / "env_diagnose.error.json", state.env_diagnose_payload)
        write_json_fn(step_dir / "env_diagnose.json", state.env_diagnose_payload)
        return "continue"

    if tool_name != "apply_recovery_action":
        write_json_fn(
            step_dir / "tool_call.invalid.json",
            {
                "tool_call": tool_call,
                "reason": f"unsupported_tool: {tool_name}",
            },
        )
        return "break"

    action_name = str(tool_call.get("args", {}).get("action", ""))
    if action_name not in available_actions:
        write_json_fn(
            step_dir / "tool_call.invalid.json",
            {
                "tool_call": tool_call,
                "reason": f"invalid_or_unavailable_recovery_action: {action_name}",
                "available_actions": available_actions,
            },
        )
        return "break"

    try:
        recovery = rr.execute_recovery_handler(
            recipe=state.workspace_recipe,
            repo_root=state.isolated_repo_root,
            workspace_root=state.workspace_root,
            python_exe=state.python_executable,
            run_dir=state.run_dir,
            attempt=state.current_attempt,
            action_name=action_name,
            ordinal=len(state.recoveries) + 1,
        )
    except Exception as exc:
        error = exception_payload_fn(exc, "step.execute_recovery_handler")
        write_json_fn(step_dir / "recovery.handler.error.json", error)
        recovery = None

    builtin_recovery: dict[str, Any] | None = None
    if recovery is None and state.builtin_recovery_enabled:
        builtin_recovery = execute_builtin_recovery_fn(
            action_name=action_name,
            attempt=state.current_attempt,
            workspace_root=state.workspace_root,
            step_dir=step_dir,
            python_exe=state.python_executable,
            source_repo_root=state.repo_root,
            attempt_recoveries=state.recoveries,
        )
        builtin_only = builtin_only_actions or set()
        if builtin_recovery is not None and (
            builtin_recovery.get("applied") or action_name in builtin_only
        ):
            recovery = builtin_recovery

    if recovery is None:
        try:
            recovery = execute_model_recovery_fn(
                recipe=state.workspace_recipe,
                attempt=state.current_attempt,
                action_name=action_name,
                workspace_root=state.workspace_root,
                paper_context=state.paper_context,
                api_base=state.api_base,
                api_key=state.api_key,
                model=state.model,
                step_dir=step_dir,
            )
        except Exception as exc:
            error = exception_payload_fn(exc, "step.execute_model_recovery")
            write_json_fn(step_dir / "recovery.model.error.json", error)
            recovery = None

    if recovery is None and state.rule_fallback_enabled and state.builtin_recovery_enabled:
        recovery = builtin_recovery or execute_builtin_recovery_fn(
            action_name=action_name,
            attempt=state.current_attempt,
            workspace_root=state.workspace_root,
            step_dir=step_dir,
            python_exe=state.python_executable,
            source_repo_root=state.repo_root,
            attempt_recoveries=state.recoveries,
        )

    if recovery is None:
        recovery = {
            "action": action_name,
            "record_path": str(step_dir / "recovery.noop.json"),
            "result": {},
            "applied": False,
            "apply_result": {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": ["recovery_not_available"],
            },
        }
        write_json_fn(step_dir / "recovery.noop.json", recovery)

    state.recoveries.append(recovery)
    if recovery.get("applied"):
        result = recovery.get("result") or {}
        selected_python = str(result.get("python_executable", "")).strip()
        if selected_python:
            state.python_executable = selected_python
    write_json_fn(step_dir / "recovery.json", recovery)
    if recovery.get("applied"):
        rerun_attempt_fn(step_dir)
    return "continue"
