#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import run_reproduce as rr

REPRO_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = REPRO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from repro_toolkit import load_state  # noqa: E402


DEFAULT_MODEL = "Qwen2.5-14B-Instruct"
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_OUTPUT_ROOT = "temp/agent_framework/reproduce_tool_loop/runs"

TERMINAL_ACTIONS = {"done", "stop", "terminal_failed", "escalate_human_review"}

ACTION_TO_TOOL = {
    "repro_preflight": "tool_preflight.py",
    "repro_run_once": "tool_run_once.py",
    "classify_failure": "tool_classify_failure.py",
    "apply_fix": "tool_apply_fix.py",
    "repo_path_probe": "tool_repo_path_probe.py",
    "build_original_runner": "tool_build_original_runner.py",
    "simulation_backend_fix": "tool_simulation_backend_fix.py",
    "dependency_runtime_fix": "tool_dependency_runtime_fix.py",
    "source_path_fix": "tool_source_path_fix.py",
    "source_fix": "tool_source_fix.py",
    "runtime_python_select": "tool_runtime_python_select.py",
    "runtime_env_select": "tool_runtime_env_select.py",
    "runtime_trace_fix": "tool_runtime_trace_fix.py",
    "constraint_resolve": "tool_constraint_resolve.py",
    "env_fix": "tool_env_fix.py",
    "fix_validate": "tool_fix_validate.py",
    "test_plan": "tool_test_plan.py",
    "run_unit_tests": "tool_run_unit_tests.py",
    "run_regression_tests": "tool_run_regression_tests.py",
    "verify_claim": "tool_verify_claim.py",
    "figure_visual_compare": "tool_figure_visual_compare.py",
    "original_pipeline_probe": "tool_original_pipeline_probe.py",
    "render_artifacts": "tool_render_artifacts.py",
}

ACTION_PRIORITY = [
    "classify_failure",
    "original_pipeline_probe",
    "repo_path_probe",
    "build_original_runner",
    "simulation_backend_fix",
    "dependency_runtime_fix",
    "source_path_fix",
    "source_fix",
    "constraint_resolve",
    "runtime_trace_fix",
    "runtime_python_select",
    "runtime_env_select",
    "env_fix",
    "fix_validate",
    "apply_fix",
    "test_plan",
    "run_unit_tests",
    "run_regression_tests",
    "verify_claim",
    "figure_visual_compare",
    "original_pipeline_probe",
    "repro_run_once",
    "repro_preflight",
    "render_artifacts",
    "done",
    "terminal_failed",
]

PYTHON_EXECUTABLE_ACTIONS = {
    "repro_preflight",
    "repro_run_once",
    "apply_fix",
    "repo_path_probe",
    "build_original_runner",
    "simulation_backend_fix",
    "dependency_runtime_fix",
    "source_path_fix",
    "source_fix",
    "runtime_python_select",
    "runtime_env_select",
    "runtime_trace_fix",
    "env_fix",
}

OUTPUT_ROOT_ACTIONS = {
    "repro_preflight",
    "repro_run_once",
    "repo_path_probe",
    "original_pipeline_probe",
    "verify_claim",
}


def action_completed(state: dict[str, Any], action: str) -> bool:
    if action == "repo_path_probe":
        return isinstance(state.get("last_repo_path_probe"), dict)
    if action == "original_pipeline_probe":
        return isinstance(state.get("last_original_pipeline_probe"), dict)
    if action == "build_original_runner":
        return bool(str(state.get("original_runner_path") or "").strip())
    if action in {"runtime_env_select", "runtime_python_select"}:
        return bool(str(state.get("runtime_python_executable") or "").strip())
    return False


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--state")
    parser.add_argument("--max-agent-steps", type=int, default=8)
    parser.add_argument("--model", default=os.getenv("REPRO_AGENT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-base", default=os.getenv("REPRO_AGENT_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.getenv("REPRO_AGENT_API_KEY", "EMPTY"))
    parser.add_argument(
        "--allow-offline-agent",
        action="store_true",
        default=parse_bool_env("REPRO_AGENT_ALLOW_OFFLINE_AGENT", False),
    )
    parser.add_argument(
        "--exit-policy",
        choices=["strict", "lenient"],
        default=os.getenv("REPRO_AGENT_EXIT_POLICY", "strict"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_stamp() -> str:
    return rr.utc_stamp()


def exception_payload(exc: Exception, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def probe_openai_endpoint(api_base: str, api_key: str) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/models"
    request = urllib.request.Request(
        url=url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "ok": True,
                "url": url,
                "status_code": getattr(response, "status", 200),
                "body_preview": body[:500],
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "url": url, "status_code": exc.code, "error": f"HTTPError: {exc}"}
    except urllib.error.URLError as exc:
        return {"ok": False, "url": url, "status_code": None, "error": f"URLError: {exc}"}
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "status_code": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def call_openai_compatible(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url=api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def choose_action_fallback(state: dict[str, Any], allowed_actions: list[str]) -> str:
    fsm_state = str(state.get("fsm_state", ""))
    last_recovery = state.get("last_recovery")
    recovery_payload = last_recovery.get("recovery") if isinstance(last_recovery, dict) else None
    has_applied_recovery = bool((recovery_payload or {}).get("applied"))
    apply_fix_preference = (
        ["fix_validate", "repro_run_once", "repro_preflight"]
        if has_applied_recovery
        else ["repro_run_once", "repro_preflight", "fix_validate"]
    )
    suggested_actions: list[str] = []
    last_classification = state.get("last_classification")
    if isinstance(last_classification, dict):
        suggested_actions.extend(
            str(action)
            for action in last_classification.get("suggested_recovery_actions", [])
            if str(action)
        )
    last_failure_classification = state.get("last_failure_classification")
    if isinstance(last_failure_classification, dict):
        suggested_actions.extend(
            str(action)
            for action in last_failure_classification.get("recommended_actions", [])
            if str(action)
        )
    last_visual_comparison = state.get("last_visual_comparison")
    if isinstance(last_visual_comparison, dict):
        suggested_actions.extend(
            str(action)
            for action in last_visual_comparison.get("recommended_actions", [])
            if str(action)
        )
    last_dependency_runtime_fix = state.get("last_dependency_runtime_fix")
    if isinstance(last_dependency_runtime_fix, dict):
        suggested_actions.extend(
            str(action)
            for action in last_dependency_runtime_fix.get("recommended_next_actions", [])
            if str(action)
        )
    suggested_actions = [
        action for action in suggested_actions if not action_completed(state, action)
    ]
    full_fig11_without_original_runner = (
        "qos_fig11_full" in str(state.get("recipe_name") or state.get("recipe_path") or "")
        and not action_completed(state, "build_original_runner")
    )
    if full_fig11_without_original_runner and fsm_state == "CLASSIFY_FAILURE":
        for action in ("original_pipeline_probe", "repo_path_probe", "build_original_runner"):
            if action in allowed_actions and not action_completed(state, action):
                return action
    classification_text = json.dumps(
        {
            "last_classification": last_classification,
            "last_failure_classification": last_failure_classification,
            "last_attempt_payload": state.get("last_attempt_payload"),
        },
        sort_keys=True,
        default=str,
    )
    missing_generated_runner_metrics = (
        ("metrics_parse_failure" in classification_text or "metric_validation_failure" in classification_text)
        and (
            "debug_simulation" in classification_text
            or "metric_provenance" in classification_text
            or "semantic" in classification_text.lower()
            or "fig11c_metric_unit_is_pair" in classification_text
            or "fig11c_uses_joint_multiprogramming_simulation" in classification_text
            or "qos_fig11_full" in str(state.get("recipe_name") or state.get("recipe_path") or "")
        )
        and not action_completed(state, "build_original_runner")
    )
    if missing_generated_runner_metrics:
        for action in ("original_pipeline_probe", "repo_path_probe", "build_original_runner"):
            if action in allowed_actions and not action_completed(state, action):
                return action

    preferred_by_state: dict[str, list[str]] = {
        "INIT": ["repro_preflight"],
        "PREFLIGHT": ["repro_run_once", "test_plan", "classify_failure"],
        "PREFLIGHT_FAILED": ["classify_failure", "test_plan"],
        "RUN_ONCE": ["verify_claim", "test_plan", "classify_failure"],
        "RUN_FAILED": ["classify_failure", "test_plan"],
        "CLASSIFY_FAILURE": [
            *suggested_actions,
            "original_pipeline_probe",
            "repo_path_probe",
            "build_original_runner",
            "simulation_backend_fix",
            "dependency_runtime_fix",
            "source_path_fix",
            "source_fix",
            "apply_fix",
            "terminal_failed",
        ],
        "APPLY_FIX": apply_fix_preference,
        "FIX_VALIDATE": ["verify_claim", "test_plan", "classify_failure"],
        "TEST_PLAN": ["run_unit_tests", "run_regression_tests"],
        "UNIT_TEST": ["verify_claim", "classify_failure"],
        "REGRESSION_TEST": ["verify_claim", "classify_failure"],
        "SUCCESS": ["figure_visual_compare", "render_artifacts", "done"],
        "ARTIFACTS_RENDERED": ["figure_visual_compare", "done"],
        "BUDGET_EXHAUSTED": ["render_artifacts", "terminal_failed"],
        "TERMINAL_FAILED": ["render_artifacts", "done"],
    }
    preferred = preferred_by_state.get(fsm_state, [])
    for action in preferred:
        if action in allowed_actions and not action_completed(state, action):
            return action
    for action in ACTION_PRIORITY:
        if action in allowed_actions and not action_completed(state, action):
            return action
    return allowed_actions[0] if allowed_actions else "terminal_failed"


def choose_action_with_model(
    *,
    api_base: str,
    api_key: str,
    model: str,
    step_index: int,
    recipe_path: str,
    state: dict[str, Any],
    allowed_actions: list[str],
) -> tuple[str, dict[str, Any]]:
    prompt_payload = {
        "step_index": step_index,
        "recipe_path": recipe_path,
        "state_snapshot": {
            "fsm_state": state.get("fsm_state"),
            "last_status": state.get("last_status"),
            "last_step": state.get("last_step"),
            "fix_budget": state.get("fix_budget"),
        },
        "allowed_actions": allowed_actions,
        "instruction": (
            "Pick exactly one action from allowed_actions. Prefer progress toward reproduction "
            "success with minimum risk. Return JSON: "
            '{"action":"<allowed>","reason":"short reason"}'
        ),
    }
    response = call_openai_compatible(
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict reproduce orchestrator. Output valid JSON only.",
            },
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ],
    )
    content = (
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    parsed = extract_json_object(content)
    action = str(parsed.get("action", "")).strip()
    if action not in allowed_actions:
        action = choose_action_fallback(state, allowed_actions)
        parsed = {
            "action": action,
            "reason": "model_selected_invalid_action_fallback",
        }
    return action, parsed


def run_tool_action(
    *,
    action: str,
    recipe_path: Path,
    state_path: Path,
    tool_python_executable: str,
    target_python_executable: str,
    output_root: Path,
    cwd: Path,
) -> dict[str, Any]:
    script = ACTION_TO_TOOL[action]
    command = [tool_python_executable, str(TOOLS_ROOT / script), "--state", str(state_path)]
    if action in OUTPUT_ROOT_ACTIONS:
        command.extend(["--output-root", str(output_root)])
    if action in PYTHON_EXECUTABLE_ACTIONS:
        command.extend(["--python-executable", target_python_executable])
    if action not in {"run_unit_tests", "run_regression_tests", "render_artifacts"}:
        command.extend(["--recipe", str(recipe_path)])
    proc = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout.strip()
    parsed: dict[str, Any] = {}
    if stdout:
        try:
            parsed = extract_json_object(stdout)
        except Exception:
            parsed = {"raw_stdout": stdout}
    return {
        "action": action,
        "command": command,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout": parsed,
        "stderr_tail": proc.stderr.strip().splitlines()[-40:],
    }


def infer_allowed_actions(state: dict[str, Any]) -> list[str]:
    history = state.get("history")
    if isinstance(history, list) and history:
        last_payload = history[-1].get("payload", {}) if isinstance(history[-1], dict) else {}
        next_actions = last_payload.get("next_allowed_actions")
        if isinstance(next_actions, list) and next_actions:
            return [str(x) for x in next_actions]
    status = str(state.get("last_status", ""))
    if status == "blocked":
        return ["terminal_failed"]
    return ["terminal_failed"]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    recipe_path = Path(args.recipe)
    if not recipe_path.is_absolute():
        recipe_path = (repo_root / recipe_path).resolve()
    recipe = load_json(recipe_path)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    run_dir = output_root / f"{recipe['name']}_tool_loop_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state).resolve() if args.state else run_dir / "state.json"
    step_log_path = run_dir / "tool_loop_steps.json"

    endpoint_probe = probe_openai_endpoint(args.api_base, args.api_key)
    if not endpoint_probe.get("ok", False) and not args.allow_offline_agent:
        write_json(
            run_dir / "final_summary.json",
            {
                "status": "failed",
                "reason": "endpoint_probe_failed",
                "endpoint_probe": endpoint_probe,
            },
        )
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "endpoint_probe_failed",
                    "endpoint_probe": endpoint_probe,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    steps: list[dict[str, Any]] = []
    for step_index in range(1, args.max_agent_steps + 1):
        state = load_state(state_path)
        if step_index == 1 and not state:
            allowed_actions = ["repro_preflight"]
        else:
            allowed_actions = infer_allowed_actions(state)

        if not allowed_actions:
            allowed_actions = ["terminal_failed"]

        if all(action in TERMINAL_ACTIONS for action in allowed_actions):
            steps.append(
                {
                    "step": step_index,
                    "decision": "terminal",
                    "allowed_actions": allowed_actions,
                }
            )
            break

        decision_payload: dict[str, Any] = {"action": choose_action_fallback(state, allowed_actions)}
        if endpoint_probe.get("ok", False):
            try:
                action, model_payload = choose_action_with_model(
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=args.model,
                    step_index=step_index,
                    recipe_path=str(recipe_path),
                    state=state,
                    allowed_actions=allowed_actions,
                )
                decision_payload = model_payload
                decision_payload["action"] = action
            except Exception as exc:
                decision_payload = {
                    "action": choose_action_fallback(state, allowed_actions),
                    "reason": f"model_error_fallback: {type(exc).__name__}",
                    "error": exception_payload(exc, "choose_action_with_model"),
                }

        action = str(decision_payload.get("action", "")).strip()
        if action not in allowed_actions:
            action = choose_action_fallback(state, allowed_actions)
            decision_payload["action"] = action
            decision_payload["reason"] = "invalid_action_fallback"

        if action in TERMINAL_ACTIONS:
            steps.append(
                {
                    "step": step_index,
                    "decision": decision_payload,
                    "allowed_actions": allowed_actions,
                    "terminal_action": action,
                }
            )
            break

        target_python_executable = str(
            state.get("runtime_python_executable") or args.python_executable
        )
        result = run_tool_action(
            action=action,
            recipe_path=recipe_path,
            state_path=state_path,
            tool_python_executable=args.python_executable,
            target_python_executable=target_python_executable,
            output_root=run_dir,
            cwd=repo_root,
        )
        step_record = {
            "step": step_index,
            "allowed_actions": allowed_actions,
            "decision": decision_payload,
            "result": result,
        }
        steps.append(step_record)
        write_json(step_log_path, {"steps": steps})

    final_state = load_state(state_path)
    final_status = str(final_state.get("last_status", "unknown"))
    final_fsm_state = str(final_state.get("fsm_state", "unknown"))
    strict_success = final_fsm_state == "SUCCESS" and final_status in {
        "verification_success",
        "figure_visual_compare_success",
        "artifacts_rendered",
    }
    summary = {
        "status": "success" if strict_success else "failed",
        "exit_policy": args.exit_policy,
        "strict_success": strict_success,
        "final_status": final_status,
        "final_fsm_state": final_fsm_state,
        "state_path": str(state_path),
        "run_dir": str(run_dir),
        "steps": len(steps),
        "endpoint_probe": endpoint_probe,
    }
    write_json(run_dir / "final_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.exit_policy == "lenient":
        return 0
    return 0 if strict_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
