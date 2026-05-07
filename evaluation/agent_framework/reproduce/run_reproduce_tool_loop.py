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
    "fix_validate": "tool_fix_validate.py",
    "test_plan": "tool_test_plan.py",
    "run_unit_tests": "tool_run_unit_tests.py",
    "run_regression_tests": "tool_run_regression_tests.py",
    "verify_claim": "tool_verify_claim.py",
    "render_artifacts": "tool_render_artifacts.py",
}

ACTION_PRIORITY = [
    "classify_failure",
    "apply_fix",
    "fix_validate",
    "test_plan",
    "run_unit_tests",
    "run_regression_tests",
    "verify_claim",
    "repro_run_once",
    "repro_preflight",
    "render_artifacts",
    "done",
    "terminal_failed",
]


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
    preferred_by_state: dict[str, list[str]] = {
        "INIT": ["repro_preflight"],
        "PREFLIGHT": ["repro_run_once", "test_plan", "classify_failure"],
        "PREFLIGHT_FAILED": ["classify_failure", "test_plan"],
        "RUN_ONCE": ["verify_claim", "test_plan", "classify_failure"],
        "RUN_FAILED": ["classify_failure", "test_plan"],
        "CLASSIFY_FAILURE": ["apply_fix", "terminal_failed"],
        "APPLY_FIX": ["fix_validate", "repro_run_once", "repro_preflight"],
        "FIX_VALIDATE": ["verify_claim", "test_plan", "classify_failure"],
        "TEST_PLAN": ["run_unit_tests", "run_regression_tests"],
        "UNIT_TEST": ["verify_claim", "classify_failure"],
        "REGRESSION_TEST": ["verify_claim", "classify_failure"],
        "SUCCESS": ["render_artifacts", "done"],
        "BUDGET_EXHAUSTED": ["render_artifacts", "terminal_failed"],
        "TERMINAL_FAILED": ["render_artifacts", "done"],
    }
    preferred = preferred_by_state.get(fsm_state, [])
    for action in preferred:
        if action in allowed_actions:
            return action
    for action in ACTION_PRIORITY:
        if action in allowed_actions:
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
    python_executable: str,
    output_root: Path,
    cwd: Path,
) -> dict[str, Any]:
    script = ACTION_TO_TOOL[action]
    command = [python_executable, str(TOOLS_ROOT / script), "--state", str(state_path)]
    if action in {"repro_preflight", "repro_run_once", "verify_claim"}:
        command.extend(["--output-root", str(output_root)])
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

        result = run_tool_action(
            action=action,
            recipe_path=recipe_path,
            state_path=state_path,
            python_executable=args.python_executable,
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
