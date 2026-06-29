#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
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


ACTION = "openevolve_target_semantic_select"
TOOL_NAME = "repro_openevolve_target_semantic_select"
DEFAULT_MODEL = "Qwen2.5-14B-Instruct"
DEFAULT_API_BASE = "http://localhost:8000/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument(
        "--output-root",
        default="temp/agent_framework/reproduce/tools/runs",
        help="Root directory for semantic-selection artifacts.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_json_object(text: str) -> dict[str, Any]:
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


def _call_openai_compatible(
    *,
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


def _candidate_packet(candidate: dict[str, Any]) -> dict[str, Any]:
    source = str(candidate.get("source") or "")
    return {
        "entrypoint": candidate.get("entrypoint"),
        "source_path": candidate.get("source_path"),
        "line": candidate.get("line"),
        "class": candidate.get("class"),
        "function": candidate.get("function"),
        "signature": candidate.get("signature"),
        "static_retrieval_score": candidate.get("score"),
        "static_evidence": candidate.get("evidence") or [],
        "static_penalties": candidate.get("penalties") or [],
        "called_by": candidate.get("called_by") or [],
        "source_excerpt": source[:4000],
    }


def _build_prompt(state: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    param_probe = state.get("last_openevolve_param_probe")
    objective_requirements = {}
    if isinstance(param_probe, dict):
        objective_requirements = param_probe.get("objective_requirements") or {}
    if not objective_requirements:
        objective_requirements = {
            "selection_target": "multiprogramming_pair_selection",
            "must_include_metrics": ["effective_utilization", "relative_fidelity"],
            "aggregation": "selected_topk_average_pareto_rank",
            "primary_score": "inv_avg_pareto_rank",
            "topk_selection": True,
            "diagnostic_metrics": ["rank_agreement", "top_rank_overlap"],
            "forbid_objective_gating_by_front_overlap": True,
            "forbid_required_rank1_overlap": True,
            "forbid_single_metric_fidelity_only": True,
        }
    return {
        "task": "Bind the reproduction contract's semantic target to the best evolvable repository function.",
        "contract_target": "multiprogramming_pair_selection",
        "objective_requirements": objective_requirements,
        "selection_rules": [
            "Do semantic reasoning over the function role, signature, source, and callers.",
            "Do not choose solely because a name/path contains matching keywords.",
            "Prefer a small, evolvable function that directly scores or ranks multiprogramming pairs.",
            "Reject orchestration, plotting, harness, and verification functions unless they are the only valid target.",
            "Return exactly one selected_entrypoint from candidates.",
        ],
        "candidates": [_candidate_packet(candidate) for candidate in candidates[:15]],
        "required_json": {
            "selected_entrypoint": "one entrypoint copied exactly from candidates",
            "confidence": "low|medium|high",
            "reasoning": "short semantic justification",
            "semantic_evidence": ["why this function implements or controls pair selection"],
            "rejected_alternatives": [
                {"entrypoint": "candidate entrypoint", "reason": "why rejected"}
            ],
        },
    }


def _select_with_llm(prompt_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    api_base = os.getenv("REPRO_TARGET_SELECT_API_BASE") or os.getenv("REPRO_AGENT_API_BASE") or DEFAULT_API_BASE
    api_key = os.getenv("REPRO_TARGET_SELECT_API_KEY") or os.getenv("REPRO_AGENT_API_KEY") or "EMPTY"
    model = os.getenv("REPRO_TARGET_SELECT_MODEL") or os.getenv("REPRO_AGENT_MODEL") or DEFAULT_MODEL
    response = _call_openai_compatible(
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a semantic code-understanding reviewer for a QOS-agent reproduction workflow. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ],
    )
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    return _extract_json_object(content), {"api_base": api_base, "model": model}


def build_selection(state: dict[str, Any]) -> dict[str, Any]:
    probe = state.get("last_openevolve_target_probe")
    if not isinstance(probe, dict) or not probe.get("success"):
        return {
            "success": False,
            "reason": "missing_successful_openevolve_target_probe",
            "selection_mode": "llm_semantic_selection",
        }
    qos_target = probe.get("qos_evolution_target") if isinstance(probe.get("qos_evolution_target"), dict) else {}
    candidates = qos_target.get("candidates") if isinstance(qos_target.get("candidates"), list) else []
    if not candidates:
        return {
            "success": False,
            "reason": "missing_qos_target_candidates",
            "selection_mode": "llm_semantic_selection",
        }

    prompt_payload = _build_prompt(state, candidates)
    try:
        llm_payload, llm_metadata = _select_with_llm(prompt_payload)
    except Exception as exc:
        return {
            "success": False,
            "reason": "llm_semantic_selection_failed",
            "selection_mode": "llm_semantic_selection",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "prompt_payload": prompt_payload,
            "note": "No heuristic fallback is used by default; target selection requires LLM semantic reasoning.",
        }

    selected_entrypoint = str(llm_payload.get("selected_entrypoint") or "").strip()
    selected = None
    for candidate in candidates:
        if str(candidate.get("entrypoint") or "") == selected_entrypoint:
            selected = dict(candidate)
            break
    if not selected:
        return {
            "success": False,
            "reason": "llm_selected_entrypoint_not_in_candidates",
            "selection_mode": "llm_semantic_selection",
            "selected_entrypoint": selected_entrypoint,
            "llm_payload": llm_payload,
            "prompt_payload": prompt_payload,
        }

    selected["selection_mode"] = "llm_semantic_selection"
    selected["semantic_reasoning"] = llm_payload.get("reasoning")
    selected["semantic_evidence"] = llm_payload.get("semantic_evidence") or []
    selected["confidence"] = llm_payload.get("confidence")
    return {
        "success": True,
        "selection_mode": "llm_semantic_selection",
        "selected": selected,
        "selected_entrypoint": selected_entrypoint,
        "llm_payload": llm_payload,
        "llm_metadata": llm_metadata,
        "prompt_payload": prompt_payload,
        "validation": {
            "selected_from_candidate_scan": True,
            "source_present": bool(selected.get("source")),
            "entrypoint_present": bool(selected.get("entrypoint")),
        },
    }


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
    artifact_dir = run_root / "openevolve_target_semantic_select"

    payload = build_selection(state)
    payload["tool"] = TOOL_NAME
    payload["action"] = ACTION
    payload["fsm_state_before"] = get_fsm_state(state)
    payload["artifact_path"] = str(artifact_dir / "openevolve_target_semantic_select.json")
    _write_json(artifact_dir / "openevolve_target_semantic_select.json", payload)

    state["last_openevolve_target_semantic_selection"] = payload
    if payload.get("success"):
        probe = state.get("last_openevolve_target_probe")
        if isinstance(probe, dict) and isinstance(probe.get("qos_evolution_target"), dict):
            probe["qos_evolution_target"]["selected"] = payload.get("selected")
            probe["qos_evolution_target"]["selection_mode"] = "llm_semantic_selection"
            probe["qos_evolution_target"]["selection_required"] = "satisfied"
    state["last_step"] = TOOL_NAME
    state["last_status"] = (
        "openevolve_target_semantic_select_succeeded"
        if payload.get("success")
        else "openevolve_target_semantic_select_failed"
    )
    set_fsm_state(state, "APPLY_FIX" if payload.get("success") else "CLASSIFY_FAILURE")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, payload)
    write_state(state_path, state)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
