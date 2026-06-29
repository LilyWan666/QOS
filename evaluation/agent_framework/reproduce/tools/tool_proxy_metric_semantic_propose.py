#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

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


ACTION = "proxy_metric_semantic_propose"
TOOL_NAME = "repro_proxy_metric_semantic_propose"
DEFAULT_MODEL = "Qwen2.5-14B-Instruct"
DEFAULT_API_BASE = "http://localhost:8000/v1"

MATERIALIZED_FEATURES = [
    "depth_ratio",
    "depth_density",
    "cnot_ratio",
    "cnot_density",
    "nonlocal_ratio",
    "nonlocal_density",
    "measure_ratio",
    "measure_density",
    "instr_ratio",
    "instr_density",
    "critical_depth_ratio",
    "critical_depth_density",
    "qubit_imbalance",
    "scale_ratio",
]

MATERIALIZED_FEATURE_SEMANTICS = {
    "depth_ratio": "min(left.depth, right.depth) / max(left.depth, right.depth); depth balance, not depth density",
    "depth_density": "(left.depth + right.depth) / (left.num_qubits + right.num_qubits)",
    "cnot_ratio": "min(left.num_cnot_gates, right.num_cnot_gates) / max(left.num_cnot_gates, right.num_cnot_gates)",
    "cnot_density": "(left.num_cnot_gates + right.num_cnot_gates) / (left.num_qubits + right.num_qubits)",
    "nonlocal_ratio": "min(left.num_nonlocal_gates, right.num_nonlocal_gates) / max(left.num_nonlocal_gates, right.num_nonlocal_gates)",
    "nonlocal_density": "(left.num_nonlocal_gates + right.num_nonlocal_gates) / (left.num_qubits + right.num_qubits)",
    "measure_ratio": "min(left.num_measurements, right.num_measurements) / max(left.num_measurements, right.num_measurements)",
    "measure_density": "(left.num_measurements + right.num_measurements) / (left.num_qubits + right.num_qubits)",
    "instr_ratio": "min(left.number_instructions, right.number_instructions) / max(left.number_instructions, right.number_instructions)",
    "instr_density": "(left.number_instructions + right.number_instructions) / (left.num_qubits + right.num_qubits)",
    "critical_depth_ratio": "min(left.critical_depth, right.critical_depth) / max(left.critical_depth, right.critical_depth)",
    "critical_depth_density": "(left.critical_depth + right.critical_depth) / (left.num_qubits + right.num_qubits)",
    "qubit_imbalance": "abs(left.num_qubits - right.num_qubits) / (left.num_qubits + right.num_qubits)",
    "scale_ratio": "(left.num_qubits + right.num_qubits) / selected_or_backend_qubits",
}

METADATA_KEYS = [
    "depth",
    "num_qubits",
    "number_instructions",
    "num_measurements",
    "num_cnot_gates",
    "num_nonlocal_gates",
    "critical_depth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output-root", default="temp/agent_framework/reproduce/tools/runs")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.startswith("```")).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
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
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url=api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def _paper_context(state: dict[str, Any]) -> dict[str, Any]:
    evaluator = state.get("last_openevolve_evaluator_semantic_selection") or {}
    spec = evaluator.get("paper_metric_spec") if isinstance(evaluator, dict) else {}
    snippets = []
    if isinstance(spec, dict):
        for item in spec.get("paper_snippets") or []:
            for snippet in item.get("snippets") or []:
                snippets.append(str(snippet)[:900])
    return {
        "figure_id": "fig11",
        "objective_axes": (spec or {}).get("objective_axes"),
        "aggregation": (spec or {}).get("aggregation"),
        "selection_operator": (spec or {}).get("selection_operator"),
        "snippets": snippets[:6],
    }


def _repo_context(repo_root: Path) -> dict[str, Any]:
    paths = [
        repo_root / "qos" / "multiprogrammer" / "multiprogrammer.py",
        repo_root / "qos" / "types" / "types.py",
        repo_root / "qos" / "error_mitigator" / "analyser.py",
    ]
    excerpts = []
    for path in paths:
        if path.exists():
            excerpts.append(
                {
                    "path": path.relative_to(repo_root).as_posix(),
                    "excerpt": path.read_text(encoding="utf-8", errors="replace")[:5000],
                }
            )
    return {
        "available_benchmark_source": "evaluation/benchmarks/<application>/<qubits>.qasm",
        "available_metadata_keys": METADATA_KEYS,
        "materialized_feature_names": MATERIALIZED_FEATURES,
        "source_excerpts": excerpts,
    }


def _prompt(repo_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    param_probe = state.get("last_openevolve_param_probe") or {}
    objective_requirements = param_probe.get("objective_requirements") if isinstance(param_probe, dict) else {}
    factor_brainstorm = state.get("last_proxy_semantic_factor_brainstorm")
    factor_context = None
    if isinstance(factor_brainstorm, dict) and factor_brainstorm.get("success"):
        factor_context = {
            "brainstorm_mode": factor_brainstorm.get("brainstorm_mode"),
            "semantic_factors": factor_brainstorm.get("semantic_factors") or [],
            "missing_materializer_requests": factor_brainstorm.get("missing_materializer_requests") or [],
        }
    return {
        "task": "Propose cheap, computable proxy metrics for the expensive fidelity-like objective in QOS Fig11 multiprogramming pair selection.",
        "important_rule": (
            "Do not use manual reproduction choices or prior human-selected proxy names as authority. "
            "Derive candidates from the semantic factor brainstorm, paper semantics, repository metadata APIs, "
            "and available circuit/QASM metadata. If the strongest semantic factor is not currently materialized, "
            "report it as a rejected/missing-materializer candidate instead of silently dropping the reasoning."
        ),
        "paper_context": _paper_context(state),
        "objective_requirements": objective_requirements,
        "semantic_factor_brainstorm": factor_context,
        "repo_context": _repo_context(repo_root),
        "materialized_feature_semantics": MATERIALIZED_FEATURE_SEMANTICS,
        "selection_target": "multiprogramming_pair_selection",
        "required_json_schema": {
            "proposal_mode": "llm_semantic_metric_proposal",
            "semantic_factor_source": "llm_semantic_factor_brainstorm",
            "proxy_role": "pre_evolution_expensive_metric_substitute",
            "candidates": [
                {
                    "name": "short semantic name",
                    "materialized_feature_name": "one value from materialized_feature_names",
                    "formula": "pair-level formula over left/right metadata",
                    "direction": "direct|inverse",
                    "semantic_factor_name": "semantic factor that motivated this proxy",
                    "required_metadata_keys": ["metadata keys needed"],
                    "paper_rationale": "why this approximates fidelity/crosstalk penalty",
                    "repo_rationale": "where the metadata can be computed in this repo",
                }
            ],
            "rejected_metrics": [
                {"name": "metric name", "reason": "why not a suitable fidelity-like proxy"}
            ],
        },
    }


def _select_with_llm(prompt_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    api_base = os.getenv("REPRO_PROXY_METRIC_PROPOSE_API_BASE") or os.getenv("REPRO_AGENT_API_BASE") or DEFAULT_API_BASE
    api_key = os.getenv("REPRO_PROXY_METRIC_PROPOSE_API_KEY") or os.getenv("REPRO_AGENT_API_KEY") or "EMPTY"
    model = os.getenv("REPRO_PROXY_METRIC_PROPOSE_MODEL") or os.getenv("REPRO_AGENT_MODEL") or DEFAULT_MODEL
    response = _call_openai_compatible(
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a semantic research-agent planner. Return valid JSON only. "
                    "Do not copy manual experiments; propose proxy metrics from the supplied paper/repo context."
                ),
            },
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ],
    )
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    return _extract_json_object(content), {"api_base": api_base, "model": model}


def _slurm_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _slurm_script(
    *,
    repo_root: Path,
    recipe_path: Path,
    state_path: Path,
    output_root: Path,
    log_dir: Path,
) -> str:
    model_dir = os.getenv("REPRO_PROXY_METRIC_SLURM_MODEL_DIR", "")
    vllm_img = os.getenv("REPRO_PROXY_METRIC_SLURM_VLLM_IMG", "")
    wait_script = os.getenv("REPRO_PROXY_METRIC_SLURM_WAIT_SCRIPT", "")
    python_bin = os.getenv("REPRO_PROXY_METRIC_SLURM_PYTHON", sys.executable)
    model = os.getenv("REPRO_PROXY_METRIC_PROPOSE_MODEL", DEFAULT_MODEL)
    api_key = os.getenv("REPRO_PROXY_METRIC_PROPOSE_API_KEY", "")
    port = os.getenv("REPRO_PROXY_METRIC_SLURM_PORT", "8000")
    host = os.getenv("REPRO_PROXY_METRIC_SLURM_HOST", "127.0.0.1")
    partition = os.getenv("REPRO_PROXY_METRIC_SLURM_PARTITION", "")
    account = os.getenv("REPRO_PROXY_METRIC_SLURM_ACCOUNT", "")
    time_limit = os.getenv("REPRO_PROXY_METRIC_SLURM_TIME", "01:00:00")
    mem = os.getenv("REPRO_PROXY_METRIC_SLURM_MEM", "48G")
    cpus = os.getenv("REPRO_PROXY_METRIC_SLURM_CPUS", "8")
    gres = os.getenv("REPRO_PROXY_METRIC_SLURM_GRES", "gpu:1")
    required = {
        "REPRO_PROXY_METRIC_SLURM_MODEL_DIR": model_dir,
        "REPRO_PROXY_METRIC_SLURM_VLLM_IMG": vllm_img,
        "REPRO_PROXY_METRIC_SLURM_WAIT_SCRIPT": wait_script,
        "REPRO_PROXY_METRIC_PROPOSE_API_KEY": api_key,
        "REPRO_PROXY_METRIC_SLURM_PARTITION": partition,
        "REPRO_PROXY_METRIC_SLURM_ACCOUNT": account,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError("missing Slurm environment variables: " + ", ".join(missing))
    return f"""#!/bin/bash
#SBATCH --job-name=qos_proxy_metric
#SBATCH --output={log_dir.as_posix()}/%x.%j.out
#SBATCH --error={log_dir.as_posix()}/%x.%j.err
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}

set -euo pipefail

REPO_ROOT={repo_root.as_posix()!r}
MODEL_DIR={model_dir!r}
VLLM_IMG={vllm_img!r}
WAIT_SCRIPT={wait_script!r}
PYTHON_BIN={python_bin!r}
MODEL={model!r}
API_KEY={api_key!r}
HOST={host!r}
PORT={port!r}
API_BASE="http://${{HOST}}:${{PORT}}/v1"
RECIPE={recipe_path.as_posix()!r}
STATE={state_path.as_posix()!r}
OUTPUT_ROOT={output_root.as_posix()!r}

mkdir -p {log_dir.as_posix()!r}

cleanup() {{
  if [[ -n "${{VLLM_PID:-}}" ]]; then
    kill "${{VLLM_PID}}" >/dev/null 2>&1 || true
    wait "${{VLLM_PID}}" >/dev/null 2>&1 || true
  fi
}}
trap cleanup EXIT

cd "${{REPO_ROOT}}"

apptainer run --nv --bind /projects,/work,/u "${{VLLM_IMG}}" \\
  vllm serve "${{MODEL_DIR}}" \\
  --served-model-name "${{MODEL}}" \\
  --host 0.0.0.0 \\
  --port "${{PORT}}" \\
  --api-key "${{API_KEY}}" \\
  --dtype auto \\
  --tensor-parallel-size 1 \\
  --gpu-memory-utilization "${{REPRO_PROXY_METRIC_GPU_MEMORY_UTILIZATION:-0.85}}" \\
  --max-model-len "${{REPRO_PROXY_METRIC_MAX_MODEL_LEN:-32768}}" \\
  > "{log_dir.as_posix()}/vllm.${{SLURM_JOB_ID}}.log" 2>&1 &
VLLM_PID=$!

python3 -u "${{WAIT_SCRIPT}}" --vllm_host "${{HOST}}" --vllm_port "${{PORT}}"

export REPRO_PROXY_METRIC_PROPOSE_FORCE_LOCAL=1
export REPRO_PROXY_METRIC_PROPOSE_MODEL="${{MODEL}}"
export REPRO_PROXY_METRIC_PROPOSE_API_BASE="${{API_BASE}}"
export REPRO_PROXY_METRIC_PROPOSE_API_KEY="${{API_KEY}}"

"${{PYTHON_BIN}}" evaluation/agent_framework/reproduce/tools/tool_proxy_metric_semantic_propose.py \\
  --recipe "${{RECIPE}}" \\
  --state "${{STATE}}" \\
  --output-root "${{OUTPUT_ROOT}}"
"""


def _parse_sbatch_job_id(stdout: str) -> str | None:
    for item in reversed(stdout.strip().split()):
        if item.isdigit():
            return item
    return None


def _build_slurm_payload(
    *,
    repo_root: Path,
    recipe_path: Path,
    state_path: Path,
    output_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    artifact_dir = run_root / "proxy_metric_semantic_propose_slurm"
    log_dir = artifact_dir / "logs"
    script_path = artifact_dir / "run_proxy_metric_semantic_propose.slurm"
    dry_run = _slurm_bool("REPRO_PROXY_METRIC_SLURM_DRY_RUN", False)
    sbatch = shutil.which("sbatch")
    payload: dict[str, Any] = {
        "success": False,
        "execution_mode": "proxy_metric_semantic_propose_slurm",
        "selection_mode": "llm_semantic_metric_proposal",
        "proxy_role": "pre_evolution_expensive_metric_substitute",
        "manual_proxy_source_used": False,
        "dry_run": dry_run,
        "sbatch_available": bool(sbatch),
        "slurm_script": str(script_path),
        "log_dir": str(log_dir),
        "warnings": [],
    }
    script = _slurm_script(
        repo_root=repo_root,
        recipe_path=recipe_path,
        state_path=state_path,
        output_root=output_root,
        log_dir=log_dir,
    )
    _write_text(script_path, script)
    if dry_run:
        payload["status"] = "slurm_script_ready"
        payload["warnings"].append("SLURM job was not submitted; set REPRO_PROXY_METRIC_SLURM_DRY_RUN=0 to call sbatch")
        return payload
    if not sbatch:
        payload["status"] = "sbatch_unavailable"
        payload["warnings"].append("sbatch not found on PATH")
        return payload
    proc = subprocess.run([sbatch, "--wait", str(script_path)], cwd=repo_root, text=True, capture_output=True, check=False)
    job_id = _parse_sbatch_job_id(proc.stdout)
    payload.update(
        {
            "job_id": job_id,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "status": "completed" if proc.returncode == 0 else "submit_or_job_failed",
        }
    )
    if proc.returncode != 0:
        payload["warnings"].append("SLURM proposal job failed before producing a valid proposal")
        return payload
    refreshed_state = load_state(state_path)
    inner = refreshed_state.get("last_proxy_metric_semantic_proposal")
    if not isinstance(inner, dict):
        payload["warnings"].append("SLURM proposal job completed but did not update state")
        return payload
    payload.update(inner)
    payload["execution_mode"] = "proxy_metric_semantic_propose_slurm"
    payload["slurm_job_id"] = job_id
    payload["slurm_stdout"] = proc.stdout.strip()
    payload["slurm_stderr"] = proc.stderr.strip()
    payload["slurm_script"] = str(script_path)
    payload["log_dir"] = str(log_dir)
    payload["success"] = bool(inner.get("success") and inner.get("candidates"))
    return payload


def _validate_llm_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    candidates: list[dict[str, Any]] = []
    for item in payload.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        feature = str(item.get("materialized_feature_name") or "").strip()
        if feature not in MATERIALIZED_FEATURES:
            warnings.append(f"discarded unsupported materialized feature: {feature}")
            continue
        required = item.get("required_metadata_keys") or []
        if any(str(key) not in METADATA_KEYS for key in required):
            warnings.append(f"candidate {item.get('name')} references unsupported metadata keys")
            continue
        formula = str(item.get("formula") or "")
        formula_ok, formula_reason = _formula_matches_feature(feature, formula)
        if not formula_ok:
            warnings.append(
                f"discarded formula/materializer mismatch for {item.get('name')}: "
                f"{feature} expects {MATERIALIZED_FEATURE_SEMANTICS.get(feature)}; {formula_reason}"
            )
            continue
        candidate = dict(item)
        candidate["materialized_feature_name"] = feature
        candidate["materialized_feature_semantics"] = MATERIALIZED_FEATURE_SEMANTICS.get(feature)
        candidate["proposal_source"] = "llm_semantic_metric_proposal"
        candidates.append(candidate)
    return candidates, warnings


def _formula_matches_feature(feature: str, formula: str) -> tuple[bool, str]:
    text = formula.lower().replace(" ", "")
    if not text:
        return False, "empty formula"
    has_sum_shape = "+" in text and any(token in text for token in ("num_qubits", "qubits"))
    has_balance_shape = ("min(" in text and "max(" in text) or "ratio" in text or "balance" in text
    if feature.endswith("_density"):
        return (True, "") if has_sum_shape else (False, "density feature requires a sum divided by qubits")
    if feature.endswith("_ratio") and feature not in {"scale_ratio"}:
        return (True, "") if has_balance_shape and not has_sum_shape else (
            False,
            "ratio feature requires min/max balance, not aggregate density",
        )
    if feature == "qubit_imbalance":
        return (True, "") if "abs(" in text or "imbalance" in text else (
            False,
            "qubit_imbalance requires an absolute qubit-count difference",
        )
    if feature == "scale_ratio":
        return (True, "") if "selected" in text or "backend" in text or "joint" in text else (
            False,
            "scale_ratio requires selected/backend qubit capacity",
        )
    return True, ""


def build_payload(
    repo_root: Path,
    state: dict[str, Any],
    *,
    recipe_path: Path | None = None,
    state_path: Path | None = None,
    output_root: Path | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    if (
        _slurm_bool("REPRO_PROXY_METRIC_PROPOSE_USE_SLURM", False)
        and not _slurm_bool("REPRO_PROXY_METRIC_PROPOSE_FORCE_LOCAL", False)
    ):
        if recipe_path is None or state_path is None or output_root is None or run_root is None:
            return {
                "success": False,
                "execution_mode": "proxy_metric_semantic_propose_slurm",
                "selection_mode": "llm_semantic_metric_proposal",
                "reason": "missing_slurm_submit_context",
                "warnings": ["SLURM proposal requires recipe_path, state_path, output_root, and run_root."],
            }
        return _build_slurm_payload(
            repo_root=repo_root,
            recipe_path=recipe_path,
            state_path=state_path,
            output_root=output_root,
            run_root=run_root,
        )
    prompt_payload = _prompt(repo_root, state)
    try:
        llm_payload, llm_metadata = _select_with_llm(prompt_payload)
    except Exception as exc:
        return {
            "success": False,
            "execution_mode": ACTION,
            "selection_mode": "llm_semantic_metric_proposal",
            "reason": "llm_proxy_metric_proposal_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "prompt_payload": prompt_payload,
            "warnings": [
                "No heuristic/manual fallback was used; proxy metric proposal requires LLM semantic reasoning."
            ],
        }
    candidates, warnings = _validate_llm_payload(llm_payload)
    return {
        "success": bool(candidates),
        "execution_mode": ACTION,
        "selection_mode": "llm_semantic_metric_proposal",
        "proxy_role": "pre_evolution_expensive_metric_substitute",
        "manual_proxy_source_used": False,
        "llm_metadata": llm_metadata,
        "llm_payload": llm_payload,
        "candidates": candidates,
        "materialized_feature_names": [item["materialized_feature_name"] for item in candidates],
        "prompt_payload": prompt_payload,
        "warnings": warnings,
    }


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
    artifact_dir = run_root / "proxy_metric_semantic_propose"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        repo_root,
        state,
        recipe_path=recipe_path,
        state_path=state_path,
        output_root=Path(args.output_root).resolve(),
        run_root=run_root,
    )
    payload["tool"] = TOOL_NAME
    payload["action"] = ACTION
    payload["fsm_state_before"] = get_fsm_state(state)
    metrics_path = artifact_dir / "proxy_metric_semantic_propose.metrics.json"
    _write_json(metrics_path, payload)
    payload["metrics_path"] = str(metrics_path)
    state["last_proxy_metric_semantic_proposal"] = payload
    state["last_step"] = TOOL_NAME
    state["last_status"] = "proxy_metric_semantic_propose_succeeded" if payload.get("success") else "proxy_metric_semantic_propose_failed"
    set_fsm_state(state, "APPLY_FIX")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
