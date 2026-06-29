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
from tool_proxy_metric_semantic_propose import (  # noqa: E402
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    MATERIALIZED_FEATURES,
    METADATA_KEYS,
    _call_openai_compatible,
    _extract_json_object,
    _paper_context,
    _repo_context,
)


ACTION = "proxy_semantic_factor_brainstorm"
TOOL_NAME = "repro_proxy_semantic_factor_brainstorm"


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


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _prompt(repo_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    param_probe = state.get("last_openevolve_param_probe") or {}
    objective_requirements = param_probe.get("objective_requirements") if isinstance(param_probe, dict) else {}
    return {
        "task": "Brainstorm semantic factors that could explain fidelity penalty in QOS Fig11 multiprogramming pair selection before choosing a concrete proxy metric.",
        "important_rule": (
            "Do not choose from the materialized feature menu first. First infer latent physical/circuit factors "
            "from the paper and repository semantics, then map each factor to raw metadata and materialized features if possible."
        ),
        "paper_context": _paper_context(state),
        "objective_requirements": objective_requirements,
        "repo_context": _repo_context(repo_root),
        "materialization_boundary": {
            "available_raw_metadata_keys": METADATA_KEYS,
            "currently_materialized_pair_features": MATERIALIZED_FEATURES,
            "cheap_source": "benchmark QASM metadata and repository circuit metadata APIs",
            "cannot_use": [
                "new simulation",
                "new QPU execution",
                "manual reproduction outputs as authority",
            ],
        },
        "required_json_schema": {
            "brainstorm_mode": "llm_semantic_factor_brainstorm",
            "semantic_factors": [
                {
                    "name": "short factor name, e.g. temporal overlap balance",
                    "fidelity_hypothesis": "why this factor may correlate with fidelity penalty",
                    "paper_rationale": "paper-grounded rationale",
                    "repo_evidence": "repo code/metadata evidence",
                    "required_raw_metadata": ["raw metadata keys"],
                    "candidate_materialized_features": ["available materialized feature names, if any"],
                    "cheap_materializable": "true|false|unknown",
                    "missing_materializer_request": "needed computable feature if not already materialized",
                }
            ],
            "rejected_or_expensive_factors": [
                {"name": "factor", "reason": "why not cheap/materializable"}
            ],
        },
    }


def _select_with_llm(prompt_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    api_base = os.getenv("REPRO_PROXY_FACTOR_BRAINSTORM_API_BASE") or os.getenv("REPRO_AGENT_API_BASE") or DEFAULT_API_BASE
    api_key = os.getenv("REPRO_PROXY_FACTOR_BRAINSTORM_API_KEY") or os.getenv("REPRO_AGENT_API_KEY") or "EMPTY"
    model = os.getenv("REPRO_PROXY_FACTOR_BRAINSTORM_MODEL") or os.getenv("REPRO_AGENT_MODEL") or DEFAULT_MODEL
    response = _call_openai_compatible(
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a semantic research-agent planner. Return valid JSON only. "
                    "Brainstorm latent factors before mapping them to computable proxy features."
                ),
            },
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ],
    )
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    return _extract_json_object(content), {"api_base": api_base, "model": model}


def _validate_factor_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    factors: list[dict[str, Any]] = []
    missing_requests: list[dict[str, Any]] = []
    for item in payload.get("semantic_factors") or []:
        if not isinstance(item, dict):
            continue
        required = [str(key) for key in (item.get("required_raw_metadata") or [])]
        unsupported_keys = [key for key in required if key not in METADATA_KEYS]
        features = [str(name) for name in (item.get("candidate_materialized_features") or [])]
        unsupported_features = [name for name in features if name not in MATERIALIZED_FEATURES]
        factor = dict(item)
        factor["required_raw_metadata"] = required
        factor["candidate_materialized_features"] = [name for name in features if name in MATERIALIZED_FEATURES]
        factor["unsupported_raw_metadata"] = unsupported_keys
        factor["unsupported_materialized_features"] = unsupported_features
        cheap_value = str(item.get("cheap_materializable") or "").strip().lower()
        factor["cheap_materializable"] = cheap_value in {"true", "yes", "1"}
        factor["factor_source"] = "llm_semantic_factor_brainstorm"
        if unsupported_keys:
            warnings.append(f"factor {item.get('name')} references unsupported raw metadata: {unsupported_keys}")
        if unsupported_features:
            warnings.append(f"factor {item.get('name')} references unsupported materialized features: {unsupported_features}")
        request = str(item.get("missing_materializer_request") or "").strip()
        if request and not factor["candidate_materialized_features"]:
            missing_requests.append(
                {
                    "factor": item.get("name"),
                    "request": request,
                    "required_raw_metadata": required,
                    "unsupported_raw_metadata": unsupported_keys,
                }
            )
        factors.append(factor)
    return factors, missing_requests, warnings


def _parse_sbatch_job_id(stdout: str) -> str | None:
    for item in reversed(stdout.strip().split()):
        if item.isdigit():
            return item
    return None


def _slurm_script(
    *,
    repo_root: Path,
    recipe_path: Path,
    state_path: Path,
    output_root: Path,
    log_dir: Path,
) -> str:
    model_dir = os.getenv("REPRO_PROXY_FACTOR_SLURM_MODEL_DIR", "")
    vllm_img = os.getenv("REPRO_PROXY_FACTOR_SLURM_VLLM_IMG", "")
    wait_script = os.getenv("REPRO_PROXY_FACTOR_SLURM_WAIT_SCRIPT", "")
    python_bin = os.getenv("REPRO_PROXY_FACTOR_SLURM_PYTHON", sys.executable)
    model = os.getenv("REPRO_PROXY_FACTOR_BRAINSTORM_MODEL", DEFAULT_MODEL)
    api_key = os.getenv("REPRO_PROXY_FACTOR_BRAINSTORM_API_KEY", "")
    port = os.getenv("REPRO_PROXY_FACTOR_SLURM_PORT", "8000")
    host = os.getenv("REPRO_PROXY_FACTOR_SLURM_HOST", "127.0.0.1")
    partition = os.getenv("REPRO_PROXY_FACTOR_SLURM_PARTITION", "")
    account = os.getenv("REPRO_PROXY_FACTOR_SLURM_ACCOUNT", "")
    time_limit = os.getenv("REPRO_PROXY_FACTOR_SLURM_TIME", "01:00:00")
    mem = os.getenv("REPRO_PROXY_FACTOR_SLURM_MEM", "48G")
    cpus = os.getenv("REPRO_PROXY_FACTOR_SLURM_CPUS", "8")
    gres = os.getenv("REPRO_PROXY_FACTOR_SLURM_GRES", "gpu:1")
    required = {
        "REPRO_PROXY_FACTOR_SLURM_MODEL_DIR": model_dir,
        "REPRO_PROXY_FACTOR_SLURM_VLLM_IMG": vllm_img,
        "REPRO_PROXY_FACTOR_SLURM_WAIT_SCRIPT": wait_script,
        "REPRO_PROXY_FACTOR_BRAINSTORM_API_KEY": api_key,
        "REPRO_PROXY_FACTOR_SLURM_PARTITION": partition,
        "REPRO_PROXY_FACTOR_SLURM_ACCOUNT": account,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError("missing Slurm environment variables: " + ", ".join(missing))
    return f"""#!/bin/bash
#SBATCH --job-name=qos_proxy_factor
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
  --gpu-memory-utilization "${{REPRO_PROXY_FACTOR_GPU_MEMORY_UTILIZATION:-0.85}}" \\
  --max-model-len "${{REPRO_PROXY_FACTOR_MAX_MODEL_LEN:-32768}}" \\
  > "{log_dir.as_posix()}/vllm.${{SLURM_JOB_ID}}.log" 2>&1 &
VLLM_PID=$!

python3 -u "${{WAIT_SCRIPT}}" --vllm_host "${{HOST}}" --vllm_port "${{PORT}}"

export REPRO_PROXY_FACTOR_BRAINSTORM_FORCE_LOCAL=1
export REPRO_PROXY_FACTOR_BRAINSTORM_MODEL="${{MODEL}}"
export REPRO_PROXY_FACTOR_BRAINSTORM_API_BASE="${{API_BASE}}"
export REPRO_PROXY_FACTOR_BRAINSTORM_API_KEY="${{API_KEY}}"

"${{PYTHON_BIN}}" evaluation/agent_framework/reproduce/tools/tool_proxy_semantic_factor_brainstorm.py \\
  --recipe "${{RECIPE}}" \\
  --state "${{STATE}}" \\
  --output-root "${{OUTPUT_ROOT}}"
"""


def _build_slurm_payload(
    *,
    repo_root: Path,
    recipe_path: Path,
    state_path: Path,
    output_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    artifact_dir = run_root / "proxy_semantic_factor_brainstorm_slurm"
    log_dir = artifact_dir / "logs"
    script_path = artifact_dir / "run_proxy_semantic_factor_brainstorm.slurm"
    dry_run = _bool_env("REPRO_PROXY_FACTOR_SLURM_DRY_RUN", False)
    sbatch = shutil.which("sbatch")
    payload: dict[str, Any] = {
        "success": False,
        "execution_mode": "proxy_semantic_factor_brainstorm_slurm",
        "brainstorm_mode": "llm_semantic_factor_brainstorm",
        "manual_proxy_source_used": False,
        "dry_run": dry_run,
        "sbatch_available": bool(sbatch),
        "slurm_script": str(script_path),
        "log_dir": str(log_dir),
        "warnings": [],
    }
    _write_text(
        script_path,
        _slurm_script(
            repo_root=repo_root,
            recipe_path=recipe_path,
            state_path=state_path,
            output_root=output_root,
            log_dir=log_dir,
        ),
    )
    if dry_run:
        payload["status"] = "slurm_script_ready"
        payload["warnings"].append("SLURM job was not submitted; set REPRO_PROXY_FACTOR_SLURM_DRY_RUN=0 to call sbatch")
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
        payload["warnings"].append("SLURM brainstorm job failed before producing semantic factors")
        return payload
    refreshed_state = load_state(state_path)
    inner = refreshed_state.get("last_proxy_semantic_factor_brainstorm")
    if not isinstance(inner, dict):
        payload["warnings"].append("SLURM brainstorm job completed but did not update state")
        return payload
    payload.update(inner)
    payload["execution_mode"] = "proxy_semantic_factor_brainstorm_slurm"
    payload["slurm_job_id"] = job_id
    payload["slurm_stdout"] = proc.stdout.strip()
    payload["slurm_stderr"] = proc.stderr.strip()
    payload["slurm_script"] = str(script_path)
    payload["log_dir"] = str(log_dir)
    payload["success"] = bool(inner.get("success") and inner.get("semantic_factors"))
    return payload


def build_payload(
    repo_root: Path,
    state: dict[str, Any],
    *,
    recipe_path: Path | None = None,
    state_path: Path | None = None,
    output_root: Path | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    if _bool_env("REPRO_PROXY_FACTOR_BRAINSTORM_USE_SLURM", False) and not _bool_env(
        "REPRO_PROXY_FACTOR_BRAINSTORM_FORCE_LOCAL", False
    ):
        if recipe_path is None or state_path is None or output_root is None or run_root is None:
            return {
                "success": False,
                "execution_mode": "proxy_semantic_factor_brainstorm_slurm",
                "brainstorm_mode": "llm_semantic_factor_brainstorm",
                "reason": "missing_slurm_submit_context",
                "warnings": ["SLURM brainstorm requires recipe_path, state_path, output_root, and run_root."],
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
            "brainstorm_mode": "llm_semantic_factor_brainstorm",
            "reason": "llm_semantic_factor_brainstorm_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "prompt_payload": prompt_payload,
            "warnings": ["No heuristic/manual fallback was used; semantic factor brainstorming requires LLM reasoning."],
        }
    factors, missing_requests, warnings = _validate_factor_payload(llm_payload)
    return {
        "success": bool(factors),
        "execution_mode": ACTION,
        "brainstorm_mode": "llm_semantic_factor_brainstorm",
        "manual_proxy_source_used": False,
        "llm_metadata": llm_metadata,
        "llm_payload": llm_payload,
        "semantic_factors": factors,
        "missing_materializer_requests": missing_requests,
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
    artifact_dir = run_root / "proxy_semantic_factor_brainstorm"
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
    metrics_path = artifact_dir / "proxy_semantic_factor_brainstorm.metrics.json"
    _write_json(metrics_path, payload)
    payload["metrics_path"] = str(metrics_path)
    state["last_proxy_semantic_factor_brainstorm"] = payload
    state["last_step"] = TOOL_NAME
    state["last_status"] = "proxy_semantic_factor_brainstorm_succeeded" if payload.get("success") else "proxy_semantic_factor_brainstorm_failed"
    set_fsm_state(state, "APPLY_FIX")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
