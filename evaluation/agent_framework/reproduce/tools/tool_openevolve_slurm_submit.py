#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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


ACTION = "openevolve_slurm_submit"
TOOL_NAME = "repro_openevolve_slurm_submit"


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


def _artifact_rel(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def _resolve_artifact(repo_root: Path, raw: Any, *, must_exist: bool = True) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path if (path.exists() or not must_exist) else None


def _proxy_artifacts(repo_root: Path, state: dict[str, Any]) -> dict[str, Path | None]:
    proxy = state.get("last_openevolve_proxy_search") or {}
    cfg = proxy.get("evolution_config") or {}
    return {
        "initial_program": _resolve_artifact(repo_root, cfg.get("initial_program")),
        "evaluator": _resolve_artifact(repo_root, cfg.get("evaluator")),
        "config": _resolve_artifact(repo_root, cfg.get("config_path")),
        "output_dir": _resolve_artifact(repo_root, cfg.get("output_dir"), must_exist=False),
        "training_data": _resolve_artifact(repo_root, cfg.get("training_data")),
        "metrics": _resolve_artifact(repo_root, proxy.get("metrics_path")),
    }


def _slurm_script(
    *,
    repo_root: Path,
    artifacts: dict[str, Path | None],
    log_dir: Path,
) -> str:
    initial = artifacts["initial_program"]
    evaluator = artifacts["evaluator"]
    config = artifacts["config"]
    output_dir = artifacts["output_dir"]
    if initial is None or evaluator is None or config is None or output_dir is None:
        raise ValueError("missing OpenEvolve proxy-search artifacts")
    model_dir = os.getenv("REPRO_OPENEVOLVE_SLURM_MODEL_DIR", "")
    vllm_img = os.getenv("REPRO_OPENEVOLVE_SLURM_VLLM_IMG", "")
    wait_script = os.getenv("REPRO_OPENEVOLVE_SLURM_WAIT_SCRIPT", "")
    python_bin = os.getenv("REPRO_OPENEVOLVE_SLURM_PYTHON", sys.executable)
    model = os.getenv("REPRO_OPENEVOLVE_MODEL", "Qwen2.5-14B-Instruct")
    api_key = os.getenv("REPRO_OPENEVOLVE_API_KEY", "")
    port = os.getenv("REPRO_OPENEVOLVE_SLURM_PORT", "8000")
    host = os.getenv("REPRO_OPENEVOLVE_SLURM_HOST", "127.0.0.1")
    partition = os.getenv("REPRO_OPENEVOLVE_SLURM_PARTITION", "")
    account = os.getenv("REPRO_OPENEVOLVE_SLURM_ACCOUNT", "")
    time_limit = os.getenv("REPRO_OPENEVOLVE_SLURM_TIME", "12:00:00")
    mem = os.getenv("REPRO_OPENEVOLVE_SLURM_MEM", "64G")
    cpus = os.getenv("REPRO_OPENEVOLVE_SLURM_CPUS", "16")
    gres = os.getenv("REPRO_OPENEVOLVE_SLURM_GRES", "gpu:1")
    openevolve_root = repo_root / "third_party" / "openevolve"
    required = {
        "REPRO_OPENEVOLVE_SLURM_MODEL_DIR": model_dir,
        "REPRO_OPENEVOLVE_SLURM_VLLM_IMG": vllm_img,
        "REPRO_OPENEVOLVE_SLURM_WAIT_SCRIPT": wait_script,
        "REPRO_OPENEVOLVE_API_KEY": api_key,
        "REPRO_OPENEVOLVE_SLURM_PARTITION": partition,
        "REPRO_OPENEVOLVE_SLURM_ACCOUNT": account,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError("missing Slurm environment variables: " + ", ".join(missing))
    return f"""#!/bin/bash
#SBATCH --job-name=qos_oe_proxy
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
OPENEVOLVE_ROOT={openevolve_root.as_posix()!r}
INITIAL_PROGRAM={initial.as_posix()!r}
EVALUATOR={evaluator.as_posix()!r}
CONFIG={config.as_posix()!r}
OUTPUT_DIR={output_dir.as_posix()!r}

mkdir -p {log_dir.as_posix()!r} "${{OUTPUT_DIR}}"

cleanup() {{
  if [[ -n "${{VLLM_PID:-}}" ]]; then
    kill "${{VLLM_PID}}" >/dev/null 2>&1 || true
    wait "${{VLLM_PID}}" >/dev/null 2>&1 || true
  fi
}}
trap cleanup EXIT

cd "${{REPO_ROOT}}"

if [[ "${{REPRO_OPENEVOLVE_SLURM_INSTALL_DEPS:-1}}" != "0" ]]; then
  "${{PYTHON_BIN}}" -m pip install -e "${{OPENEVOLVE_ROOT}}"
fi

apptainer run --nv --bind /projects,/work,/u "${{VLLM_IMG}}" \\
  vllm serve "${{MODEL_DIR}}" \\
  --served-model-name "${{MODEL}}" \\
  --host 0.0.0.0 \\
  --port "${{PORT}}" \\
  --api-key "${{API_KEY}}" \\
  --dtype auto \\
  --tensor-parallel-size 1 \\
  --gpu-memory-utilization "${{REPRO_OPENEVOLVE_GPU_MEMORY_UTILIZATION:-0.85}}" \\
  --max-model-len "${{REPRO_OPENEVOLVE_MAX_MODEL_LEN:-32768}}" \\
  > "{log_dir.as_posix()}/vllm.${{SLURM_JOB_ID}}.log" 2>&1 &
VLLM_PID=$!

python3 -u "${{WAIT_SCRIPT}}" --vllm_host "${{HOST}}" --vllm_port "${{PORT}}"

export API_BASE API_KEY MODEL

"${{PYTHON_BIN}}" - <<'PY'
import json
import os
import sys
import time
import urllib.request

api_base = os.environ["API_BASE"].rstrip("/")
api_key = os.environ["API_KEY"]
model = os.environ["MODEL"]
deadline = time.time() + int(os.environ.get("REPRO_OPENEVOLVE_MODEL_READY_TIMEOUT", "300"))
last_error = None
while time.time() < deadline:
    try:
        req = urllib.request.Request(
            api_base + "/models",
            headers={{"Authorization": "Bearer " + api_key}},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        model_ids = [str(item.get("id") or "") for item in payload.get("data", []) if isinstance(item, dict)]
        if model in model_ids:
            print("vLLM model ready:", model)
            raise SystemExit(0)
        last_error = "available models: " + ",".join(model_ids)
    except SystemExit:
        raise
    except Exception as exc:
        last_error = type(exc).__name__ + ": " + str(exc)
    time.sleep(5)
print("Timed out waiting for served model " + model + "; " + str(last_error), file=sys.stderr)
raise SystemExit(1)
PY

export PYTHONPATH="${{OPENEVOLVE_ROOT}}:${{PYTHONPATH:-}}"
export REPRO_OPENEVOLVE_DRY_RUN=0
export REPRO_OPENEVOLVE_MODEL="${{MODEL}}"
export REPRO_OPENEVOLVE_API_BASE="${{API_BASE}}"
export REPRO_OPENEVOLVE_API_KEY="${{API_KEY}}"

"${{PYTHON_BIN}}" -m openevolve.cli \\
  "${{INITIAL_PROGRAM}}" \\
  "${{EVALUATOR}}" \\
  --config "${{CONFIG}}" \\
  --output "${{OUTPUT_DIR}}"
"""


def _parse_sbatch_job_id(stdout: str) -> str | None:
    parts = stdout.strip().split()
    for item in reversed(parts):
        if item.isdigit():
            return item
    return None


def build_payload(repo_root: Path, run_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    artifact_dir = run_root / "openevolve_slurm_submit"
    log_dir = artifact_dir / "logs"
    artifacts = _proxy_artifacts(repo_root, state)
    missing = [name for name, path in artifacts.items() if name in {"initial_program", "evaluator", "config", "output_dir"} and path is None]
    script_path = artifact_dir / "run_openevolve_proxy_search.slurm"
    dry_run = os.getenv("REPRO_OPENEVOLVE_SLURM_DRY_RUN", "1").strip().lower() not in {"0", "false", "no", "off"}
    sbatch = shutil.which("sbatch")
    payload: dict[str, Any] = {
        "success": False,
        "execution_mode": "openevolve_slurm_submit",
        "not_fig11_strict_success": True,
        "dry_run": dry_run,
        "sbatch_available": bool(sbatch),
        "slurm_script": str(script_path),
        "log_dir": str(log_dir),
        "proxy_artifacts": {name: _artifact_rel(repo_root, path) for name, path in artifacts.items()},
        "warnings": [],
    }
    if missing:
        payload["warnings"].append("missing OpenEvolve artifacts: " + ", ".join(missing))
        return payload
    script = _slurm_script(repo_root=repo_root, artifacts=artifacts, log_dir=log_dir)
    _write_text(script_path, script)
    if dry_run:
        payload["success"] = True
        payload["status"] = "slurm_script_ready"
        payload["warnings"].append("SLURM job was not submitted; set REPRO_OPENEVOLVE_SLURM_DRY_RUN=0 to call sbatch")
        return payload
    if not sbatch:
        payload["status"] = "sbatch_unavailable"
        payload["warnings"].append("sbatch not found on PATH")
        return payload
    proc = subprocess.run([sbatch, str(script_path)], cwd=repo_root, text=True, capture_output=True, check=False)
    job_id = _parse_sbatch_job_id(proc.stdout)
    payload.update(
        {
            "success": proc.returncode == 0 and bool(job_id),
            "status": "submitted" if proc.returncode == 0 and job_id else "submit_failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "job_id": job_id,
        }
    )
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
    payload = build_payload(repo_root, run_root, state)
    payload["tool"] = TOOL_NAME
    payload["action"] = ACTION
    payload["fsm_state_before"] = get_fsm_state(state)
    artifact_path = run_root / "openevolve_slurm_submit" / "openevolve_slurm_submit.json"
    payload["artifact_path"] = str(artifact_path)
    _write_json(artifact_path, payload)
    state["last_openevolve_slurm_submit"] = payload
    state["last_step"] = TOOL_NAME
    state["last_status"] = "openevolve_slurm_submit_succeeded" if payload.get("success") else "openevolve_slurm_submit_failed"
    set_fsm_state(state, "APPLY_FIX")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
