#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
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


ACTION = "openevolve_slurm_collect"
TOOL_NAME = "repro_openevolve_slurm_collect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output-root", default="temp/agent_framework/reproduce/tools/runs")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve(repo_root: Path, raw: Any) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _artifact_rel(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def _sacct(job_id: str) -> dict[str, Any]:
    sacct = shutil.which("sacct")
    if not sacct or not job_id:
        return {"available": bool(sacct), "job_id": job_id, "rows": []}
    proc = subprocess.run(
        [sacct, "-j", job_id, "--format=JobID,State,ExitCode,Elapsed", "--parsable2", "--noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 4:
            rows.append({"job_id": parts[0], "state": parts[1], "exit_code": parts[2], "elapsed": parts[3]})
    return {"available": True, "returncode": proc.returncode, "stderr": proc.stderr.strip(), "rows": rows}


def _best_program(output_dir: Path | None) -> Path | None:
    if output_dir is None or not output_dir.exists():
        return None
    candidates = []
    for pattern in ("**/best_program.py", "**/best_program*.py", "**/program.py"):
        candidates.extend(output_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_payload(repo_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    submit = state.get("last_openevolve_slurm_submit") or {}
    proxy = state.get("last_openevolve_proxy_search") or {}
    cfg = proxy.get("evolution_config") or {}
    output_dir = _resolve(repo_root, cfg.get("output_dir"))
    initial_program = _resolve(repo_root, cfg.get("initial_program"))
    job_id = str(submit.get("job_id") or "").strip()
    status = _sacct(job_id)
    best = _best_program(output_dir)
    initial_hash = _sha256(initial_program)
    best_hash = _sha256(best)
    mutation_applied = bool(initial_hash and best_hash and initial_hash != best_hash)
    payload = {
        "success": bool(best and best.exists() and mutation_applied),
        "execution_mode": "openevolve_slurm_collect",
        "not_fig11_strict_success": True,
        "job_id": job_id or None,
        "slurm_status": status,
        "openevolve_output_dir": _artifact_rel(repo_root, output_dir),
        "initial_program": _artifact_rel(repo_root, initial_program),
        "evolved_program": _artifact_rel(repo_root, best),
        "mutation_verification": {
            "initial_program_sha256": initial_hash,
            "evolved_program_sha256": best_hash,
            "mutation_applied": mutation_applied,
            "requires_code_hash_change": True,
        },
        "warnings": [],
    }
    if not job_id and not submit.get("dry_run"):
        payload["warnings"].append("missing SLURM job id")
    if submit.get("dry_run"):
        payload["warnings"].append("submit tool ran in dry-run mode; no SLURM job was launched")
    if best is None:
        payload["warnings"].append("missing evolved best_program.py from OpenEvolve output")
    if best is not None and not mutation_applied:
        payload["warnings"].append("OpenEvolve best program hash matches initial program; no-op mutation rejected")
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
    payload = build_payload(repo_root, state)
    payload["tool"] = TOOL_NAME
    payload["action"] = ACTION
    payload["fsm_state_before"] = get_fsm_state(state)
    artifact_path = run_root / "openevolve_slurm_collect" / "openevolve_slurm_collect.json"
    payload["artifact_path"] = str(artifact_path)
    _write_json(artifact_path, payload)
    state["last_openevolve_slurm_collect"] = payload
    if payload.get("success") and payload.get("evolved_program"):
        state["last_openevolve_evolved_program"] = payload.get("evolved_program")
    elif "last_openevolve_evolved_program" in state:
        state.pop("last_openevolve_evolved_program", None)
    state["last_step"] = TOOL_NAME
    state["last_status"] = "openevolve_slurm_collect_succeeded" if payload.get("success") else "openevolve_slurm_collect_failed"
    set_fsm_state(state, "APPLY_FIX")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
