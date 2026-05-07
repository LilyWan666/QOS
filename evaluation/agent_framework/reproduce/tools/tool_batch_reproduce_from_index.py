#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch reproduction from generated recipes.index.json and emit a summary report."
    )
    parser.add_argument("--index", required=True, help="Path to recipes.index.json")
    parser.add_argument("--repo-root", required=True, help="Repo root")
    parser.add_argument(
        "--mode",
        choices=["core", "openclaw_tool_loop", "openclaw_agent"],
        default="core",
        help=(
            "Execution mode: core uses run_reproduce.py; "
            "openclaw_tool_loop uses claw reproduce --tool-loop; "
            "openclaw_agent uses claw reproduce (agent recovery path)."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="temp/agent_framework/reproduce/figure_batch_runs",
        help="Base output dir for batch runs (relative to repo root allowed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run first N recipes (0 means all).",
    )
    parser.add_argument(
        "--figure-code-map",
        default="",
        help="Optional path to figure code map JSON produced by tool_figure_code_map.py",
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="When set, only run figures marked reproduce_required=true in --figure-code-map.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue remaining recipes even when one fails.",
    )
    parser.add_argument("--python-executable", default="", help="Python executable override for core mode.")
    parser.add_argument("--claw-rust-root", default="claw-code/rust", help="Rust workspace for claw binary.")
    parser.add_argument("--model", default="", help="Model for openclaw_tool_loop mode.")
    parser.add_argument("--api-base", default="", help="API base for openclaw_tool_loop mode.")
    parser.add_argument("--api-key", default="", help="API key for openclaw_tool_loop mode.")
    parser.add_argument("--max-agent-steps", type=int, default=8, help="Max steps for openclaw_tool_loop mode.")
    parser.add_argument("--allow-offline-agent", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_from_root(repo_root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (repo_root / p).resolve()


def load_required_figure_ids(path: Path) -> set[str]:
    payload = load_json(path)
    required: set[str] = set()
    for fig in payload.get("figures", []):
        if bool(fig.get("reproduce_required")):
            fid = str(fig.get("figure_id", "")).strip()
            if fid:
                required.add(fid)
    return required


def run_one_core(
    repo_root: Path,
    recipe_path: Path,
    run_output_root: Path,
    python_executable: str,
) -> tuple[int, str, str, list[str]]:
    py = python_executable or os.environ.get("PYTHON", "python")
    cmd = [
        py,
        str(repo_root / "evaluation/agent_framework/reproduce/run_reproduce.py"),
        "--recipe",
        str(recipe_path),
        "--output-root",
        str(run_output_root),
    ]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr, cmd


def run_one_openclaw(
    repo_root: Path,
    claw_rust_root: Path,
    recipe_path: Path,
    run_output_root: Path,
    model: str,
    api_base: str,
    api_key: str,
    max_agent_steps: int,
    allow_offline_agent: bool,
    python_executable: str,
) -> tuple[int, str, str, list[str]]:
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "rusty-claude-cli",
        "--",
        "reproduce",
        "--tool-loop",
        "--recipe",
        str(recipe_path),
        "--output-root",
        str(run_output_root),
        "--max-agent-steps",
        str(max_agent_steps),
        "--exit-policy",
        "strict",
    ]
    if python_executable:
        cmd.extend(["--python-executable", python_executable])
    if model:
        cmd.extend(["--model", model])
    if api_base:
        cmd.extend(["--api-base", api_base])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if allow_offline_agent:
        cmd.append("--allow-offline-agent")

    proc = subprocess.run(cmd, cwd=claw_rust_root, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr, cmd


def run_one_openclaw_agent(
    repo_root: Path,
    claw_rust_root: Path,
    recipe_path: Path,
    model: str,
    api_base: str,
    api_key: str,
    max_agent_steps: int,
    allow_offline_agent: bool,
    python_executable: str,
) -> tuple[int, str, str, list[str]]:
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "rusty-claude-cli",
        "--",
        "reproduce",
        "--recipe",
        str(recipe_path),
        "--max-agent-steps",
        str(max_agent_steps),
        "--exit-policy",
        "lenient",
    ]
    if model:
        cmd.extend(["--model", model])
    if api_base:
        cmd.extend(["--api-base", api_base])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if allow_offline_agent:
        cmd.append("--allow-offline-agent")
    if python_executable:
        cmd.extend(["--python-executable", python_executable])

    proc = subprocess.run(cmd, cwd=claw_rust_root, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr, cmd


def list_run_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {p.resolve() for p in root.iterdir() if p.is_dir()}


def extract_failure_signals(run_dir: Path) -> list[str]:
    signals: list[str] = []
    metrics_paths = sorted(run_dir.glob("attempts/attempt_*/metrics.json"))
    if metrics_paths:
        try:
            metrics = load_json(metrics_paths[-1])
            for item in metrics.get("checked_modules", []):
                if isinstance(item, dict) and not item.get("ok"):
                    module = str(item.get("module", "")).strip()
                    error = str(item.get("error", "")).strip()
                    if module and error:
                        signals.append(f"{module}: {error}")
        except Exception:
            pass
    return signals[:5]


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    index_path = resolve_from_root(repo_root, args.index)
    output_root = resolve_from_root(repo_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_run_dir = output_root / f"batch_{int(time.time())}"
    batch_run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = batch_run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    index = load_json(index_path)
    recipes_all = list(index.get("recipes", []))

    filtered_by_required = False
    required_ids: set[str] = set()
    if args.required_only:
        if not args.figure_code_map:
            raise SystemExit("--required-only requires --figure-code-map")
        map_path = resolve_from_root(repo_root, args.figure_code_map)
        required_ids = load_required_figure_ids(map_path)
        recipes = [r for r in recipes_all if str(r.get("figure_id", "")).strip() in required_ids]
        filtered_by_required = True
    else:
        recipes = recipes_all

    if args.limit and args.limit > 0:
        recipes = recipes[: args.limit]

    claw_rust_root = resolve_from_root(repo_root, args.claw_rust_root)
    summary_rows: list[dict[str, Any]] = []
    start = time.time()

    for i, item in enumerate(recipes, start=1):
        figure_id = str(item.get("figure_id", "unknown"))
        recipe_path = resolve_from_root(repo_root, str(item.get("recipe_path", "")))
        row = {
            "index": i,
            "figure_id": figure_id,
            "recipe_path": str(recipe_path),
            "mode": args.mode,
        }
        t0 = time.time()
        runs_root = batch_run_dir / "runs"
        before_runs = list_run_dirs(runs_root)
        if args.mode == "core":
            rc, stdout, stderr, cmd = run_one_core(
                repo_root=repo_root,
                recipe_path=recipe_path,
                run_output_root=runs_root,
                python_executable=args.python_executable,
            )
        elif args.mode == "openclaw_tool_loop":
            rc, stdout, stderr, cmd = run_one_openclaw(
                repo_root=repo_root,
                claw_rust_root=claw_rust_root,
                recipe_path=recipe_path,
                run_output_root=runs_root,
                model=args.model,
                api_base=args.api_base,
                api_key=args.api_key,
                max_agent_steps=args.max_agent_steps,
                allow_offline_agent=args.allow_offline_agent,
                python_executable=args.python_executable,
            )
        else:
            rc, stdout, stderr, cmd = run_one_openclaw_agent(
                repo_root=repo_root,
                claw_rust_root=claw_rust_root,
                recipe_path=recipe_path,
                model=args.model,
                api_base=args.api_base,
                api_key=args.api_key,
                max_agent_steps=args.max_agent_steps,
                allow_offline_agent=args.allow_offline_agent,
                python_executable=args.python_executable,
            )
        after_runs = list_run_dirs(runs_root)
        new_runs = sorted(after_runs - before_runs)
        run_dir = new_runs[-1] if new_runs else None
        elapsed = time.time() - t0
        stdout_path = logs_dir / f"{i:03d}_{figure_id}.stdout.log"
        stderr_path = logs_dir / f"{i:03d}_{figure_id}.stderr.log"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        row.update(
            {
                "returncode": rc,
                "status": "success" if rc == 0 else "failed",
                "elapsed_seconds": elapsed,
                "command": cmd,
                "run_dir": str(run_dir) if run_dir is not None else None,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "failure_signals": extract_failure_signals(run_dir) if run_dir is not None else [],
            }
        )
        summary_rows.append(row)

        if rc != 0 and not args.continue_on_failure:
            break

    success_count = sum(1 for x in summary_rows if x["status"] == "success")
    failed_count = len(summary_rows) - success_count
    summary = {
        "index_path": str(index_path),
        "repo_root": str(repo_root),
        "mode": args.mode,
        "batch_run_dir": str(batch_run_dir),
        "required_only": bool(args.required_only),
        "figure_code_map": args.figure_code_map if args.figure_code_map else None,
        "filtered_by_required": filtered_by_required,
        "total_index_recipes": len(recipes_all),
        "required_figure_count": len(required_ids) if args.required_only else None,
        "total_requested": len(recipes),
        "total_executed": len(summary_rows),
        "success_count": success_count,
        "failed_count": failed_count,
        "elapsed_seconds": time.time() - start,
        "rows": summary_rows,
    }
    write_json(batch_run_dir / "batch_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
