#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import agent_tool_runtime as atr
import run_reproduce as rr


DEFAULT_MODEL = "Qwen2.5-14B-Instruct"
DEFAULT_API_BASE = "http://localhost:8000/v1"
WORKSPACE_IGNORE_NAMES = {
    ".git",
    "temp",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}
SHIM_RECOVERY_ACTIONS = {
    "patch_missing_module_file",
    "patch_missing_symbol_import",
}
BUILTIN_ONLY_RECOVERY_ACTIONS: set[str] = {
    "runtime_env_select",
    "resolve_dependency_mapping",
    "install_missing_module",
    "select_or_prepare_python_environment",
    "source_path_fix",
}
STRICT_BLOCKED_RECOVERY_ACTION_PREFIXES: tuple[str, ...] = ()
MAX_FAILED_PER_ACTION_DEFAULT = 2
DEPENDENCY_IMPORT_TO_PIP: dict[str, str] = {
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "pil": "pillow",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "qiskit_aer": "qiskit-aer",
}


def maybe_redirect_to_native_tool_loop(argv: list[str]) -> int | None:
    """Route legacy runner invocations to native tool-loop by default.

    Set REPRO_LEGACY_RUNNER_MODE=legacy to force the historical python path.
    """
    mode = os.getenv("REPRO_LEGACY_RUNNER_MODE", "redirect").strip().lower()
    if mode in {"legacy", "off", "0", "false", "no"}:
        return None
    if any(token in {"-h", "--help"} for token in argv):
        return None

    repo_root = Path(__file__).resolve().parents[3]
    cargo_root = repo_root / "claw-code" / "rust"
    cargo_manifest = cargo_root / "Cargo.toml"
    if not cargo_root.exists():
        sys.stderr.write(
            "[WARN] native tool-loop workspace not found; falling back to legacy python runner\n"
        )
        return None

    cmd = [
        "cargo",
        "run",
        "-q",
        "--manifest-path",
        str(cargo_manifest),
        "-p",
        "rusty-claude-cli",
        "--",
        "reproduce",
        "--tool-loop",
        *argv,
    ]
    sys.stderr.write(
        "[DEPRECATED] run_reproduce_agent.py is now a compatibility shim.\n"
        "[DEPRECATED] Redirecting to native tool-loop.\n"
        f"[DEPRECATED] Command: {' '.join(cmd)}\n"
    )
    try:
        proc = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        sys.stderr.write(
            "[WARN] cargo not found in PATH; falling back to legacy python runner\n"
        )
        return None
    return proc.returncode


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument(
        "--output-root",
        default="temp/agent_framework/reproduce_agent/runs",
        help="Directory where reproduce-agent artifacts are written.",
    )
    parser.add_argument("--python-executable")
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=4,
        help="Maximum number of LLM decision steps after the initial attempt.",
    )
    parser.add_argument("--model", default=os.getenv("REPRO_AGENT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-base", default=os.getenv("REPRO_AGENT_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.getenv("REPRO_AGENT_API_KEY", "EMPTY"))
    parser.add_argument(
        "--allow-offline-agent",
        action="store_true",
        default=parse_bool_env("REPRO_AGENT_ALLOW_OFFLINE_AGENT", False),
        help=(
            "Allow running without a reachable OpenAI-compatible endpoint. "
            "Default behavior is fail-fast when endpoint probe fails."
        ),
    )
    parser.add_argument(
        "--allow-run-on-preflight-failure",
        action="store_true",
        help="Allow the first reproduce attempt to run even when preflight fails.",
    )
    parser.add_argument(
        "--exit-policy",
        choices=["strict", "lenient"],
        default=os.getenv("REPRO_AGENT_EXIT_POLICY", "lenient"),
        help=(
            "strict: return non-zero when final reproduce status is not success/preflight_only. "
            "lenient: return zero when the agent loop completes, even if reproduction remains unresolved."
        ),
    )
    parser.add_argument(
        "--disable-rule-fallback",
        action="store_true",
        help=(
            "Disable deterministic fallback decisions/recovery when model endpoint is unavailable "
            "or returns malformed output."
        ),
    )
    parser.add_argument(
        "--disable-builtin-recovery",
        action="store_true",
        default=parse_bool_env("REPRO_AGENT_DISABLE_BUILTIN_RECOVERY", False),
        help=(
            "Disable builtin compatibility recovery helpers. "
            "When disabled, recoveries must come from explicit handlers/model patching only."
        ),
    )
    parser.add_argument(
        "--strict-agent-tooling",
        action="store_true",
        default=parse_bool_env("REPRO_AGENT_STRICT_AGENT_TOOLING", False),
        help=(
            "Pure agent tooling mode: require live endpoint, disable rule fallback, and disable "
            "builtin compatibility recovery actions."
        ),
    )
    parser.add_argument(
        "--isolation-mode",
        choices=["git_snapshot", "live_copy"],
        default=os.getenv("REPRO_ISOLATION_MODE", "git_snapshot"),
        help=(
            "Workspace isolation strategy. git_snapshot (default) extracts from a fixed git ref; "
            "live_copy copies the current working tree."
        ),
    )
    parser.add_argument(
        "--baseline-ref",
        default=os.getenv("REPRO_BASELINE_REF", "HEAD"),
        help="Git ref used when --isolation-mode=git_snapshot (e.g., HEAD, main, <sha>).",
    )
    parser.add_argument(
        "--source-repo-root",
        default=os.getenv("REPRO_SOURCE_REPO_ROOT", ""),
        help="Optional source repository root override for isolation input.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def exception_payload(exc: Exception, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def infer_missing_modules_from_attempt(attempt: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def push(name: str) -> None:
        mod = str(name or "").strip()
        if not mod:
            return
        if mod in seen:
            return
        seen.add(mod)
        ordered.append(mod)

    classification = (attempt.get("diagnosis_payload") or {}).get("classification", {})
    for item in classification.get("evidence", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("missing_module"):
            push(str(item.get("missing_module")))
        for raw in item.get("missing_imports", []) or []:
            if isinstance(raw, str):
                push(raw)

    for raw in ((attempt.get("run_error_signals") or {}).get("missing_modules") or []):
        if isinstance(raw, str):
            push(raw)

    preflight_checks = (attempt.get("preflight_payload") or {}).get("checks", {})
    for check in preflight_checks.get("python_imports", []) or []:
        if not isinstance(check, dict):
            continue
        if check.get("ok"):
            continue
        name = str(check.get("name", "")).strip()
        if name:
            push(name)
        stderr = str(check.get("stderr", "") or "")
        for match in re.findall(r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]", stderr):
            push(match)
    return ordered


def rule_fallback_decision(
    attempt: dict[str, Any],
    available_actions: list[str],
    env_diagnose: dict[str, Any] | None,
    failed_action_counts: dict[str, int] | None,
    reason: str,
) -> dict[str, str]:
    failed_counts = failed_action_counts or {}

    category = (
        attempt.get("diagnosis_payload", {})
        .get("classification", {})
        .get("category", "")
    )
    preferred_for_category = {
        "optional_runtime_dependency_used_in_simulation_path": "patch_lazy_optional_import",
        "source_compat_missing_module_file": "source_path_fix",
        "source_compat_bad_import_statement": "source_path_fix",
        "source_compat_missing_symbol_import": "runtime_trace_fix",
        "source_compat_missing_symbol_reference": "runtime_trace_fix",
        "missing_runtime_dependency": "dependency_runtime_fix",
        "runtime_import_surface_mismatch": "runtime_trace_fix",
    }
    if (
        category == "runtime_import_surface_mismatch"
        and failed_counts.get("runtime_trace_fix", 0) >= 1
        and "runtime_env_select" in available_actions
        and failed_counts.get("runtime_env_select", 0) < 2
    ):
        return {
            "decision": "apply_recovery_action",
            "recovery_action": "runtime_env_select",
            "reason": f"rule_fallback_promote_env_select_after_runtime_trace_fix_failure: {reason}",
            "expected_outcome": "switch to a compatible python runtime after runtime-trace fix failure",
        }

    preferred_action = preferred_for_category.get(category)
    if (
        preferred_action
        and preferred_action in available_actions
        and failed_counts.get(preferred_action, 0) < 2
    ):
        return {
            "decision": "apply_recovery_action",
            "recovery_action": preferred_action,
            "reason": f"rule_fallback_after_model_error: {reason}",
            "expected_outcome": "apply a deterministic high-confidence recovery action",
        }

    if category in {
        "missing_runtime_dependency",
        "runtime_import_surface_mismatch",
        "missing_required_commands",
        "missing_required_paths",
    }:
        if env_diagnose is None:
            return {
                "decision": "run_env_diagnose",
                "recovery_action": "",
                "reason": f"rule_fallback_env_first_for_{category}: {reason}",
                "expected_outcome": "collect environment diagnostics before patching source files",
            }
        if "runtime_env_select" in available_actions and failed_counts.get("runtime_env_select", 0) < 2:
            return {
                "decision": "apply_recovery_action",
                "recovery_action": "runtime_env_select",
                "reason": f"rule_fallback_env_select_for_{category}: {reason}",
                "expected_outcome": "select a compatible python runtime and venv for this workspace",
            }
        if "resolve_dependency_mapping" in available_actions and failed_counts.get("resolve_dependency_mapping", 0) < 2:
            return {
                "decision": "apply_recovery_action",
                "recovery_action": "resolve_dependency_mapping",
                "reason": f"rule_fallback_dependency_mapping_for_{category}: {reason}",
                "expected_outcome": "map missing import to pip dependencies and install into isolated runtime",
            }
        if "install_missing_module" in available_actions and failed_counts.get("install_missing_module", 0) < 2:
            return {
                "decision": "apply_recovery_action",
                "recovery_action": "install_missing_module",
                "reason": f"rule_fallback_install_missing_module_for_{category}: {reason}",
                "expected_outcome": "install missing runtime dependency inside isolated runtime env",
            }
        if "select_or_prepare_python_environment" in available_actions:
            return {
                "decision": "apply_recovery_action",
                "recovery_action": "select_or_prepare_python_environment",
                "reason": f"rule_fallback_env_then_prepare_for_{category}: {reason}",
                "expected_outcome": "install or select missing runtime dependency in isolated workspace",
            }
        return {
            "decision": "stop",
            "recovery_action": "",
            "reason": f"rule_fallback_stop_after_env_for_{category}: {reason}",
            "expected_outcome": "none",
        }

    for action in available_actions:
        if action not in {"retry_reproduce", "retry_preflight"} and failed_counts.get(action, 0) < 2:
            return {
                "decision": "apply_recovery_action",
                "recovery_action": action,
                "reason": f"rule_fallback_pick_available_action: {reason}",
                "expected_outcome": "attempt a safe available recovery action",
            }

    if env_diagnose is None:
        return {
            "decision": "run_env_diagnose",
            "recovery_action": "",
            "reason": f"rule_fallback_collect_env: {reason}",
            "expected_outcome": "collect environment diagnostics for next recovery step",
        }

    return {
        "decision": "stop",
        "recovery_action": "",
        "reason": f"rule_fallback_stop: {reason}",
        "expected_outcome": "none",
    }


def copy_isolated_workspace(source_repo_root: Path, target_repo_root: Path) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in WORKSPACE_IGNORE_NAMES}
        if Path(directory).resolve() == source_repo_root.resolve():
            ignored.add(".git")
        return ignored

    if target_repo_root.exists():
        shutil.rmtree(target_repo_root)
    # Do not preserve source permission bits; source trees may be intentionally read-only.
    # The isolated workspace must stay writable for agent patching and runtime setup.
    shutil.copytree(
        source_repo_root,
        target_repo_root,
        ignore=ignore,
        copy_function=shutil.copyfile,
    )
    for directory in target_repo_root.rglob("*"):
        if directory.is_dir():
            mode = directory.stat().st_mode
            if not (mode & 0o200):
                os.chmod(directory, mode | 0o200)


def ensure_writable_workspace(root: Path) -> None:
    if not root.exists():
        return
    mode = root.stat().st_mode
    if not (mode & 0o200):
        os.chmod(root, mode | 0o200)
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        if not (mode & 0o200):
            os.chmod(path, mode | 0o200)


def export_git_snapshot(source_repo_root: Path, target_repo_root: Path, git_ref: str) -> None:
    if target_repo_root.exists():
        shutil.rmtree(target_repo_root)
    target_repo_root.parent.mkdir(parents=True, exist_ok=True)
    archive_path = target_repo_root.parent / "baseline_snapshot.tar"
    if archive_path.exists():
        archive_path.unlink()
    subprocess.run(
        [
            "git",
            "-C",
            str(source_repo_root),
            "archive",
            "--format=tar",
            "--output",
            str(archive_path),
            git_ref,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    target_repo_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r") as handle:
        try:
            handle.extractall(path=target_repo_root, filter="data")
        except TypeError:
            handle.extractall(path=target_repo_root)
    archive_path.unlink(missing_ok=True)
    ensure_writable_workspace(target_repo_root)


def resolve_git_commit(source_repo_root: Path, git_ref: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_repo_root), "rev-parse", "--verify", f"{git_ref}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def select_source_repo_root(repo_root: Path, override: str) -> Path:
    if override.strip():
        candidate = Path(override).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"source repo override does not exist: {candidate}")
    return repo_root


def infer_runtime_requirements(
    *,
    root_package: str,
    import_source_module: str = "",
    import_symbol_name: str = "",
) -> list[str]:
    if root_package == "qiskit":
        reqs = ["numpy<2", "qiskit==0.46.3", "qiskit-ibm-provider==0.10.0"]
        if import_source_module == "qiskit.circuit" and import_symbol_name == "Store":
            reqs.append("qiskit-aer==0.13.3")
        return reqs
    if root_package in {"qiskit_ibm_provider", "qiskit-ibm-provider"}:
        return ["numpy<2", "qiskit==0.46.3", "qiskit-ibm-provider==0.10.0"]
    if root_package in {"qiskit_aer", "qiskit-aer"}:
        return ["numpy<2", "qiskit==0.46.3", "qiskit-ibm-provider==0.10.0", "qiskit-aer==0.13.3"]
    if root_package:
        mapped = DEPENDENCY_IMPORT_TO_PIP.get(root_package.lower())
        if mapped:
            return [mapped]
        return [root_package.replace("_", "-")]
    return []


def infer_requirements_for_missing_module(
    missing_module: str,
    *,
    import_source_module: str = "",
    import_symbol_name: str = "",
) -> list[str]:
    parts = [part for part in str(missing_module).split(".") if part]
    root_pkg = parts[0] if parts else str(missing_module).strip()
    if not root_pkg:
        return []
    return infer_runtime_requirements(
        root_package=root_pkg,
        import_source_module=import_source_module,
        import_symbol_name=import_symbol_name,
    )


def relative_recipe_path(recipe_path: Path, repo_root: Path) -> Path:
    try:
        return recipe_path.relative_to(repo_root)
    except ValueError:
        return Path(recipe_path.name)


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


def probe_openai_endpoint(api_base: str, api_key: str) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/models"
    request = urllib.request.Request(
        url=url,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
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
        return {
            "ok": False,
            "url": url,
            "status_code": exc.code,
            "error": f"HTTPError: {exc}",
        }
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "url": url,
            "status_code": None,
            "error": f"URLError: {exc}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "status_code": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def available_recovery_actions(
    recipe: dict[str, Any],
    attempt: dict[str, Any],
    blocked_actions: set[str] | None = None,
) -> list[str]:
    planned = {
        action.get("action")
        for action in attempt["diagnosis_payload"].get("recovery_plan", {}).get("actions", [])
        if isinstance(action, dict)
    }
    missing_candidates = infer_missing_modules_from_attempt(attempt)
    missing_module = missing_candidates[0] if missing_candidates else ""
    if missing_module and not missing_module.startswith(("qos.", "qvm.", "Baseline_Multiprogramming")):
        planned.add("resolve_dependency_mapping")
        planned.add("install_missing_module")

    blocked = blocked_actions or set()
    return sorted(
        action
        for action in planned
        if action and action not in blocked and action not in SHIM_RECOVERY_ACTIONS
    )


def collect_failed_action_counts(recoveries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in recoveries:
        if not isinstance(item, dict):
            continue
        if item.get("applied"):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue
        result_payload = item.get("result")
        scoped_module = ""
        if isinstance(result_payload, dict):
            candidate = result_payload.get("missing_module") or result_payload.get("module")
            if isinstance(candidate, str):
                scoped_module = candidate.strip()
        counts[action] = counts.get(action, 0) + 1
        if scoped_module:
            scoped_key = f"{action}::{scoped_module}"
            counts[scoped_key] = counts.get(scoped_key, 0) + 1
    return counts


def collect_applied_action_counts(recoveries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in recoveries:
        if not isinstance(item, dict):
            continue
        if not item.get("applied"):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue
        result_payload = item.get("result")
        scoped_module = ""
        if isinstance(result_payload, dict):
            candidate = result_payload.get("missing_module") or result_payload.get("module")
            if isinstance(candidate, str):
                scoped_module = candidate.strip()
        counts[action] = counts.get(action, 0) + 1
        if scoped_module:
            scoped_key = f"{action}::{scoped_module}"
            counts[scoped_key] = counts.get(scoped_key, 0) + 1
    return counts


def action_failure_limit() -> int:
    raw = os.getenv("REPRO_AGENT_MAX_FAILED_PER_ACTION", str(MAX_FAILED_PER_ACTION_DEFAULT)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = MAX_FAILED_PER_ACTION_DEFAULT
    return max(1, value)


def collect_recovered_requirements(recoveries: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in recoveries:
        if not isinstance(item, dict):
            continue
        if not item.get("applied"):
            continue
        result = item.get("result") or {}
        reqs = result.get("requirements")
        if not isinstance(reqs, list):
            continue
        for req in reqs:
            if not isinstance(req, str):
                continue
            normalized = req.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def run_paper_extract(
    isolated_repo_root: Path,
    recipe_path: Path,
    run_dir: Path,
) -> dict[str, Any] | None:
    recipe = load_json(recipe_path)
    paper_cfg = recipe.get("paper", {})
    if not paper_cfg.get("documents"):
        return None

    step_dir = run_dir / "paper"
    step_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(isolated_repo_root / "evaluation/agent_framework/reproduce/paper/extract_paper_context.py"),
        "--recipe",
        str(recipe_path),
        "--output-dir",
        str(step_dir),
    ]
    proc = subprocess.run(
        command,
        cwd=isolated_repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload: dict[str, Any] = {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "artifact_dir": str(step_dir),
    }
    context_path = step_dir / "paper_context.json"
    if context_path.exists():
        payload["context_path"] = str(context_path)
        payload["context"] = load_json(context_path)
    else:
        payload["context"] = None
    write_json(step_dir / "manifest.json", payload)
    return payload


def compact_paper_context(paper_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not paper_context:
        return None
    compact_docs: list[dict[str, Any]] = []
    for doc in paper_context.get("documents", [])[:3]:
        compact_docs.append(
            {
                "label": doc.get("label"),
                "path": doc.get("path"),
                "title": doc.get("title"),
                "focus_snippets": (doc.get("focus_snippets") or [])[:5],
            }
        )
    return {
        "document_count": paper_context.get("document_count"),
        "guidance": paper_context.get("guidance", []),
        "documents": compact_docs,
    }


def entrypoint_hints(repo_root: Path) -> dict[str, list[str]]:
    hints: dict[str, list[str]] = {
        "test_scripts": [],
        "benchmark_scripts": [],
        "reproduce_harnesses": [],
    }
    test_root = repo_root / "test"
    if test_root.exists():
        for pattern in ("reproduce*.py", "run_*.py"):
            for path in sorted(test_root.glob(pattern)):
                hints["test_scripts"].append(str(path.relative_to(repo_root)))
    bench_root = repo_root / "evaluation" / "benchmarks"
    if bench_root.exists():
        for path in sorted(bench_root.rglob("*.py"))[:20]:
            hints["benchmark_scripts"].append(str(path.relative_to(repo_root)))
    harness_root = repo_root / "evaluation" / "agent_framework" / "reproduce" / "harnesses"
    if harness_root.exists():
        for path in sorted(harness_root.glob("*.py")):
            hints["reproduce_harnesses"].append(str(path.relative_to(repo_root)))
    return hints


def _normalize_entrypoint_values(values: list[Any], repo_root: Path) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        value = _normalize_entry_script(item, repo_root)
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def resolve_recipe_authoring_config(
    cfg: dict[str, Any],
    repo_root: Path,
    hints: dict[str, list[str]],
) -> tuple[dict[str, Any], list[str]]:
    effective = dict(cfg)
    warnings: list[str] = []

    blocked_raw = cfg.get("blocked_entrypoints", [])
    blocked_norm = _normalize_entrypoint_values(blocked_raw if isinstance(blocked_raw, list) else [], repo_root)
    effective["blocked_entrypoints"] = blocked_norm
    blocked_set = set(blocked_norm)

    allowed_raw = cfg.get("allowed_entrypoints", [])
    allowed_norm = _normalize_entrypoint_values(allowed_raw if isinstance(allowed_raw, list) else [], repo_root)
    allowed_existing = [path for path in allowed_norm if (repo_root / path).exists()]
    missing_allowed = [path for path in allowed_norm if not (repo_root / path).exists()]
    if missing_allowed:
        warnings.append(f"missing_allowed_entrypoints_in_snapshot={missing_allowed}")

    if allowed_norm and not allowed_existing:
        fallback_candidates = [
            *hints.get("reproduce_harnesses", []),
            *hints.get("test_scripts", []),
        ]
        fallback_norm = _normalize_entrypoint_values(fallback_candidates, repo_root)
        fallback_existing = [
            path
            for path in fallback_norm
            if (repo_root / path).exists() and path.endswith(".py") and path not in blocked_set
        ]
        if fallback_existing:
            effective["allowed_entrypoints"] = fallback_existing
            warnings.append(
                "allowed_entrypoints_all_missing_in_snapshot; "
                f"fallback_allowed_entrypoints={fallback_existing}"
            )
        else:
            effective["allowed_entrypoints"] = []
            warnings.append("allowed_entrypoints_all_missing_in_snapshot_and_no_fallback_found")
    else:
        effective["allowed_entrypoints"] = allowed_existing
    return effective, warnings


def _extract_entry_script(command: list[Any]) -> str | None:
    for token in command:
        if not isinstance(token, str):
            continue
        candidate = token.strip()
        if not candidate or candidate.startswith("{") or candidate.startswith("-"):
            continue
        if candidate.endswith(".py"):
            return candidate
    return None


def _normalize_entry_script(script: str, repo_root: Path) -> str:
    path = Path(script)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            return str(path.resolve())
    return str(path)


def validate_generated_recipe(
    recipe: dict[str, Any],
    repo_root: Path,
    seed_recipe_authoring_cfg: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    required_top = [
        "name",
        "description",
        "workspace_root",
        "entry",
        "preflight",
        "parsing",
        "success_criteria",
        "recovery",
    ]
    for key in required_top:
        if key not in recipe:
            errors.append(f"missing top-level key: {key}")

    entry = recipe.get("entry", {})
    command = entry.get("command")
    if not isinstance(command, list) or not command:
        errors.append("entry.command must be a non-empty list")
        command = []
    entry_script = _extract_entry_script(command)
    if entry_script is None:
        errors.append("entry.command must include a python script path (*.py)")
    else:
        normalized_script = _normalize_entry_script(entry_script, repo_root)
        cfg = seed_recipe_authoring_cfg or {}
        allowed = cfg.get("allowed_entrypoints", [])
        blocked = cfg.get("blocked_entrypoints", [])
        if isinstance(allowed, list) and allowed:
            allowed_norm = {_normalize_entry_script(str(item), repo_root) for item in allowed if isinstance(item, str)}
            if normalized_script not in allowed_norm:
                errors.append(
                    "entry.command script not in allowed_entrypoints: "
                    f"{normalized_script}; allowed={sorted(allowed_norm)}"
                )
        if isinstance(blocked, list) and blocked:
            blocked_norm = {_normalize_entry_script(str(item), repo_root) for item in blocked if isinstance(item, str)}
            if normalized_script in blocked_norm:
                errors.append(
                    "entry.command script is blocked by blocked_entrypoints: "
                    f"{normalized_script}"
                )

    preflight = recipe.get("preflight", {})
    paths_exist = preflight.get("paths_exist", [])
    if not isinstance(paths_exist, list):
        errors.append("preflight.paths_exist must be a list")
        paths_exist = []
    for raw in paths_exist:
        if not isinstance(raw, str):
            errors.append("preflight.paths_exist entries must be strings")
            continue
        if raw.startswith("/"):
            errors.append(f"preflight path should be repo-relative, got absolute: {raw}")
            continue
        if not (repo_root / raw).exists():
            errors.append(f"preflight path does not exist: {raw}")

    parsing = recipe.get("parsing", {})
    if parsing.get("kind") != "json_file":
        errors.append("parsing.kind must be json_file for current runner")

    success_criteria = recipe.get("success_criteria", {})
    for key in ("require_exit_code_zero", "require_metrics", "require_metric_success"):
        if key not in success_criteria:
            errors.append(f"success_criteria missing key: {key}")
    if success_criteria.get("require_exit_code_zero") is not True:
        errors.append("success_criteria.require_exit_code_zero must be true")
    if success_criteria.get("require_metrics") is not True:
        errors.append("success_criteria.require_metrics must be true")
    if success_criteria.get("require_metric_success") is not True:
        errors.append("success_criteria.require_metric_success must be true")
    return errors


def build_recipe_author_messages(
    base_recipe: dict[str, Any],
    paper_context: dict[str, Any] | None,
    hints: dict[str, list[str]],
    effective_recipe_authoring_cfg: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    recipe_authoring_cfg = (
        effective_recipe_authoring_cfg
        if effective_recipe_authoring_cfg is not None
        else base_recipe.get("recipe_authoring", {})
    )
    payload = {
        "base_recipe": base_recipe,
        "paper_context": compact_paper_context(paper_context),
        "repo_hints": hints,
        "recipe_authoring_config": recipe_authoring_cfg,
    }
    system = """You are a recipe-authoring agent for paper reproduction.
Generate a single reproduction recipe JSON that is executable by the current reproduce runner.
Return strict JSON with this schema:
{
  "reason": "short explanation",
  "recipe": { ... full recipe object ... }
}
Rules:
- Keep recipe workspace_root as "." unless you are certain another value is needed.
- entry.command must call a repo-local script and use "{python_executable}".
- If recipe_authoring_config.allowed_entrypoints is non-empty, entry.command script MUST be one of them.
- entry.command script MUST NOT be in recipe_authoring_config.blocked_entrypoints.
- Do not use repo_hints.benchmark_scripts as entry.command unless explicitly allowed.
- Prefer simulation-friendly settings when possible.
- preflight.paths_exist must contain repo-relative paths that actually exist.
- preflight.paths_exist should include required input files only, not future output artifacts.
- Do not invent files or directories that are not present.
- Keep recovery.mode as "agent_guided".
- Keep parsing.kind as "json_file".
- Keep success_criteria.require_exit_code_zero/require_metrics/require_metric_success as true.
- If recipe_authoring_config.target_figure_id is set, optimize the recipe specifically for that figure.
- If recipe_authoring_config.required_methods exists, ensure recipe metadata preserves those method names.
- If recipe_authoring_config.required_outputs exists, include them in metadata (labels), not as mandatory preflight paths.
- Do not return markdown fences."""
    user = json.dumps(payload, indent=2, sort_keys=True)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def run_recipe_authoring(
    recipe: dict[str, Any],
    repo_root: Path,
    run_dir: Path,
    paper_context: dict[str, Any] | None,
    api_base: str,
    api_key: str,
    model: str,
) -> dict[str, Any] | None:
    cfg = recipe.get("recipe_authoring", {})
    if not bool(cfg.get("enabled", False)):
        return None

    author_dir = run_dir / "recipe_authoring"
    author_dir.mkdir(parents=True, exist_ok=True)
    hints = entrypoint_hints(repo_root)
    effective_cfg, cfg_warnings = resolve_recipe_authoring_config(cfg, repo_root, hints)
    prompt_recipe = dict(recipe)
    prompt_recipe["recipe_authoring"] = effective_cfg
    messages = build_recipe_author_messages(
        prompt_recipe,
        paper_context,
        hints,
        effective_recipe_authoring_cfg=effective_cfg,
    )
    write_json(author_dir / "prompt.json", {"messages": messages})

    payload: dict[str, Any] = {
        "applied": False,
        "errors": [],
        "generated_recipe_path": str(author_dir / "recipe.generated.json"),
        "effective_recipe_authoring_config": effective_cfg,
        "config_warnings": cfg_warnings,
    }
    try:
        response = call_openai_compatible(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
        )
    except urllib.error.URLError as exc:
        payload["errors"].append(f"model_endpoint_error: {exc}")
        write_json(author_dir / "manifest.json", payload)
        return payload

    write_json(author_dir / "response.raw.json", response)
    raw = response["choices"][0]["message"]["content"]
    parsed = extract_json_object(raw)
    write_json(author_dir / "response.parsed.json", parsed)

    candidate = parsed.get("recipe") if isinstance(parsed, dict) else None
    if not isinstance(candidate, dict):
        if isinstance(parsed, dict):
            candidate = parsed
        else:
            payload["errors"].append("recipe_authoring output is not a JSON object")
            write_json(author_dir / "manifest.json", payload)
            return payload

    validation_errors = validate_generated_recipe(candidate, repo_root, effective_cfg)
    if validation_errors:
        write_json(author_dir / "recipe.generated.rejected.json", candidate)
        payload["errors"].extend(validation_errors)
        write_json(author_dir / "manifest.json", payload)
        return payload

    write_json(author_dir / "recipe.generated.json", candidate)
    payload["applied"] = True
    payload["reason"] = parsed.get("reason", "") if isinstance(parsed, dict) else ""
    payload["recipe"] = candidate
    write_json(author_dir / "manifest.json", payload)
    return payload


def workspace_relative_path(path: Path, workspace_root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        return None


def gather_repair_context(workspace_root: Path, attempt: dict[str, Any]) -> list[dict[str, str]]:
    run_error_signals = attempt.get("run_error_signals") or {}
    seen: set[str] = set()
    candidate_paths: list[Path] = []

    for file_ref in run_error_signals.get("file_refs", []):
        path = Path(file_ref["path"])
        rel = workspace_relative_path(path, workspace_root)
        if rel is None or rel in seen or not path.exists():
            continue
        seen.add(rel)
        candidate_paths.append(path)

    for module_name in run_error_signals.get("missing_modules", []):
        module_token = module_name.split(".")[-1]
        if not module_token:
            continue
        token_pattern = re.compile(rf"\\b{re.escape(module_token)}\\b")
        for path in workspace_root.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if module_name in text or token_pattern.search(text):
                rel = workspace_relative_path(path, workspace_root)
                if rel is None or rel in seen:
                    continue
                seen.add(rel)
                candidate_paths.append(path)
                if len(candidate_paths) >= 6:
                    break
        if len(candidate_paths) >= 6:
            break

    contexts: list[dict[str, str]] = []
    for path in candidate_paths[:6]:
        rel = workspace_relative_path(path, workspace_root)
        if rel is None:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        contexts.append({"path": rel, "content": content[:12000]})
    return contexts


def recovery_policy(
    action_name: str,
    attempt: dict[str, Any],
    workspace_root: Path,
) -> dict[str, Any]:
    allowed_kinds = ["replace"]
    allowed_paths: list[str] = []

    if action_name == "patch_missing_module_file":
        allowed_kinds = ["create"]
        for evidence in attempt["diagnosis_payload"].get("classification", {}).get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            for candidate in evidence.get("module_candidates", []):
                candidate_path = Path(candidate)
                rel = workspace_relative_path(candidate_path, workspace_root)
                if rel is not None:
                    allowed_paths.append(rel)
                else:
                    try:
                        rel = str(candidate_path.resolve().relative_to(workspace_root.resolve()))
                        allowed_paths.append(rel)
                    except ValueError:
                        continue
        allowed_paths = sorted(set(allowed_paths))
    else:
        allowed_paths = sorted(
            {
                item["path"]
                for item in gather_repair_context(workspace_root, attempt)
                if item.get("path")
            }
        )

    return {
        "allowed_kinds": allowed_kinds,
        "allowed_paths": allowed_paths,
        "max_edits": 1,
    }


def build_patch_messages(
    recipe: dict[str, Any],
    attempt: dict[str, Any],
    action_name: str,
    workspace_root: Path,
    paper_context: dict[str, Any] | None,
) -> list[dict[str, str]]:
    run_error_signals = attempt.get("run_error_signals") or {}
    policy = recovery_policy(action_name, attempt, workspace_root)
    payload = {
        "recipe": {
            "name": recipe.get("name"),
            "description": recipe.get("description"),
            "simulation_only": bool(recipe.get("recovery", {}).get("simulation_only")),
        },
        "recovery_action": action_name,
        "diagnosis": attempt["diagnosis_payload"],
        "status": attempt["status_payload"],
        "stderr_tail": run_error_signals.get("stderr_tail", []),
        "stdout_tail": run_error_signals.get("stdout_tail", []),
        "paper_context": paper_context,
        "patch_policy": policy,
        "repair_context": gather_repair_context(workspace_root, attempt),
    }

    system = """You are a code-repair agent working inside an isolated paper-reproduction workspace.
Produce the smallest safe patch that helps the current reproduce step make progress.
Prefer simulation-only compatibility when simulation_only=true.
Do not invent files.
Return strict JSON with this schema:
{
  "summary": "short explanation",
  "edits": [
    {
      "kind": "replace" | "create",
      "path": "relative/path.py",
      "find": "required for replace edits",
      "replace": "required for replace edits",
      "content": "required for create edits"
    }
  ]
}
Rules:
- Keep edits to at most 1 edit or return an empty edits list.
- Use relative workspace paths only.
- Only use kinds listed in `patch_policy.allowed_kinds`.
- Only use paths listed in `patch_policy.allowed_paths`.
- For replace edits, the `find` text must match exactly once.
- Do not return markdown fences."""
    user = json.dumps(payload, indent=2, sort_keys=True)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def apply_patch_plan(
    workspace_root: Path,
    patch_plan: dict[str, Any],
    step_dir: Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    edits = patch_plan.get("edits", [])
    records: list[dict[str, Any]] = []
    applied = True
    errors: list[str] = []
    backups: list[tuple[Path, str | None]] = []

    max_edits = int(policy.get("max_edits", 1) or 1)
    allowed_kinds = set(policy.get("allowed_kinds", []))
    allowed_paths = set(policy.get("allowed_paths", []))

    if len(edits) > max_edits:
        return {
            "applied": False,
            "summary": patch_plan.get("summary", ""),
            "edits": [],
            "errors": [f"expected at most {max_edits} edit, got {len(edits)}"],
        }

    for index, edit in enumerate(edits, start=1):
        kind = edit.get("kind")
        rel_path = edit.get("path", "")
        if not rel_path:
            applied = False
            errors.append(f"edit_{index}: missing path")
            continue
        target_path = (workspace_root / rel_path).resolve()
        if workspace_root.resolve() not in target_path.parents and target_path != workspace_root.resolve():
            applied = False
            errors.append(f"edit_{index}: path escapes workspace: {rel_path}")
            continue
        if allowed_kinds and kind not in allowed_kinds:
            applied = False
            errors.append(f"edit_{index}: disallowed kind {kind}; allowed={sorted(allowed_kinds)}")
            continue
        if allowed_paths and rel_path not in allowed_paths:
            applied = False
            errors.append(f"edit_{index}: disallowed path {rel_path}; allowed={sorted(allowed_paths)}")
            continue

        if kind == "replace":
            find_text = edit.get("find", "")
            replace_text = edit.get("replace", "")
            if not target_path.exists():
                applied = False
                errors.append(f"edit_{index}: replace target missing: {rel_path}")
                continue
            original = target_path.read_text(encoding="utf-8")
            occurrences = original.count(find_text)
            if occurrences != 1:
                applied = False
                errors.append(
                    f"edit_{index}: expected find text exactly once in {rel_path}, saw {occurrences}"
                )
                continue
            backups.append((target_path, original))
            target_path.write_text(original.replace(find_text, replace_text, 1), encoding="utf-8")
            records.append({"kind": kind, "path": rel_path})
            continue

        if kind == "create":
            if target_path.exists():
                applied = False
                errors.append(f"edit_{index}: create target already exists: {rel_path}")
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            backups.append((target_path, None))
            target_path.write_text(edit.get("content", ""), encoding="utf-8")
            records.append({"kind": kind, "path": rel_path})
            continue

        applied = False
        errors.append(f"edit_{index}: unsupported kind: {kind}")

    if applied and records:
        changed_paths = [workspace_root / record["path"] for record in records]
        validate_cmd = [sys.executable, "-m", "py_compile", *[str(path) for path in changed_paths]]
        proc = subprocess.run(
            validate_cmd,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            applied = False
            errors.append(f"py_compile failed: {proc.stderr.strip() or proc.stdout.strip()}")
            for target_path, original in reversed(backups):
                if original is None:
                    try:
                        target_path.unlink()
                    except FileNotFoundError:
                        pass
                else:
                    target_path.write_text(original, encoding="utf-8")
        write_json(
            step_dir / "patch.validate.json",
            {
                "command": validate_cmd,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
        )

    result = {
        "applied": applied and bool(records),
        "summary": patch_plan.get("summary", ""),
        "edits": records,
        "errors": errors,
    }
    write_json(step_dir / "patch.apply.json", result)
    return result


def _safe_names_from_import_clause(import_clause: str) -> list[str]:
    names: list[str] = []
    for token in import_clause.split(","):
        cleaned = token.strip()
        if not cleaned or cleaned == "*":
            continue
        alias = cleaned.split(" as ")
        if len(alias) == 2:
            names.append(alias[1].strip())
        else:
            names.append(alias[0].strip())
    return [name for name in names if name.isidentifier()]


def _infer_imported_symbols_for_module(
    attempt: dict[str, Any], workspace_root: Path, missing_module: str
) -> list[str]:
    symbols: list[str] = []
    file_refs = (attempt.get("run_error_signals") or {}).get("file_refs", [])
    pattern = re.compile(rf"from\s+{re.escape(missing_module)}\s+import\s+(.+)")
    for file_ref in file_refs:
        raw_path = str(file_ref.get("path", ""))
        if not raw_path.endswith(".py"):
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        rel = workspace_relative_path(path, workspace_root)
        if rel is None:
            continue
        line_no = int(file_ref.get("line", 0) or 0)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        if line_no <= 0 or line_no > len(lines):
            continue
        line = lines[line_no - 1].strip()
        match = pattern.search(line)
        if not match:
            continue
        symbols.extend(_safe_names_from_import_clause(match.group(1)))
    dedup: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        dedup.append(symbol)
    return dedup


def _build_missing_module_stub(missing_module: str, symbols: list[str]) -> str:
    lines = [
        '"""Auto-generated compatibility shim by reproduce agent."""',
        "from __future__ import annotations",
        "from typing import Any, Dict",
        "",
    ]
    if missing_module == "qos.backends.types":
        lines.extend(
            [
                "class QPU:",
                "    def __init__(self) -> None:",
                "        self.args: Dict[str, Any] = {}",
                "",
            ]
        )
        return "\n".join(lines)

    if not symbols:
        symbols = ["Placeholder"]
    for symbol in symbols:
        if symbol and symbol[0].isupper():
            lines.extend(
                [
                    f"class {symbol}:",
                    "    def __init__(self, *args: Any, **kwargs: Any) -> None:",
                    "        pass",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"def {symbol}(*args: Any, **kwargs: Any) -> Any:",
                    "    return None",
                    "",
                ]
            )
    return "\n".join(lines)


def _resolve_module_target_file(workspace_root: Path, module_name: str) -> Path:
    parts = [part for part in module_name.split(".") if part]
    package_dir = workspace_root.joinpath(*parts)
    if package_dir.is_dir():
        return package_dir / "__init__.py"
    py_file = workspace_root.joinpath(*parts).with_suffix(".py")
    return py_file


def _build_symbol_stub(symbol_name: str) -> str:
    if symbol_name == "DAG":
        return (
            "class DAG:\n"
            "    def __init__(self, circuit=None) -> None:\n"
            "        self._circuit = circuit\n\n"
            "    def to_circuit(self):\n"
            "        return self._circuit\n"
        )
    if symbol_name and symbol_name[0].isupper():
        return (
            f"class {symbol_name}:\n"
            "    def __init__(self, *args, **kwargs) -> None:\n"
            "        pass\n"
        )
    return (
        f"def {symbol_name}(*args, **kwargs):\n"
        "    return None\n"
    )


def _build_inline_symbol_fallback(symbols: list[str]) -> str:
    lines: list[str] = []
    fallback_symbols = symbols or ["Placeholder"]
    for symbol in fallback_symbols:
        clean = str(symbol or "").strip()
        if not clean.isidentifier():
            continue
        if clean[0].isupper():
            lines.extend(
                [
                    f"    class {clean}:",
                    "        def __init__(self, *args, **kwargs):",
                    "            pass",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"    def {clean}(*args, **kwargs):",
                    "        return None",
                    "",
                ]
            )
    if not lines:
        lines = ["    pass", ""]
    return "\n".join(lines).rstrip()


def _patch_missing_module_imports_in_place(
    *,
    workspace_root: Path,
    attempt: dict[str, Any],
    missing_module: str,
) -> dict[str, Any]:
    candidate_files: list[Path] = []
    seen: set[Path] = set()

    for file_ref in (attempt.get("run_error_signals") or {}).get("file_refs", []):
        raw_path = str(file_ref.get("path", ""))
        if not raw_path.endswith(".py"):
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            path.resolve().relative_to(workspace_root.resolve())
        except ValueError:
            continue
        if path in seen:
            continue
        seen.add(path)
        candidate_files.append(path)

    from_pat = re.compile(rf"(?m)^from\s+{re.escape(missing_module)}\s+import\s+(.+)$")
    import_pat = re.compile(
        rf"(?m)^import\s+{re.escape(missing_module)}(?:\s+as\s+([A-Za-z_]\w*))?\s*$"
    )
    edits: list[dict[str, str]] = []

    for path in candidate_files:
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        patched = original
        summary = ""
        from_match = from_pat.search(original)
        if from_match:
            clause = from_match.group(1).strip()
            safe_names = _safe_names_from_import_clause(clause)
            fallback = _build_inline_symbol_fallback(safe_names)
            replacement = (
                "try:\n"
                f"    from {missing_module} import {clause}\n"
                "except ModuleNotFoundError:\n"
                f"{fallback}"
            )
            patched = from_pat.sub(replacement, original, count=1)
            summary = f"guarded from-import in {workspace_relative_path(path, workspace_root) or str(path)}"
        else:
            import_match = import_pat.search(original)
            if import_match:
                alias = import_match.group(1) or missing_module.split(".")[-1]
                replacement = (
                    "try:\n"
                    f"    import {missing_module}"
                    + (f" as {alias}" if import_match.group(1) else "")
                    + "\n"
                    "except ModuleNotFoundError:\n"
                    f"    {alias} = None"
                )
                patched = import_pat.sub(replacement, original, count=1)
                summary = f"guarded import in {workspace_relative_path(path, workspace_root) or str(path)}"

        if patched == original:
            continue

        path.write_text(patched, encoding="utf-8")
        validate = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if validate.returncode != 0:
            path.write_text(original, encoding="utf-8")
            continue
        rel = workspace_relative_path(path, workspace_root) or str(path)
        edits.append({"kind": "replace", "path": rel, "summary": summary})

    return {
        "applied": len(edits) > 0,
        "edits": edits,
    }


def execute_builtin_recovery(
    action_name: str,
    attempt: dict[str, Any],
    workspace_root: Path,
    step_dir: Path,
    python_exe: str,
    source_repo_root: Path | None = None,
    attempt_recoveries: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if action_name in SHIM_RECOVERY_ACTIONS:
        return None

    if action_name == "inspect_traceback_and_retry":
        run_error_signals = attempt.get("run_error_signals") or {}
        stderr_tail = run_error_signals.get("stderr_tail") or []
        joined_stderr = "\n".join(str(line) for line in stderr_tail)
        key_match = re.search(r"KeyError:\s*[\"']([^\"']+)[\"']", joined_stderr)
        if not key_match:
            return None
        missing_key = key_match.group(1)

        file_refs = run_error_signals.get("file_refs") or []
        target_path: Path | None = None
        target_line_no = 0
        original_line = ""
        updated_line = ""

        for file_ref in reversed(file_refs):
            raw_path = str(file_ref.get("path", ""))
            line_no = int(file_ref.get("line", 0) or 0)
            if line_no <= 0:
                continue
            path = Path(raw_path)
            if not path.exists():
                continue
            rel = workspace_relative_path(path, workspace_root)
            if rel is None:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            if line_no > len(lines):
                continue

            line = lines[line_no - 1]
            patched = re.sub(
                rf'([A-Za-z_][A-Za-z0-9_]*)\[\s*[\'"]{re.escape(missing_key)}[\'"]\s*\]',
                rf'\1.get("{missing_key}", 0.0)',
                line,
            )
            if patched == line:
                continue
            target_path = path
            target_line_no = line_no
            original_line = line
            updated_line = patched
            break

        if target_path is None:
            return None

        lines = target_path.read_text(encoding="utf-8").splitlines()
        lines[target_line_no - 1] = updated_line
        target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        validate = subprocess.run(
            [sys.executable, "-m", "py_compile", str(target_path)],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if validate.returncode != 0:
            lines[target_line_no - 1] = original_line
            target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"py_compile_failed_for_keyerror_patch: {target_path}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_key": missing_key},
                "applied": False,
                "apply_result": result,
            }

        rel = workspace_relative_path(target_path, workspace_root) or str(target_path)
        apply_result = {
            "applied": True,
            "summary": f"patched KeyError access for '{missing_key}' in traceback line",
            "edits": [{"kind": "replace", "path": rel}],
            "errors": [],
        }
        payload = {
            "missing_key": missing_key,
            "target_file": rel,
            "line": target_line_no,
            "original_line": original_line,
            "updated_line": updated_line,
            "validate": {
                "returncode": validate.returncode,
                "stdout": validate.stdout,
                "stderr": validate.stderr,
            },
        }
        write_json(step_dir / "patch.builtin.plan.json", payload)
        write_json(step_dir / "patch.builtin.apply.json", apply_result)
        return {
            "action": action_name,
            "record_path": str(step_dir / "patch.builtin.apply.json"),
            "result": payload,
            "applied": True,
            "apply_result": apply_result,
        }

    if action_name == "runtime_env_select":
        missing_candidates = infer_missing_modules_from_attempt(attempt)
        missing_module = missing_candidates[0] if missing_candidates else ""

        raw_candidates = os.getenv(
            "REPRO_RUNTIME_ENV_CANDIDATES", "python3.11,python3.10,python3.12,python3"
        )
        candidates = [item.strip() for item in raw_candidates.split(",") if item.strip()]
        if python_exe and python_exe not in candidates:
            candidates = [python_exe, *candidates]

        module_parts = [part for part in missing_module.split(".") if part]
        root_pkg = module_parts[0] if module_parts else ""
        probe_modules: list[str] = []
        if root_pkg:
            probe_modules.append(root_pkg)
        if root_pkg == "qiskit":
            for item in ["qiskit", "qiskit.circuit"]:
                if item not in probe_modules:
                    probe_modules.append(item)
        if missing_module:
            probe_modules.append(missing_module)
        if not probe_modules:
            probe_modules = [item for item in missing_candidates if item][:3] or ["qiskit"]

        base_requirements = infer_runtime_requirements(root_package=root_pkg)
        historical_requirements = collect_recovered_requirements(attempt_recoveries or [])
        requirements: list[str] = []
        seen_requirements: set[str] = set()
        for req in [*historical_requirements, *base_requirements]:
            if req in seen_requirements:
                continue
            seen_requirements.add(req)
            requirements.append(req)

        run_root = step_dir.parent.parent
        env_root = run_root / "runtime_envs"
        env_root.mkdir(parents=True, exist_ok=True)
        attempts_log: list[dict[str, Any]] = []
        selected_python: str | None = None

        for candidate in candidates:
            resolved = shutil.which(candidate) if "/" not in candidate else candidate
            record: dict[str, Any] = {"candidate": candidate, "resolved": resolved}
            if not resolved:
                record["status"] = "missing_interpreter"
                attempts_log.append(record)
                continue

            resolved_path = Path(resolved).expanduser()
            resolved_str = str(resolved_path)
            reuse_existing_venv = (
                "/runtime_envs/" in resolved_str and resolved_str.endswith("/bin/python")
            )
            if reuse_existing_venv:
                venv_python = resolved_path
                venv_dir = venv_python.parent.parent
            else:
                try:
                    resolved_abs = str(resolved_path.resolve())
                except Exception:
                    resolved_abs = resolved_str
                digest = hashlib.sha1(resolved_abs.encode("utf-8")).hexdigest()[:10]
                safe_name = f"{resolved_path.name}_{digest}"
                venv_dir = env_root / safe_name
                venv_python = venv_dir / "bin" / "python"

            if not venv_python.exists():
                create_proc = subprocess.run(
                    [resolved, "-m", "venv", str(venv_dir)],
                    cwd=workspace_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                record["venv_create"] = {
                    "returncode": create_proc.returncode,
                    "stdout": create_proc.stdout,
                    "stderr": create_proc.stderr,
                }
                if create_proc.returncode != 0:
                    record["status"] = "venv_create_failed"
                    attempts_log.append(record)
                    continue

            install_records: list[dict[str, Any]] = []
            install_ok = True
            if requirements:
                proc = subprocess.run(
                    [
                        str(venv_python),
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        *requirements,
                    ],
                    cwd=workspace_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                install_records.append(
                    {
                        "requirements": requirements,
                        "returncode": proc.returncode,
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                    }
                )
                if proc.returncode != 0:
                    install_ok = False
            record["installs"] = install_records
            if not install_ok:
                record["status"] = "dependency_install_failed"
                attempts_log.append(record)
                continue

            import_script = (
                "import importlib, json\n"
                f"mods = {probe_modules!r}\n"
                "failed = []\n"
                "for m in mods:\n"
                "    try:\n"
                "        importlib.import_module(m)\n"
                "    except Exception as exc:\n"
                "        failed.append({'module': m, 'error': f'{type(exc).__name__}: {exc}'})\n"
                "print(json.dumps({'ok': len(failed)==0, 'failed': failed}))\n"
                "raise SystemExit(0 if len(failed)==0 else 1)\n"
            )
            verify_proc = subprocess.run(
                [str(venv_python), "-c", import_script],
                cwd=workspace_root,
                capture_output=True,
                text=True,
                check=False,
            )
            record["verify"] = {
                "returncode": verify_proc.returncode,
                "stdout": verify_proc.stdout,
                "stderr": verify_proc.stderr,
                "probe_modules": probe_modules,
            }
            if verify_proc.returncode == 0:
                record["status"] = "selected"
                selected_python = str(venv_python)
                attempts_log.append(record)
                break

            record["status"] = "probe_failed"
            attempts_log.append(record)

        write_json(
            step_dir / "patch.builtin.runtime_env_select.json",
            {
                "missing_module": missing_module,
                "probe_modules": probe_modules,
                "requirements": requirements,
                "historical_requirements": historical_requirements,
                "attempts": attempts_log,
            },
        )

        if not selected_python:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"runtime_env_select_no_compatible_python_for_module: {missing_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": missing_module, "probe_modules": probe_modules},
                "applied": False,
                "apply_result": result,
            }

        apply_result = {
            "applied": True,
            "summary": f"selected compatible runtime python: {selected_python}",
            "edits": [{"kind": "environment", "path": str(workspace_root)}],
            "errors": [],
        }
        payload = {
            "missing_module": missing_module,
            "python_executable": selected_python,
            "probe_modules": probe_modules,
            "requirements": requirements,
            "historical_requirements": historical_requirements,
            "record": str(step_dir / "patch.builtin.runtime_env_select.json"),
        }
        write_json(step_dir / "patch.builtin.plan.json", payload)
        write_json(step_dir / "patch.builtin.apply.json", apply_result)
        return {
            "action": action_name,
            "record_path": str(step_dir / "patch.builtin.apply.json"),
            "result": payload,
            "applied": True,
            "apply_result": apply_result,
        }

    if action_name == "resolve_dependency_mapping":
        missing_candidates = infer_missing_modules_from_attempt(attempt)
        missing_module = missing_candidates[0] if missing_candidates else ""
        import_source_module = ""
        import_symbol_name = ""
        evidence = (
            attempt.get("diagnosis_payload", {})
            .get("classification", {})
            .get("evidence", [])
        )
        for item in evidence:
            if not isinstance(item, dict):
                continue
            if item.get("from") and not import_source_module:
                import_source_module = str(item["from"])
            if item.get("name") and not import_symbol_name:
                import_symbol_name = str(item["name"])
        if not missing_module:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": ["resolve_dependency_mapping_no_missing_module_found"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": None},
                "applied": False,
                "apply_result": result,
            }
        if missing_module.startswith(("qos.", "qvm.", "Baseline_Multiprogramming")):
            return None

        requirements = infer_requirements_for_missing_module(
            missing_module,
            import_source_module=import_source_module,
            import_symbol_name=import_symbol_name,
        )
        if not requirements:
            parts = [part for part in missing_module.split(".") if part]
            root_pkg = parts[0] if parts else missing_module
            if root_pkg:
                requirements = [root_pkg.replace("_", "-")]
        requirements = [item for item in requirements if isinstance(item, str) and item.strip()]
        if not requirements:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"resolve_dependency_mapping_no_requirements_for_module: {missing_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": missing_module},
                "applied": False,
                "apply_result": result,
            }

        run_root = step_dir.parent.parent
        runtime_env_root = run_root / "runtime_envs"
        runtime_env_root.mkdir(parents=True, exist_ok=True)
        runtime_python = python_exe
        try:
            runtime_python_resolved = str(Path(runtime_python).resolve())
        except Exception:
            runtime_python_resolved = runtime_python
        if "/runtime_envs/" not in runtime_python_resolved:
            mapping_venv = runtime_env_root / "dep_mapping"
            mapping_python = mapping_venv / "bin" / "python"
            if not mapping_python.exists():
                create_proc = subprocess.run(
                    [runtime_python, "-m", "venv", str(mapping_venv)],
                    cwd=workspace_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                write_json(
                    step_dir / "patch.builtin.resolve_dependency_mapping.venv_create.json",
                    {
                        "command": [runtime_python, "-m", "venv", str(mapping_venv)],
                        "returncode": create_proc.returncode,
                        "stdout": create_proc.stdout,
                        "stderr": create_proc.stderr,
                    },
                )
                if create_proc.returncode != 0:
                    result = {
                        "applied": False,
                        "summary": "",
                        "edits": [],
                        "errors": ["resolve_dependency_mapping_venv_create_failed"],
                    }
                    write_json(step_dir / "patch.builtin.apply.json", result)
                    return {
                        "action": action_name,
                        "record_path": str(step_dir / "patch.builtin.apply.json"),
                        "result": {"missing_module": missing_module, "requirements": requirements},
                        "applied": False,
                        "apply_result": result,
                    }
            runtime_python = str(mapping_python)

        install_cmd = [
            runtime_python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            *requirements,
        ]
        install_proc = subprocess.run(
            install_cmd,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        write_json(
            step_dir / "patch.builtin.resolve_dependency_mapping.install.json",
            {
                "command": install_cmd,
                "missing_module": missing_module,
                "requirements": requirements,
                "returncode": install_proc.returncode,
                "stdout": install_proc.stdout,
                "stderr": install_proc.stderr,
            },
        )
        if install_proc.returncode != 0:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"resolve_dependency_mapping_install_failed: {missing_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": missing_module, "requirements": requirements},
                "applied": False,
                "apply_result": result,
            }

        probe_modules = []
        root_pkg = missing_module.split(".")[0]
        if root_pkg:
            probe_modules.append(root_pkg)
        if missing_module and missing_module not in probe_modules:
            probe_modules.append(missing_module)
        verify_script = (
            "import importlib, json\n"
            f"mods = {probe_modules!r}\n"
            "failed = []\n"
            "for m in mods:\n"
            "    try:\n"
            "        importlib.import_module(m)\n"
            "    except Exception as exc:\n"
            "        failed.append({'module': m, 'error': f'{type(exc).__name__}: {exc}'})\n"
            "print(json.dumps({'ok': len(failed)==0, 'failed': failed}))\n"
            "raise SystemExit(0 if len(failed)==0 else 1)\n"
        )
        verify_proc = subprocess.run(
            [runtime_python, "-c", verify_script],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        write_json(
            step_dir / "patch.builtin.resolve_dependency_mapping.verify.json",
            {
                "python_executable": runtime_python,
                "probe_modules": probe_modules,
                "returncode": verify_proc.returncode,
                "stdout": verify_proc.stdout,
                "stderr": verify_proc.stderr,
            },
        )
        if verify_proc.returncode != 0:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"resolve_dependency_mapping_verify_failed: {missing_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {
                    "missing_module": missing_module,
                    "requirements": requirements,
                    "probe_modules": probe_modules,
                },
                "applied": False,
                "apply_result": result,
            }

        apply_result = {
            "applied": True,
            "summary": f"resolved dependency mapping and installed requirements for: {missing_module}",
            "edits": [{"kind": "environment", "path": str(workspace_root)}],
            "errors": [],
        }
        payload = {
            "missing_module": missing_module,
            "requirements": requirements,
            "probe_modules": probe_modules,
            "python_executable": runtime_python,
            "install_record": str(step_dir / "patch.builtin.resolve_dependency_mapping.install.json"),
            "verify_record": str(step_dir / "patch.builtin.resolve_dependency_mapping.verify.json"),
        }
        write_json(step_dir / "patch.builtin.plan.json", payload)
        write_json(step_dir / "patch.builtin.apply.json", apply_result)
        return {
            "action": action_name,
            "record_path": str(step_dir / "patch.builtin.apply.json"),
            "result": payload,
            "applied": True,
            "apply_result": apply_result,
        }

    if action_name == "install_missing_module":
        missing_candidates = infer_missing_modules_from_attempt(attempt)
        missing_module = missing_candidates[0] if missing_candidates else ""
        if not missing_module:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": ["install_missing_module_no_missing_module_found"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": None},
                "applied": False,
                "apply_result": result,
            }
        if missing_module.startswith(("qos.", "qvm.", "Baseline_Multiprogramming")):
            return None

        parts = [part for part in missing_module.split(".") if part]
        root_pkg = parts[0] if parts else missing_module
        requirements = infer_runtime_requirements(root_package=root_pkg) or [root_pkg.replace("_", "-")]
        historical_requirements = collect_recovered_requirements(attempt_recoveries or [])
        merged_requirements: list[str] = []
        seen_req: set[str] = set()
        for req in [*historical_requirements, *requirements]:
            if req in seen_req:
                continue
            seen_req.add(req)
            merged_requirements.append(req)

        run_root = step_dir.parent.parent
        runtime_env_root = run_root / "runtime_envs"
        runtime_env_root.mkdir(parents=True, exist_ok=True)
        runtime_python = python_exe
        try:
            runtime_python_resolved = str(Path(runtime_python).resolve())
        except Exception:
            runtime_python_resolved = runtime_python
        if "/runtime_envs/" not in runtime_python_resolved:
            install_venv = runtime_env_root / "install_missing_module"
            install_python = install_venv / "bin" / "python"
            if not install_python.exists():
                create_proc = subprocess.run(
                    [runtime_python, "-m", "venv", str(install_venv)],
                    cwd=workspace_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                write_json(
                    step_dir / "patch.builtin.install_missing_module.venv_create.json",
                    {
                        "command": [runtime_python, "-m", "venv", str(install_venv)],
                        "returncode": create_proc.returncode,
                        "stdout": create_proc.stdout,
                        "stderr": create_proc.stderr,
                    },
                )
                if create_proc.returncode != 0:
                    result = {
                        "applied": False,
                        "summary": "",
                        "edits": [],
                        "errors": ["install_missing_module_venv_create_failed"],
                    }
                    write_json(step_dir / "patch.builtin.apply.json", result)
                    return {
                        "action": action_name,
                        "record_path": str(step_dir / "patch.builtin.apply.json"),
                        "result": {"missing_module": missing_module},
                        "applied": False,
                        "apply_result": result,
                    }
            runtime_python = str(install_python)

        install_cmd = [
            runtime_python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            *merged_requirements,
        ]
        install_proc = subprocess.run(
            install_cmd,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        write_json(
            step_dir / "patch.builtin.install_missing_module.install.json",
            {
                "command": install_cmd,
                "requirements": merged_requirements,
                "returncode": install_proc.returncode,
                "stdout": install_proc.stdout,
                "stderr": install_proc.stderr,
            },
        )
        if install_proc.returncode != 0:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"install_missing_module_pip_install_failed: {missing_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": missing_module, "requirements": merged_requirements},
                "applied": False,
                "apply_result": result,
            }

        probe_modules: list[str] = []
        if root_pkg:
            probe_modules.append(root_pkg)
        if missing_module and missing_module not in probe_modules:
            probe_modules.append(missing_module)
        verify_script = (
            "import importlib, json\n"
            f"mods = {probe_modules!r}\n"
            "failed = []\n"
            "for m in mods:\n"
            "    try:\n"
            "        importlib.import_module(m)\n"
            "    except Exception as exc:\n"
            "        failed.append({'module': m, 'error': f'{type(exc).__name__}: {exc}'})\n"
            "print(json.dumps({'ok': len(failed)==0, 'failed': failed}))\n"
            "raise SystemExit(0 if len(failed)==0 else 1)\n"
        )
        verify_proc = subprocess.run(
            [runtime_python, "-c", verify_script],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        write_json(
            step_dir / "patch.builtin.install_missing_module.verify.json",
            {
                "python_executable": runtime_python,
                "probe_modules": probe_modules,
                "returncode": verify_proc.returncode,
                "stdout": verify_proc.stdout,
                "stderr": verify_proc.stderr,
            },
        )
        if verify_proc.returncode != 0:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"install_missing_module_verify_failed: {missing_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": missing_module, "probe_modules": probe_modules},
                "applied": False,
                "apply_result": result,
            }

        apply_result = {
            "applied": True,
            "summary": f"installed missing module in isolated runtime env: {missing_module}",
            "edits": [{"kind": "environment", "path": str(workspace_root)}],
            "errors": [],
        }
        payload = {
            "missing_module": missing_module,
            "requirements": merged_requirements,
            "probe_modules": probe_modules,
            "python_executable": runtime_python,
            "install_record": str(step_dir / "patch.builtin.install_missing_module.install.json"),
            "verify_record": str(step_dir / "patch.builtin.install_missing_module.verify.json"),
        }
        write_json(step_dir / "patch.builtin.plan.json", payload)
        write_json(step_dir / "patch.builtin.apply.json", apply_result)
        return {
            "action": action_name,
            "record_path": str(step_dir / "patch.builtin.apply.json"),
            "result": payload,
            "applied": True,
            "apply_result": apply_result,
        }

    if action_name == "select_or_prepare_python_environment":
        missing_candidates = infer_missing_modules_from_attempt(attempt)
        missing_module = missing_candidates[0] if missing_candidates else ""
        if not missing_module:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": ["builtin_env_recovery_missing_module_not_found"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": None},
                "applied": False,
                "apply_result": result,
            }

        if missing_module.startswith(("qos.", "qvm.", "Baseline_Multiprogramming")):
            return None

        requirements = infer_requirements_for_missing_module(missing_module)
        if not requirements:
            requirements = [missing_module.replace("_", "-")]

        run_root = step_dir.parent.parent
        runtime_env_root = run_root / "runtime_envs"
        runtime_env_root.mkdir(parents=True, exist_ok=True)
        runtime_python = python_exe
        try:
            runtime_python_resolved = str(Path(runtime_python).resolve())
        except Exception:
            runtime_python_resolved = runtime_python
        if "/runtime_envs/" not in runtime_python_resolved:
            env_venv = runtime_env_root / "select_or_prepare_python_environment"
            env_python = env_venv / "bin" / "python"
            if not env_python.exists():
                create_proc = subprocess.run(
                    [runtime_python, "-m", "venv", str(env_venv)],
                    cwd=workspace_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                write_json(
                    step_dir / "patch.builtin.env.venv_create.json",
                    {
                        "command": [runtime_python, "-m", "venv", str(env_venv)],
                        "returncode": create_proc.returncode,
                        "stdout": create_proc.stdout,
                        "stderr": create_proc.stderr,
                    },
                )
                if create_proc.returncode != 0:
                    result = {
                        "applied": False,
                        "summary": "",
                        "edits": [],
                        "errors": ["select_or_prepare_python_environment_venv_create_failed"],
                    }
                    write_json(step_dir / "patch.builtin.apply.json", result)
                    return {
                        "action": action_name,
                        "record_path": str(step_dir / "patch.builtin.apply.json"),
                        "result": {"missing_module": missing_module},
                        "applied": False,
                        "apply_result": result,
                    }
            runtime_python = str(env_python)

        install_cmd = [
            runtime_python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            *requirements,
        ]
        install_proc = subprocess.run(
            install_cmd,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        write_json(
            step_dir / "patch.builtin.env.install.json",
            {
                "command": install_cmd,
                "requirements": requirements,
                "returncode": install_proc.returncode,
                "stdout": install_proc.stdout,
                "stderr": install_proc.stderr,
            },
        )
        if install_proc.returncode != 0:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"pip_install_failed_for_module: {missing_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": missing_module},
                "applied": False,
                "apply_result": result,
            }

        verify_script = (
            "import importlib, json\n"
            f"mods = {[missing_module.split('.')[0], missing_module]!r}\n"
            "seen = set()\n"
            "ordered = []\n"
            "for m in mods:\n"
            "    if m and m not in seen:\n"
            "        seen.add(m)\n"
            "        ordered.append(m)\n"
            "failed = []\n"
            "for m in ordered:\n"
            "    try:\n"
            "        importlib.import_module(m)\n"
            "    except Exception as exc:\n"
            "        failed.append({'module': m, 'error': f'{type(exc).__name__}: {exc}'})\n"
            "print(json.dumps({'ok': len(failed)==0, 'failed': failed}))\n"
            "raise SystemExit(0 if len(failed)==0 else 1)\n"
        )
        verify_proc = subprocess.run(
            [runtime_python, "-c", verify_script],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        write_json(
            step_dir / "patch.builtin.env.verify.json",
            {
                "returncode": verify_proc.returncode,
                "stdout": verify_proc.stdout,
                "stderr": verify_proc.stderr,
            },
        )
        if verify_proc.returncode != 0:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"import_verify_failed_after_install: {missing_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": missing_module},
                "applied": False,
                "apply_result": result,
            }

        apply_result = {
            "applied": True,
            "summary": f"installed missing runtime module into workspace: {missing_module}",
            "edits": [{"kind": "environment", "path": str(workspace_root)}],
            "errors": [],
        }
        payload = {
            "missing_module": missing_module,
            "requirements": requirements,
            "python_executable": runtime_python,
            "install_record": str(step_dir / "patch.builtin.env.install.json"),
            "verify_record": str(step_dir / "patch.builtin.env.verify.json"),
        }
        write_json(step_dir / "patch.builtin.plan.json", payload)
        write_json(step_dir / "patch.builtin.apply.json", apply_result)
        return {
            "action": action_name,
            "record_path": str(step_dir / "patch.builtin.apply.json"),
            "result": payload,
            "applied": True,
            "apply_result": apply_result,
        }

    if action_name == "runtime_trace_fix":
        missing_module = ""
        import_source_module = ""
        import_symbol_name = ""
        evidence = (
            attempt.get("diagnosis_payload", {})
            .get("classification", {})
            .get("evidence", [])
        )
        for item in evidence:
            if isinstance(item, dict) and item.get("missing_module"):
                missing_module = str(item["missing_module"])
                break
            if isinstance(item, dict) and item.get("from") and not import_source_module:
                import_source_module = str(item["from"])
            if isinstance(item, dict) and item.get("name") and not import_symbol_name:
                import_symbol_name = str(item["name"])
        if not missing_module:
            missing_modules = (attempt.get("run_error_signals") or {}).get("missing_modules", [])
            if missing_modules:
                missing_module = str(missing_modules[0])

        target_module = missing_module or import_source_module

        if not target_module:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": ["builtin_runtime_trace_fix_target_module_not_found"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": None, "import_source_module": None},
                "applied": False,
                "apply_result": result,
            }

        module_parts = [part for part in target_module.split(".") if part]
        root_package = module_parts[0] if module_parts else target_module
        if not root_package.isidentifier():
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"builtin_runtime_trace_fix_invalid_module_name: {target_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": missing_module, "import_source_module": import_source_module},
                "applied": False,
                "apply_result": result,
            }

        requirements = infer_runtime_requirements(
            root_package=root_package,
            import_source_module=import_source_module,
            import_symbol_name=import_symbol_name,
        )
        preflight_checks = (attempt.get("preflight_payload") or {}).get("checks", {})
        for item in preflight_checks.get("python_imports", []):
            if not isinstance(item, dict):
                continue
            if item.get("required") is False:
                continue
            module_name = str(item.get("name", "")).strip()
            if not module_name:
                continue
            module_root = module_name.split(".")[0]
            inferred = infer_runtime_requirements(root_package=module_root)
            if inferred:
                for dep in inferred:
                    if dep not in requirements:
                        requirements.append(dep)
            else:
                fallback_dep = module_root.replace("_", "-")
                if fallback_dep and fallback_dep not in requirements:
                    requirements.append(fallback_dep)

        run_root = step_dir.parent.parent
        runtime_env_root = run_root / "runtime_envs"
        runtime_env_root.mkdir(parents=True, exist_ok=True)
        runtime_python = python_exe
        if "/runtime_envs/" not in str(Path(runtime_python).resolve()):
            tracefix_venv = runtime_env_root / "tracefix"
            tracefix_python = tracefix_venv / "bin" / "python"
            if not tracefix_python.exists():
                create_proc = subprocess.run(
                    [python_exe, "-m", "venv", str(tracefix_venv)],
                    cwd=workspace_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                write_json(
                    step_dir / "patch.builtin.runtime_trace_fix.venv_create.json",
                    {
                        "command": [python_exe, "-m", "venv", str(tracefix_venv)],
                        "returncode": create_proc.returncode,
                        "stdout": create_proc.stdout,
                        "stderr": create_proc.stderr,
                    },
                )
                if create_proc.returncode != 0:
                    result = {
                        "applied": False,
                        "summary": "",
                        "edits": [],
                        "errors": ["runtime_trace_fix_venv_create_failed"],
                    }
                    write_json(step_dir / "patch.builtin.apply.json", result)
                    return {
                        "action": action_name,
                        "record_path": str(step_dir / "patch.builtin.apply.json"),
                        "result": {
                            "missing_module": missing_module,
                            "import_source_module": import_source_module,
                            "python_executable": runtime_python,
                        },
                        "applied": False,
                        "apply_result": result,
                    }
            runtime_python = str(tracefix_python)

        install_records: list[dict[str, Any]] = []
        install_ok = True
        if requirements:
            install_cmd = [
                runtime_python,
                "-m",
                "pip",
                "install",
                "--upgrade",
                *requirements,
            ]
            proc = subprocess.run(
                install_cmd,
                cwd=workspace_root,
                capture_output=True,
                text=True,
                check=False,
            )
            install_records.append(
                {
                    "requirements": requirements,
                    "command": install_cmd,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            )
            if proc.returncode != 0:
                install_ok = False

        write_json(
            step_dir / "patch.builtin.runtime_trace_fix.install.json",
            {"missing_module": missing_module, "installs": install_records},
        )
        if not install_ok:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"runtime_trace_fix_install_failed_for_module: {target_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {
                    "missing_module": missing_module,
                    "import_source_module": import_source_module,
                    "requirements": requirements,
                },
                "applied": False,
                "apply_result": result,
            }

        verify_module = target_module
        verify_script = (
            "import importlib; "
            f"importlib.import_module({verify_module!r}); "
            "print('ok')"
        )
        verify_proc = subprocess.run(
            [runtime_python, "-c", verify_script],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        write_json(
            step_dir / "patch.builtin.runtime_trace_fix.verify.json",
            {
                "missing_module": missing_module,
                "import_source_module": import_source_module,
                "verify_module": verify_module,
                "returncode": verify_proc.returncode,
                "stdout": verify_proc.stdout,
                "stderr": verify_proc.stderr,
            },
        )
        if verify_proc.returncode != 0:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"runtime_trace_fix_verify_failed_for_module: {verify_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {
                    "missing_module": missing_module,
                    "import_source_module": import_source_module,
                    "requirements": requirements,
                },
                "applied": False,
                "apply_result": result,
            }

        apply_result = {
            "applied": True,
            "summary": f"runtime import-surface fix applied for module: {verify_module}",
            "edits": [{"kind": "environment", "path": str(workspace_root)}],
            "errors": [],
        }
        payload = {
            "missing_module": missing_module,
            "import_source_module": import_source_module,
            "verify_module": verify_module,
            "requirements": requirements,
            "python_executable": runtime_python,
            "install_record": str(step_dir / "patch.builtin.runtime_trace_fix.install.json"),
            "verify_record": str(step_dir / "patch.builtin.runtime_trace_fix.verify.json"),
        }
        write_json(step_dir / "patch.builtin.plan.json", payload)
        write_json(step_dir / "patch.builtin.apply.json", apply_result)
        return {
            "action": action_name,
            "record_path": str(step_dir / "patch.builtin.apply.json"),
            "result": payload,
            "applied": True,
            "apply_result": apply_result,
        }

    if action_name == "source_path_fix":
        missing_module = ""
        evidence = (
            attempt.get("diagnosis_payload", {})
            .get("classification", {})
            .get("evidence", [])
        )
        for item in evidence:
            if isinstance(item, dict) and item.get("missing_module"):
                missing_module = str(item["missing_module"])
                break
        if not missing_module:
            missing_modules = (attempt.get("run_error_signals") or {}).get("missing_modules", [])
            if missing_modules:
                missing_module = str(missing_modules[0])

        if not missing_module:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": ["builtin_source_path_fix_missing_module_not_found"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": None},
                "applied": False,
                "apply_result": result,
            }

        if source_repo_root is None:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": ["builtin_source_path_fix_source_repo_missing"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"missing_module": missing_module},
                "applied": False,
                "apply_result": result,
            }

        source_target = _resolve_module_target_file(source_repo_root, missing_module)
        target = _resolve_module_target_file(workspace_root, missing_module)
        if not source_target.exists():
            patched = _patch_missing_module_imports_in_place(
                workspace_root=workspace_root,
                attempt=attempt,
                missing_module=missing_module,
            )
            if patched.get("applied"):
                edits = patched.get("edits", [])
                apply_result = {
                    "applied": True,
                    "summary": (
                        "source module missing; applied guarded import fallback in existing files "
                        f"for module: {missing_module}"
                    ),
                    "edits": [{"kind": edit.get("kind", "replace"), "path": edit.get("path", "")} for edit in edits],
                    "errors": [],
                }
                payload = {
                    "missing_module": missing_module,
                    "source_file": str(source_target),
                    "patched_files": [edit.get("path", "") for edit in edits],
                }
                write_json(step_dir / "patch.builtin.plan.json", payload)
                write_json(step_dir / "patch.builtin.apply.json", apply_result)
                return {
                    "action": action_name,
                    "record_path": str(step_dir / "patch.builtin.apply.json"),
                    "result": payload,
                    "applied": True,
                    "apply_result": apply_result,
                }
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"builtin_source_path_fix_source_module_not_found: {missing_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {
                    "missing_module": missing_module,
                    "source_file": str(source_target),
                },
                "applied": False,
                "apply_result": result,
            }

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source_target.read_text(encoding="utf-8"), encoding="utf-8")
        validate = subprocess.run(
            [sys.executable, "-m", "py_compile", str(target)],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if validate.returncode != 0:
            target.unlink(missing_ok=True)
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"py_compile_failed_for_source_path_fix: {target}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {
                    "missing_module": missing_module,
                    "source_file": str(source_target),
                    "target_file": str(target),
                },
                "applied": False,
                "apply_result": result,
            }

        rel = workspace_relative_path(target, workspace_root) or str(target)
        apply_result = {
            "applied": True,
            "summary": f"restored missing module from source repo via source_path_fix: {missing_module}",
            "edits": [{"kind": "create", "path": rel}],
            "errors": [],
        }
        payload = {
            "missing_module": missing_module,
            "source_file": str(source_target),
            "target_file": rel,
            "validate": {
                "returncode": validate.returncode,
                "stdout": validate.stdout,
                "stderr": validate.stderr,
            },
        }
        write_json(step_dir / "patch.builtin.plan.json", payload)
        write_json(step_dir / "patch.builtin.apply.json", apply_result)
        return {
            "action": action_name,
            "record_path": str(step_dir / "patch.builtin.apply.json"),
            "result": payload,
            "applied": True,
            "apply_result": apply_result,
        }

    if action_name == "patch_missing_symbol_import":
        symbol_name = ""
        module_name = ""
        evidence = (
            attempt.get("diagnosis_payload", {})
            .get("classification", {})
            .get("evidence", [])
        )
        for item in evidence:
            if not isinstance(item, dict):
                continue
            if item.get("name") and not symbol_name:
                symbol_name = str(item["name"])
            if item.get("from") and not module_name:
                module_name = str(item["from"])
        if not symbol_name or not module_name:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": ["builtin_recovery_missing_symbol_evidence_not_found"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"symbol": symbol_name, "module": module_name},
                "applied": False,
                "apply_result": result,
            }

        inferred_symbols = _infer_imported_symbols_for_module(
            attempt=attempt,
            workspace_root=workspace_root,
            missing_module=module_name,
        )
        symbols_to_consider: list[str] = []
        for candidate in [symbol_name, *inferred_symbols]:
            cleaned = str(candidate or "").strip()
            if not cleaned or not cleaned.isidentifier() or cleaned in symbols_to_consider:
                continue
            symbols_to_consider.append(cleaned)
        if not symbols_to_consider:
            symbols_to_consider = [symbol_name]

        target = _resolve_module_target_file(workspace_root, module_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        original = target.read_text(encoding="utf-8") if target.exists() else ""
        missing_symbols = [
            name
            for name in symbols_to_consider
            if not re.search(rf"(?m)^(class|def)\s+{re.escape(name)}\b", original)
        ]
        if not missing_symbols:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"symbol_already_defined: {module_name}.{symbol_name}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"symbol": symbol_name, "module": module_name},
                "applied": False,
                "apply_result": result,
            }

        prefix = ""
        if not original:
            prefix = (
                '"""Auto-generated compatibility shim by reproduce agent."""\n'
                "from __future__ import annotations\n\n"
            )
        separator = "\n" if original and not original.endswith("\n") else ""
        stubs = "\n\n".join(_build_symbol_stub(name).rstrip("\n") for name in missing_symbols)
        updated = f"{original}{separator}{prefix}{stubs}\n"
        target.write_text(updated, encoding="utf-8")

        validate = subprocess.run(
            [sys.executable, "-m", "py_compile", str(target)],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if validate.returncode != 0:
            if original:
                target.write_text(original, encoding="utf-8")
            else:
                target.unlink(missing_ok=True)
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"py_compile_failed_for_symbol_patch: {module_name}.{symbol_name}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"symbol": symbol_name, "module": module_name},
                "applied": False,
                "apply_result": result,
            }

        rel = workspace_relative_path(target, workspace_root) or str(target)
        apply_result = {
            "applied": True,
            "summary": f"added missing symbols to {module_name}: {', '.join(missing_symbols)}",
            "edits": [{"kind": "replace" if original else "create", "path": rel}],
            "errors": [],
        }
        payload = {
            "module": module_name,
            "symbol": symbol_name,
            "symbols_added": missing_symbols,
            "target_file": rel,
            "validate": {
                "returncode": validate.returncode,
                "stdout": validate.stdout,
                "stderr": validate.stderr,
            },
        }
        write_json(step_dir / "patch.builtin.plan.json", payload)
        write_json(step_dir / "patch.builtin.apply.json", apply_result)
        return {
            "action": action_name,
            "record_path": str(step_dir / "patch.builtin.apply.json"),
            "result": payload,
            "applied": True,
            "apply_result": apply_result,
        }

    if action_name == "patch_missing_module_file":
        missing_module = ""
        module_candidates: list[str] = []
        evidence = (
            attempt.get("diagnosis_payload", {})
            .get("classification", {})
            .get("evidence", [])
        )
        for item in evidence:
            if not isinstance(item, dict):
                continue
            if item.get("missing_module") and not missing_module:
                missing_module = str(item["missing_module"])
            for raw in item.get("module_candidates", []):
                module_candidates.append(str(raw))
        target: Path | None = None
        for raw in module_candidates:
            path = Path(raw)
            try:
                rel = path.resolve().relative_to(workspace_root.resolve())
            except ValueError:
                continue
            candidate = workspace_root / rel
            if candidate.exists():
                continue
            target = candidate
            break
        if target is None:
            if missing_module:
                guessed = _resolve_module_target_file(workspace_root, missing_module)
                if not guessed.exists():
                    target = guessed
        if target is None:
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": ["builtin_recovery_no_missing_module_candidate_path"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"module": missing_module},
                "applied": False,
                "apply_result": result,
            }

        source_target: Path | None = None
        if source_repo_root is not None and missing_module:
            candidate = _resolve_module_target_file(source_repo_root, missing_module)
            if candidate.exists():
                source_target = candidate

        if source_target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source_target.read_text(encoding="utf-8"), encoding="utf-8")
            validate = subprocess.run(
                [sys.executable, "-m", "py_compile", str(target)],
                cwd=workspace_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if validate.returncode != 0:
                target.unlink(missing_ok=True)
                result = {
                    "applied": False,
                    "summary": "",
                    "edits": [],
                    "errors": [f"py_compile_failed_for_restored_module: {target}"],
                }
                write_json(step_dir / "patch.builtin.apply.json", result)
                return {
                    "action": action_name,
                    "record_path": str(step_dir / "patch.builtin.apply.json"),
                    "result": {"module": missing_module},
                    "applied": False,
                    "apply_result": result,
                }

            rel = workspace_relative_path(target, workspace_root) or str(target)
            apply_result = {
                "applied": True,
                "summary": f"restored missing module from source repo: {missing_module}",
                "edits": [{"kind": "create", "path": rel}],
                "errors": [],
            }
            payload = {
                "module": missing_module,
                "target_file": rel,
                "source_file": str(source_target),
                "validate": {
                    "returncode": validate.returncode,
                    "stdout": validate.stdout,
                    "stderr": validate.stderr,
                },
            }
            write_json(step_dir / "patch.builtin.plan.json", payload)
            write_json(step_dir / "patch.builtin.apply.json", apply_result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": payload,
                "applied": True,
                "apply_result": apply_result,
            }

        if missing_module.startswith(("qos.", "qvm.")):
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"internal_module_missing_no_source_template: {missing_module}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"module": missing_module},
                "applied": False,
                "apply_result": result,
            }

        symbols = _infer_imported_symbols_for_module(attempt, workspace_root, missing_module)
        content = _build_missing_module_stub(missing_module, symbols)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        validate = subprocess.run(
            [sys.executable, "-m", "py_compile", str(target)],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if validate.returncode != 0:
            target.unlink(missing_ok=True)
            result = {
                "applied": False,
                "summary": "",
                "edits": [],
                "errors": [f"py_compile_failed_for_created_module: {target}"],
            }
            write_json(step_dir / "patch.builtin.apply.json", result)
            return {
                "action": action_name,
                "record_path": str(step_dir / "patch.builtin.apply.json"),
                "result": {"module": missing_module},
                "applied": False,
                "apply_result": result,
            }

        rel = workspace_relative_path(target, workspace_root) or str(target)
        apply_result = {
            "applied": True,
            "summary": f"created compatibility shim for missing module {missing_module}",
            "edits": [{"kind": "create", "path": rel}],
            "errors": [],
        }
        payload = {
            "module": missing_module,
            "target_file": rel,
            "symbols": symbols,
            "validate": {
                "returncode": validate.returncode,
                "stdout": validate.stdout,
                "stderr": validate.stderr,
            },
        }
        write_json(step_dir / "patch.builtin.plan.json", payload)
        write_json(step_dir / "patch.builtin.apply.json", apply_result)
        return {
            "action": action_name,
            "record_path": str(step_dir / "patch.builtin.apply.json"),
            "result": payload,
            "applied": True,
            "apply_result": apply_result,
        }

    if action_name != "patch_lazy_optional_import":
        return None

    missing_module = ""
    evidence = (
        attempt.get("diagnosis_payload", {})
        .get("classification", {})
        .get("evidence", [])
    )
    for item in evidence:
        if isinstance(item, dict) and item.get("missing_module"):
            missing_module = str(item["missing_module"])
            break
    if not missing_module:
        missing_modules = (attempt.get("run_error_signals") or {}).get("missing_modules", [])
        if missing_modules:
            missing_module = str(missing_modules[0])
    if not missing_module:
        result = {
            "applied": False,
            "summary": "",
            "edits": [],
            "errors": ["builtin_recovery_missing_module_not_found"],
        }
        write_json(step_dir / "patch.builtin.apply.json", result)
        return {
            "action": action_name,
            "record_path": str(step_dir / "patch.builtin.apply.json"),
            "result": {"module": None},
            "applied": False,
            "apply_result": result,
        }

    candidates: list[Path] = []
    seen: set[str] = set()
    for file_ref in (attempt.get("run_error_signals") or {}).get("file_refs", []):
        raw_path = str(file_ref.get("path", ""))
        if not raw_path.endswith(".py"):
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        rel = workspace_relative_path(path, workspace_root)
        if rel is None or rel in seen:
            continue
        seen.add(rel)
        candidates.append(path)
    for path in workspace_root.rglob("*.py"):
        if len(candidates) >= 20:
            break
        rel = workspace_relative_path(path, workspace_root)
        if rel is None or rel in seen:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if missing_module in text:
            seen.add(rel)
            candidates.append(path)

    for path in candidates:
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        from_pat = re.compile(rf"(?m)^from\s+{re.escape(missing_module)}\s+import\s+(.+)$")
        import_pat = re.compile(
            rf"(?m)^import\s+{re.escape(missing_module)}(?:\s+as\s+([A-Za-z_]\w*))?\s*$"
        )

        edit_applied = False
        replacement_summary = ""
        patched = original

        from_match = from_pat.search(original)
        if from_match:
            clause = from_match.group(1).strip()
            safe_names = _safe_names_from_import_clause(clause)
            if safe_names:
                assignments = "\n".join(f"    {name} = None" for name in safe_names)
            else:
                assignments = "    pass"
            replacement = (
                "try:\n"
                f"    from {missing_module} import {clause}\n"
                "except ModuleNotFoundError:\n"
                f"{assignments}"
            )
            patched = from_pat.sub(replacement, original, count=1)
            edit_applied = patched != original
            replacement_summary = f"guarded from-import for missing optional module {missing_module}"
        else:
            import_match = import_pat.search(original)
            if import_match:
                alias = import_match.group(1) or missing_module.split(".")[-1]
                replacement = (
                    "try:\n"
                    f"    import {missing_module}"
                    + (f" as {alias}" if import_match.group(1) else "")
                    + "\n"
                    "except ModuleNotFoundError:\n"
                    f"    {alias} = None"
                )
                patched = import_pat.sub(replacement, original, count=1)
                edit_applied = patched != original
                replacement_summary = (
                    f"guarded import for missing optional module {missing_module}"
                )

        if not edit_applied:
            continue

        path.write_text(patched, encoding="utf-8")
        validate = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if validate.returncode != 0:
            path.write_text(original, encoding="utf-8")
            continue

        rel = workspace_relative_path(path, workspace_root) or str(path)
        apply_result = {
            "applied": True,
            "summary": replacement_summary,
            "edits": [{"kind": "replace", "path": rel}],
            "errors": [],
        }
        payload = {
            "module": missing_module,
            "target_file": rel,
            "summary": replacement_summary,
            "validate": {
                "returncode": validate.returncode,
                "stdout": validate.stdout,
                "stderr": validate.stderr,
            },
        }
        write_json(step_dir / "patch.builtin.plan.json", payload)
        write_json(step_dir / "patch.builtin.apply.json", apply_result)
        return {
            "action": action_name,
            "record_path": str(step_dir / "patch.builtin.apply.json"),
            "result": payload,
            "applied": True,
            "apply_result": apply_result,
        }

    result = {
        "applied": False,
        "summary": "",
        "edits": [],
        "errors": [f"builtin_recovery_no_patch_candidate_for_module: {missing_module}"],
    }
    write_json(step_dir / "patch.builtin.apply.json", result)
    return {
        "action": action_name,
        "record_path": str(step_dir / "patch.builtin.apply.json"),
        "result": {"module": missing_module},
        "applied": False,
        "apply_result": result,
    }


def execute_model_recovery(
    recipe: dict[str, Any],
    attempt: dict[str, Any],
    action_name: str,
    workspace_root: Path,
    paper_context: dict[str, Any] | None,
    api_base: str,
    api_key: str,
    model: str,
    step_dir: Path,
) -> dict[str, Any]:
    policy = recovery_policy(action_name, attempt, workspace_root)
    messages = build_patch_messages(recipe, attempt, action_name, workspace_root, paper_context)
    write_json(step_dir / "patch.prompt.json", {"messages": messages})
    response = call_openai_compatible(
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=messages,
    )
    write_json(step_dir / "patch.response.raw.json", response)
    patch_plan = extract_json_object(response["choices"][0]["message"]["content"])
    write_json(step_dir / "patch.plan.json", patch_plan)
    write_json(step_dir / "patch.policy.json", policy)
    apply_result = apply_patch_plan(workspace_root, patch_plan, step_dir, policy)
    return {
        "action": action_name,
        "record_path": str(step_dir / "patch.apply.json"),
        "result": patch_plan,
        "applied": apply_result["applied"],
        "apply_result": apply_result,
    }


def run_env_diagnose(
    repo_root: Path,
    recipe_path: Path,
    python_executable: str | None,
    step_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(repo_root / "evaluation/agent_framework/reproduce/env/diagnose_repro_env.py"),
        "--recipe",
        str(recipe_path),
    ]
    if python_executable:
        command.extend(["--python-executable", python_executable])

    stdout_path = step_dir / "env_diagnose.stdout.log"
    stderr_path = step_dir / "env_diagnose.stderr.log"
    proc = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    payload: dict[str, Any] = {
        "command": command,
        "returncode": proc.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    try:
        payload["report"] = extract_json_object(proc.stdout)
    except Exception:
        payload["report"] = None
    return payload


def build_messages(
    recipe: dict[str, Any],
    attempt: dict[str, Any],
    available_actions: list[str],
    env_diagnose: dict[str, Any] | None,
    paper_context: dict[str, Any] | None,
) -> list[dict[str, str]]:
    stderr_tail = []
    stdout_tail = []
    if attempt["run_error_signals"] is not None:
        stderr_tail = attempt["run_error_signals"].get("stderr_tail", [])
        stdout_tail = attempt["run_error_signals"].get("stdout_tail", [])

    prompt_payload = {
        "recipe": {
            "name": recipe.get("name"),
            "description": recipe.get("description"),
            "workspace_root": recipe.get("workspace_root"),
            "simulation_only": bool(recipe.get("recovery", {}).get("simulation_only")),
        },
        "attempt_status": attempt["status"],
        "attempt_status_payload": attempt["status_payload"],
        "diagnosis": attempt["diagnosis_payload"],
        "available_recovery_actions": available_actions,
        "tool_registry": atr.build_registry(available_actions),
        "stderr_tail": stderr_tail,
        "stdout_tail": stdout_tail,
        "env_diagnose": env_diagnose,
        "paper_context": paper_context,
    }

    system = """You are a reproduce orchestrator that picks tools in a multi-step agent loop.
Your job is to choose exactly one next tool call after reading structured artifacts from a reproduce runner.
Prefer simulation-only progress when simulation_only=true.
Do not invent capabilities. Only choose tools listed in tool_registry.
Return strict JSON with this schema:
{
  "tool": "finish" | "apply_recovery_action" | "run_env_diagnose" | "stop",
  "args": {},
  "reason": "short explanation",
  "expected_outcome": "what this should unblock"
}
Rules:
- For apply_recovery_action, args must be {"action": "<one of available_recovery_actions>"}.
- For finish/stop/run_env_diagnose, args must be {}.
- Choose finish only if current attempt is already successful or no further action is needed."""
    user = json.dumps(prompt_payload, indent=2, sort_keys=True)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_internal_failure_attempt(
    recipe_name: str,
    attempt_dir: Path,
    reason: str,
    error: dict[str, Any],
) -> dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    status_payload = {
        "success": False,
        "reason": reason,
        "status": "run_failed",
        "recipe_name": recipe_name,
        "run_dir": str(attempt_dir),
        "completed_at": rr.iso_now(),
    }
    diagnosis_payload = {
        "status": "run_failed",
        "summary": "Agent internal error while executing attempt.",
        "hints": [reason],
        "classification": {
            "category": "agent_internal_error",
            "recoverable": True,
            "confidence": 0.7,
            "evidence": [error],
        },
        "recovery_plan": {
            "category": "agent_internal_error",
            "recoverable": True,
            "mode": "agent_guided",
            "actions": [],
        },
    }
    write_json(attempt_dir / "status.json", status_payload)
    write_json(attempt_dir / "diagnosis.json", diagnosis_payload)
    write_json(attempt_dir / "attempt.error.json", error)
    manifest = {
        "recipe_name": recipe_name,
        "attempt_dir": str(attempt_dir),
        "workspace_root": "",
        "status": "run_failed",
        "artifacts": {
            "preflight": None,
            "run": None,
            "stdout": None,
            "stderr": None,
            "metrics": None,
            "status": str(attempt_dir / "status.json"),
            "diagnosis": str(attempt_dir / "diagnosis.json"),
            "error": str(attempt_dir / "attempt.error.json"),
        },
    }
    write_json(attempt_dir / "manifest.json", manifest)
    return {
        "status": "run_failed",
        "status_payload": status_payload,
        "diagnosis_payload": diagnosis_payload,
        "manifest": manifest,
        "preflight_payload": {
            "ok": False,
            "checks": {"commands_exist": [], "paths_exist": [], "python_imports": []},
        },
        "run_payload": None,
        "metrics_payload": None,
        "parse_error": reason,
        "run_error_signals": {
            "stderr_tail": [reason],
            "stdout_tail": [],
            "missing_modules": [],
            "import_name_failures": [],
            "name_errors": [],
            "file_refs": [],
        },
        "attempt_dir": attempt_dir,
        "artifacts": manifest["artifacts"],
    }


def main() -> int:
    redirected = maybe_redirect_to_native_tool_loop(sys.argv[1:])
    if redirected is not None:
        return int(redirected)

    args = parse_args()
    if args.strict_agent_tooling:
        # Strict mode should exercise real model/tool orchestration only.
        args.allow_offline_agent = False
        args.disable_rule_fallback = True
        args.disable_builtin_recovery = True
    rule_fallback_enabled = not args.disable_rule_fallback
    builtin_recovery_enabled = not args.disable_builtin_recovery
    require_live_endpoint = not args.allow_offline_agent
    blocked_recovery_actions: set[str] = set()
    if args.strict_agent_tooling:
        blocked_recovery_actions = {
            action
            for action in BUILTIN_ONLY_RECOVERY_ACTIONS
            if any(action.startswith(prefix) for prefix in STRICT_BLOCKED_RECOVERY_ACTION_PREFIXES)
        }
    repo_root = Path(__file__).resolve().parents[3]
    source_repo_root = select_source_repo_root(repo_root, args.source_repo_root)
    baseline_commit = None
    if args.isolation_mode == "git_snapshot":
        baseline_commit = resolve_git_commit(source_repo_root, args.baseline_ref)
    isolation_info = {
        "mode": args.isolation_mode,
        "source_repo_root": str(source_repo_root),
        "baseline_ref": args.baseline_ref if args.isolation_mode == "git_snapshot" else None,
        "baseline_commit": baseline_commit,
    }
    recipe_path = Path(args.recipe).resolve()
    recipe = load_json(recipe_path)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()

    run_id = f"{recipe['name']}_agent_{rr.utc_stamp()}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "recipe.snapshot.json", recipe)
    endpoint_probe = probe_openai_endpoint(args.api_base, args.api_key)
    write_json(run_dir / "model_endpoint_probe.json", endpoint_probe)
    llm_endpoint_usable = bool(endpoint_probe.get("ok"))
    if require_live_endpoint and not llm_endpoint_usable:
        final_status = {
            "agent_status": "endpoint_unavailable",
            "status": "endpoint_unavailable",
            "attempt_count": 0,
            "recovery_count": 0,
            "decision_count": 0,
            "tool_call_count": 0,
            "final_attempt_dir": None,
            "model": args.model,
            "api_base": args.api_base,
            "isolated_repo_root": None,
            "exit_policy": args.exit_policy,
            "rule_fallback_enabled": rule_fallback_enabled,
            "builtin_recovery_enabled": builtin_recovery_enabled,
            "allow_offline_agent": args.allow_offline_agent,
            "strict_agent_tooling": args.strict_agent_tooling,
            "blocked_recovery_actions": sorted(blocked_recovery_actions),
            "require_live_endpoint": require_live_endpoint,
            "endpoint_probe": endpoint_probe,
            "isolation": isolation_info,
        }
        write_json(run_dir / "status.json", final_status)
        manifest = {
            "recipe_name": recipe["name"],
            "run_id": run_id,
            "run_dir": str(run_dir),
            "workspace_root": None,
            "isolated_repo_root": None,
            "status": "endpoint_unavailable",
            "model": args.model,
            "isolation": isolation_info,
            "attempts": [],
            "recovery": [],
            "decisions": [],
            "final_attempt": None,
            "artifacts": {
                "recipe_snapshot": str(run_dir / "recipe.snapshot.json"),
                "recipe_generated_snapshot": None,
                "status": str(run_dir / "status.json"),
                "paper": None,
                "recipe_authoring": None,
                "endpoint_probe": str(run_dir / "model_endpoint_probe.json"),
            },
        }
        write_json(run_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 2

    isolated_repo_root = run_dir / "workspace" / "repo"
    if args.isolation_mode == "git_snapshot":
        if baseline_commit is None:
            raise RuntimeError("baseline commit unresolved for git_snapshot mode")
        export_git_snapshot(source_repo_root, isolated_repo_root, baseline_commit)
    else:
        copy_isolated_workspace(source_repo_root, isolated_repo_root)
    workspace_recipe_path = isolated_repo_root / relative_recipe_path(recipe_path, repo_root)
    if not workspace_recipe_path.exists():
        workspace_recipe_path.parent.mkdir(parents=True, exist_ok=True)
        workspace_recipe_path.write_text(recipe_path.read_text(encoding="utf-8"), encoding="utf-8")
    workspace_recipe = load_json(workspace_recipe_path)
    paper_payload = run_paper_extract(isolated_repo_root, workspace_recipe_path, run_dir)
    paper_context = None if paper_payload is None else paper_payload.get("context")
    recipe_authoring_payload = run_recipe_authoring(
        recipe=workspace_recipe,
        repo_root=isolated_repo_root,
        run_dir=run_dir,
        paper_context=paper_context,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
    )
    recipe_authoring_cfg = workspace_recipe.get("recipe_authoring", {})
    strict_recipe_authoring = bool(recipe_authoring_cfg.get("enabled", False)) and bool(
        recipe_authoring_cfg.get("strict_paper_alignment", False)
    )
    if strict_recipe_authoring and not (
        recipe_authoring_payload and recipe_authoring_payload.get("applied")
    ):
        final_status = {
            "status": "recipe_authoring_failed",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "workspace_root": str(isolated_repo_root),
            "reason": "strict recipe_authoring requested but no generated recipe was applied",
            "recipe_authoring": recipe_authoring_payload,
            "model": args.model,
            "api_base": args.api_base,
            "isolation": isolation_info,
        }
        write_json(run_dir / "status.json", final_status)
        manifest = {
            "recipe_name": workspace_recipe.get("name", recipe.get("name", "unknown_recipe")),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "workspace_root": str(isolated_repo_root),
            "isolated_repo_root": str(isolated_repo_root),
            "status": "recipe_authoring_failed",
            "model": args.model,
            "isolation": isolation_info,
            "attempts": [],
            "recovery": [],
            "decisions": [],
            "final_attempt": None,
            "artifacts": {
                "recipe_snapshot": str(run_dir / "recipe.snapshot.json"),
                "recipe_generated_snapshot": None,
                "status": str(run_dir / "status.json"),
                "paper": None if paper_payload is None else str(run_dir / "paper" / "paper_context.json"),
                "recipe_authoring": str(run_dir / "recipe_authoring" / "manifest.json"),
                "endpoint_probe": str(run_dir / "model_endpoint_probe.json"),
            },
        }
        write_json(run_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 2
    if recipe_authoring_payload and recipe_authoring_payload.get("applied"):
        workspace_recipe = recipe_authoring_payload["recipe"]
        write_json(run_dir / "recipe.generated.snapshot.json", workspace_recipe)

    workspace_root = rr.resolve_workspace_root(isolated_repo_root, workspace_recipe.get("workspace_root", "."))

    python_executable = (
        args.python_executable
        or workspace_recipe.get("runtime", {}).get("python_executable")
        or sys.executable
    )

    attempts: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    attempts_dir = run_dir / "attempts"
    steps_dir = run_dir / "agent_steps"

    try:
        current_attempt = rr.run_attempt(
            recipe=workspace_recipe,
            repo_root=isolated_repo_root,
            workspace_root=workspace_root,
            python_exe=python_executable,
            attempt_dir=attempts_dir / "attempt_1",
            preflight_only=False,
            allow_run_on_preflight_failure=args.allow_run_on_preflight_failure,
        )
    except Exception as exc:
        error = exception_payload(exc, "attempt.initial_run")
        write_json(run_dir / "attempt.initial_run.error.json", error)
        current_attempt = build_internal_failure_attempt(
            recipe_name=workspace_recipe.get("name", recipe.get("name", "unknown_recipe")),
            attempt_dir=attempts_dir / "attempt_1",
            reason=f"initial_run_exception: {error['error_type']}: {error['error']}",
            error=error,
        )
    attempts.append(current_attempt["manifest"])
    loop_state: atr.AgentLoopState | None = None

    def rerun_attempt_from_step(step_dir: Path) -> None:
        nonlocal current_attempt, loop_state
        attempt_dir = attempts_dir / f"attempt_{len(attempts) + 1}"
        selected_python = (
            loop_state.python_executable if loop_state is not None else python_executable
        )
        try:
            current_attempt = rr.run_attempt(
                recipe=workspace_recipe,
                repo_root=isolated_repo_root,
                workspace_root=workspace_root,
                python_exe=selected_python,
                attempt_dir=attempt_dir,
                preflight_only=False,
                allow_run_on_preflight_failure=False,
            )
        except Exception as exc:
            error = exception_payload(exc, "attempt.post_recovery_run")
            write_json(step_dir / "attempt.post_recovery_run.error.json", error)
            current_attempt = build_internal_failure_attempt(
                recipe_name=workspace_recipe.get("name", recipe.get("name", "unknown_recipe")),
                attempt_dir=attempt_dir,
                reason=f"post_recovery_run_exception: {error['error_type']}: {error['error']}",
                error=error,
            )
        if loop_state is not None:
            loop_state.current_attempt = current_attempt
        attempts.append(current_attempt["manifest"])

    loop_state = atr.AgentLoopState(
        workspace_recipe=workspace_recipe,
        isolated_repo_root=isolated_repo_root,
        workspace_recipe_path=workspace_recipe_path,
        workspace_root=workspace_root,
        python_executable=python_executable,
        args_python_executable=args.python_executable,
        run_dir=run_dir,
        paper_context=paper_context,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        repo_root=source_repo_root,
        rule_fallback_enabled=rule_fallback_enabled,
        builtin_recovery_enabled=builtin_recovery_enabled,
        current_attempt=current_attempt,
        env_diagnose_payload=None,
        recoveries=recoveries,
    )

    for step_index in range(1, args.max_agent_steps + 1):
        if loop_state.current_attempt["status"] in {"success", "preflight_only"}:
            break

        step_dir = steps_dir / f"step_{step_index:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        raw_available_actions = available_recovery_actions(
            workspace_recipe,
            loop_state.current_attempt,
            blocked_actions=blocked_recovery_actions,
        )
        current_missing_candidates = infer_missing_modules_from_attempt(loop_state.current_attempt)
        current_missing_module = current_missing_candidates[0] if current_missing_candidates else ""
        failed_counts = collect_failed_action_counts(recoveries)
        applied_counts = collect_applied_action_counts(recoveries)
        fail_limit = action_failure_limit()
        exhausted_actions: list[str] = []
        for action in raw_available_actions:
            scoped_key = f"{action}::{current_missing_module}" if current_missing_module else action
            scoped_fail_count = failed_counts.get(scoped_key, 0)
            global_fail_count = failed_counts.get(action, 0)
            if current_missing_module and scoped_key in failed_counts:
                effective_fail_count = scoped_fail_count
            else:
                effective_fail_count = global_fail_count
            if effective_fail_count >= fail_limit:
                exhausted_actions.append(action)
                continue
            if action == "runtime_env_select":
                scoped_applied_count = applied_counts.get(scoped_key, 0)
                global_applied_count = applied_counts.get(action, 0)
                if current_missing_module and scoped_key in applied_counts:
                    effective_applied_count = scoped_applied_count
                else:
                    effective_applied_count = global_applied_count
                if effective_applied_count >= 1:
                    exhausted_actions.append(action)
        exhausted_actions = sorted(set(exhausted_actions))
        available_actions = [
            action for action in raw_available_actions if action not in exhausted_actions
        ]
        write_json(
            step_dir / "action_filter.json",
            {
                "raw_available_actions": raw_available_actions,
                "current_missing_module": current_missing_module,
                "failed_action_counts": failed_counts,
                "applied_action_counts": applied_counts,
                "max_failed_per_action": fail_limit,
                "exhausted_actions": exhausted_actions,
                "available_actions": available_actions,
            },
        )
        if not available_actions and loop_state.env_diagnose_payload is not None:
            tool_call = {
                "tool": "stop",
                "args": {},
                "reason": "no_available_actions_after_action_filter_and_env_already_diagnosed",
                "expected_outcome": "none",
            }
            decisions.append(tool_call)
            write_json(step_dir / "tool_call.default.json", tool_call)
            break
        messages = build_messages(
            workspace_recipe,
            loop_state.current_attempt,
            available_actions,
            loop_state.env_diagnose_payload,
            paper_context,
        )
        write_json(step_dir / "prompt.json", {"messages": messages})

        tool_call: dict[str, Any] | None = None

        if not llm_endpoint_usable and rule_fallback_enabled:
            failed_counts = collect_failed_action_counts(recoveries)
            fallback_payload = rule_fallback_decision(
                attempt=loop_state.current_attempt,
                available_actions=available_actions,
                env_diagnose=loop_state.env_diagnose_payload,
                failed_action_counts=failed_counts,
                reason=f"model_endpoint_probe_failed: {endpoint_probe.get('error', 'unreachable')}",
            )
            write_json(step_dir / "decision.fallback.json", fallback_payload)
            try:
                tool_call = atr.normalize_tool_call(fallback_payload, available_actions)
            except Exception as exc:
                error = exception_payload(exc, "step.normalize_fallback_tool_call")
                write_json(step_dir / "tool_call.normalize.error.json", error)
                tool_call = {
                    "tool": "stop",
                    "args": {},
                    "reason": f"fallback_tool_call_normalization_failed: {error['error']}",
                    "expected_outcome": "none",
                }
            write_json(step_dir / "tool_call.fallback.json", tool_call)

        if tool_call is None:
            model_exception: dict[str, Any] | None = None
            response: dict[str, Any] | None = None
            try:
                response = call_openai_compatible(
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=args.model,
                    messages=messages,
                )
            except Exception as exc:
                model_exception = exception_payload(exc, "step.call_openai_compatible")
                write_json(step_dir / "response.exception.json", model_exception)

            if response is None:
                if rule_fallback_enabled:
                    failed_counts = collect_failed_action_counts(recoveries)
                    fallback_payload = rule_fallback_decision(
                        attempt=loop_state.current_attempt,
                        available_actions=available_actions,
                        env_diagnose=loop_state.env_diagnose_payload,
                        failed_action_counts=failed_counts,
                        reason=(
                            "model_endpoint_error: "
                            f"{model_exception['error_type']}: {model_exception['error']}"
                        ),
                    )
                    write_json(step_dir / "decision.fallback.json", fallback_payload)
                    try:
                        tool_call = atr.normalize_tool_call(
                            fallback_payload, available_actions
                        )
                    except Exception as exc:
                        error = exception_payload(exc, "step.normalize_fallback_tool_call")
                        write_json(step_dir / "tool_call.normalize.error.json", error)
                        tool_call = {
                            "tool": "stop",
                            "args": {},
                            "reason": (
                                "fallback_tool_call_normalization_failed: "
                                f"{error['error']}"
                            ),
                            "expected_outcome": "none",
                        }
                    write_json(step_dir / "tool_call.fallback.json", tool_call)
                else:
                    tool_call = {
                        "tool": "stop",
                        "args": {},
                        "reason": (
                            "model_endpoint_error: "
                            f"{model_exception['error_type']}: {model_exception['error']}"
                        ),
                        "expected_outcome": "none",
                    }
            else:
                write_json(step_dir / "response.raw.json", response)
                try:
                    raw_content = response["choices"][0]["message"]["content"]
                    parsed = extract_json_object(raw_content)
                    write_json(step_dir / "decision.json", parsed)
                    tool_call = atr.normalize_tool_call(parsed, available_actions)
                    write_json(step_dir / "tool_call.json", tool_call)
                except Exception as exc:
                    error = exception_payload(exc, "step.parse_or_normalize_model_output")
                    write_json(step_dir / "decision.parse.error.json", error)
                    if rule_fallback_enabled:
                        failed_counts = collect_failed_action_counts(recoveries)
                        fallback_payload = rule_fallback_decision(
                            attempt=loop_state.current_attempt,
                            available_actions=available_actions,
                            env_diagnose=loop_state.env_diagnose_payload,
                            failed_action_counts=failed_counts,
                            reason=(
                                "model_decision_parse_exception: "
                                f"{error['error_type']}: {error['error']}"
                            ),
                        )
                        write_json(step_dir / "decision.fallback.json", fallback_payload)
                        try:
                            tool_call = atr.normalize_tool_call(
                                fallback_payload, available_actions
                            )
                        except Exception as normalize_exc:
                            normalize_error = exception_payload(
                                normalize_exc, "step.normalize_fallback_tool_call"
                            )
                            write_json(
                                step_dir / "tool_call.normalize.error.json",
                                normalize_error,
                            )
                            tool_call = {
                                "tool": "stop",
                                "args": {},
                                "reason": (
                                    "fallback_tool_call_normalization_failed: "
                                    f"{normalize_error['error']}"
                                ),
                                "expected_outcome": "none",
                            }
                        write_json(step_dir / "tool_call.fallback.json", tool_call)
                    else:
                        tool_call = {
                            "tool": "stop",
                            "args": {},
                            "reason": (
                                "model_decision_parse_exception: "
                                f"{error['error_type']}: {error['error']}"
                            ),
                            "expected_outcome": "none",
                        }

        if tool_call is None:
            tool_call = {
                "tool": "stop",
                "args": {},
                "reason": "tool_call_missing_after_decision",
                "expected_outcome": "none",
            }
            write_json(step_dir / "tool_call.default.json", tool_call)

        decisions.append(tool_call)
        execute_status = atr.execute_tool_call(
            state=loop_state,
            tool_call=tool_call,
            available_actions=available_actions,
            step_dir=step_dir,
            run_env_diagnose_fn=run_env_diagnose,
            execute_builtin_recovery_fn=execute_builtin_recovery,
            execute_model_recovery_fn=execute_model_recovery,
            rerun_attempt_fn=rerun_attempt_from_step,
            write_json_fn=write_json,
            exception_payload_fn=exception_payload,
            builtin_only_actions=BUILTIN_ONLY_RECOVERY_ACTIONS,
        )
        current_attempt = loop_state.current_attempt
        if execute_status == "break":
            break
        continue

    final_status = {
        "agent_status": "completed",
        "status": current_attempt["status"],
        "attempt_count": len(attempts),
        "recovery_count": len(recoveries),
        "decision_count": len(decisions),
        "tool_call_count": len(decisions),
        "final_attempt_dir": str(current_attempt["attempt_dir"]),
        "model": args.model,
        "api_base": args.api_base,
        "isolated_repo_root": str(isolated_repo_root),
        "exit_policy": args.exit_policy,
        "rule_fallback_enabled": rule_fallback_enabled,
        "builtin_recovery_enabled": builtin_recovery_enabled,
        "allow_offline_agent": args.allow_offline_agent,
        "strict_agent_tooling": args.strict_agent_tooling,
        "blocked_recovery_actions": sorted(blocked_recovery_actions),
        "require_live_endpoint": require_live_endpoint,
        "isolation": isolation_info,
    }
    write_json(run_dir / "status.json", final_status)

    manifest = {
        "recipe_name": workspace_recipe["name"],
        "run_id": run_id,
        "run_dir": str(run_dir),
        "workspace_root": str(workspace_root),
        "isolated_repo_root": str(isolated_repo_root),
        "status": current_attempt["status"],
        "model": args.model,
        "isolation": isolation_info,
        "attempts": attempts,
        "recovery": recoveries,
        "decisions": decisions,
        "final_attempt": current_attempt["manifest"],
        "artifacts": {
            "recipe_snapshot": str(run_dir / "recipe.snapshot.json"),
            "recipe_generated_snapshot": (
                str(run_dir / "recipe.generated.snapshot.json")
                if (run_dir / "recipe.generated.snapshot.json").exists()
                else None
            ),
            "status": str(run_dir / "status.json"),
            "paper": None if paper_payload is None else str(run_dir / "paper" / "paper_context.json"),
            "recipe_authoring": (
                str(run_dir / "recipe_authoring" / "manifest.json")
                if recipe_authoring_payload is not None
                else None
            ),
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    reproduce_ok = current_attempt["status"] in {"success", "preflight_only"}
    if args.exit_policy == "strict":
        return 0 if reproduce_ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
