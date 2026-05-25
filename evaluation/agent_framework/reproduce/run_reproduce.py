#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from figure_claim import evaluate_figure_contract


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def resolve_workspace_root(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (repo_root / path).resolve()


def expand_template(value: str, context: dict[str, str]) -> str:
    return value.format(**context)


def build_workspace_env(base_env: dict[str, str] | None, workspace_root: Path) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env["PYTHONPATH"] = str(workspace_root)
    return env


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_rel_path(path: str, workspace_root: Path) -> str:
    raw = str(path).strip()
    if not raw:
        return ""
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(workspace_root.resolve()).as_posix()
        except ValueError:
            return candidate.as_posix()
    return candidate.as_posix()


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_strings(item))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    return []


def original_code_path_requirements(recipe: dict[str, Any]) -> dict[str, Any]:
    requirements = recipe.get("reproduction_requirements") or {}
    original = requirements.get("original_code_path") or {}
    return {
        "required": bool(requirements.get("original_code_path_required")),
        "min_original_modules": int(original.get("min_original_modules", 1)),
        "forbidden_path_prefixes": [
            str(item).strip().rstrip("/") + "/"
            for item in _as_list(original.get("forbidden_path_prefixes"))
            if str(item).strip()
        ],
        "preferred_path_prefixes": [
            str(item).strip().rstrip("/") + "/"
            for item in _as_list(original.get("preferred_path_prefixes"))
            if str(item).strip()
        ],
        "required_modules": [
            str(item).strip()
            for item in _as_list(original.get("required_modules"))
            if str(item).strip()
        ],
    }


def collect_original_code_path_evidence(
    recipe: dict[str, Any],
    workspace_root: Path,
    run_payload: dict[str, Any] | None,
    metrics_payload: dict[str, Any] | None,
    verification_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    req = original_code_path_requirements(recipe)
    command_parts = [str(item) for item in (run_payload or {}).get("command", [])]
    command_paths = [_normalize_rel_path(item, workspace_root) for item in command_parts]
    command_blob = "\n".join(command_paths)
    payload_strings = "\n".join(
        _flatten_strings(metrics_payload or {}) + _flatten_strings(verification_payload or {})
    )
    evidence_blob = command_blob + "\n" + payload_strings

    forbidden_hits = [
        prefix
        for prefix in req["forbidden_path_prefixes"]
        if any(path.startswith(prefix) for path in command_paths)
    ]
    preferred_hits = [
        prefix
        for prefix in req["preferred_path_prefixes"]
        if prefix in evidence_blob
    ]
    module_hits = [
        module
        for module in req["required_modules"]
        if module in evidence_blob or module.replace(".", "/") in evidence_blob
    ]
    evidence_count = len(set(preferred_hits + module_hits))
    ok = (not req["required"]) or (
        not forbidden_hits and evidence_count >= req["min_original_modules"]
    )
    return {
        "required": req["required"],
        "ok": ok,
        "min_original_modules": req["min_original_modules"],
        "evidence_count": evidence_count,
        "forbidden_path_prefixes": req["forbidden_path_prefixes"],
        "preferred_path_prefixes": req["preferred_path_prefixes"],
        "required_modules": req["required_modules"],
        "forbidden_hits": forbidden_hits,
        "preferred_hits": preferred_hits,
        "module_hits": module_hits,
        "command_paths": command_paths,
    }


def proxy_harness_entry_violation(
    recipe: dict[str, Any],
    workspace_root: Path,
    command: list[str],
) -> dict[str, Any] | None:
    req = original_code_path_requirements(recipe)
    if not req["required"]:
        return None

    command_paths = [_normalize_rel_path(item, workspace_root) for item in command]
    forbidden_hits = [
        prefix
        for prefix in req["forbidden_path_prefixes"]
        if any(path.startswith(prefix) for path in command_paths)
    ]
    harness_hits = [
        path
        for path in command_paths
        if path.startswith("evaluation/agent_framework/reproduce/harnesses/")
    ]
    if not forbidden_hits and not harness_hits:
        return None

    return {
        "reason": "proxy harness entry is forbidden when original code path is required",
        "command_paths": command_paths,
        "forbidden_hits": forbidden_hits,
        "harness_hits": harness_hits,
    }


def run_import_check(python_exe: str, workspace_root: Path, module_name: str) -> dict[str, Any]:
    script = (
        "import importlib, json, sys; "
        f"sys.path.insert(0, {workspace_root.as_posix()!r}); "
        f"importlib.import_module({module_name!r}); "
        "print(json.dumps({'ok': True}))"
    )
    started = time.time()
    proc = subprocess.run(
        [python_exe, "-c", script],
        capture_output=True,
        text=True,
        cwd=workspace_root,
        env=build_workspace_env(os.environ.copy(), workspace_root),
        check=False,
    )
    duration = time.time() - started
    if proc.returncode == 0:
        return {
            "name": module_name,
            "ok": True,
            "duration_seconds": duration,
        }
    return {
        "name": module_name,
        "ok": False,
        "duration_seconds": duration,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
        "stdout": proc.stdout.strip(),
    }


def preflight(recipe: dict[str, Any], workspace_root: Path, python_exe: str) -> dict[str, Any]:
    checks: dict[str, list[dict[str, Any]]] = {
        "commands_exist": [],
        "paths_exist": [],
        "python_imports": [],
    }

    preflight_cfg = recipe.get("preflight", {})

    for command_name in preflight_cfg.get("commands_exist", []):
        resolved = shutil.which(command_name)
        checks["commands_exist"].append(
            {
                "name": command_name,
                "ok": resolved is not None,
                "resolved_path": resolved,
            }
        )

    for raw_path in preflight_cfg.get("paths_exist", []):
        full_path = (workspace_root / raw_path).resolve()
        checks["paths_exist"].append(
            {
                "path": raw_path,
                "ok": full_path.exists(),
                "resolved_path": str(full_path),
            }
        )

    optional_imports = set(preflight_cfg.get("optional_python_imports", []))
    for module_name in preflight_cfg.get("python_imports", []):
        result = run_import_check(python_exe, workspace_root, module_name)
        result["required"] = module_name not in optional_imports
        result["optional"] = module_name in optional_imports
        checks["python_imports"].append(result)

    ok = (
        all(item["ok"] for item in checks["commands_exist"])
        and all(item["ok"] for item in checks["paths_exist"])
        and all(item["ok"] for item in checks["python_imports"] if item.get("required", True))
    )
    return {
        "checked_at": iso_now(),
        "workspace_root": str(workspace_root),
        "python_executable": python_exe,
        "ok": ok,
        "checks": checks,
    }


def collect_run_error_signals(run_dir: Path) -> dict[str, Any]:
    stderr_path = run_dir / "stderr.log"
    stdout_path = run_dir / "stdout.log"
    metrics_path = run_dir / "metrics.json"
    stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
    stdout = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    metrics_error_chunks: list[str] = []
    if metrics_path.exists():
        try:
            metrics_payload = load_json(metrics_path)
        except Exception:
            metrics_payload = {}
        for item in metrics_payload.get("checked_modules", []):
            if not isinstance(item, dict):
                continue
            err = str(item.get("error", "")).strip()
            tb = str(item.get("traceback", "")).strip()
            if err:
                metrics_error_chunks.append(err)
            if tb:
                metrics_error_chunks.append(tb)
    combined = "\n".join(part for part in (stderr, stdout, "\n".join(metrics_error_chunks)) if part)

    missing_modules = re.findall(
        r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
        combined,
    )
    import_name_failures = re.findall(
        r"ImportError: cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]",
        combined,
    )
    name_errors = re.findall(
        r"NameError: name ['\"]([^'\"]+)['\"] is not defined",
        combined,
    )
    file_refs = re.findall(r'File "([^"]+)", line (\d+)', combined)

    return {
        "stderr_tail": stderr.strip().splitlines()[-20:],
        "stdout_tail": stdout.strip().splitlines()[-20:],
        "missing_modules": missing_modules,
        "import_name_failures": [
            {"name": name, "from": source} for name, source in import_name_failures
        ],
        "name_errors": name_errors,
        "file_refs": [
            {"path": path, "line": int(line)} for path, line in file_refs[-10:]
        ],
    }


def candidate_module_paths(workspace_root: Path, module_name: str) -> list[Path]:
    parts = [part for part in module_name.split(".") if part]
    if not parts:
        return []
    package_path = workspace_root.joinpath(*parts, "__init__.py")
    module_path = workspace_root.joinpath(*parts).with_suffix(".py")
    return [module_path, package_path]


def module_exists_in_workspace(workspace_root: Path, module_name: str) -> bool:
    return any(path.exists() for path in candidate_module_paths(workspace_root, module_name))


def is_likely_runtime_import_surface_mismatch(module_name: str) -> bool:
    parts = [part for part in module_name.split(".") if part]
    if len(parts) < 2:
        return False
    if not all(part.isidentifier() for part in parts):
        return False
    internal_roots = {"qos", "qvm", "Baseline_Multiprogramming", "settings", "data", "src"}
    if parts[0] in internal_roots:
        return False
    return True


def classify_failure(
    recipe: dict[str, Any],
    workspace_root: Path,
    status: str,
    preflight_payload: dict[str, Any],
    run_payload: dict[str, Any] | None,
    metrics_payload: dict[str, Any] | None,
    run_error_signals: dict[str, Any] | None,
    parse_error: str | None,
    status_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recovery_cfg = recipe.get("recovery", {})
    optional_imports = set(recipe.get("preflight", {}).get("optional_python_imports", []))
    optional_runtime_modules = set(recovery_cfg.get("optional_runtime_modules", []))
    simulation_only = bool(recovery_cfg.get("simulation_only"))

    if status == "success":
        return {
            "category": "success",
            "recoverable": False,
            "confidence": 1.0,
            "evidence": [],
        }

    if status == "run_failed" and status_payload is not None:
        evidence = status_payload.get("original_code_path_evidence")
        if status_payload.get("reason") == "original code path requirement was not satisfied":
            return {
                "category": "original_code_path_not_exercised",
                "recoverable": True,
                "confidence": 0.98,
                "evidence": [evidence] if isinstance(evidence, dict) else [],
            }

    if status == "run_failed" and run_payload is not None and run_payload.get("proxy_harness_forbidden"):
        return {
            "category": "original_code_path_not_exercised",
            "recoverable": True,
            "confidence": 0.99,
            "evidence": [run_payload.get("proxy_harness_forbidden")],
        }

    if status == "preflight_failed":
        failed_imports = [
            item["name"]
            for item in preflight_payload["checks"]["python_imports"]
            if not item["ok"] and item.get("required", True)
        ]
        missing_paths = [
            item["path"]
            for item in preflight_payload["checks"]["paths_exist"]
            if not item["ok"]
        ]
        missing_commands = [
            item["name"]
            for item in preflight_payload["checks"]["commands_exist"]
            if not item["ok"]
        ]
        if failed_imports:
            category = "missing_runtime_dependency"
            recoverable = True
            if simulation_only and all(name in optional_imports for name in failed_imports):
                category = "optional_runtime_dependency_for_simulation"
            return {
                "category": category,
                "recoverable": recoverable,
                "confidence": 0.95,
                "evidence": [{"missing_imports": failed_imports}],
            }
        if missing_paths:
            return {
                "category": "missing_required_paths",
                "recoverable": True,
                "confidence": 0.95,
                "evidence": [{"missing_paths": missing_paths}],
            }
        if missing_commands:
            return {
                "category": "missing_required_commands",
                "recoverable": True,
                "confidence": 0.95,
                "evidence": [{"missing_commands": missing_commands}],
            }
        return {
            "category": "preflight_failure_unknown",
            "recoverable": True,
            "confidence": 0.5,
            "evidence": [],
        }

    if status == "run_failed" and run_error_signals is not None:
        missing_modules = run_error_signals.get("missing_modules", [])
        if missing_modules:
            first = missing_modules[0]
            if first.startswith(("qos.", "qvm.", "Baseline_Multiprogramming", "data.", "settings.")):
                category = (
                    "source_compat_bad_import_statement"
                    if module_exists_in_workspace(workspace_root, first)
                    else "source_compat_missing_module_file"
                )
                return {
                    "category": category,
                    "recoverable": True,
                    "confidence": 0.98,
                    "evidence": [
                        {
                            "missing_module": first,
                            "module_candidates": [
                                str(path) for path in candidate_module_paths(workspace_root, first)
                            ],
                        }
                    ],
                }
            if simulation_only and (first in optional_imports or first in optional_runtime_modules):
                return {
                    "category": "optional_runtime_dependency_used_in_simulation_path",
                    "recoverable": True,
                    "confidence": 0.9,
                    "evidence": [{"missing_module": first}],
                }
            if is_likely_runtime_import_surface_mismatch(first):
                return {
                    "category": "runtime_import_surface_mismatch",
                    "recoverable": True,
                    "confidence": 0.9,
                    "evidence": [{"missing_module": first}],
                }
            return {
                "category": "missing_runtime_dependency",
                "recoverable": True,
                "confidence": 0.95,
                "evidence": [{"missing_module": first}],
            }

        import_name_failures = run_error_signals.get("import_name_failures", [])
        if import_name_failures:
            first = import_name_failures[0]
            source = first.get("from", "")
            if source.startswith(
                ("qos.", "qvm.", "Baseline_Multiprogramming", "settings.", "data.")
            ):
                return {
                    "category": "source_compat_missing_symbol_import",
                    "recoverable": True,
                    "confidence": 0.95,
                    "evidence": [first],
                }
            return {
                "category": "runtime_import_surface_mismatch",
                "recoverable": True,
                "confidence": 0.9,
                "evidence": [first],
            }

        name_errors = run_error_signals.get("name_errors", [])
        if name_errors:
            first = name_errors[0]
            file_refs = run_error_signals.get("file_refs", [])
            in_repo_source = any(
                "/qos/" in item.get("path", "")
                or "/qvm/" in item.get("path", "")
                or "/Baseline_Multiprogramming/" in item.get("path", "")
                for item in file_refs
            )
            if in_repo_source:
                return {
                    "category": "source_compat_missing_symbol_reference",
                    "recoverable": True,
                    "confidence": 0.9,
                    "evidence": [{"name": first}],
                }

        if run_payload is not None and run_payload.get("returncode") not in (0, None):
            return {
                "category": "runtime_execution_failure",
                "recoverable": True,
                "confidence": 0.6,
                "evidence": [{"returncode": run_payload["returncode"]}],
            }

    if status == "parse_failed":
        return {
            "category": "metrics_parse_failure",
            "recoverable": True,
            "confidence": 0.9,
            "evidence": [{"parse_error": parse_error}] if parse_error else [],
        }

    if status == "metric_failed":
        if isinstance(metrics_payload, dict) and metrics_payload.get("failure_category") == "original_pipeline_not_executable":
            return {
                "category": "original_pipeline_not_executable",
                "recoverable": True,
                "confidence": 0.95,
                "evidence": [
                    {
                        "reason": metrics_payload.get("reason"),
                        "recommended_next_actions": metrics_payload.get("recommended_next_actions", []),
                        "original_pipeline_evidence": metrics_payload.get("original_pipeline_evidence", {}),
                    }
                ],
            }
        return {
            "category": "metric_validation_failure",
            "recoverable": True,
            "confidence": 0.9,
            "evidence": [],
        }

    return {
        "category": "unknown_failure",
        "recoverable": True,
        "confidence": 0.3,
        "evidence": [],
    }


def build_recovery_plan(
    recipe: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    recovery_cfg = recipe.get("recovery", {})
    category = classification["category"]
    plan = {
        "category": category,
        "recoverable": classification["recoverable"],
        "mode": recovery_cfg.get("mode", "agent_guided"),
        "actions": [],
    }

    if category == "optional_runtime_dependency_for_simulation":
        plan["actions"] = [
            {
                "action": "downgrade_missing_import_to_optional",
                "detail": "Do not fail simulation-only preflight on this dependency.",
            },
            {
                "action": "retry_reproduce",
                "detail": "Re-run reproduce without blocking on the optional import.",
            },
        ]
    elif category == "optional_runtime_dependency_used_in_simulation_path":
        plan["actions"] = [
            {
                "action": "patch_lazy_optional_import",
                "detail": "Move the optional runtime dependency behind a lazy import or guard in the simulation path.",
            },
            {
                "action": "retry_reproduce",
                "detail": "Re-run the same smoke recipe after the compatibility patch.",
            },
        ]
    elif category == "source_compat_missing_module_file":
        plan["actions"] = [
            {
                "action": "source_path_fix",
                "detail": "Repair source/package path configuration when import resolution cannot find a module file.",
            },
            {
                "action": "retry_reproduce",
                "detail": "Re-run the same smoke recipe after the compatibility patch.",
            },
        ]
    elif category == "source_compat_bad_import_statement":
        plan["actions"] = [
            {
                "action": "patch_bad_import_statement",
                "detail": "Repair the import statement or package path when the target module exists but import resolution still fails.",
            },
            {
                "action": "retry_reproduce",
                "detail": "Re-run the same smoke recipe after the compatibility patch.",
            },
        ]
    elif category == "source_compat_missing_symbol_import":
        plan["actions"] = [
            {
                "action": "runtime_trace_fix",
                "detail": "Repair symbol import compatibility in existing source code without creating shim modules.",
            },
            {
                "action": "retry_reproduce",
                "detail": "Re-run the same smoke recipe after the compatibility patch.",
            },
        ]
    elif category == "source_compat_missing_symbol_reference":
        plan["actions"] = [
            {
                "action": "runtime_trace_fix",
                "detail": "Repair undefined in-repo symbol references by editing existing files only.",
            },
            {
                "action": "retry_reproduce",
                "detail": "Re-run the same smoke recipe after the compatibility patch.",
            },
        ]
    elif category == "missing_runtime_dependency":
        plan["actions"] = [
            {
                "action": "runtime_env_select",
                "detail": "Select a compatible Python runtime/venv for the current reproduce workspace before package-level fixes.",
            },
            {
                "action": "select_or_prepare_python_environment",
                "detail": "Use env diagnosis to choose an interpreter or install the missing dependency.",
            },
            {
                "action": "retry_preflight",
                "detail": "Re-run preflight before the full reproduce attempt.",
            },
        ]
    elif category == "runtime_import_surface_mismatch":
        plan["actions"] = [
            {
                "action": "runtime_trace_fix",
                "detail": "Repair import-surface/version compatibility for missing submodules before environment installs.",
            },
            {
                "action": "runtime_env_select",
                "detail": "If import-surface repair is insufficient, switch to a compatible Python runtime/venv.",
            },
            {
                "action": "select_or_prepare_python_environment",
                "detail": "If import-surface repair is insufficient, adjust runtime environment.",
            },
            {
                "action": "retry_preflight",
                "detail": "Re-run preflight before the full reproduce attempt.",
            },
        ]
    elif category == "missing_required_paths":
        plan["actions"] = [
            {
                "action": "repair_recipe_paths",
                "detail": "Update the recipe to point at paths that exist in the current repo.",
            }
        ]
    elif category == "missing_required_commands":
        plan["actions"] = [
            {
                "action": "install_or_select_required_command",
                "detail": "Provide the missing executable in PATH or use a different environment.",
            }
        ]
    elif category == "metrics_parse_failure":
        plan["actions"] = [
            {
                "action": "fix_parser_or_harness",
                "detail": "Update the harness output or parser contract so required metrics are emitted.",
            }
        ]
    elif category == "metric_validation_failure":
        plan["actions"] = [
            {
                "action": "inspect_metrics_and_thresholds",
                "detail": "Check whether the recipe success criteria are too strict for this smoke test.",
            }
        ]
    elif category == "runtime_execution_failure":
        plan["actions"] = [
            {
                "action": "inspect_traceback_and_retry",
                "detail": "Use stderr/stdout plus failure classification to identify the next targeted fix.",
            }
        ]
    elif category == "original_code_path_not_exercised":
        plan["actions"] = [
            {
                "action": "repo_path_probe",
                "detail": "Inspect original repository execution paths before accepting proxy-only artifacts.",
            },
            {
                "action": "original_pipeline_probe",
                "detail": "Map the figure claim to original repository metric-producing modules before building a runner.",
            },
            {
                "action": "build_original_runner",
                "detail": "Route the reproduction through original repository modules and collect code-path evidence.",
            },
        ]
    elif category == "original_pipeline_not_executable":
        evidence_text = json.dumps(classification.get("evidence", []), sort_keys=True)
        if "invalid environment" in evidence_text and "qiskit-terra" in evidence_text:
            plan["actions"] = [
                {
                    "action": "runtime_env_select",
                    "detail": "Select a Python environment with a consistent Qiskit install before patching source.",
                },
                {
                    "action": "dependency_runtime_fix",
                    "detail": "Repair dependency/runtime selection if no compatible interpreter is already available.",
                },
                {
                    "action": "retry_reproduce",
                    "detail": "Re-run after the runtime no longer mixes Qiskit >=1.0 with old qiskit-terra packages.",
                },
            ]
        elif "No module named 'mqt'" in evidence_text or 'No module named "mqt"' in evidence_text:
            plan["actions"] = [
                {
                    "action": "source_fix",
                    "detail": "Treat mqt.predictor as optional in simulation-only reproduction and install a lightweight feature fallback in the isolated workspace.",
                },
                {
                    "action": "build_original_runner",
                    "detail": "Regenerate the original-code runner after the optional predictor fallback is patched.",
                },
                {
                    "action": "retry_reproduce",
                    "detail": "Re-run without installing mqt into the legacy Qiskit runtime.",
                },
            ]
        elif "No module named 'qos." in evidence_text or "No module named 'qvm." in evidence_text:
            plan["actions"] = [
                {
                    "action": "source_path_fix",
                    "detail": "Repair in-repo package paths or compatibility modules needed by the original QOS pipeline.",
                },
                {
                    "action": "runtime_trace_fix",
                    "detail": "Patch narrow import-surface mismatches exposed while importing original QOS modules.",
                },
                {
                    "action": "retry_reproduce",
                    "detail": "Re-run after the original in-repo imports resolve.",
                },
            ]
        elif "cannot import name" in evidence_text and ("from 'qos." in evidence_text or "from 'qvm." in evidence_text):
            plan["actions"] = [
                {
                    "action": "runtime_trace_fix",
                    "detail": "Patch narrow in-repo symbol export/import mismatches in the original QOS pipeline.",
                },
                {
                    "action": "source_path_fix",
                    "detail": "Repair package layout if symbol fixes require compatibility exports.",
                },
                {
                    "action": "retry_reproduce",
                    "detail": "Re-run after the original in-repo imports resolve.",
                },
            ]
        elif "No module named" in evidence_text:
            plan["actions"] = [
                {
                    "action": "dependency_runtime_fix",
                    "detail": "Install or select runtime dependencies required by the original pipeline.",
                },
                {
                    "action": "retry_reproduce",
                    "detail": "Re-run after missing third-party modules are available.",
                },
            ]
        else:
            plan["actions"] = [
                {
                    "action": "simulation_backend_fix",
                    "detail": "Replace remote QPU/IBM backend surfaces in the isolated workspace so original modules can execute offline.",
                },
                {
                    "action": "source_fix",
                    "detail": "Patch the isolated runner/source to call original metric-producing functions instead of proxy formulas.",
                },
                {
                    "action": "build_original_runner",
                    "detail": "Regenerate the original-code runner after simulation/source fixes.",
                },
                {
                    "action": "retry_reproduce",
                    "detail": "Re-run and require metrics to come from original repository pipeline evidence.",
                },
            ]

    return plan


def evaluate_status(
    recipe: dict[str, Any],
    preflight_payload: dict[str, Any],
    run_payload: dict[str, Any] | None,
    metrics_payload: dict[str, Any] | None,
    verification_payload: dict[str, Any] | None,
    parse_error: str | None,
    original_code_path_evidence: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    criteria = recipe.get("success_criteria", {})

    if not preflight_payload["ok"]:
        return "preflight_failed", {
            "success": False,
            "reason": "preflight checks failed",
        }

    if run_payload is None:
        return "preflight_only", {
            "success": True,
            "reason": "preflight completed; execution skipped",
        }

    if criteria.get("require_exit_code_zero", True) and run_payload["returncode"] != 0:
        return "run_failed", {
            "success": False,
            "reason": "entry command returned non-zero exit code",
            "returncode": run_payload["returncode"],
        }

    if parse_error is not None:
        return "parse_failed", {
            "success": False,
            "reason": parse_error,
        }

    if criteria.get("require_metrics", True) and metrics_payload is None:
        return "parse_failed", {
            "success": False,
            "reason": "metrics were required but not produced",
        }

    if criteria.get("require_metric_success", False) and not bool(metrics_payload.get("success")):
        return "metric_failed", {
            "success": False,
            "reason": "metrics indicate unsuccessful reproduction",
        }

    if criteria.get("require_verification_success", False) and verification_payload is None:
        return "parse_failed", {
            "success": False,
            "reason": "verification was required but no verdict was produced",
        }

    if (
        criteria.get("require_verification_success", False)
        and verification_payload is not None
        and not bool(verification_payload.get("success"))
    ):
        return "metric_failed", {
            "success": False,
            "reason": "figure claim verification failed",
        }

    if (
        original_code_path_evidence is not None
        and original_code_path_evidence.get("required")
        and not original_code_path_evidence.get("ok")
    ):
        return "run_failed", {
            "success": False,
            "reason": "original code path requirement was not satisfied",
            "original_code_path_evidence": original_code_path_evidence,
        }

    return "success", {
        "success": True,
        "reason": "reproduce workflow completed successfully",
        "original_code_path_evidence": original_code_path_evidence,
    }


def build_diagnosis(
    recipe: dict[str, Any],
    workspace_root: Path,
    status: str,
    preflight_payload: dict[str, Any],
    run_payload: dict[str, Any] | None,
    metrics_payload: dict[str, Any] | None,
    verification_payload: dict[str, Any] | None,
    parse_error: str | None,
    run_error_signals: dict[str, Any] | None,
    status_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    classification = classify_failure(
        recipe,
        workspace_root,
        status,
        preflight_payload,
        run_payload,
        metrics_payload,
        run_error_signals,
        parse_error,
        status_payload,
    )
    recovery_plan = build_recovery_plan(recipe, classification)
    diagnosis: dict[str, Any] = {
        "status": status,
        "summary": "",
        "hints": [],
        "classification": classification,
        "recovery_plan": recovery_plan,
    }
    if status == "success":
        diagnosis["summary"] = "Recipe completed and emitted valid metrics."
        return diagnosis
    if status == "preflight_only":
        diagnosis["summary"] = "Preflight completed; no run was attempted."
        return diagnosis
    if status == "preflight_failed":
        diagnosis["summary"] = "Environment is not ready for this recipe."
        for item in preflight_payload["checks"]["python_imports"]:
            if not item["ok"]:
                qualifier = "optional" if item.get("optional") else "required"
                diagnosis["hints"].append(
                    f"Fix {qualifier} python import failure: {item['name']}"
                )
        for item in preflight_payload["checks"]["paths_exist"]:
            if not item["ok"]:
                diagnosis["hints"].append(f"Missing required path: {item['path']}")
        for item in preflight_payload["checks"]["commands_exist"]:
            if not item["ok"]:
                diagnosis["hints"].append(f"Missing required command: {item['name']}")
        return diagnosis
    if status == "run_failed":
        if classification["category"] == "original_code_path_not_exercised":
            diagnosis["summary"] = "Run produced artifacts but did not exercise required original repository code paths."
            diagnosis["hints"].append(
                "Run repo_path_probe to identify original repo entry points and route the reproduction away from proxy-only harnesses."
            )
            return diagnosis
        diagnosis["summary"] = "Entry command failed."
        if run_payload is not None:
            diagnosis["hints"].append(
                f"Inspect stderr.log for the failing command (exit code {run_payload['returncode']})."
            )
        if run_error_signals is not None:
            for module_name in run_error_signals.get("missing_modules", []):
                diagnosis["hints"].append(f"Traceback missing module: {module_name}")
        if classification["category"] == "source_compat_missing_module_file":
            diagnosis["hints"].append(
                "The imported module path appears to be missing from the source tree."
            )
        if classification["category"] == "source_compat_bad_import_statement":
            diagnosis["hints"].append(
                "The target module file exists; inspect the import statement or package layout instead of creating new files."
            )
        if classification["category"] == "source_compat_missing_symbol_import":
            diagnosis["hints"].append(
                "An import statement refers to a symbol that the target module does not export; prefer a minimal import fix over broader edits."
            )
        if classification["category"] == "source_compat_missing_symbol_reference":
            diagnosis["hints"].append(
                "A symbol is referenced in repo code without a valid definition in scope; prefer adding the smallest missing import or alias."
            )
        if classification["category"] == "runtime_import_surface_mismatch":
            diagnosis["hints"].append(
                "A third-party submodule path is missing; prefer runtime/import-surface compatibility repair before pip installs."
            )
        return diagnosis
    if status == "parse_failed":
        diagnosis["summary"] = "Run finished but metrics could not be parsed."
        if parse_error:
            diagnosis["hints"].append(parse_error)
        return diagnosis
    if status == "metric_failed":
        diagnosis["summary"] = "Metrics were produced but do not satisfy success criteria."
        if classification["category"] == "original_pipeline_not_executable":
            diagnosis["summary"] = "Original repository pipeline candidates were found, but no safe raw-metric execution path was built yet."
            diagnosis["hints"].append(
                "Use simulation_backend_fix/source_fix to replace QPU-only surfaces and wire the runner to original metric-producing functions."
            )
            return diagnosis
        if metrics_payload is not None and not metrics_payload.get("success", True):
            diagnosis["hints"].append("Inspect metrics.json for module-level failures.")
        if verification_payload is not None and not verification_payload.get("success", True):
            diagnosis["hints"].append(
                "Inspect figure_verdict.json for claim-level check failures."
            )
        return diagnosis

    diagnosis["summary"] = "Unknown reproduce failure state."
    return diagnosis


def run_attempt(
    recipe: dict[str, Any],
    repo_root: Path,
    workspace_root: Path,
    python_exe: str,
    attempt_dir: Path,
    preflight_only: bool,
    allow_run_on_preflight_failure: bool,
) -> dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    metrics_path = attempt_dir / "metrics.json"
    preflight_path = attempt_dir / "preflight.json"
    run_json_path = attempt_dir / "run.json"
    status_path = attempt_dir / "status.json"
    diagnosis_path = attempt_dir / "diagnosis.json"
    manifest_path = attempt_dir / "manifest.json"
    verification_path = attempt_dir / "figure_verdict.json"

    context = {
        "repo_root": str(repo_root),
        "workspace_root": str(workspace_root),
        "run_dir": str(attempt_dir),
        "metrics_path": str(metrics_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "python_executable": str(python_exe),
    }

    preflight_payload = preflight(recipe, workspace_root, python_exe)
    write_json(preflight_path, preflight_payload)

    run_payload: dict[str, Any] | None = None
    metrics_payload: dict[str, Any] | None = None
    verification_payload: dict[str, Any] | None = None
    original_code_path_evidence: dict[str, Any] | None = None
    parse_error: str | None = None
    run_error_signals: dict[str, Any] | None = None

    should_run = not preflight_only
    if not preflight_payload["ok"] and not allow_run_on_preflight_failure:
        should_run = False

    if should_run:
        entry = recipe["entry"]
        timeout_seconds = int(entry.get("timeout_seconds", 300))
        command = [expand_template(part, context) for part in entry["command"]]
        env = build_workspace_env(os.environ.copy(), workspace_root)
        env.update(
            {
                "REPRO_RECIPE_NAME": recipe["name"],
                "REPRO_RUN_DIR": str(attempt_dir),
                "REPRO_WORKSPACE_ROOT": str(workspace_root),
                "REPRO_METRICS_PATH": str(metrics_path),
                "REPRO_SIMULATION_ONLY": "1" if bool(recipe.get("recovery", {}).get("simulation_only")) else "0",
                "QOS_SIMULATION_ONLY": "1" if bool(recipe.get("recovery", {}).get("simulation_only")) else "0",
            }
        )
        for key, value in entry.get("env", {}).items():
            env[key] = expand_template(str(value), context)

        started_at = iso_now()
        started = time.time()
        proxy_violation = proxy_harness_entry_violation(recipe, workspace_root, command)
        if proxy_violation is not None:
            duration = time.time() - started
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(
                "[reproduce] proxy harness entry refused\n"
                + json.dumps(proxy_violation, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            returncode = 2
            timed_out = False
        else:
            try:
                proc = subprocess.run(
                    command,
                    cwd=workspace_root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                duration = time.time() - started
                stdout_path.write_text(proc.stdout, encoding="utf-8")
                stderr_path.write_text(proc.stderr, encoding="utf-8")
                returncode = proc.returncode
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                duration = time.time() - started
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                stderr = (
                    str(stderr)
                    + f"\n[reproduce] command timed out after {timeout_seconds} seconds\n"
                )
                stdout_path.write_text(str(stdout), encoding="utf-8")
                stderr_path.write_text(stderr, encoding="utf-8")
                returncode = 124
                timed_out = True
        run_payload = {
            "command": command,
            "cwd": str(workspace_root),
            "started_at": started_at,
            "duration_seconds": duration,
            "returncode": returncode,
            "timeout_seconds": timeout_seconds,
            "timed_out": timed_out,
        }
        if proxy_violation is not None:
            run_payload["proxy_harness_forbidden"] = proxy_violation
        write_json(run_json_path, run_payload)
        run_error_signals = collect_run_error_signals(attempt_dir)

        parsing = recipe.get("parsing", {})
        if parsing.get("kind") == "json_file":
            parse_path = Path(expand_template(parsing.get("path", "{metrics_path}"), context))
            try:
                metrics_payload = load_json(parse_path)
                required_keys = parsing.get("required_metric_keys", [])
                missing_keys = [key for key in required_keys if key not in metrics_payload]
                if missing_keys:
                    parse_error = f"metrics file missing required keys: {missing_keys}"
            except FileNotFoundError:
                parse_error = f"metrics file not found: {parse_path}"
            except json.JSONDecodeError as exc:
                parse_error = f"metrics file is not valid JSON: {exc}"
        else:
            parse_error = f"unsupported parsing kind: {parsing.get('kind')}"

        verification_cfg = recipe.get("verification", {})
        if (
            parse_error is None
            and metrics_payload is not None
            and bool(verification_cfg.get("enabled", False))
        ):
            verification_kind = verification_cfg.get("kind", "figure_contract")
            if verification_kind != "figure_contract":
                parse_error = f"unsupported verification kind: {verification_kind}"
            else:
                raw_contract_path = verification_cfg.get("contract_path")
                if not raw_contract_path:
                    parse_error = "verification.contract_path is required when verification is enabled"
                else:
                    contract_path = Path(expand_template(str(raw_contract_path), context))
                    if not contract_path.is_absolute():
                        contract_path = (repo_root / contract_path).resolve()
                    if not contract_path.exists():
                        parse_error = f"verification contract not found: {contract_path}"
                    else:
                        try:
                            contract_payload = load_json(contract_path)
                            verification_payload = evaluate_figure_contract(
                                contract_payload,
                                metrics_payload,
                            )
                            verification_payload["contract_path"] = str(contract_path)
                            out_template = verification_cfg.get(
                                "output_path",
                                "{run_dir}/figure_verdict.json",
                            )
                            verification_path = Path(expand_template(out_template, context))
                            if not verification_path.is_absolute():
                                verification_path = (repo_root / verification_path).resolve()
                            write_json(verification_path, verification_payload)
                        except (ValueError, TypeError, json.JSONDecodeError) as exc:
                            parse_error = f"verification failed: {exc}"

    original_code_path_evidence = collect_original_code_path_evidence(
        recipe,
        workspace_root,
        run_payload,
        metrics_payload,
        verification_payload,
    )
    status, status_payload = evaluate_status(
        recipe,
        preflight_payload,
        run_payload,
        metrics_payload,
        verification_payload,
        parse_error,
        original_code_path_evidence,
    )
    status_payload.update(
        {
            "status": status,
            "recipe_name": recipe["name"],
            "run_dir": str(attempt_dir),
            "completed_at": iso_now(),
        }
    )
    write_json(status_path, status_payload)

    diagnosis_payload = build_diagnosis(
        recipe,
        workspace_root,
        status,
        preflight_payload,
        run_payload,
        metrics_payload,
        verification_payload,
        parse_error,
        run_error_signals,
        status_payload,
    )
    write_json(diagnosis_path, diagnosis_payload)

    manifest = {
        "recipe_name": recipe["name"],
        "attempt_dir": str(attempt_dir),
        "workspace_root": str(workspace_root),
        "status": status,
        "artifacts": {
            "preflight": str(preflight_path),
            "run": str(run_json_path) if run_payload is not None else None,
            "stdout": str(stdout_path) if stdout_path.exists() else None,
            "stderr": str(stderr_path) if stderr_path.exists() else None,
            "metrics": str(metrics_path) if metrics_path.exists() else None,
            "verification": str(verification_path) if verification_path.exists() else None,
            "status": str(status_path),
            "diagnosis": str(diagnosis_path),
        },
    }
    write_json(manifest_path, manifest)

    return {
        "status": status,
        "status_payload": status_payload,
        "diagnosis_payload": diagnosis_payload,
        "manifest": manifest,
        "preflight_payload": preflight_payload,
        "run_payload": run_payload,
        "metrics_payload": metrics_payload,
        "verification_payload": verification_payload,
        "original_code_path_evidence": original_code_path_evidence,
        "parse_error": parse_error,
        "run_error_signals": run_error_signals,
        "attempt_dir": attempt_dir,
        "artifacts": manifest["artifacts"],
    }


def build_recovery_context(
    repo_root: Path,
    workspace_root: Path,
    python_exe: str,
    run_dir: Path,
    attempt: dict[str, Any],
    action_name: str,
    result_path: Path,
) -> dict[str, str]:
    classification = attempt["diagnosis_payload"].get("classification", {})
    context = {
        "repo_root": str(repo_root),
        "workspace_root": str(workspace_root),
        "run_dir": str(run_dir),
        "attempt_dir": str(attempt["attempt_dir"]),
        "python_executable": str(python_exe),
        "recovery_action": action_name,
        "recovery_result_path": str(result_path),
    }
    for evidence in classification.get("evidence", []):
        if isinstance(evidence, dict):
            for key, value in evidence.items():
                context[key] = str(value)
    return context


def execute_recovery_handler(
    recipe: dict[str, Any],
    repo_root: Path,
    workspace_root: Path,
    python_exe: str,
    run_dir: Path,
    attempt: dict[str, Any],
    action_name: str,
    ordinal: int,
) -> dict[str, Any] | None:
    recovery_cfg = recipe.get("recovery", {})
    handlers = recovery_cfg.get("handlers", {})
    handler_cfg = handlers.get(action_name)
    if handler_cfg is None:
        return None

    recovery_dir = run_dir / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = recovery_dir / f"{ordinal:02d}_{action_name}.stdout.log"
    stderr_path = recovery_dir / f"{ordinal:02d}_{action_name}.stderr.log"
    result_path = recovery_dir / f"{ordinal:02d}_{action_name}.result.json"
    record_path = recovery_dir / f"{ordinal:02d}_{action_name}.json"

    context = build_recovery_context(
        repo_root,
        workspace_root,
        python_exe,
        run_dir,
        attempt,
        action_name,
        result_path,
    )
    command = [expand_template(part, context) for part in handler_cfg["command"]]
    env = build_workspace_env(os.environ.copy(), workspace_root)
    env.update(
        {
            "REPRO_RECOVERY_ACTION": action_name,
            "REPRO_WORKSPACE_ROOT": str(workspace_root),
            "REPRO_RUN_DIR": str(run_dir),
            "REPRO_ATTEMPT_DIR": str(attempt["attempt_dir"]),
        }
    )
    for key, value in handler_cfg.get("env", {}).items():
        env[key] = expand_template(str(value), context)

    timeout_seconds = int(handler_cfg.get("timeout_seconds", 120))
    started_at = iso_now()
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=workspace_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    duration = time.time() - started

    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    result_payload: dict[str, Any] | None = None
    if result_path.exists():
        try:
            result_payload = load_json(result_path)
        except (FileNotFoundError, json.JSONDecodeError):
            result_payload = None

    record = {
        "action": action_name,
        "command": command,
        "started_at": started_at,
        "duration_seconds": duration,
        "returncode": proc.returncode,
        "timeout_seconds": timeout_seconds,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "result_path": str(result_path),
        "result": result_payload,
        "applied": proc.returncode == 0 and bool(result_payload is None or result_payload.get("success", True)),
    }
    write_json(record_path, record)
    record["record_path"] = str(record_path)
    return record


def retry_requested(recovery_plan: dict[str, Any]) -> bool:
    return any(
        action.get("action") in {"retry_reproduce", "retry_preflight"}
        for action in recovery_plan.get("actions", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument(
        "--output-root",
        default="temp/agent_framework/reproduce/runs",
        help="Directory where reproduce artifacts are written.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight checks only; do not execute the recipe command.",
    )
    parser.add_argument(
        "--allow-run-on-preflight-failure",
        action="store_true",
        help="Attempt the run even when preflight fails.",
    )
    parser.add_argument(
        "--auto-recover",
        action="store_true",
        help="Apply recipe-defined safe recovery handlers and retry once when possible.",
    )
    parser.add_argument(
        "--python-executable",
        help="Override the Python interpreter used for preflight import checks and recipe execution.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    recipe_path = Path(args.recipe).resolve()
    recipe = load_json(recipe_path)

    workspace_root = resolve_workspace_root(repo_root, recipe.get("workspace_root", "."))
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()

    run_id = f"{recipe['name']}_{utc_stamp()}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    recipe_snapshot_path = run_dir / "recipe.snapshot.json"
    status_path = run_dir / "status.json"
    diagnosis_path = run_dir / "diagnosis.json"
    manifest_path = run_dir / "manifest.json"

    write_json(recipe_snapshot_path, recipe)

    python_exe = (
        args.python_executable
        or recipe.get("runtime", {}).get("python_executable")
        or sys.executable
    )
    attempts_dir = run_dir / "attempts"
    attempts: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    current_attempt = run_attempt(
        recipe=recipe,
        repo_root=repo_root,
        workspace_root=workspace_root,
        python_exe=python_exe,
        attempt_dir=attempts_dir / "attempt_1",
        preflight_only=args.preflight_only,
        allow_run_on_preflight_failure=args.allow_run_on_preflight_failure,
    )
    attempts.append(current_attempt["manifest"])

    auto_recover = bool(args.auto_recover or recipe.get("recovery", {}).get("auto_apply_safe_fixes"))
    max_auto_recoveries = int(recipe.get("recovery", {}).get("max_auto_recoveries", 1))
    recovery_ordinal = 1
    next_attempt_index = 2
    while (
        auto_recover
        and current_attempt["status"] not in {"success", "preflight_only"}
        and recovery_ordinal <= max_auto_recoveries
    ):
        recovery_plan = current_attempt["diagnosis_payload"].get("recovery_plan", {})
        recovery_record = None
        for action in recovery_plan.get("actions", []):
            recovery_record = execute_recovery_handler(
                recipe=recipe,
                repo_root=repo_root,
                workspace_root=workspace_root,
                python_exe=python_exe,
                run_dir=run_dir,
                attempt=current_attempt,
                action_name=action["action"],
                ordinal=recovery_ordinal,
            )
            if recovery_record is None:
                continue
            recoveries.append(recovery_record)
            break
        if recovery_record is None:
            break
        recovery_ordinal += 1
        if not recovery_record.get("applied") or not retry_requested(recovery_plan):
            break

        current_attempt = run_attempt(
            recipe=recipe,
            repo_root=repo_root,
            workspace_root=workspace_root,
            python_exe=python_exe,
            attempt_dir=attempts_dir / f"attempt_{next_attempt_index}",
            preflight_only=args.preflight_only,
            allow_run_on_preflight_failure=False,
        )
        attempts.append(current_attempt["manifest"])
        next_attempt_index += 1

    final_status_payload = dict(current_attempt["status_payload"])
    final_status_payload.update(
        {
            "final_attempt_dir": str(current_attempt["attempt_dir"]),
            "attempt_count": len(attempts),
            "recovery_count": len(recoveries),
        }
    )
    write_json(status_path, final_status_payload)
    final_diagnosis_payload = dict(current_attempt["diagnosis_payload"])
    final_diagnosis_payload["recovery_records"] = recoveries
    write_json(diagnosis_path, final_diagnosis_payload)

    manifest = {
        "recipe_name": recipe["name"],
        "run_id": run_id,
        "run_dir": str(run_dir),
        "workspace_root": str(workspace_root),
        "status": current_attempt["status"],
        "artifacts": {
            "recipe_snapshot": str(recipe_snapshot_path),
            "status": str(status_path),
            "diagnosis": str(diagnosis_path),
        },
        "attempts": attempts,
        "recovery": recoveries,
        "final_attempt": current_attempt["manifest"],
    }
    write_json(manifest_path, manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if current_attempt["status"] in {"success", "preflight_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
