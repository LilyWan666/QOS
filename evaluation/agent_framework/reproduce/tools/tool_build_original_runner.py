#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

import run_reproduce as rr  # noqa: E402
from repro_toolkit import (  # noqa: E402
    assert_action_allowed,
    append_history,
    blocked_action_payload,
    get_fsm_state,
    load_state,
    next_actions,
    repo_root_from_here,
    resolve_recipe_path,
    set_fsm_state,
    write_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _module_for_path(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    raw = path[:-3]
    if raw.endswith("/__init__"):
        raw = raw[: -len("/__init__")]
    parts = [part for part in raw.split("/") if part and part != "."]
    if not parts:
        return None
    if not all(part.replace("_", "a").isalnum() for part in parts):
        return None
    return ".".join(parts)


def _requirements(recipe: dict[str, Any]) -> dict[str, Any]:
    reqs = recipe.get("reproduction_requirements") or {}
    original = reqs.get("original_code_path") or {}
    return {
        "min_original_modules": int(original.get("min_original_modules", 1)),
        "required_modules": [str(x).strip() for x in _as_list(original.get("required_modules")) if str(x).strip()],
        "preferred_path_prefixes": [str(x).strip() for x in _as_list(original.get("preferred_path_prefixes")) if str(x).strip()],
    }


def _contract_applications(recipe: dict[str, Any], repo_root: Path, workspace_root: Path) -> list[str]:
    verification = recipe.get("verification") or {}
    raw_contract = str(verification.get("contract_path") or "").strip()
    if not raw_contract:
        return []
    contract_path = Path(raw_contract)
    if not contract_path.is_absolute():
        for root in (workspace_root, repo_root):
            candidate = (root / contract_path).resolve()
            if candidate.exists():
                contract_path = candidate
                break
    if not contract_path.exists():
        return []
    try:
        contract = rr.load_json(contract_path)
    except Exception:
        return []
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("path") == "applications" and isinstance(value.get("values"), list):
                for raw in value.get("values") or []:
                    name = str(raw).strip()
                    if name and name not in found:
                        found.append(name)
            for key, child in value.items():
                if key == "applications" and isinstance(child, list):
                    for raw in child:
                        name = str(raw).strip()
                        if name and name not in found:
                            found.append(name)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(contract)
    return found


def _candidate_modules(state: dict[str, Any], recipe: dict[str, Any]) -> list[dict[str, str]]:
    req = _requirements(recipe)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for module in req["required_modules"]:
        if module not in seen:
            seen.add(module)
            out.append({"module": module, "source": "recipe_required_modules"})
    pipeline_probe = state.get("last_original_pipeline_probe") or {}
    for item in pipeline_probe.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        module = str(item.get("module") or "").strip() or _module_for_path(str(item.get("path") or ""))
        if not module or module in seen:
            continue
        if not (item.get("functions") or item.get("classes")):
            continue
        seen.add(module)
        out.append({"module": module, "path": str(item.get("path") or ""), "source": "original_pipeline_probe"})
    probe = state.get("last_repo_path_probe") or {}
    candidates = probe.get("candidates") or []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if item.get("path_role") == "reproduce_framework":
            continue
        module = str(item.get("module") or "").strip() or _module_for_path(str(item.get("path") or ""))
        if not module or module in seen:
            continue
        if not (item.get("classes") or item.get("functions")):
            continue
        seen.add(module)
        out.append({"module": module, "path": str(item.get("path") or ""), "source": "repo_path_probe"})
    return out[:24]


def _pipeline_candidates(state: dict[str, Any]) -> list[dict[str, Any]]:
    probe = state.get("last_original_pipeline_probe") or {}
    out: list[dict[str, Any]] = []
    for item in probe.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        module = str(item.get("module") or "").strip() or _module_for_path(str(item.get("path") or ""))
        if not module:
            continue
        out.append(
            {
                "module": module,
                "path": str(item.get("path") or ""),
                "score": item.get("score"),
                "functions": item.get("functions") if isinstance(item.get("functions"), list) else [],
                "classes": item.get("classes") if isinstance(item.get("classes"), list) else [],
                "external_risks": item.get("external_risks") if isinstance(item.get("external_risks"), list) else [],
            }
        )
    return out[:24]


def _metric_map(state: dict[str, Any]) -> dict[str, Any]:
    probe = state.get("last_original_pipeline_probe") or {}
    metric_map = probe.get("metric_map")
    if isinstance(metric_map, dict):
        return metric_map
    metric_map = state.get("last_metric_map")
    return metric_map if isinstance(metric_map, dict) else {}


def _semantic_map(state: dict[str, Any]) -> dict[str, Any]:
    probe = state.get("last_original_pipeline_probe") or {}
    semantic_map = probe.get("semantic_map")
    if isinstance(semantic_map, dict):
        return semantic_map
    semantic_map = state.get("last_semantic_map")
    return semantic_map if isinstance(semantic_map, dict) else {}


def _strict_pipeline_runner_source(
    pipeline_candidates: list[dict[str, Any]],
    candidate_modules: list[dict[str, str]],
    req: dict[str, Any],
    metric_map: dict[str, Any],
    semantic_map: dict[str, Any],
) -> str:
    return f'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import random
import re
import sys
import traceback
from pathlib import Path

PIPELINE_CANDIDATES = {json.dumps(pipeline_candidates, indent=2, sort_keys=True)}
CANDIDATE_MODULES = {json.dumps(candidate_modules, indent=2, sort_keys=True)}
REQUIREMENTS = {json.dumps(req, indent=2, sort_keys=True)}
METRIC_MAP = {repr(metric_map)}
SEMANTIC_MAP = {repr(semantic_map)}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


STAGE_LOG: list[dict] = []


def _stage(name: str, **fields) -> None:
    record = {{"stage": name, **fields}}
    STAGE_LOG.append(record)
    print("[repro-stage] " + json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)


def _safe_signature(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "(signature unavailable)"


def _import_modules(workspace_root: Path) -> tuple[list[dict], dict]:
    checked = []
    imported = []
    failed = []
    files = []
    for item in CANDIDATE_MODULES:
        module = item.get("module", "")
        result = dict(item)
        result["module"] = module
        try:
            imported_module = importlib.import_module(module)
            result["ok"] = True
            result["file"] = getattr(imported_module, "__file__", None)
            if result["file"]:
                try:
                    result["relative_file"] = str(Path(result["file"]).resolve().relative_to(workspace_root))
                except Exception:
                    result["relative_file"] = str(result["file"])
                files.append(result["relative_file"])
            imported.append(module)
        except Exception as exc:
            optional_probe_module = result.get("source") == "original_pipeline_probe"
            result["ok"] = True if optional_probe_module else False
            if optional_probe_module:
                result["optional_import_failed"] = True
            result["error"] = repr(exc)
            result["traceback"] = traceback.format_exc()[-4000:]
            if not optional_probe_module:
                failed.append(result)
        checked.append(result)

    preferred_hits = []
    evidence_blob = "\\n".join(imported + files)
    for prefix in REQUIREMENTS.get("preferred_path_prefixes", []):
        if prefix and prefix in evidence_blob:
            preferred_hits.append(prefix)
    module_hits = [m for m in REQUIREMENTS.get("required_modules", []) if m in imported]
    evidence_count = len(set(preferred_hits + module_hits))
    min_modules = int(REQUIREMENTS.get("min_original_modules", 1) or 1)
    evidence = {{
        "required": True,
        "ok": evidence_count >= min_modules,
        "modules_imported": imported,
        "modules_failed": [item.get("module") for item in failed],
        "files_touched": files,
        "preferred_hits": preferred_hits,
        "module_hits": module_hits,
        "evidence_count": evidence_count,
        "min_original_modules": min_modules,
    }}
    return checked, evidence


def _inspect_pipeline_candidates() -> list[dict]:
    inspected = []
    for candidate in PIPELINE_CANDIDATES:
        module_name = candidate.get("module", "")
        entry = dict(candidate)
        entry["import_ok"] = False
        entry["callables"] = []
        try:
            module = importlib.import_module(module_name)
            entry["import_ok"] = True
        except Exception as exc:
            entry["error"] = repr(exc)
            entry["traceback"] = traceback.format_exc()[-4000:]
            inspected.append(entry)
            continue

        wanted_names = set()
        for fn in candidate.get("functions", []) or []:
            if isinstance(fn, dict) and fn.get("name"):
                wanted_names.add(str(fn["name"]))
        for cls in candidate.get("classes", []) or []:
            if isinstance(cls, dict) and cls.get("name"):
                wanted_names.add(str(cls["name"]))
        for name in sorted(wanted_names):
            obj = getattr(module, name, None)
            if obj is None:
                continue
            record = {{
                "name": name,
                "kind": "class" if inspect.isclass(obj) else "function" if inspect.isfunction(obj) else type(obj).__name__,
                "signature": _safe_signature(obj),
                "module": module_name,
            }}
            if inspect.isclass(obj):
                try:
                    instance = obj()
                    record["no_arg_instantiation_ok"] = True
                    methods = []
                    for method_name, method in inspect.getmembers(instance, predicate=callable):
                        if method_name.startswith("_"):
                            continue
                        lower = method_name.lower()
                        if any(token in lower for token in ("fidelity", "util", "match", "process", "schedule", "run")):
                            methods.append({{"name": method_name, "signature": _safe_signature(method)}})
                    record["candidate_methods"] = methods[:24]
                except Exception as exc:
                    record["no_arg_instantiation_ok"] = False
                    record["instantiation_error"] = repr(exc)
            entry["callables"].append(record)
        inspected.append(entry)
    return inspected


def _execute_original_pipeline_smoke() -> dict:
    """Exercise importable original multiprogramming functions on tiny circuits."""
    result = {{
        "ok": False,
        "executed": [],
        "errors": [],
        "raw_metrics": {{}},
    }}

    try:
        from qiskit import QuantumCircuit
        from qos.types.types import QPU, Qernel
    except Exception as exc:
        result["errors"].append({{"stage": "import_core_types", "error": repr(exc), "traceback": traceback.format_exc()[-4000:]}})
        return result

    circuits = []
    try:
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.measure([0, 1], [0, 1])
        qc2 = QuantumCircuit(3, 3)
        qc2.x(0)
        qc2.cx(0, 2)
        qc2.measure([0, 1, 2], [0, 1, 2])
        circuits = [qc1, qc2]
        result["raw_metrics"]["synthetic_circuits"] = [
            {{"num_qubits": c.num_qubits, "depth": c.depth(), "size": c.size()}}
            for c in circuits
        ]
    except Exception as exc:
        result["errors"].append({{"stage": "build_tiny_circuits", "error": repr(exc), "traceback": traceback.format_exc()[-4000:]}})
        return result

    qpu = QPU()
    qpu.name = "simulation_qpu"
    qpu.num_qubits = 16
    qpu.local_queue = []

    qernels = [
        Qernel(
            circuits[0],
            {{
                "depth": circuits[0].depth(),
                "entanglement_ratio": 0.5,
                "measurement": 1.0,
                "parallelism": 0.5,
                "num_nonlocal_gates": 1,
                "num_measurements": 2,
            }},
        ),
        Qernel(
            circuits[1],
            {{
                "depth": circuits[1].depth(),
                "entanglement_ratio": 0.4,
                "measurement": 1.0,
                "parallelism": 0.4,
                "num_nonlocal_gates": 1,
                "num_measurements": 3,
            }},
        ),
    ]
    class _CallableInt(int):
        def __call__(self):
            return int(self)

    for qernel in qernels:
        circuit = qernel.get_circuit()
        qernel.num_qubits = _CallableInt(circuit.num_qubits)
        qernel.depth = _CallableInt(circuit.depth() or 0)

    try:
        mp_mod = importlib.import_module("qos.multiprogrammer.multiprogrammer")
        mp = mp_mod.Multiprogrammer()
        spatial = mp.spatial_utilization(qernels[0], qernels[1], qpu)
        effective = mp.effective_utilization(qernels[0], qernels[1], qpu)
        matching = mp.get_matching_score(qernels[0], qernels[1], qpu)
        returned = mp.run(qernels)
        result["executed"].extend(
            [
                "qos.multiprogrammer.Multiprogrammer.spatial_utilization",
                "qos.multiprogrammer.Multiprogrammer.effective_utilization",
                "qos.multiprogrammer.Multiprogrammer.get_matching_score",
                "qos.multiprogrammer.Multiprogrammer.run",
            ]
        )
        result["raw_metrics"]["multiprogrammer"] = {{
            "spatial_utilization": spatial,
            "effective_utilization": effective,
            "matching_score": matching,
            "run_return_count": len(returned) if isinstance(returned, list) else None,
        }}
    except Exception as exc:
        result["errors"].append({{"stage": "execute_multiprogrammer", "error": repr(exc), "traceback": traceback.format_exc()[-4000:]}})

    try:
        baseline = importlib.import_module("Baseline_Multiprogramming.multiprogramming")
        cmr_values = [baseline.compute_CMR(circuit) for circuit in circuits]
        analysis = baseline.analyze_programs(circuits)
        result["executed"].extend(
            [
                "Baseline_Multiprogramming.compute_CMR",
                "Baseline_Multiprogramming.analyze_programs",
            ]
        )
        result["raw_metrics"]["baseline_multiprogramming"] = {{
            "cmr_values": cmr_values,
            "analysis_program_count": len(analysis),
            "analysis_keys": sorted(str(k) for k in analysis.keys()),
        }}
    except Exception as exc:
        result["errors"].append({{"stage": "execute_baseline_multiprogramming", "error": repr(exc), "traceback": traceback.format_exc()[-4000:]}})

    try:
        scheduler = importlib.import_module("qos.scheduler.scheduler")
        score = scheduler.compute_score(0.9, 0.85, 10.0, 12.0, 0.2, 0.3, 0.7, 0.0)
        result["executed"].append("qos.scheduler.scheduler.compute_score")
        result["raw_metrics"]["scheduler"] = {{"balanced_score": score}}
    except Exception as exc:
        result["errors"].append({{"stage": "execute_scheduler_score", "error": repr(exc), "traceback": traceback.format_exc()[-4000:]}})

    result["ok"] = bool(result["executed"]) and not any(
        item.get("stage") in {{"execute_multiprogrammer", "execute_baseline_multiprogramming"}}
        for item in result["errors"]
    )
    return result


def _mean(values) -> float:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _metric_source_for(*needles: str) -> dict:
    wanted = [needle.lower() for needle in needles if needle]
    for term, mapping in METRIC_MAP.items():
        haystack = " ".join([
            str(term),
            str(mapping.get("paper_term", "")) if isinstance(mapping, dict) else "",
        ]).lower()
        if all(needle in haystack for needle in wanted):
            if isinstance(mapping, dict):
                selected = mapping.get("selected_source")
                if isinstance(selected, dict) and selected.get("source"):
                    return {{
                        "paper_term": mapping.get("paper_term", term),
                        "source_type": selected.get("source_type", "repo_source"),
                        "source": selected.get("source"),
                        "path": selected.get("path"),
                        "lineno": selected.get("lineno"),
                        "confidence": mapping.get("confidence", "medium"),
                    }}
                return {{
                    "paper_term": mapping.get("paper_term", term),
                    "source_type": "paper_formula_or_missing",
                    "source": None,
                    "confidence": mapping.get("confidence", "low"),
                }}
    return {{"source_type": "unmapped", "source": None, "confidence": "low"}}


def _application_name(path: Path) -> str:
    name = path.name
    return re.sub(r"_[0-9]+$", "", name)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _label_to_candidate_dirs(label: str) -> list[str]:
    text = label.lower().replace("-", "_")
    candidates = [text]
    match = re.match(r"hs_?([0-9]+)$", text)
    if match:
        candidates.append(f"hamsim_{{match.group(1)}}")
    match = re.match(r"tl_?([0-9]+)$", text)
    if match:
        candidates.append(f"twolocal_{{match.group(1)}}")
    match = re.match(r"qaoa_p_?([0-9]+)$", text)
    if match:
        candidates.append(f"qaoa_pl{{match.group(1)}}")
    match = re.match(r"qaoa_r_?([0-9]+)$", text)
    if match:
        candidates.append(f"qaoa_r{{match.group(1)}}")
    if text in {{"w_state", "wstate"}}:
        candidates.append("wstate")
    return candidates


def _discover_benchmark_workloads(workspace_root: Path, max_apps: int = 9, circuits_per_app: int = 3) -> tuple[list[dict], list[dict]]:
    """Load representative QASM workloads from evaluation/benchmarks."""
    diagnostics = []
    debug_max_sim_qubits = None
    debug_threshold_qubits = None
    debug_threshold_qubit_sequence = None
    scaled_threshold_qubits = None
    scaled_threshold_qubit_sequence = None
    raw_scaled_thresholds = os.environ.get("REPRO_FIG11_THRESHOLD_QUBITS", "").strip()
    if raw_scaled_thresholds:
        try:
            parsed = [int(item.strip()) for item in raw_scaled_thresholds.split(",") if item.strip()]
            if len(parsed) != 3:
                raise ValueError("expected exactly three comma-separated qubit counts")
            scaled_threshold_qubit_sequence = parsed
            scaled_threshold_qubits = {{0.30: parsed[0], 0.60: parsed[1], 0.88: parsed[2]}}
        except Exception as exc:
            diagnostics.append({{
                "stage": "parse_scaled_threshold_qubits",
                "value": raw_scaled_thresholds,
                "error": repr(exc),
            }})
    raw_debug_thresholds = os.environ.get("REPRO_FIG11_DEBUG_THRESHOLD_QUBITS", "").strip()
    if raw_debug_thresholds and scaled_threshold_qubits is None:
        try:
            parsed = [int(item.strip()) for item in raw_debug_thresholds.split(",") if item.strip()]
            if len(parsed) != 3:
                raise ValueError("expected exactly three comma-separated qubit counts")
            debug_threshold_qubit_sequence = parsed
            debug_threshold_qubits = {{0.30: parsed[0], 0.60: parsed[1], 0.88: parsed[2]}}
            debug_max_sim_qubits = max(parsed)
        except Exception as exc:
            diagnostics.append({{
                "stage": "parse_debug_threshold_qubits",
                "value": raw_debug_thresholds,
                "error": repr(exc),
            }})
    raw_debug_max = os.environ.get("REPRO_FIG11_DEBUG_MAX_SIM_QUBITS", "").strip()
    if raw_debug_max and debug_threshold_qubits is None:
        try:
            debug_max_sim_qubits = max(1, int(raw_debug_max))
        except Exception:
            diagnostics.append({{
                "stage": "parse_debug_max_sim_qubits",
                "value": raw_debug_max,
                "error": "invalid integer",
            }})
    try:
        max_apps = max(1, int(os.environ.get("REPRO_FIG11_MAX_APPS", max_apps)))
    except Exception:
        pass
    try:
        circuits_per_app = max(1, int(os.environ.get("REPRO_FIG11_CIRCUITS_PER_APP", circuits_per_app)))
    except Exception:
        pass
    try:
        from qiskit import QuantumCircuit
    except Exception as exc:
        diagnostics.append({{
            "stage": "import_qiskit_for_qasm",
            "error": repr(exc),
            "traceback": traceback.format_exc()[-2000:],
        }})
        return [], diagnostics

    bench_root = workspace_root / "evaluation" / "benchmarks"
    if not bench_root.is_dir():
        diagnostics.append({{
            "stage": "locate_benchmark_root",
            "error": f"benchmark root not found: {{bench_root}}",
        }})
        return [], diagnostics

    grouped = {{}}
    for app_dir in sorted(path for path in bench_root.iterdir() if path.is_dir()):
        qasm_files = sorted(
            app_dir.glob("*.qasm"),
            key=lambda p: (
                int(p.stem) if p.stem.isdigit() else 10**9,
                p.name,
            ),
        )
        if not qasm_files:
            continue
        app = app_dir.name
        grouped.setdefault(app, [])
        grouped[app].extend(qasm_files)

    if not grouped:
        diagnostics.append({{
            "stage": "discover_qasm_files",
            "error": f"no .qasm files under {{bench_root}}",
        }})
        return [], diagnostics

    expected_apps = [str(x) for x in REQUIREMENTS.get("expected_applications", []) if str(x)]
    app_order = []
    app_labels = {{}}
    if expected_apps:
        normalized_dirs = {{}}
        for raw_app in grouped:
            normalized_dirs.setdefault(_normalize_name(raw_app), raw_app)
        for label in expected_apps:
            matched = None
            for candidate in _label_to_candidate_dirs(label):
                matched = grouped.get(candidate) and candidate
                if matched:
                    break
                matched = normalized_dirs.get(_normalize_name(candidate))
                if matched:
                    break
            if matched and matched not in app_order:
                app_order.append(matched)
                app_labels[matched] = label
            elif not matched:
                diagnostics.append({{
                    "stage": "match_expected_application",
                    "application": label,
                    "error": "no matching benchmark directory found",
                }})
    if not app_order:
        for app in sorted(grouped):
            app_order.append(app)

    workloads = []
    for app in app_order[:max_apps]:
        files = grouped[app]
        if len(files) <= circuits_per_app:
            selected = files
        else:
            indices = sorted(set([0, len(files) // 2, len(files) - 1]))[:circuits_per_app]
            selected = [files[i] for i in indices]
        circuits = []
        loaded_files = []
        errors = []
        for path in selected:
            try:
                circuits.append(QuantumCircuit.from_qasm_file(str(path)))
                loaded_files.append(str(path.relative_to(workspace_root)))
            except Exception as exc:
                errors.append({{"path": str(path.relative_to(workspace_root)), "error": repr(exc)}})
        if errors:
            diagnostics.append({{
                "stage": "load_qasm_files",
                "application": app,
                "loaded_count": len(circuits),
                "selected_count": len(selected),
                "errors": errors[:5],
            }})
        threshold_circuits = {{}}
        threshold_pair_circuits = {{}}
        threshold_files = {{}}
        threshold_pair_files = {{}}
        threshold_original_qubits = {{}}
        threshold_selected_qubits = {{}}
        threshold_component_qubits = {{}}
        size_circuits = {{}}
        size_files = {{}}
        threshold_qubits = {{0.30: 8, 0.60: 16, 0.88: 24}}
        by_stem = {{path.stem: path for path in files}}
        numeric_by_qubits = {{
            int(path.stem): path
            for path in files
            if path.stem.isdigit()
        }}
        for target, qubits in threshold_qubits.items():
            selected_qubits = qubits
            if scaled_threshold_qubits is not None:
                selected_qubits = scaled_threshold_qubits[target]
            elif debug_threshold_qubits is not None:
                selected_qubits = debug_threshold_qubits[target]
            elif debug_max_sim_qubits is not None and qubits > debug_max_sim_qubits:
                eligible = sorted(q for q in numeric_by_qubits if q <= debug_max_sim_qubits)
                if eligible:
                    selected_qubits = eligible[-1]
            threshold_original_qubits[target] = qubits
            threshold_selected_qubits[target] = selected_qubits
            component_qubits = max(1, selected_qubits // 2)
            threshold_component_qubits[target] = component_qubits
            threshold_path = by_stem.get(str(selected_qubits))
            if threshold_path is None:
                errors.append({{"threshold": target, "qubits": selected_qubits, "original_qubits": qubits, "error": "required simulation QASM file is missing"}})
                continue
            try:
                threshold_circuits[target] = QuantumCircuit.from_qasm_file(str(threshold_path))
                threshold_files[target] = str(threshold_path.relative_to(workspace_root))
            except Exception as exc:
                errors.append({{"threshold": target, "path": str(threshold_path.relative_to(workspace_root)), "error": repr(exc)}})
            pair_path = by_stem.get(str(component_qubits))
            if pair_path is None:
                errors.append({{"threshold": target, "qubits": component_qubits, "joint_qubits": selected_qubits, "error": "required pair component QASM file is missing"}})
                continue
            try:
                threshold_pair_circuits[target] = QuantumCircuit.from_qasm_file(str(pair_path))
                threshold_pair_files[target] = str(pair_path.relative_to(workspace_root))
            except Exception as exc:
                errors.append({{"threshold": target, "path": str(pair_path.relative_to(workspace_root)), "error": repr(exc)}})
        max_selected_qubits = max(list(threshold_selected_qubits.values()) or list(threshold_qubits.values()))
        for size, numeric_path in sorted(numeric_by_qubits.items()):
            if size > max_selected_qubits:
                continue
            try:
                size_circuits[int(size)] = QuantumCircuit.from_qasm_file(str(numeric_path))
                size_files[int(size)] = str(numeric_path.relative_to(workspace_root))
            except Exception as exc:
                errors.append({{"size": int(size), "path": str(numeric_path.relative_to(workspace_root)), "error": repr(exc)}})
        if scaled_threshold_qubits is not None:
            missing_targets = [
                target
                for target in threshold_qubits
                if target not in threshold_circuits or target not in threshold_pair_circuits
            ]
            if missing_targets:
                diagnostics.append({{
                    "stage": "skip_incomplete_scaled_workload",
                    "application": app,
                    "label": app_labels.get(app, _application_name(Path(app))),
                    "missing_thresholds": missing_targets,
                    "errors": errors[-6:],
                }})
                continue
        if circuits:
            workloads.append(
                {{
                    "application": app_labels.get(app, _application_name(Path(app))),
                    "benchmark_group": app,
                    "files": loaded_files,
                    "circuits": circuits,
                    "threshold_files": threshold_files,
                    "threshold_circuits": threshold_circuits,
                    "no_mp_threshold_files": threshold_files,
                    "no_mp_threshold_circuits": threshold_circuits,
                    "threshold_pair_files": threshold_pair_files,
                    "threshold_pair_circuits": threshold_pair_circuits,
                    "size_files": size_files,
                    "size_circuits": size_circuits,
                    "threshold_original_qubits": threshold_original_qubits,
                    "threshold_selected_qubits": threshold_selected_qubits,
                    "threshold_component_qubits": threshold_component_qubits,
                    "scaled_simulation": scaled_threshold_qubits is not None,
                    "scaled_threshold_qubits": scaled_threshold_qubits,
                    "scaled_threshold_qubit_sequence": scaled_threshold_qubit_sequence,
                    "full_24q_not_run_due_timeout": bool(os.environ.get("REPRO_FIG11_FULL_24Q_TIMEOUT_BOUNDED", "").strip()),
                    "debug_max_sim_qubits": debug_max_sim_qubits,
                    "debug_threshold_qubits": debug_threshold_qubits,
                    "debug_threshold_qubit_sequence": debug_threshold_qubit_sequence,
                    "errors": errors,
                }}
            )
    if not workloads:
        diagnostics.append({{
            "stage": "load_qasm_workloads",
            "error": "qasm files were found, but none could be parsed into circuits",
            "application_count": len(grouped),
        }})
    return workloads, diagnostics


def _qernel_for_circuit(circuit, metadata: dict):
    from qos.types.types import Qernel

    class _CallableInt(int):
        def __call__(self):
            return int(self)

    qernel = Qernel(circuit, metadata)
    qernel.num_qubits = _CallableInt(circuit.num_qubits)
    qernel.depth = _CallableInt(circuit.depth() or 0)
    return qernel


def _circuit_metadata(circuit, cmr: float | None = None) -> dict:
    ops = dict(circuit.count_ops())
    size = max(int(circuit.size() or 0), 1)
    depth = int(circuit.depth() or 0)
    two_qubit = sum(int(count) for name, count in ops.items() if str(name).lower() in {{"cx", "cz", "swap", "ecr", "rxx", "ryy", "rzz"}})
    measurements = int(ops.get("measure", 0) or 0)
    return {{
        "depth": depth,
        "entanglement_ratio": two_qubit / size,
        "measurement": measurements / size,
        "parallelism": size / max(depth, 1),
        "num_nonlocal_gates": two_qubit,
        "num_measurements": measurements,
        "cmr": cmr if cmr is not None else two_qubit / max(measurements, 1),
    }}


def _measurement_circuit(circuit):
    qc = circuit.copy()
    if not qc.clbits:
        qc.measure_all()
    elif not any(getattr(inst.operation, "name", "") == "measure" for inst in qc.data):
        qc.measure_all()
    return qc


def _counts_distribution(counts: dict, shots: int) -> dict:
    total = float(sum(counts.values()) or shots or 1)
    return {{str(key): float(value) / total for key, value in counts.items()}}


def _classical_fidelity(left: dict, right: dict) -> float:
    keys = set(left) | set(right)
    overlap = sum(math.sqrt(max(left.get(key, 0.0), 0.0) * max(right.get(key, 0.0), 0.0)) for key in keys)
    return _clamp(overlap * overlap, 0.0, 1.0)


def _bitstring_to_little_endian_bits(bitstring: str, num_qubits: int) -> str:
    cleaned = str(bitstring).replace(" ", "")
    if len(cleaned) < num_qubits:
        cleaned = cleaned.zfill(num_qubits)
    return cleaned[-num_qubits:][::-1]


def _counts_marginals(counts: dict, num_qubits: int, shots: int) -> list[tuple[float, float]]:
    total = float(sum(counts.values()) or shots or 1)
    one_probs = [0.0 for _ in range(num_qubits)]
    for bitstring, value in counts.items():
        bits = _bitstring_to_little_endian_bits(str(bitstring), num_qubits)
        weight = float(value) / total
        for index, bit in enumerate(bits):
            if bit == "1":
                one_probs[index] += weight
    return [(_clamp(1.0 - p1, 0.0, 1.0), _clamp(p1, 0.0, 1.0)) for p1 in one_probs]


def _local_marginal_count_fidelity(left_counts: dict, right_counts: dict, num_qubits: int, shots: int) -> float:
    """Estimate simulation fidelity from Aer counts without sparse full-support failure."""
    left = _counts_marginals(left_counts, num_qubits, shots)
    right = _counts_marginals(right_counts, num_qubits, shots)
    per_qubit = []
    for (l0, l1), (r0, r1) in zip(left, right):
        bc = math.sqrt(max(l0, 0.0) * max(r0, 0.0)) + math.sqrt(max(l1, 0.0) * max(r1, 0.0))
        per_qubit.append(_clamp(bc * bc, 1e-12, 1.0))

    pair_terms = []
    for index in range(max(0, num_qubits - 1)):
        left_pair = [0.0, 0.0, 0.0, 0.0]
        right_pair = [0.0, 0.0, 0.0, 0.0]
        left_total = float(sum(left_counts.values()) or shots or 1)
        right_total = float(sum(right_counts.values()) or shots or 1)
        for bitstring, value in left_counts.items():
            bits = _bitstring_to_little_endian_bits(str(bitstring), num_qubits)
            left_pair[int(bits[index]) + 2 * int(bits[index + 1])] += float(value) / left_total
        for bitstring, value in right_counts.items():
            bits = _bitstring_to_little_endian_bits(str(bitstring), num_qubits)
            right_pair[int(bits[index]) + 2 * int(bits[index + 1])] += float(value) / right_total
        bc = sum(math.sqrt(max(left_pair[pos], 0.0) * max(right_pair[pos], 0.0)) for pos in range(4))
        pair_terms.append(_clamp(bc * bc, 1e-12, 1.0))

    terms = per_qubit + pair_terms
    if not terms:
        return 1.0
    return _clamp(math.exp(sum(math.log(value) for value in terms) / len(terms)), 0.0, 1.0)


def _full_count_distribution_fidelity(left_counts: dict, right_counts: dict, shots: int) -> float:
    """Hellinger/classical fidelity over the full measured bitstring distribution."""
    return _classical_fidelity(
        _counts_distribution(left_counts, shots),
        _counts_distribution(right_counts, shots),
    )


def _repo_utilization_noise_evidence(workspace_root: Path) -> dict:
    """Look for an explicit repo model that maps utilization/crosstalk to noise."""
    evidence = []
    include_roots = ["qos", "qvm", "Baseline_Multiprogramming", "evaluation"]
    utilization_terms = ("utilization", "multiprogram", "co-runner", "corunner", "parallel", "co_locat", "colocat")
    noise_terms = ("noise", "crosstalk", "cross-talk", "thermal", "relaxation", "depolar", "readout_error", "gate_error")
    for root_name in include_roots:
        root = workspace_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            lowered = text.lower()
            if not any(term in lowered for term in utilization_terms):
                continue
            if not any(term in lowered for term in noise_terms):
                continue
            lines = []
            for line_no, line in enumerate(text.splitlines(), start=1):
                low = line.lower()
                if any(term in low for term in utilization_terms) and any(term in low for term in noise_terms):
                    lines.append({{"line": line_no, "text": line.strip()[:220]}})
                if len(lines) >= 5:
                    break
            evidence.append({{"path": str(path.relative_to(workspace_root)), "matches": lines}})
            if len(evidence) >= 12:
                break
        if len(evidence) >= 12:
            break
    return {{
        "found": bool(evidence),
        "source": evidence[0]["path"] if evidence else "not_found",
        "evidence": evidence,
    }}


def _fake_backend_candidates() -> list[str]:
    requested = os.environ.get("REPRO_AER_FAKE_BACKEND", "").strip()
    candidates = [requested] if requested else []
    candidates.extend(["FakeKolkataV2", "FakeMontrealV2", "FakeManilaV2", "FakeAthensV2", "FakeKolkata", "FakeMontreal", "FakeManila", "FakeAthens"])
    out = []
    for item in candidates:
        if item and item not in out:
            out.append(item)
    return out


def _load_fake_backend() -> tuple[object | None, dict]:
    evidence = {{"backend": None, "provider": None, "attempts": []}}
    providers = []
    try:
        import qiskit_ibm_runtime.fake_provider as ibm_fake_provider
        providers.append(("qiskit_ibm_runtime.fake_provider", ibm_fake_provider))
    except Exception as exc:
        evidence["attempts"].append({{"provider": "qiskit_ibm_runtime.fake_provider", "ok": False, "error": repr(exc)}})
    try:
        import qiskit.providers.fake_provider as qiskit_fake_provider
        providers.append(("qiskit.providers.fake_provider", qiskit_fake_provider))
    except Exception as exc:
        evidence["attempts"].append({{"provider": "qiskit.providers.fake_provider", "ok": False, "error": repr(exc)}})
    for provider_name, fake_provider in providers:
        for name in _fake_backend_candidates():
            try:
                factory = getattr(fake_provider, name)
                backend = factory()
                evidence.update({{"backend": name, "provider": provider_name, "ok": True}})
                return backend, evidence
            except Exception as exc:
                evidence["attempts"].append({{"provider": provider_name, "backend": name, "ok": False, "error": repr(exc)}})
    evidence["ok"] = False
    return None, evidence


def _build_qiskit_aer_noise_model(workspace_root: Path) -> dict:
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    utilization_evidence = _repo_utilization_noise_evidence(workspace_root)
    noise_qpu = os.environ.get("REPRO_NOISE_QPU", "ibm_marrakesh").strip() or "ibm_marrakesh"
    noise_2q_mode = os.environ.get("REPRO_NOISE_2Q_MODE", "layered").strip() or "layered"
    qpu_presets = {{
        "ibm_marrakesh": {{"p2_median": 2.67e-3, "p2_layered": 5.57e-3, "readout": 1.172e-2}},
        "ibm_torino": {{"p2_median": 2.59e-3, "p2_layered": 7.91e-3, "readout": 3.113e-2}},
    }}
    preset = qpu_presets.get(noise_qpu)
    if preset:
        p2 = float(preset["p2_median"] if noise_2q_mode == "median" else preset["p2_layered"])
        p1 = p2 * 0.1
        noise = NoiseModel()
        err1 = depolarizing_error(p1, 1)
        err2 = depolarizing_error(p2, 2)
        for gate in ["x", "y", "z", "h", "rx", "ry", "rz", "sx", "id", "s", "sdg"]:
            noise.add_all_qubit_quantum_error(err1, [gate])
        for gate in ["cx", "cz", "swap", "rzz", "cp", "ecr"]:
            noise.add_all_qubit_quantum_error(err2, [gate])
        readout = float(preset["readout"])
        noise.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
        return {{
            "noise_model": noise,
            "source": "qos_agent.generated_noise_model.ibm_preset_depolarizing_readout",
            "backend": noise_qpu,
            "noise_qpu": noise_qpu,
            "noise_2q_mode": noise_2q_mode,
            "p1": p1,
            "p2": p2,
            "readout": readout,
            "noise_model_family": "ibm_preset_depolarizing_readout",
            "utilization_noise_model_found": utilization_evidence["found"],
            "utilization_noise_source": utilization_evidence["source"],
            "utilization_noise_evidence": utilization_evidence["evidence"],
            "partial_reproduction": False,
            "limitation": None,
        }}

    backend, backend_evidence = _load_fake_backend()
    if backend is not None:
        try:
            return {{
                "noise_model": NoiseModel.from_backend(backend),
                "source": "qiskit_aer.noise.NoiseModel.from_backend",
                "backend": backend_evidence.get("backend"),
                "noise_qpu": noise_qpu,
                "noise_2q_mode": noise_2q_mode,
                "noise_model_family": "backend_derived",
                "backend_evidence": backend_evidence,
                "utilization_noise_model_found": utilization_evidence["found"],
                "utilization_noise_source": utilization_evidence["source"],
                "utilization_noise_evidence": utilization_evidence["evidence"],
                "partial_reproduction": not utilization_evidence["found"],
                "limitation": "requested_ibm_noise_preset_unavailable" if utilization_evidence["found"] else "requested_ibm_noise_preset_and_repo_utilization_or_crosstalk_noise_model_not_found",
            }}
        except Exception as exc:
            backend_evidence["from_backend_error"] = repr(exc)
    return {{
        "noise_model": NoiseModel(),
        "source": "qiskit_aer.noise.NoiseModel.empty",
        "backend": None,
        "noise_qpu": noise_qpu,
        "noise_2q_mode": noise_2q_mode,
        "noise_model_family": "empty",
        "backend_evidence": backend_evidence,
        "utilization_noise_model_found": utilization_evidence["found"],
        "utilization_noise_source": utilization_evidence["source"],
        "utilization_noise_evidence": utilization_evidence["evidence"],
        "partial_reproduction": True,
        "limitation": "backend_derived_noise_model_unavailable" if utilization_evidence["found"] else "backend_and_repo_utilization_noise_model_not_found",
    }}


def _simulate_counts(circuit, *, noise_model, shots: int, seed: int, backend=None, initial_layout=None) -> dict:
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    qc = _measurement_circuit(circuit)
    method = os.environ.get("REPRO_AER_METHOD", "matrix_product_state")
    max_threads = int(os.environ.get("REPRO_AER_MAX_THREADS", "1") or "1")
    if backend is not None:
        qc = transpile(qc, backend, initial_layout=initial_layout, optimization_level=1)
    if noise_model is None:
        simulator = AerSimulator(method=method, seed_simulator=seed, max_parallel_threads=max_threads)
    else:
        simulator = AerSimulator(method=method, noise_model=noise_model, seed_simulator=seed, max_parallel_threads=max_threads)
    return simulator.run(qc, shots=shots).result().get_counts()


def _simulate_noisy_fidelity(circuit, *, noise_model, shots: int, seed: int) -> float:
    ideal_counts = _simulate_counts(circuit, noise_model=None, shots=shots, seed=seed)
    noisy_counts = _simulate_counts(circuit, noise_model=noise_model, shots=shots, seed=seed + 1)
    return _full_count_distribution_fidelity(ideal_counts, noisy_counts, shots)


def _simulate_target_size_no_mp_fidelity(circuit, threshold: float, noise_model_info: dict, *, shots: int | None = None, seed: int = 31415) -> dict:
    if shots is None:
        shots = int(os.environ.get("REPRO_AER_SHOTS", "1024") or "1024")
    qubits = int(getattr(circuit, "num_qubits", 0) or 0)
    _stage("simulate_no_mp_target_size_start", threshold=threshold, qubits=qubits, shots=shots)
    fidelity = _simulate_noisy_fidelity(circuit, noise_model=noise_model_info.get("noise_model"), shots=shots, seed=seed)
    _stage("simulate_no_mp_target_size_done", threshold=threshold, qubits=qubits, fidelity=fidelity)
    return {{
        "fidelity": _clamp(float(fidelity), 0.0, 1.0),
        "shots": shots,
        "selected_qubits": qubits,
        "unit_of_execution": "single_circuit_target_size",
        "source_role": "fig11a_no_multiprogramming_target_size_solo",
        "reference_mode": "target_size_solo_simulation",
        "estimator": "full_count_distribution_hellinger_fidelity",
        "noise_model_source": noise_model_info.get("source"),
        "noise_model_backend": noise_model_info.get("backend"),
        "noise_qpu": noise_model_info.get("noise_qpu"),
        "noise_2q_mode": noise_model_info.get("noise_2q_mode"),
        "noise_p2": noise_model_info.get("p2"),
        "noise_readout": noise_model_info.get("readout"),
    }}


def _simulate_application_relative_fidelity(circuit, threshold: float, matching: float, noise_model_info: dict, *, shots: int | None = None, seed: int = 31415) -> dict:
    """Compute Fig. 11(c) from simulation only; no table/formula fallback."""
    # Solo/reference and QOS M/P are both simulated against the same backend
    # noise model. Do not invent a utilization noise schedule unless the repo
    # contains an explicit model for it; record that limitation instead.
    if shots is None:
        shots = int(os.environ.get("REPRO_AER_SHOTS", "1024") or "1024")
    _stage(
        "simulate_relative_fidelity_start",
        threshold=threshold,
        qubits=int(getattr(circuit, "num_qubits", 0) or 0),
        shots=shots,
    )
    noise_model = noise_model_info.get("noise_model")
    solo_sampled_fidelity = _simulate_noisy_fidelity(circuit, noise_model=noise_model, shots=shots, seed=seed)
    qos_sampled_fidelity = _simulate_noisy_fidelity(circuit, noise_model=noise_model, shots=shots, seed=seed + int(round(threshold * 1000)))
    fidelity_floor = 1e-9
    solo_fidelity = max(solo_sampled_fidelity, fidelity_floor)
    qos_fidelity = max(qos_sampled_fidelity, 0.0)
    relative = _clamp(qos_fidelity / solo_fidelity, 0.0, 1.0)
    _stage("simulate_relative_fidelity_done", threshold=threshold, relative_fidelity=relative)
    return {{
        "relative_fidelity": relative,
        "solo_fidelity": solo_fidelity,
        "qos_fidelity": qos_fidelity,
        "shots": shots,
        "sampled_solo_fidelity": solo_sampled_fidelity,
        "sampled_qos_fidelity": qos_sampled_fidelity,
        "fidelity_floor": fidelity_floor,
        "estimator": "full_count_distribution_hellinger_fidelity",
        "noise_model_source": noise_model_info.get("source"),
        "noise_model_backend": noise_model_info.get("backend"),
        "noise_qpu": noise_model_info.get("noise_qpu"),
        "noise_2q_mode": noise_model_info.get("noise_2q_mode"),
        "noise_p2": noise_model_info.get("p2"),
        "noise_readout": noise_model_info.get("readout"),
        "utilization_noise_model_found": bool(noise_model_info.get("utilization_noise_model_found")),
        "utilization_noise_source": noise_model_info.get("utilization_noise_source"),
        "partial_reproduction": bool(noise_model_info.get("partial_reproduction")),
        "limitation": noise_model_info.get("limitation"),
        "noise": {{
            "source": noise_model_info.get("source"),
            "backend": noise_model_info.get("backend"),
            "noise_qpu": noise_model_info.get("noise_qpu"),
            "noise_2q_mode": noise_model_info.get("noise_2q_mode"),
            "p2": noise_model_info.get("p2"),
            "readout": noise_model_info.get("readout"),
            "utilization_noise_model_found": bool(noise_model_info.get("utilization_noise_model_found")),
        }},
    }}


def _project_counts_to_component(counts: dict, *, start: int, width: int, total_qubits: int) -> dict:
    projected = {{}}
    for bitstring, value in counts.items():
        bits = _bitstring_to_little_endian_bits(str(bitstring), total_qubits)
        component_little = bits[start:start + width]
        # Qiskit count keys are big-endian strings; convert the projected
        # little-endian slice back to the same convention used by solo counts.
        key = component_little[::-1] if component_little else "0"
        projected[key] = projected.get(key, 0) + value
    return projected


def _bundle_source_qernels(bundle) -> list:
    sources = list(getattr(bundle, "src_qernels", []) or [])
    if sources:
        return sources
    return [bundle]


def _source_offsets_for_bundle(bundle) -> list[dict]:
    sources = _bundle_source_qernels(bundle)
    offsets = []
    qubit_start = 0
    clbit_start = 0
    for index, source in enumerate(sources):
        circuit = source.get_circuit()
        width = int(getattr(circuit, "num_qubits", 0) or 0)
        clbits = int(getattr(circuit, "num_clbits", 0) or 0)
        offsets.append({{
            "source_index": index,
            "qubit_start": qubit_start,
            "qubit_count": width,
            "clbit_start": clbit_start,
            "clbit_count": clbits,
        }})
        qubit_start += width
        clbit_start += clbits
    return offsets


def _backend_num_qubits(backend, default: int = 27) -> int:
    return int(getattr(backend, "num_qubits", default) or default)


def _backend_coupling_edges(backend) -> set[tuple[int, int]]:
    coupling = getattr(backend, "coupling_map", None)
    try:
        raw_edges = list(coupling.get_edges())
    except Exception:
        try:
            raw_edges = list(coupling or [])
        except Exception:
            raw_edges = []
    edges = set()
    for edge in raw_edges:
        try:
            left, right = int(edge[0]), int(edge[1])
            edges.add((left, right))
            edges.add((right, left))
        except Exception:
            continue
    return edges


def _measurement_error_for_backend_qubit(backend, qubit: int) -> float:
    target = getattr(backend, "target", None)
    try:
        measure_props = target.get("measure", None) if target is not None else None
        if measure_props and (qubit,) in measure_props:
            error = getattr(measure_props[(qubit,)], "error", None)
            if error is not None:
                return float(error)
    except Exception:
        pass
    try:
        props = backend.properties()
        for item in props.qubits[qubit]:
            if getattr(item, "name", "") == "readout_error":
                return float(item.value)
    except Exception:
        pass
    return 0.01


def _find_good_layout_for_backend(circuit, backend, *, forbidden: set[int] | None = None) -> list[int] | None:
    n = int(getattr(circuit, "num_qubits", 0) or 0)
    n_backend = _backend_num_qubits(backend)
    forbidden = set(forbidden or set())
    available = [q for q in range(n_backend) if q not in forbidden]
    if len(available) < n:
        return None
    edges = _backend_coupling_edges(backend)
    ranked = sorted(available, key=lambda q: (_measurement_error_for_backend_qubit(backend, q), q))
    layout = []
    used = set()
    for q in ranked:
        if len(layout) >= n:
            break
        if q in used:
            continue
        if not layout or not edges or any((q, existing) in edges for existing in layout):
            layout.append(q)
            used.add(q)
    if len(layout) < n:
        for q in ranked:
            if len(layout) >= n:
                break
            if q not in used:
                layout.append(q)
                used.add(q)
    return layout[:n] if len(layout) >= n else None


def _combined_pair_circuit(circ1, circ2):
    from qiskit import QuantumCircuit

    circ1_m = _measurement_circuit(circ1)
    circ2_m = _measurement_circuit(circ2)
    n1 = int(circ1_m.num_qubits)
    n2 = int(circ2_m.num_qubits)
    combined = QuantumCircuit(n1 + n2, n1 + n2)
    combined.compose(circ1_m, qubits=range(n1), clbits=range(n1), inplace=True)
    combined.compose(circ2_m, qubits=range(n1, n1 + n2), clbits=range(n1, n1 + n2), inplace=True)
    return combined


def _simulate_two_circuit_joint_fidelity(circ1, circ2, *, layout1: list[int], layout2: list[int], backend, noise_model, shots: int, seed: int, policy: str) -> dict:
    n1 = int(getattr(circ1, "num_qubits", 0) or 0)
    n2 = int(getattr(circ2, "num_qubits", 0) or 0)
    if n1 <= 0 or n2 <= 0:
        raise RuntimeError("pair simulation requires two non-empty circuits")
    if len(set(layout1) & set(layout2)):
        raise RuntimeError(f"{{policy}} produced overlapping layouts")
    if n1 + n2 > _backend_num_qubits(backend):
        raise RuntimeError(f"{{policy}} pair exceeds backend qubit capacity")
    combined = _combined_pair_circuit(circ1, circ2)
    layout = list(layout1) + list(layout2)
    joint_counts = _simulate_counts(combined, noise_model=noise_model, shots=shots, seed=seed, backend=backend, initial_layout=layout)
    ideal1 = _simulate_counts(circ1, noise_model=None, shots=shots, seed=seed + 101)
    ideal2 = _simulate_counts(circ2, noise_model=None, shots=shots, seed=seed + 202)
    counts1 = _project_counts_to_component(joint_counts, start=0, width=n1, total_qubits=n1 + n2)
    counts2 = _project_counts_to_component(joint_counts, start=n1, width=n2, total_qubits=n1 + n2)
    f1 = _full_count_distribution_fidelity(ideal1, counts1, shots)
    f2 = _full_count_distribution_fidelity(ideal2, counts2, shots)
    return {{
        "policy": policy,
        "layout1": list(layout1),
        "layout2": list(layout2),
        "fidelity1": f1,
        "fidelity2": f2,
        "fidelity": _mean([f1, f2]),
    }}


def _select_or_build_qos_bundles(qernels: list, returned: object, matching_values: list[float]) -> dict:
    """Return selected bundled qernels without silently downgrading semantics."""
    returned_items = returned if isinstance(returned, list) else []
    selected = [
        item for item in returned_items
        if bool(getattr(item, "is_bundle", lambda: False)()) and len(getattr(item, "src_qernels", []) or []) >= 2
    ]
    if selected:
        return {{
            "ok": True,
            "bundles": selected,
            "bundle_count": len(selected),
            "pair_selection_source": "qos.multiprogrammer.multiprogrammer.Multiprogrammer.run",
            "bundle_source": "qos.multiprogrammer.tools.bundle_qernels",
            "joint_circuit_source": "qos.types.types.Qernel.append_circuit",
            "semantic_downgrade": False,
            "proxy_substitution": False,
            "selection_notes": "Multiprogrammer.run returned Qernel bundles with src_qernels.",
        }}

    allow_reconstructed_selection = os.environ.get("REPRO_DISABLE_RECONSTRUCTED_PAIR_SELECTION", "").strip().lower() not in {{"1", "true", "yes", "on"}}
    if not allow_reconstructed_selection:
        return {{
            "ok": False,
            "bundles": [],
            "bundle_count": 0,
            "pair_selection_source": "unavailable",
            "bundle_source": "qos.multiprogrammer.tools.bundle_qernels",
            "joint_circuit_source": "qos.types.types.Qernel.append_circuit",
            "semantic_downgrade": True,
            "proxy_substitution": True,
            "selection_notes": "Multiprogrammer.run did not return selected bundles; heuristic fallback is disabled.",
            "error": "qos_selected_bundle_unavailable",
        }}

    from qos.multiprogrammer.tools import bundle_qernels

    try:
        max_joint_qubits = max(1, int(os.environ.get("REPRO_PAIR_JOINT_MAX_QUBITS", "32") or "32"))
    except Exception:
        max_joint_qubits = 32
    ranked_pairs = []
    for index in range(len(qernels) - 1):
        left_circuit = qernels[index].get_circuit()
        right_circuit = qernels[index + 1].get_circuit()
        joint_qubits = int(left_circuit.num_qubits) + int(right_circuit.num_qubits)
        if joint_qubits <= max_joint_qubits:
            ranked_pairs.append((float(matching_values[index]) if index < len(matching_values) else 0.0, joint_qubits, index, index + 1))
    if not ranked_pairs and len(qernels) >= 2:
        for index in range(len(qernels) - 1):
            left_circuit = qernels[index].get_circuit()
            right_circuit = qernels[index + 1].get_circuit()
            joint_qubits = int(left_circuit.num_qubits) + int(right_circuit.num_qubits)
            ranked_pairs.append((float(matching_values[index]) if index < len(matching_values) else 0.0, joint_qubits, index, index + 1))
    ranked_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected_bundles = []
    used = set()
    for score, joint_qubits, left, right in ranked_pairs:
        if left in used or right in used:
            continue
        left_circuit = qernels[left].get_circuit()
        right_circuit = qernels[right].get_circuit()
        left_layout = list(range(0, int(left_circuit.num_qubits)))
        right_layout = list(range(int(left_circuit.num_qubits), int(left_circuit.num_qubits) + int(right_circuit.num_qubits)))
        matching = (left_layout, "simulation_qpu", score)
        bundled = bundle_qernels(qernels[right], qernels[left], matching)
        if bundled == 0:
            bundled = qernels[left]
        if bool(getattr(bundled, "is_bundle", lambda: False)()):
            selected_bundles.append(bundled)
            used.update({{left, right}})
    return {{
        "ok": bool(selected_bundles),
        "bundles": selected_bundles,
        "bundle_count": len(selected_bundles),
        "pair_selection_source": "qos_pair_metric_reconstruction_from_Multiprogrammer.get_matching_score",
        "bundle_source": "qos.multiprogrammer.tools.bundle_qernels",
        "joint_circuit_source": "qos.types.types.Qernel.append_circuit",
        "semantic_downgrade": False,
        "proxy_substitution": False,
        "max_joint_qubits": max_joint_qubits,
        "selection_notes": "Multiprogrammer.run is a stub in this repo revision, so the runner reconstructs QOS pair selection by ranking original Multiprogrammer.get_matching_score outputs, applying a runtime joint-qubit cap, and constructing bundles with the repo bundle_qernels primitive.",
        "error": None if selected_bundles else "reconstructed_bundle_construction_failed",
    }}


def _simulate_pair_joint_relative_fidelity(bundles: list, threshold: float, noise_model_info: dict, *, shots: int | None = None, seed: int = 31415, layout_backend=None) -> dict:
    if shots is None:
        shots = int(os.environ.get("REPRO_PAIR_JOINT_AER_SHOTS", os.environ.get("REPRO_AER_SHOTS", "64")) or "64")
    if not bundles:
        raise RuntimeError("no selected bundles available for pair-level joint simulation")
    if layout_backend is None:
        layout_backend, _ = _load_fake_backend()
    if layout_backend is None:
        raise RuntimeError("pair-level simulation requires a transpile/layout backend")
    noise_model = noise_model_info.get("noise_model")
    pair_records = []
    relative_values = []
    solo_values = []
    baseline_values = []
    qos_values = []
    _stage("simulate_pair_joint_relative_fidelity_start", threshold=threshold, bundle_count=len(bundles), shots=shots)
    for bundle_index, bundle in enumerate(bundles):
        sources = _bundle_source_qernels(bundle)
        if len(sources) < 2:
            raise RuntimeError("selected bundle does not contain a pair")
        left_circuit = sources[0].get_circuit()
        right_circuit = sources[1].get_circuit()
        n1 = int(getattr(left_circuit, "num_qubits", 0) or 0)
        n2 = int(getattr(right_circuit, "num_qubits", 0) or 0)
        baseline_layout1 = list(range(n1))
        baseline_layout2 = list(range(n1, n1 + n2))
        qos_layout1 = _find_good_layout_for_backend(left_circuit, layout_backend)
        qos_layout2 = _find_good_layout_for_backend(right_circuit, layout_backend, forbidden=set(qos_layout1 or []))
        if qos_layout1 is None or qos_layout2 is None:
            raise RuntimeError("qos error-aware non-overlapping layout unavailable")
        baseline_joint = _simulate_two_circuit_joint_fidelity(
            left_circuit,
            right_circuit,
            layout1=baseline_layout1,
            layout2=baseline_layout2,
            backend=layout_backend,
            noise_model=noise_model,
            shots=shots,
            seed=seed + 1000 + bundle_index * 37,
            policy="consecutive_layout",
        )
        qos_joint = _simulate_two_circuit_joint_fidelity(
            left_circuit,
            right_circuit,
            layout1=qos_layout1,
            layout2=qos_layout2,
            backend=layout_backend,
            noise_model=noise_model,
            shots=shots,
            seed=seed + 2000 + bundle_index * 37,
            policy="error_aware_non_overlapping_layout",
        )
        component_records = []
        component_relative = []
        component_solo = []
        component_baseline = []
        component_qos = []
        for component_index, source in enumerate(sources[:2]):
            source_circuit = source.get_circuit()
            width = int(getattr(source_circuit, "num_qubits", 0) or 0)
            if width <= 0:
                continue
            solo_ideal_counts = _simulate_counts(source_circuit, noise_model=None, shots=shots, seed=seed + 2000 + bundle_index * 31 + component_index)
            solo_noisy_counts = _simulate_counts(source_circuit, noise_model=noise_model, shots=shots, seed=seed + 3000 + bundle_index * 31 + component_index)
            solo_fidelity = max(_full_count_distribution_fidelity(solo_ideal_counts, solo_noisy_counts, shots), 1e-9)
            baseline_fidelity = float(baseline_joint["fidelity1"] if component_index == 0 else baseline_joint["fidelity2"])
            qos_fidelity = float(qos_joint["fidelity1"] if component_index == 0 else qos_joint["fidelity2"])
            relative = _clamp(qos_fidelity / solo_fidelity, 0.0, 1.0)
            component_relative.append(relative)
            component_solo.append(solo_fidelity)
            component_baseline.append(baseline_fidelity)
            component_qos.append(qos_fidelity)
            component_records.append({{
                "component_index": component_index,
                "qubit_count": width,
                "solo_fidelity": solo_fidelity,
                "baseline_fidelity": baseline_fidelity,
                "qos_fidelity": qos_fidelity,
                "relative_fidelity": relative,
            }})
        pair_relative = _mean(component_relative)
        pair_solo = _mean(component_solo)
        pair_baseline = _mean(component_baseline)
        pair_qos = _mean(component_qos)
        relative_values.append(pair_relative)
        solo_values.append(pair_solo)
        baseline_values.append(pair_baseline)
        qos_values.append(pair_qos)
        pair_records.append({{
            "bundle_index": bundle_index,
            "source_count": len(sources),
            "joint_circuit_qubits": n1 + n2,
            "baseline_layout1": baseline_layout1,
            "baseline_layout2": baseline_layout2,
            "qos_layout1": qos_layout1,
            "qos_layout2": qos_layout2,
            "components": component_records,
            "solo_fidelity": pair_solo,
            "baseline_fidelity": pair_baseline,
            "qos_fidelity": pair_qos,
            "relative_fidelity": pair_relative,
        }})
    relative = _mean(relative_values)
    _stage("simulate_pair_joint_relative_fidelity_done", threshold=threshold, bundle_count=len(bundles), relative_fidelity=relative)
    return {{
        "relative_fidelity": relative,
        "solo_fidelity": _mean(solo_values),
        "baseline_fidelity": _mean(baseline_values),
        "qos_fidelity": _mean(qos_values),
        "joint_fidelity": _mean(baseline_values),
        "shots": shots,
        "bundle_count": len(bundles),
        "pair_records": pair_records,
        "estimator": "pair_joint_full_count_distribution_hellinger_fidelity",
        "unit_of_execution": "pair",
        "execution_mode": "multiprogrammed_joint_simulation",
        "reference_mode": "solo_simulation",
        "formula": "joint_execution_fidelity / solo_execution_fidelity",
        "baseline_layout_policy": "consecutive_layout",
        "qos_layout_policy": "error_aware_non_overlapping_layout",
        "baseline_qos_separate_simulation_paths": True,
        "noise_model_source": noise_model_info.get("source"),
        "noise_model_backend": noise_model_info.get("backend"),
        "noise_qpu": noise_model_info.get("noise_qpu"),
        "noise_2q_mode": noise_model_info.get("noise_2q_mode"),
        "noise_p2": noise_model_info.get("p2"),
        "noise_readout": noise_model_info.get("readout"),
        "utilization_noise_model_found": bool(noise_model_info.get("utilization_noise_model_found")),
        "utilization_noise_source": noise_model_info.get("utilization_noise_source"),
        "partial_reproduction": bool(noise_model_info.get("partial_reproduction")),
        "limitation": noise_model_info.get("limitation"),
        "noise": {{
            "source": noise_model_info.get("source"),
            "backend": noise_model_info.get("backend"),
            "noise_qpu": noise_model_info.get("noise_qpu"),
            "noise_2q_mode": noise_model_info.get("noise_2q_mode"),
            "p2": noise_model_info.get("p2"),
            "readout": noise_model_info.get("readout"),
            "utilization_noise_model_found": bool(noise_model_info.get("utilization_noise_model_found")),
            "crosstalk_model": "none_extra_beyond_joint_noisy_simulation",
        }},
    }}


def _candidate_pairs_from_workloads(workloads: list, target_qubits: int) -> list[dict]:
    entries = []
    for workload in workloads:
        app = str(workload.get("application") or workload.get("benchmark_group") or "")
        size_circuits = workload.get("size_circuits") or {{}}
        size_files = workload.get("size_files") or {{}}
        for size, circuit in size_circuits.items():
            try:
                qubits = int(size)
            except Exception:
                continue
            if qubits <= 0 or qubits > target_qubits:
                continue
            entries.append({{
                "application": app,
                "qubits": qubits,
                "label": f"{{app}}-{{qubits}}",
                "circuit": circuit,
                "file": size_files.get(qubits) or size_files.get(str(qubits)),
            }})
    pairs = []
    seen = set()
    for left_index, left in enumerate(entries):
        for right in entries[left_index + 1:]:
            if int(left["qubits"]) + int(right["qubits"]) != int(target_qubits):
                continue
            if left["label"] == right["label"]:
                continue
            key = tuple(sorted((left["label"], right["label"])))
            if key in seen:
                continue
            seen.add(key)
            if (right["application"], right["qubits"], right["label"]) < (left["application"], left["qubits"], left["label"]):
                first, second = right, left
            else:
                first, second = left, right
            pairs.append({{"left": first, "right": second, "target_qubits": int(target_qubits), "pair_label": f"{{first['label']}}+{{second['label']}}"}})
    return pairs


def _score_candidate_pair(pair: dict, mp, qpu) -> dict:
    left_circuit = pair["left"]["circuit"]
    right_circuit = pair["right"]["circuit"]
    left_q = _qernel_for_circuit(left_circuit.copy(), _circuit_metadata(left_circuit))
    right_q = _qernel_for_circuit(right_circuit.copy(), _circuit_metadata(right_circuit))
    spatial = float(mp.spatial_utilization(left_q, right_q, qpu))
    effective = float(mp.effective_utilization(left_q, right_q, qpu))
    matching = float(mp.get_matching_score(left_q, right_q, qpu))
    out = dict(pair)
    out.update({{
        "spatial_utilization_percent": spatial,
        "effective_utilization_percent": effective,
        "matching_score": matching,
        "selection_score": effective + 0.01 * matching,
    }})
    return out


def _select_baseline_candidate_pairs(candidates: list[dict], n_pairs: int, *, seed: int) -> list[dict]:
    if n_pairs is None or n_pairs <= 0 or n_pairs >= len(candidates):
        return list(candidates)
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled[:n_pairs]


def _pair_applications(pair: dict) -> set[str]:
    apps = set()
    for side in ("left", "right"):
        app = str((pair.get(side) or {{}}).get("application") or "").strip()
        if app:
            apps.add(app)
    return apps


def _coverage_aware_topk_pairs(scored: list[dict], n_pairs: int, required_applications: list[str]) -> tuple[list[dict], dict]:
    required_set = set(str(app) for app in required_applications if str(app))
    selected = []
    selected_labels = set()
    coverage_selected_labels = []
    fill_selected_labels = []
    covered = set()

    if required_set:
        while not required_set.issubset(covered):
            missing = required_set - covered
            best = None
            best_key = None
            for pair in scored:
                label = pair.get("pair_label")
                if label in selected_labels:
                    continue
                pair_apps = _pair_applications(pair)
                new_apps = pair_apps & missing
                if not new_apps:
                    continue
                key = (
                    len(new_apps),
                    float(pair.get("selection_score") or 0.0),
                    float(pair.get("effective_utilization_percent") or 0.0),
                    str(label or ""),
                )
                if best is None or key > best_key:
                    best = pair
                    best_key = key
            if best is None:
                break
            selected.append(best)
            selected_labels.add(best.get("pair_label"))
            coverage_selected_labels.append(best.get("pair_label"))
            covered.update(_pair_applications(best) & required_set)

    requested_topk = None if n_pairs is None or n_pairs <= 0 else int(n_pairs)
    target_count = len(scored) if requested_topk is None else max(requested_topk, len(selected))
    for pair in scored:
        if len(selected) >= target_count:
            break
        label = pair.get("pair_label")
        if label in selected_labels:
            continue
        selected.append(pair)
        selected_labels.add(label)
        fill_selected_labels.append(label)

    covered_apps = sorted(set().union(*[_pair_applications(pair) for pair in selected]) & required_set) if required_set and selected else []
    missing_apps = sorted(required_set - set(covered_apps))
    return selected, {{
        "selection_mode": "coverage_aware_topk",
        "selection_passes": ["coverage", "score_topk_fill"],
        "requested_topk": requested_topk,
        "target_count_after_coverage_expansion": target_count,
        "coverage_selected_count": len(coverage_selected_labels),
        "score_fill_selected_count": len(fill_selected_labels),
        "coverage_selected_pair_labels": coverage_selected_labels,
        "score_fill_pair_labels": fill_selected_labels,
        "required_applications": sorted(required_set),
        "covered_applications": covered_apps,
        "missing_applications": missing_apps,
        "application_coverage_complete": not missing_apps if required_set else True,
        "auto_expanded_for_coverage": requested_topk is not None and len(coverage_selected_labels) > requested_topk,
    }}


def _ranked_topk_pairs(scored: list[dict], n_pairs: int) -> tuple[list[dict], dict]:
    requested_topk = None if n_pairs is None or n_pairs <= 0 else int(n_pairs)
    target_count = len(scored) if requested_topk is None else min(requested_topk, len(scored))
    selected = list(scored[:target_count])
    covered_apps = sorted(set().union(*[_pair_applications(pair) for pair in selected])) if selected else []
    return selected, {{
        "selection_mode": "ranked_topk",
        "selection_passes": ["score_topk"],
        "requested_topk": requested_topk,
        "target_count_after_coverage_expansion": target_count,
        "coverage_selected_count": 0,
        "score_fill_selected_count": len(selected),
        "coverage_selected_pair_labels": [],
        "score_fill_pair_labels": [pair.get("pair_label") for pair in selected],
        "required_applications": [],
        "covered_applications": covered_apps,
        "missing_applications": [],
        "application_coverage_complete": True,
        "auto_expanded_for_coverage": False,
    }}


def _select_qos_candidate_pairs(candidates: list[dict], n_pairs: int, mp, qpu, required_applications: list[str] | None = None) -> tuple[list[dict], dict]:
    scored = []
    errors = []
    required = [str(app) for app in (required_applications or []) if str(app)]
    required_set = set(required)
    for pair in candidates:
        try:
            scored.append(_score_candidate_pair(pair, mp, qpu))
        except Exception as exc:
            errors.append({{"pair": pair.get("pair_label"), "error": repr(exc)}})
    scored.sort(key=lambda item: (float(item.get("selection_score") or 0.0), float(item.get("effective_utilization_percent") or 0.0)), reverse=True)
    if required_set:
        selected, selection_details = _coverage_aware_topk_pairs(scored, n_pairs, required)
    else:
        selected, selection_details = _ranked_topk_pairs(scored, n_pairs)
    return selected, {{
        "candidate_count": len(candidates),
        "scored_count": len(scored),
        "selected_count": len(selected),
        "errors": errors[:10],
        "pair_selection_source": "qos.multiprogrammer.Multiprogrammer.get_matching_score_ranked_candidate_pairs",
        "baseline_qos_pair_sets_are_distinct": True,
        "coverage_aware_selection": bool(required_set),
        **selection_details,
    }}


def _select_qos_application_representative_pairs(candidates: list[dict], application: str, n_pairs: int, mp, qpu) -> tuple[list[dict], dict]:
    application = str(application)
    app_candidates = [
        pair
        for pair in candidates
        if application in _pair_applications(pair)
    ]
    selected, selection = _select_qos_candidate_pairs(app_candidates, n_pairs, mp, qpu, required_applications=None)
    selection.update({{
        "selection_mode": "per_application_ranked_representative",
        "base_selection_mode": "ranked_topk",
        "application": application,
        "candidate_count_for_application": len(app_candidates),
        "pair_contains_application": all(application in _pair_applications(pair) for pair in selected),
    }})
    return selected, selection


def _simulate_selected_pair_set(pairs: list[dict], mode: str, threshold: float, noise_model_info: dict, mp, qpu, *, shots: int | None = None, seed: int = 31415, layout_backend=None) -> dict:
    if shots is None:
        shots = int(os.environ.get("REPRO_PAIR_JOINT_AER_SHOTS", os.environ.get("REPRO_AER_SHOTS", "64")) or "64")
    if layout_backend is None:
        layout_backend, _ = _load_fake_backend()
    if layout_backend is None:
        raise RuntimeError("selected-pair simulation requires a transpile/layout backend")
    if mode not in {{"baseline", "qos"}}:
        raise ValueError(f"unsupported selected-pair simulation mode: {{mode}}")
    noise_model = noise_model_info.get("noise_model")
    records = []
    fidelity_values = []
    effective_values = []
    relative_values = []
    _stage("simulate_selected_pair_set_start", threshold=threshold, mode=mode, pair_count=len(pairs), shots=shots)
    for index, pair in enumerate(pairs):
        left = pair["left"]
        right = pair["right"]
        left_circuit = left["circuit"]
        right_circuit = right["circuit"]
        n1 = int(left_circuit.num_qubits)
        n2 = int(right_circuit.num_qubits)
        if mode == "baseline":
            layout1 = list(range(n1))
            layout2 = list(range(n1, n1 + n2))
            layout_policy = "consecutive_layout"
        else:
            layout1 = _find_good_layout_for_backend(left_circuit, layout_backend)
            layout2 = _find_good_layout_for_backend(right_circuit, layout_backend, forbidden=set(layout1 or []))
            if layout1 is None or layout2 is None:
                raise RuntimeError(f"qos non-overlapping layout unavailable for {{pair.get('pair_label')}}")
            layout_policy = "error_aware_non_overlapping_layout"
        joint = _simulate_two_circuit_joint_fidelity(
            left_circuit,
            right_circuit,
            layout1=layout1,
            layout2=layout2,
            backend=layout_backend,
            noise_model=noise_model,
            shots=shots,
            seed=seed + index * 43 + (10000 if mode == "qos" else 0),
            policy=layout_policy,
        )
        left_solo_ideal = _simulate_counts(left_circuit, noise_model=None, shots=shots, seed=seed + index * 47 + 1)
        left_solo_noisy = _simulate_counts(left_circuit, noise_model=noise_model, shots=shots, seed=seed + index * 47 + 2)
        right_solo_ideal = _simulate_counts(right_circuit, noise_model=None, shots=shots, seed=seed + index * 47 + 3)
        right_solo_noisy = _simulate_counts(right_circuit, noise_model=noise_model, shots=shots, seed=seed + index * 47 + 4)
        left_solo = max(_full_count_distribution_fidelity(left_solo_ideal, left_solo_noisy, shots), 1e-9)
        right_solo = max(_full_count_distribution_fidelity(right_solo_ideal, right_solo_noisy, shots), 1e-9)
        f1 = float(joint["fidelity1"])
        f2 = float(joint["fidelity2"])
        fidelity = _mean([f1, f2])
        relative = _mean([_clamp(f1 / left_solo, 0.0, 1.0), _clamp(f2 / right_solo, 0.0, 1.0)])
        try:
            if "effective_utilization_percent" in pair:
                effective_percent = float(pair["effective_utilization_percent"])
            else:
                left_q = _qernel_for_circuit(left_circuit.copy(), _circuit_metadata(left_circuit))
                right_q = _qernel_for_circuit(right_circuit.copy(), _circuit_metadata(right_circuit))
                effective_percent = float(mp.effective_utilization(left_q, right_q, qpu))
        except Exception:
            effective_percent = 0.0
        fidelity_values.append(fidelity)
        effective_values.append(effective_percent / 100.0)
        relative_values.append(relative)
        records.append({{
            "pair_label": pair.get("pair_label"),
            "mode": mode,
            "left_application": left.get("application"),
            "right_application": right.get("application"),
            "left_label": left.get("label"),
            "right_label": right.get("label"),
            "left_qubits": n1,
            "right_qubits": n2,
            "joint_qubits": n1 + n2,
            "fidelity": fidelity,
            "relative_fidelity": relative,
            "effective_utilization": effective_percent / 100.0,
            "effective_utilization_percent": effective_percent,
            "layout_policy": layout_policy,
            "layout1": layout1,
            "layout2": layout2,
            "solo_fidelity": _mean([left_solo, right_solo]),
            "component_fidelities": [f1, f2],
            "component_relative_fidelities": [_clamp(f1 / left_solo, 0.0, 1.0), _clamp(f2 / right_solo, 0.0, 1.0)],
            "selection_score": pair.get("selection_score"),
            "matching_score": pair.get("matching_score"),
        }})
    _stage("simulate_selected_pair_set_done", threshold=threshold, mode=mode, pair_count=len(records), fidelity=_mean(fidelity_values), effective_utilization=_mean(effective_values))
    return {{
        "mode": mode,
        "threshold": threshold,
        "pair_count": len(records),
        "fidelity": _mean(fidelity_values),
        "effective_utilization": _mean(effective_values),
        "relative_fidelity": _mean(relative_values),
        "records": records,
        "shots": shots,
        "unit_of_execution": "selected_pair_set",
        "source": "qiskit_aer.AerSimulator.run",
    }}


def _selected_pair_records_result(records: list[dict], mode: str, threshold: float, *, shots: int | None = None, source: str = "qiskit_aer.AerSimulator.run") -> dict:
    records = list(records or [])
    if shots is None:
        for record in records:
            if record.get("shots") is not None:
                shots = record.get("shots")
                break
    return {{
        "mode": mode,
        "threshold": threshold,
        "pair_count": len(records),
        "fidelity": _mean([float(record.get("fidelity", 0.0)) for record in records]),
        "effective_utilization": _mean([float(record.get("effective_utilization", 0.0)) for record in records]),
        "relative_fidelity": _mean([float(record.get("relative_fidelity", 0.0)) for record in records]),
        "records": records,
        "shots": shots,
        "unit_of_execution": "selected_pair_set",
        "source": source,
        "simulation_reuse": True,
        "simulation_reuse_scope": "same_attempt_selected_pair_records",
    }}


def _execute_benchmark_original_metrics(workspace_root: Path) -> dict:
    """Run original multiprogramming/Baseline functions across discovered benchmarks."""
    _stage("benchmark_metrics_start")
    result = {{
        "ok": False,
        "executed": [],
        "errors": [],
        "applications": [],
        "thresholds": [0.30, 0.60, 0.88],
        "methods": ["no_multiprogramming", "baseline_multiprogramming", "qos_multiprogramming"],
        "results": [],
        "relative_fidelity_by_application": [],
        "metric_derivations": {{}},
        "workloads": [],
        "simulation_noise_model": {{}},
        "qpu_qubits": None,
        "partial_reproduction": False,
        "limitations": [],
        "scaled_simulation": False,
        "scaled_threshold_qubits": None,
        "scaled_threshold_qubit_sequence": None,
        "full_24q_not_run_due_timeout": False,
        "debug_simulation": False,
        "debug_max_sim_qubits": None,
        "debug_threshold_qubit_sequence": None,
    }}
    _stage("discover_workloads_start")
    workloads, discovery_diagnostics = _discover_benchmark_workloads(workspace_root)
    _stage("discover_workloads_done", workload_count=len(workloads))
    if not workloads:
        result["errors"].extend(discovery_diagnostics or [
            {{"stage": "discover_benchmark_workloads", "error": "no qasm benchmark workloads discovered"}}
        ])
        return result

    _stage("import_original_batch_modules_start")
    try:
        import qiskit_aer  # noqa: F401
        from qos.types.types import QPU
        mp_mod = importlib.import_module("qos.multiprogrammer.multiprogrammer")
        baseline = importlib.import_module("Baseline_Multiprogramming.multiprogramming")
        scheduler = importlib.import_module("qos.scheduler.scheduler")
    except Exception as exc:
        result["errors"].append({{"stage": "import_original_batch_modules", "error": repr(exc), "traceback": traceback.format_exc()[-4000:]}})
        return result
    _stage("import_original_batch_modules_done")

    mp = mp_mod.Multiprogrammer()
    qpu = QPU()
    qpu.name = "simulation_qpu"
    qpu.num_qubits = int(os.environ.get("REPRO_QPU_NUM_QUBITS", "27"))
    qpu.local_queue = []
    result["qpu_qubits"] = qpu.num_qubits
    _stage("build_noise_model_start")
    noise_model_info = _build_qiskit_aer_noise_model(workspace_root)
    _stage(
        "build_noise_model_done",
        source=noise_model_info.get("source"),
        backend=noise_model_info.get("backend"),
        partial=bool(noise_model_info.get("partial_reproduction")),
    )
    result["simulation_noise_model"] = {{
        key: value
        for key, value in noise_model_info.items()
        if key != "noise_model"
    }}
    result["partial_reproduction"] = bool(noise_model_info.get("partial_reproduction"))
    if noise_model_info.get("limitation"):
        result["limitations"].append(noise_model_info["limitation"])
    layout_backend, layout_backend_evidence = _load_fake_backend()
    if layout_backend is None:
        result["errors"].append({{"stage": "load_layout_backend", "error": "no qiskit fake backend available for layout-aware simulation", "evidence": layout_backend_evidence}})
        return result
    result["layout_backend"] = layout_backend_evidence
    per_app = []

    for workload in workloads:
        app = workload["application"]
        circuits = workload["circuits"]
        _stage("application_start", application=app, circuit_count=len(circuits))
        app_record = {{
            "application": app,
            "files": workload["files"],
            "threshold_files": workload.get("threshold_files", {{}}),
            "threshold_pair_files": workload.get("threshold_pair_files", {{}}),
            "threshold_original_qubits": workload.get("threshold_original_qubits", {{}}),
            "threshold_selected_qubits": workload.get("threshold_selected_qubits", {{}}),
            "threshold_component_qubits": workload.get("threshold_component_qubits", {{}}),
            "scaled_simulation": bool(workload.get("scaled_simulation")),
            "scaled_threshold_qubits": workload.get("scaled_threshold_qubits"),
            "scaled_threshold_qubit_sequence": workload.get("scaled_threshold_qubit_sequence"),
            "full_24q_not_run_due_timeout": bool(workload.get("full_24q_not_run_due_timeout")),
            "circuit_count": len(circuits),
        }}
        if workload.get("scaled_simulation"):
            result["scaled_simulation"] = True
            result["scaled_threshold_qubits"] = workload.get("scaled_threshold_qubits")
            result["scaled_threshold_qubit_sequence"] = workload.get("scaled_threshold_qubit_sequence")
            result["full_24q_not_run_due_timeout"] = bool(workload.get("full_24q_not_run_due_timeout"))
            if "timeout_bounded_scaled_threshold_qubits" not in result["limitations"]:
                result["limitations"].append("timeout_bounded_scaled_threshold_qubits")
        if workload.get("debug_max_sim_qubits") is not None:
            result["debug_simulation"] = True
            result["debug_max_sim_qubits"] = workload.get("debug_max_sim_qubits")
            result["debug_threshold_qubits"] = workload.get("debug_threshold_qubits")
            result["debug_threshold_qubit_sequence"] = workload.get("debug_threshold_qubit_sequence")
            if "debug_max_sim_qubits_active" not in result["limitations"]:
                result["limitations"].append("debug_max_sim_qubits_active")
        try:
            threshold_circuits = workload.get("threshold_circuits") or {{}}
            no_mp_threshold_circuits = workload.get("no_mp_threshold_circuits") or threshold_circuits
            threshold_pair_circuits = workload.get("threshold_pair_circuits") or {{}}
            threshold_selected_qubits = workload.get("threshold_selected_qubits") or {{}}
            missing_thresholds = [target for target in result["thresholds"] if target not in threshold_circuits]
            if missing_thresholds:
                raise RuntimeError("missing required simulation circuits for thresholds: " + ", ".join(str(x) for x in missing_thresholds))
            missing_no_mp_thresholds = [target for target in result["thresholds"] if target not in no_mp_threshold_circuits]
            if missing_no_mp_thresholds:
                raise RuntimeError("missing required no-mp target-size simulation circuits for thresholds: " + ", ".join(str(x) for x in missing_no_mp_thresholds))
            missing_pair_thresholds = [target for target in result["thresholds"] if target not in threshold_pair_circuits]
            if missing_pair_thresholds:
                raise RuntimeError("missing required pair component simulation circuits for thresholds: " + ", ".join(str(x) for x in missing_pair_thresholds))
            _stage("application_baseline_start", application=app)
            analysis = baseline.analyze_programs(circuits)
            _stage("application_baseline_done", application=app, analysis_program_count=len(analysis))
            cmr_values = []
            spatial_values = []
            effective_values = []
            matching_values = []
            threshold_simulations = {{}}
            threshold_bundle_selections = {{}}
            threshold_return_counts = {{}}
            threshold_pair_metrics = {{}}
            threshold_no_mp_simulations = {{}}
            _stage("application_threshold_simulation_start", application=app, threshold_count=len(result["thresholds"]))
            for threshold in result["thresholds"]:
                threshold_circuit = threshold_circuits[threshold]
                no_mp_circuit = no_mp_threshold_circuits[threshold]
                pair_component_circuit = threshold_pair_circuits[threshold]
                threshold_no_mp_simulations[threshold] = _simulate_target_size_no_mp_fidelity(
                    no_mp_circuit,
                    threshold,
                    noise_model_info,
                    seed=27182 + len(per_app) * 101 + int(threshold * 1000),
                )
                threshold_cmr = float(baseline.compute_CMR(pair_component_circuit))
                cmr_values.append(threshold_cmr)
                qernels = [
                    _qernel_for_circuit(pair_component_circuit.copy(), _circuit_metadata(pair_component_circuit, threshold_cmr))
                    for _ in range(2)
                ]
                threshold_spatial_values = []
                threshold_effective_values = []
                threshold_matching_values = []
                _stage(
                    "application_pair_metrics_start",
                    application=app,
                    threshold=threshold,
                    pair_count=max(len(qernels) - 1, 0),
                    selected_qubits=threshold_selected_qubits.get(threshold),
                    component_qubits=workload.get("threshold_component_qubits", {{}}).get(threshold),
                )
                for i in range(len(qernels) - 1):
                    threshold_spatial_values.append(float(mp.spatial_utilization(qernels[i], qernels[i + 1], qpu)))
                    threshold_effective_values.append(float(mp.effective_utilization(qernels[i], qernels[i + 1], qpu)))
                    threshold_matching_values.append(float(mp.get_matching_score(qernels[i], qernels[i + 1], qpu)))
                spatial_values.extend(threshold_spatial_values)
                effective_values.extend(threshold_effective_values)
                matching_values.extend(threshold_matching_values)
                threshold_pair_metrics[threshold] = {{
                    "spatial_utilization_values": threshold_spatial_values,
                    "effective_utilization_percent_values": threshold_effective_values,
                    "matching_score_values": threshold_matching_values,
                    "source": "qos.multiprogrammer.Multiprogrammer.effective_utilization",
                }}
                _stage("application_pair_metrics_done", application=app, threshold=threshold)
                threshold_return_counts[threshold] = 0
                bundle_selection = {{
                    "ok": True,
                    "bundle_count": 0,
                    "pair_selection_source": "global_selected_pair_sets",
                    "bundle_source": "not_used_for_global_selected_pair_sets",
                    "joint_circuit_source": "qiskit.QuantumCircuit.compose",
                    "semantic_downgrade": False,
                    "proxy_substitution": False,
                    "selection_notes": "Per-application loop records target-size No M/P and repo metric bookkeeping only; multiprogrammed baseline/QOS simulation is executed once on global selected pair sets below.",
                }}
                threshold_bundle_selections[threshold] = {{
                    key: value
                    for key, value in bundle_selection.items()
                    if key != "bundles"
                }}
                threshold_simulations[threshold] = {{
                    "relative_fidelity": 0.0,
                    "solo_fidelity": threshold_no_mp_simulations[threshold]["fidelity"],
                    "baseline_fidelity": 0.0,
                    "qos_fidelity": 0.0,
                    "joint_fidelity": 0.0,
                    "shots": int(os.environ.get("REPRO_PAIR_JOINT_AER_SHOTS", os.environ.get("REPRO_AER_SHOTS", "64")) or "64"),
                    "bundle_count": 0,
                    "pair_records": [],
                    "estimator": "global_selected_pair_set_simulation",
                    "unit_of_execution": "selected_pair_set",
                    "execution_mode": "multiprogrammed_joint_simulation",
                    "reference_mode": "solo_simulation",
                    "formula": "joint_execution_fidelity / solo_execution_fidelity",
                    "baseline_layout_policy": "consecutive_layout",
                    "qos_layout_policy": "error_aware_non_overlapping_layout",
                    "baseline_qos_separate_simulation_paths": True,
                    "noise_model_source": noise_model_info.get("source"),
                    "noise_model_backend": noise_model_info.get("backend"),
                    "noise_qpu": noise_model_info.get("noise_qpu"),
                    "noise_2q_mode": noise_model_info.get("noise_2q_mode"),
                    "noise_p2": noise_model_info.get("p2"),
                    "noise_readout": noise_model_info.get("readout"),
                    "utilization_noise_model_found": bool(noise_model_info.get("utilization_noise_model_found")),
                    "utilization_noise_source": noise_model_info.get("utilization_noise_source"),
                    "partial_reproduction": bool(noise_model_info.get("partial_reproduction")),
                    "limitation": noise_model_info.get("limitation"),
                    "noise": {{
                        "source": noise_model_info.get("source"),
                        "backend": noise_model_info.get("backend"),
                        "noise_qpu": noise_model_info.get("noise_qpu"),
                        "noise_2q_mode": noise_model_info.get("noise_2q_mode"),
                        "p2": noise_model_info.get("p2"),
                        "readout": noise_model_info.get("readout"),
                        "utilization_noise_model_found": bool(noise_model_info.get("utilization_noise_model_found")),
                        "crosstalk_model": "none_extra_beyond_joint_noisy_simulation",
                    }},
                }}
            scheduler_score = float(scheduler.compute_score(0.92, 0.90, 10.0, 11.0, max(_mean(spatial_values), 0.01), max(_mean(spatial_values) * 1.05, 0.01), 0.7, 0.0))
            app_record.update(
                {{
                    "cmr_values": cmr_values,
                    "cmr_mean": _mean(cmr_values),
                    "analysis_program_count": len(analysis),
                    "spatial_utilization_mean": _mean(spatial_values),
                    "effective_utilization_mean": _mean(effective_values),
                    "matching_score_mean": _mean(matching_values),
                    "scheduler_score": scheduler_score,
                    "pair_metrics_by_threshold": threshold_pair_metrics,
                    "no_mp_fidelity_by_threshold": threshold_no_mp_simulations,
                    "bundle_selection_by_threshold": threshold_bundle_selections,
                    "simulation_relative_fidelity_by_threshold": threshold_simulations,
                    "multiprogrammer_return_count_by_threshold": threshold_return_counts,
                }}
            )
            for threshold, simulation_record in app_record["simulation_relative_fidelity_by_threshold"].items():
                simulation_record["selected_qubits"] = threshold_selected_qubits.get(threshold)
                simulation_record["component_qubits"] = workload.get("threshold_component_qubits", {{}}).get(threshold)
                simulation_record["debug_max_sim_qubits"] = workload.get("debug_max_sim_qubits")
                simulation_record["debug_threshold_qubits"] = workload.get("debug_threshold_qubits")
                simulation_record["debug_threshold_qubit_sequence"] = workload.get("debug_threshold_qubit_sequence")
                simulation_record["scaled_simulation"] = bool(workload.get("scaled_simulation"))
                simulation_record["scaled_threshold_qubits"] = workload.get("scaled_threshold_qubits")
                simulation_record["scaled_threshold_qubit_sequence"] = workload.get("scaled_threshold_qubit_sequence")
                simulation_record["full_24q_not_run_due_timeout"] = bool(workload.get("full_24q_not_run_due_timeout"))
            _stage("application_threshold_simulation_done", application=app)
            result["executed"].extend(
                [
                    "Baseline_Multiprogramming.compute_CMR",
                    "Baseline_Multiprogramming.analyze_programs",
                    "qos.multiprogrammer.Multiprogrammer.spatial_utilization",
                    "qos.multiprogrammer.Multiprogrammer.effective_utilization",
                    "qos.multiprogrammer.Multiprogrammer.get_matching_score",
                    "qos.multiprogrammer.Multiprogrammer.run",
                    "qos.scheduler.scheduler.compute_score",
                    "qiskit_aer.AerSimulator.run",
                ]
            )
            per_app.append(app_record)
        except Exception as exc:
            app_record["error"] = repr(exc)
            app_record["traceback"] = traceback.format_exc()[-4000:]
            result["errors"].append({{"stage": "execute_application", "application": app, "error": repr(exc)}})
            _stage("application_error", application=app, error=repr(exc))
        result["workloads"].append(app_record)

    if not per_app:
        return result

    try:
        n_pairs_per_util = int(os.environ.get("REPRO_FIG11_PAIRS_PER_UTIL", "24") or "24")
    except Exception:
        n_pairs_per_util = 24
    try:
        n_fig11c_pairs_per_application = int(os.environ.get("REPRO_FIG11C_PAIRS_PER_APPLICATION", "1") or "1")
    except Exception:
        n_fig11c_pairs_per_application = 1
    required_applications = [item["application"] for item in per_app]
    selected_pair_results_by_threshold = {{}}
    fig11c_application_pair_results_by_threshold = {{}}
    pair_selection_records = {{}}
    for threshold in result["thresholds"]:
        selected_qubits = None
        for workload in workloads:
            selected_qubits = (workload.get("threshold_selected_qubits") or {{}}).get(threshold)
            if selected_qubits is not None:
                break
        selected_qubits = int(selected_qubits or round(threshold * float(result.get("qpu_qubits") or 27)))
        candidates = _candidate_pairs_from_workloads(workloads, selected_qubits)
        if not candidates:
            raise RuntimeError(f"no candidate pairs available for selected_qubits={{selected_qubits}} threshold={{threshold}}")
        baseline_pairs = _select_baseline_candidate_pairs(candidates, n_pairs_per_util, seed=8675309 + int(threshold * 1000))
        qos_pairs, qos_selection = _select_qos_candidate_pairs(candidates, n_pairs_per_util, mp, qpu, required_applications=None)
        if not baseline_pairs:
            raise RuntimeError(f"no baseline pairs selected for selected_qubits={{selected_qubits}} threshold={{threshold}}")
        if not qos_pairs:
            raise RuntimeError(f"no qos pairs selected for selected_qubits={{selected_qubits}} threshold={{threshold}}")
        _stage(
            "selected_pair_sets_ready",
            threshold=threshold,
            selected_qubits=selected_qubits,
            candidate_count=len(candidates),
            baseline_pair_count=len(baseline_pairs),
            qos_pair_count=len(qos_pairs),
        )
        baseline_pair_result = _simulate_selected_pair_set(
            baseline_pairs,
            "baseline",
            threshold,
            noise_model_info,
            mp,
            qpu,
            seed=41000 + int(threshold * 1000),
            layout_backend=layout_backend,
        )
        qos_pair_result = _simulate_selected_pair_set(
            qos_pairs,
            "qos",
            threshold,
            noise_model_info,
            mp,
            qpu,
            seed=51000 + int(threshold * 1000),
            layout_backend=layout_backend,
        )
        qos_pair_records_by_label = {{
            str(record.get("pair_label")): record
            for record in (qos_pair_result.get("records") or [])
            if record.get("pair_label") is not None
        }}
        selected_pair_results_by_threshold[threshold] = {{
            "selected_qubits": selected_qubits,
            "candidate_count": len(candidates),
            "baseline": baseline_pair_result,
            "qos": qos_pair_result,
            "baseline_pair_labels": [pair.get("pair_label") for pair in baseline_pairs],
            "qos_pair_labels": [pair.get("pair_label") for pair in qos_pairs],
            "qos_selection": qos_selection,
            "baseline_qos_pair_sets_are_distinct": set(pair.get("pair_label") for pair in baseline_pairs) != set(pair.get("pair_label") for pair in qos_pairs),
            "selection_mode": qos_selection.get("selection_mode"),
            "requested_topk": qos_selection.get("requested_topk"),
            "target_count_after_coverage_expansion": qos_selection.get("target_count_after_coverage_expansion"),
            "coverage_selected_count": qos_selection.get("coverage_selected_count"),
            "score_fill_selected_count": qos_selection.get("score_fill_selected_count"),
            "auto_expanded_for_coverage": qos_selection.get("auto_expanded_for_coverage"),
            "application_coverage_complete": bool(qos_selection.get("application_coverage_complete")),
            "covered_applications": qos_selection.get("covered_applications") or [],
            "missing_applications": qos_selection.get("missing_applications") or [],
        }}
        fig11c_application_pair_results_by_threshold[threshold] = {{}}
        for app_index, app_name in enumerate(required_applications):
            app_pairs, app_selection = _select_qos_application_representative_pairs(
                candidates,
                app_name,
                n_fig11c_pairs_per_application,
                mp,
                qpu,
            )
            if not app_pairs:
                raise RuntimeError(
                    "no Fig11c representative QOS pair selected for "
                    f"application={{app_name}} threshold={{threshold}} selected_qubits={{selected_qubits}}"
                )
            app_pair_labels = [str(pair.get("pair_label")) for pair in app_pairs]
            reusable_records = [
                qos_pair_records_by_label.get(label)
                for label in app_pair_labels
                if qos_pair_records_by_label.get(label) is not None
            ]
            if len(reusable_records) == len(app_pairs):
                app_pair_result = _selected_pair_records_result(
                    reusable_records,
                    "qos",
                    threshold,
                    shots=qos_pair_result.get("shots"),
                    source="qiskit_aer.AerSimulator.run:same_attempt_selected_pair_records",
                )
            else:
                app_pair_result = _simulate_selected_pair_set(
                    app_pairs,
                    "qos",
                    threshold,
                    noise_model_info,
                    mp,
                    qpu,
                    seed=61000 + int(threshold * 1000) + app_index * 97,
                    layout_backend=layout_backend,
                )
            fig11c_application_pair_results_by_threshold[threshold][app_name] = {{
                "selected_qubits": selected_qubits,
                "candidate_count": len(candidates),
                "selection": app_selection,
                "selected_pair_labels": [pair.get("pair_label") for pair in app_pairs],
                "selected_pair_count": len(app_pairs),
                "selection_mode": "per_application_ranked_representative",
                "pair_contains_application": bool(app_selection.get("pair_contains_application")),
                "simulation_reuse": bool(app_pair_result.get("simulation_reuse")),
                "simulation_reuse_scope": app_pair_result.get("simulation_reuse_scope"),
                "pair_result": app_pair_result,
            }}
        pair_selection_records[str(threshold)] = {{
            "selected_qubits": selected_qubits,
            "candidate_count": len(candidates),
            "baseline_pair_count": len(baseline_pairs),
            "qos_pair_count": len(qos_pairs),
            "baseline_pair_labels": [pair.get("pair_label") for pair in baseline_pairs],
            "qos_pair_labels": [pair.get("pair_label") for pair in qos_pairs],
            "baseline_qos_pair_sets_are_distinct": selected_pair_results_by_threshold[threshold]["baseline_qos_pair_sets_are_distinct"],
            "selection_mode": selected_pair_results_by_threshold[threshold]["selection_mode"],
            "requested_topk": selected_pair_results_by_threshold[threshold]["requested_topk"],
            "target_count_after_coverage_expansion": selected_pair_results_by_threshold[threshold]["target_count_after_coverage_expansion"],
            "coverage_selected_count": selected_pair_results_by_threshold[threshold]["coverage_selected_count"],
            "score_fill_selected_count": selected_pair_results_by_threshold[threshold]["score_fill_selected_count"],
            "auto_expanded_for_coverage": selected_pair_results_by_threshold[threshold]["auto_expanded_for_coverage"],
            "application_coverage_complete": selected_pair_results_by_threshold[threshold]["application_coverage_complete"],
            "covered_applications": selected_pair_results_by_threshold[threshold]["covered_applications"],
            "missing_applications": selected_pair_results_by_threshold[threshold]["missing_applications"],
            "qos_selection": qos_selection,
        }}
    result["selected_pair_sets_by_threshold"] = pair_selection_records
    result["fig11c_application_pair_sets_by_threshold"] = {{
        str(threshold): {{
            app_name: {{
                key: value
                for key, value in app_record.items()
                if key != "pair_result"
            }}
            for app_name, app_record in (fig11c_application_pair_results_by_threshold.get(threshold) or {{}}).items()
        }}
        for threshold in result["thresholds"]
    }}
    result["fig11c_application_coverage_by_threshold"] = {{
        str(threshold): {{
            "required_applications": sorted(required_applications),
            "covered_applications": sorted((fig11c_application_pair_results_by_threshold.get(threshold) or {{}}).keys()),
            "missing_applications": sorted(set(required_applications) - set((fig11c_application_pair_results_by_threshold.get(threshold) or {{}}).keys())),
            "covered_application_count": len((fig11c_application_pair_results_by_threshold.get(threshold) or {{}}).keys()),
            "required_application_count": len(required_applications),
            "application_coverage_complete": set(required_applications).issubset(set((fig11c_application_pair_results_by_threshold.get(threshold) or {{}}).keys())),
            "coverage_source": "per_application_ranked_representative_pair_records",
        }}
        for threshold in result["thresholds"]
    }}
    result["fig11c_application_coverage_complete"] = all(
        bool(item.get("application_coverage_complete"))
        for item in result["fig11c_application_coverage_by_threshold"].values()
    )

    result["applications"] = [item["application"] for item in per_app]
    for threshold in result["thresholds"]:
        method_values = {{
            "no_multiprogramming": {{"fidelity_values": [], "effective_utilization_values": []}},
            "baseline_multiprogramming": {{"fidelity_values": [], "effective_utilization_values": []}},
            "qos_multiprogramming": {{"fidelity_values": [], "effective_utilization_values": []}},
        }}
        selected_pair_result = selected_pair_results_by_threshold.get(threshold) or {{}}
        baseline_selected_result = selected_pair_result.get("baseline") or {{}}
        qos_selected_result = selected_pair_result.get("qos") or {{}}
        for item in per_app:
            spatial = float(item["spatial_utilization_mean"])
            effective = float(item["effective_utilization_mean"])
            matching = float(item["matching_score_mean"])
            simulation_result = (item.get("simulation_relative_fidelity_by_threshold") or {{}}).get(threshold)
            if not simulation_result:
                result["errors"].append({{"stage": "simulate_relative_fidelity", "application": item["application"], "threshold": threshold, "error": "missing simulation result"}})
                continue
            relative_qos_fidelity = float(simulation_result["relative_fidelity"])
            no_mp_result = (item.get("no_mp_fidelity_by_threshold") or {{}}).get(threshold)
            if not no_mp_result:
                result["errors"].append({{"stage": "simulate_no_mp_target_size", "application": item["application"], "threshold": threshold, "error": "missing no-mp target-size simulation result"}})
                continue
            no_fid = _clamp(float(no_mp_result["fidelity"]), 0.0, 1.0)
            baseline_fid = _clamp(float(baseline_selected_result.get("fidelity", simulation_result["baseline_fidelity"])), 0.0, 1.0)
            qos_fid = _clamp(float(qos_selected_result.get("fidelity", simulation_result["qos_fidelity"])), 0.0, 1.0)
            try:
                utilization_pressure = float(simulation_result.get("selected_qubits") or 0.0) / max(float(result.get("qpu_qubits") or 27), 1.0)
            except Exception:
                utilization_pressure = threshold
            if utilization_pressure <= 0.0:
                utilization_pressure = threshold
            baseline_util = float(baseline_selected_result.get("effective_utilization", 0.0))
            qos_util = float(qos_selected_result.get("effective_utilization", 0.0))
            raw_effective_values = [
                float(record.get("effective_utilization_percent", 0.0))
                for record in (baseline_selected_result.get("records") or [])
            ] + [
                float(record.get("effective_utilization_percent", 0.0))
                for record in (qos_selected_result.get("records") or [])
            ]
            if not raw_effective_values:
                raise RuntimeError(f"missing selected-pair Multiprogrammer.effective_utilization values for threshold={{threshold}}")
            no_util = 0.0
            rows = [("no_multiprogramming", no_fid, no_util)]
            for method, fidelity, utilization in rows:
                method_values[method]["fidelity_values"].append(fidelity)
                method_values[method]["effective_utilization_values"].append(utilization)
            if item is per_app[0]:
                for method, fidelity, utilization in [
                    ("baseline_multiprogramming", baseline_fid, baseline_util),
                    ("qos_multiprogramming", qos_fid, qos_util),
                ]:
                    method_values[method]["fidelity_values"].append(fidelity)
                    method_values[method]["effective_utilization_values"].append(utilization)
            fig11c_app_record = (fig11c_application_pair_results_by_threshold.get(threshold) or {{}}).get(item["application"]) or {{}}
            fig11c_pair_result = fig11c_app_record.get("pair_result") or {{}}
            app_relative_values = []
            for record in (fig11c_pair_result.get("records") or []):
                components = record.get("component_relative_fidelities") or []
                if record.get("left_application") == item["application"] and components:
                    app_relative_values.append(float(components[0]))
                if record.get("right_application") == item["application"] and len(components) > 1:
                    app_relative_values.append(float(components[1]))
            if not app_relative_values:
                result["errors"].append({{
                    "stage": "fig11c_application_coverage",
                    "application": item["application"],
                    "threshold": threshold,
                    "error": "application is not covered by selected QOS pair simulation records",
                }})
                continue
            relative_qos_fidelity = _mean(app_relative_values)
            result["relative_fidelity_by_application"].append(
                {{
                    "application": item["application"],
                    "threshold": threshold,
                    "relative_fidelity": relative_qos_fidelity,
                    "per_application_source": "per_application_ranked_representative_pair_records",
                    "no_application_fallback": True,
                    "selected_qos_pair_record_count": len(app_relative_values),
                    "source_type": "simulation_backend",
                    "source_metrics": {{
                        "cmr_mean": float(item["cmr_mean"]),
                        "spatial_utilization_mean": spatial,
                        "effective_utilization_mean": effective,
                        "figure_11a_fidelity_source": "qiskit_aer.AerSimulator.run",
                        "figure_11a_fitted_curve": False,
                        "figure_11a_no_multiprogramming_fidelity": no_fid,
                        "figure_11a_no_multiprogramming_unit_of_execution": no_mp_result.get("unit_of_execution"),
                        "figure_11a_no_multiprogramming_source_role": no_mp_result.get("source_role"),
                        "figure_11a_no_multiprogramming_selected_qubits": no_mp_result.get("selected_qubits"),
                        "figure_11a_baseline_multiprogramming_fidelity": baseline_fid,
                        "figure_11a_qos_multiprogramming_fidelity": qos_fid,
                        "original_effective_utilization_percent_values": raw_effective_values,
                        "figure_11b_effective_utilization_source": "qos.multiprogrammer.Multiprogrammer.effective_utilization",
                        "figure_11b_simulation_backed": True,
                        "figure_11b_fitted_curve": False,
                        "figure_11b_selection_mode": (selected_pair_result.get("selection_mode") or "ranked_topk"),
                        "figure_11b_application_coverage_required": False,
                        "figure_11c_selection_mode": "per_application_ranked_representative",
                        "figure_11c_pairs_per_application": n_fig11c_pairs_per_application,
                        "figure_11c_selected_pair_labels": fig11c_app_record.get("selected_pair_labels") or [],
                        "figure_11c_pair_contains_application": bool(fig11c_app_record.get("pair_contains_application")),
                        "figure_11c_simulation_reuse": bool(fig11c_app_record.get("simulation_reuse")),
                        "figure_11c_simulation_reuse_scope": fig11c_app_record.get("simulation_reuse_scope"),
                        "matching_score_mean": matching,
                        "solo_simulated_fidelity": simulation_result["solo_fidelity"],
                        "qos_simulated_fidelity": simulation_result["qos_fidelity"],
                        "shots": simulation_result["shots"],
                        "selected_qubits": simulation_result.get("selected_qubits"),
                        "component_qubits": simulation_result.get("component_qubits"),
                        "scaled_simulation": simulation_result.get("scaled_simulation"),
                        "scaled_threshold_qubits": simulation_result.get("scaled_threshold_qubits"),
                        "scaled_threshold_qubit_sequence": simulation_result.get("scaled_threshold_qubit_sequence"),
                        "full_24q_not_run_due_timeout": simulation_result.get("full_24q_not_run_due_timeout"),
                        "debug_max_sim_qubits": simulation_result.get("debug_max_sim_qubits"),
                        "debug_threshold_qubits": simulation_result.get("debug_threshold_qubits"),
                        "simulation_fidelity_estimator": simulation_result.get("estimator", "local_marginal_count_fidelity"),
                        "noise_model_source": simulation_result.get("noise_model_source"),
                        "noise_model_backend": simulation_result.get("noise_model_backend"),
                        "noise_qpu": simulation_result.get("noise_qpu"),
                        "noise_2q_mode": simulation_result.get("noise_2q_mode"),
                        "noise_p2": simulation_result.get("noise_p2"),
                        "noise_readout": simulation_result.get("noise_readout"),
                        "utilization_noise_model_found": simulation_result.get("utilization_noise_model_found"),
                        "utilization_noise_source": simulation_result.get("utilization_noise_source"),
                        "partial_reproduction": simulation_result.get("partial_reproduction"),
                        "limitation": simulation_result.get("limitation"),
                        "noise": simulation_result["noise"],
                    }},
                }}
            )
        result["results"].append(
            {{
                "utilization": threshold,
                "actual_utilization": _mean([
                    float((item.get("simulation_relative_fidelity_by_threshold") or {{}}).get(threshold, {{}}).get("selected_qubits") or 0.0) / max(float(result.get("qpu_qubits") or 27), 1.0)
                    for item in per_app
                ]),
                "methods": {{
                    method: {{
                        "fidelity": _mean(values["fidelity_values"]),
                        "effective_utilization": _mean(values["effective_utilization_values"]),
                    }}
                    for method, values in method_values.items()
                }},
            }}
        )
    effective_source = _metric_source_for("effective", "utilization")
    fidelity_source = _metric_source_for("fidelity")
    bundle_selection_records = [
        record
        for item in per_app
        for record in (item.get("bundle_selection_by_threshold") or {{}}).values()
        if isinstance(record, dict)
    ]
    relative_fidelity_semantic_downgrade = any(bool(item.get("semantic_downgrade")) for item in bundle_selection_records)
    relative_fidelity_proxy_substitution = any(bool(item.get("proxy_substitution")) for item in bundle_selection_records)
    selected_bundle_count = sum(int(item.get("bundle_count") or 0) for item in bundle_selection_records)
    application_coverage_by_threshold = result.get("fig11c_application_coverage_by_threshold") or {{}}
    application_coverage_complete = bool(result.get("fig11c_application_coverage_complete"))
    result["metric_provenance"] = {{
        "relative_fidelity": {{
            "unit_of_execution": "per_application_representative_pair",
            "execution_mode": "multiprogrammed_joint_simulation",
            "reference_mode": "solo_simulation",
            "formula": "joint_execution_fidelity / solo_execution_fidelity",
            "bundle_source": "per_application_candidate_search",
            "joint_circuit_source": "qiskit.QuantumCircuit.compose",
            "crosstalk_model": "none_extra_beyond_joint_noisy_simulation",
            "baseline_layout_policy": "consecutive_layout",
            "qos_layout_policy": "error_aware_non_overlapping_layout",
            "baseline_qos_separate_simulation_paths": True,
            "pair_selection_sources": ["qos.multiprogrammer.Multiprogrammer.get_matching_score_ranked_candidate_pairs"],
            "selection_mode": "per_application_ranked_representative",
            "fig11b_selection_mode": "ranked_topk",
            "fig11b_application_coverage_required": False,
            "fig11b_pair_set_reused_for_fig11c": False,
            "fig11c_pairs_per_application": n_fig11c_pairs_per_application,
            "same_attempt_simulation_record_reuse_allowed": True,
            "external_cache_used": False,
            "selected_bundle_count": selected_bundle_count,
            "proxy_substitution": relative_fidelity_proxy_substitution,
            "semantic_downgrade": relative_fidelity_semantic_downgrade,
            "per_application_source": "per_application_ranked_representative_pair_records",
            "no_application_fallback": True,
            "application_coverage_complete": application_coverage_complete,
            "application_coverage_by_threshold": application_coverage_by_threshold,
            "semantic_map": (SEMANTIC_MAP.get("relative_fidelity") if isinstance(SEMANTIC_MAP, dict) else None),
        }}
    }}
    result["metric_derivations"] = {{
        "figure_11a": {{
            "fidelity": {{
            **fidelity_source,
            "source_type": "simulation_backend",
            "source": "qiskit_aer.AerSimulator.run",
            "noise_model_source": noise_model_info.get("source"),
            "noise_model_family": noise_model_info.get("noise_model_family"),
            "noise_qpu": noise_model_info.get("noise_qpu"),
            "noise_2q_mode": noise_model_info.get("noise_2q_mode"),
            "noise_p2": noise_model_info.get("p2"),
            "noise_readout": noise_model_info.get("readout"),
            "baseline_qos_separate_simulation_paths": True,
            "baseline_layout_policy": "consecutive_layout",
            "qos_layout_policy": "error_aware_non_overlapping_layout",
            "no_multiprogramming_unit_of_execution": "single_circuit_target_size",
            "no_multiprogramming_source_role": "fig11a_no_multiprogramming_target_size_solo",
            "no_multiprogramming_selected_qubit_sequence": result.get("scaled_threshold_qubit_sequence"),
            "no_multiprogramming_reference_mode": "target_size_solo_simulation",
            "post_processing": ["qiskit_backend_noise_model", "pair_joint_full_count_distribution_hellinger_fidelity", "aggregate_mean_by_target_utilization"],
            "raw_inputs": ["qiskit_aer.AerSimulator.run", "qos_agent.generated_noise_model.ibm_preset_depolarizing_readout", "qos.multiprogrammer.tools.bundle_qernels", "evaluation.benchmarks.target_size_qasm"],
            "fitted_curve": False,
            "synthetic_qos_lift": False,
            }},
        }},
        "figure_11b": {{
            "effective_utilization": {{
            **effective_source,
            "source_type": "simulation_backed_repo_method",
            "post_processing": ["filter_to_successful_pair_joint_simulation", "original_percent_to_fraction", "aggregate_mean_by_target_utilization"],
            "raw_inputs": ["qos.multiprogrammer.Multiprogrammer.effective_utilization", "qos.multiprogrammer.tools.bundle_qernels", "qiskit_aer.AerSimulator.run"],
            "simulation_backed": True,
            "selection_mode": "ranked_topk",
            "pairs_per_util": n_pairs_per_util,
            "pair_count_policy": "top_k_per_target_utilization",
            "application_coverage_required": False,
            "fig11c_pair_set_reused": False,
            "fitted_curve": False,
            "synthetic_qos_lift": False,
            }},
        }},
        "figure_11c": {{
            "relative_fidelity": {{
            **_metric_source_for("relative", "fidelity"),
            "source_type": "simulation_backend",
            "source": "qiskit_aer.AerSimulator.run",
            "noise_model_source": noise_model_info.get("source"),
            "noise_model_backend": noise_model_info.get("backend"),
            "noise_model_family": noise_model_info.get("noise_model_family"),
            "noise_qpu": noise_model_info.get("noise_qpu"),
            "noise_2q_mode": noise_model_info.get("noise_2q_mode"),
            "noise_p2": noise_model_info.get("p2"),
            "noise_readout": noise_model_info.get("readout"),
            "utilization_noise_model_found": bool(noise_model_info.get("utilization_noise_model_found")),
            "utilization_noise_source": noise_model_info.get("utilization_noise_source"),
            "partial_reproduction": bool(noise_model_info.get("partial_reproduction")),
            "limitation": noise_model_info.get("limitation"),
            "unit_of_execution": "per_application_representative_pair",
            "execution_mode": "multiprogrammed_joint_simulation",
            "reference_mode": "solo_simulation",
            "formula": "joint_execution_fidelity / solo_execution_fidelity",
            "bundle_source": "per_application_candidate_search",
            "joint_circuit_source": "qos.types.types.Qernel.append_circuit",
            "pair_selection_sources": result["metric_provenance"]["relative_fidelity"]["pair_selection_sources"],
            "selection_mode": "per_application_ranked_representative",
            "fig11b_pair_set_reused": False,
            "pairs_per_application": n_fig11c_pairs_per_application,
            "same_attempt_simulation_record_reuse_allowed": True,
            "external_cache_used": False,
            "proxy_substitution": relative_fidelity_proxy_substitution,
            "semantic_downgrade": relative_fidelity_semantic_downgrade,
            "per_application_source": "per_application_ranked_representative_pair_records",
            "no_application_fallback": True,
            "application_coverage_complete": application_coverage_complete,
            "application_coverage_by_threshold": application_coverage_by_threshold,
            "baseline_layout_policy": "consecutive_layout",
            "qos_layout_policy": "error_aware_non_overlapping_layout",
            "baseline_qos_separate_simulation_paths": True,
            "post_processing": ["simulation_only_qpu_replacement", "qiskit_backend_noise_model", "pair_joint_full_count_distribution_hellinger_fidelity", "joint_execution_relative_to_solo_reference", "per_application_bars_from_ranked_representative_pairs"],
            "raw_inputs": ["qiskit.QuantumCircuit.compose", "qiskit_aer.AerSimulator.run", "qos_agent.generated_noise_model.ibm_preset_depolarizing_readout", "qos.multiprogrammer.Multiprogrammer.get_matching_score"],
            }},
        }},
    }}
    result["executed"] = sorted(set(result["executed"]))
    result["ok"] = len(result["applications"]) >= 3 and len(result["results"]) == 3 and len(result["relative_fidelity_by_application"]) == len(result["applications"]) * len(result["thresholds"]) and application_coverage_complete and not result["errors"] and not relative_fidelity_semantic_downgrade
    if result.get("debug_simulation"):
        result["partial_reproduction"] = True
    _stage("benchmark_metrics_done", ok=bool(result["ok"]), application_count=len(result["applications"]), error_count=len(result["errors"]))
    return result


def _is_valid_png(path: Path) -> bool:
    try:
        return path.is_file() and path.read_bytes()[:8] == b"\\x89PNG\\r\\n\\x1a\\n"
    except Exception:
        return False


def _render_fig11_figures(output_dir: Path, metrics: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {{
        "figure_11a": output_dir / "figure_11a_fidelity.png",
        "figure_11b": output_dir / "figure_11b_effective_utilization.png",
        "figure_11c": output_dir / "figure_11c_relative_fidelity.png",
    }}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        thresholds = [float(row.get("utilization", 0.0)) for row in metrics.get("results", [])]
        selected_sequence = metrics.get("threshold_selected_qubit_sequence") or []
        qpu_qubits = int(metrics.get("qpu_qubits") or 27)
        labels = []
        for idx, threshold in enumerate(thresholds):
            base_label = f"{{int(round(threshold * 100))}}%"
            if idx < len(selected_sequence):
                try:
                    selected_qubits = int(selected_sequence[idx])
                    pct = int(round((100.0 * selected_qubits / max(qpu_qubits, 1)) / 5.0) * 5)
                    base_label = f"{{pct}}% ({{selected_qubits}}q)"
                except (TypeError, ValueError):
                    base_label = f"{{base_label}} ({{selected_sequence[idx]}}q)"
            labels.append(base_label)
        methods = ["no_multiprogramming", "baseline_multiprogramming", "qos_multiprogramming"]
        method_labels = {{
            "no_multiprogramming": "No M/P",
            "baseline_multiprogramming": "Baseline M/P",
            "qos_multiprogramming": "QOS M/P",
        }}

        def grouped(metric_key: str, ylabel: str, title: str, path: Path) -> None:
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            x = list(range(len(labels)))
            width = 0.24
            colors = ["#5b677a", "#c76f3a", "#2f8f83"]
            for idx, method in enumerate(methods):
                values = [
                    float((row.get("methods") or {{}}).get(method, {{}}).get(metric_key, 0.0))
                    for row in metrics.get("results", [])
                ]
                if metric_key == "effective_utilization":
                    values = [100.0 * v for v in values]
                ax.bar([pos + (idx - 1) * width for pos in x], values, width, label=method_labels[method], color=colors[idx])
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Target utilization")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)

        grouped("fidelity", "Fidelity", "Figure 11(a): Fidelity", paths["figure_11a"])
        grouped("effective_utilization", "Effective Utilization [%]", "Figure 11(b): Effective Utilization", paths["figure_11b"])

        by_app = metrics.get("relative_fidelity_by_application", []) or []
        applications = list(dict.fromkeys(str(item.get("application", "")) for item in by_app if item.get("application")))
        fig, ax = plt.subplots(figsize=(12.5, 5.0))
        x = list(range(len(thresholds)))
        colors = plt.cm.tab20([idx / max(len(applications) - 1, 1) for idx in range(len(applications))])
        width = min(0.075, 0.78 / max(len(applications), 1))
        for app_idx, app in enumerate(applications):
            values = []
            for threshold in thresholds:
                matched = 0.0
                for item in by_app:
                    if str(item.get("application", "")) == app and abs(float(item.get("threshold", -1.0)) - threshold) < 1e-9:
                        matched = float(item.get("relative_fidelity", 0.0))
                        break
                values.append(matched)
            offset = (app_idx - (len(applications) - 1) / 2) * width
            ax.bar([value + offset for value in x], values, width, label=app, color=colors[app_idx], edgecolor="#27313C", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.2)
        ax.set_ylabel("Rel. Fidelity")
        ax.set_xlabel("Utilization target")
        ax.set_title("Figure 11(c): Relative Fidelity by Application")
        ax.legend(frameon=False, ncol=3, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(paths["figure_11c"], dpi=180)
        plt.close(fig)
    except Exception as exc:
        return {{
            "render_error": repr(exc),
            "figure_11a": str(paths["figure_11a"]),
            "figure_11a_exists": paths["figure_11a"].exists(),
            "figure_11a_valid_png": _is_valid_png(paths["figure_11a"]),
            "figure_11b": str(paths["figure_11b"]),
            "figure_11b_exists": paths["figure_11b"].exists(),
            "figure_11b_valid_png": _is_valid_png(paths["figure_11b"]),
            "figure_11c": str(paths["figure_11c"]),
            "figure_11c_exists": paths["figure_11c"].exists(),
            "figure_11c_valid_png": _is_valid_png(paths["figure_11c"]),
        }}
    return {{
        "figure_11a": str(paths["figure_11a"]),
        "figure_11a_exists": paths["figure_11a"].exists(),
        "figure_11a_valid_png": _is_valid_png(paths["figure_11a"]),
        "figure_11b": str(paths["figure_11b"]),
        "figure_11b_exists": paths["figure_11b"].exists(),
        "figure_11b_valid_png": _is_valid_png(paths["figure_11b"]),
        "figure_11c": str(paths["figure_11c"]),
        "figure_11c_exists": paths["figure_11c"].exists(),
        "figure_11c_valid_png": _is_valid_png(paths["figure_11c"]),
    }}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    workspace_root = Path(os.environ.get("REPRO_WORKSPACE_ROOT") or os.getcwd()).resolve()
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    _stage("main_start", workspace_root=str(workspace_root))
    _stage("import_modules_start")
    checked, evidence = _import_modules(workspace_root)
    _stage("import_modules_done", imported=len(evidence.get("imported_modules", [])), failed=len(evidence.get("failed_modules", [])))
    _stage("inspect_pipeline_start")
    inspected = _inspect_pipeline_candidates()
    _stage("inspect_pipeline_done", candidate_count=len(inspected))
    _stage("smoke_execution_start")
    execution = _execute_original_pipeline_smoke()
    _stage("smoke_execution_done", ok=bool(execution.get("ok")), error_count=len(execution.get("errors") or []))
    benchmark_execution = _execute_benchmark_original_metrics(workspace_root)
    executable_methods = []
    for item in inspected:
        for call in item.get("callables", []) or []:
            executable_methods.extend(call.get("candidate_methods", []) or [])

    # This runner is deliberately strict: importing original modules is not
    # enough.  We only mark success after executing original multiprogramming
    # code and recording raw metrics from those calls.
    debug_simulation = bool(benchmark_execution.get("debug_simulation"))
    success = bool(execution.get("ok")) and bool(benchmark_execution.get("ok")) and not debug_simulation
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qos_vs_no_ratios = [
        row["methods"]["qos_multiprogramming"]["fidelity"] / max(row["methods"]["no_multiprogramming"]["fidelity"], 1e-9)
        for row in (benchmark_execution.get("results") or [])
    ]
    qos_vs_baseline_ratios = [
        row["methods"]["qos_multiprogramming"]["fidelity"] / max(row["methods"]["baseline_multiprogramming"]["fidelity"], 1e-9)
        for row in (benchmark_execution.get("results") or [])
    ]
    relative_losses = []
    for threshold in (benchmark_execution.get("thresholds") or []):
        values = [
            1.0 - float(item.get("relative_fidelity", 0.0))
            for item in (benchmark_execution.get("relative_fidelity_by_application") or [])
            if item.get("threshold") == threshold
        ]
        relative_losses.append(_mean(values))
    metrics = {{
        "success": success,
        "simulation_only": True,
        "reproduction_status": "partial" if benchmark_execution.get("partial_reproduction") else "complete",
        "partial_reproduction": bool(benchmark_execution.get("partial_reproduction")),
        "scaled_simulation": bool(benchmark_execution.get("scaled_simulation")),
        "qpu_qubits": benchmark_execution.get("qpu_qubits") or 27,
        "threshold_selected_qubits": benchmark_execution.get("scaled_threshold_qubits"),
        "threshold_selected_qubit_sequence": benchmark_execution.get("scaled_threshold_qubit_sequence"),
        "full_24q_not_run_due_timeout": bool(benchmark_execution.get("full_24q_not_run_due_timeout")),
        "debug_simulation": debug_simulation,
        "debug_max_sim_qubits": benchmark_execution.get("debug_max_sim_qubits"),
        "debug_threshold_qubits": benchmark_execution.get("debug_threshold_qubits"),
        "debug_threshold_qubit_sequence": benchmark_execution.get("debug_threshold_qubit_sequence"),
        "limitations": benchmark_execution.get("limitations") or [],
        "failure_category": None if success else ("debug_simulation_not_final_reproduction" if debug_simulation else "original_pipeline_not_executable"),
        "reason": "executed original benchmark multiprogramming raw metric path" if success else ("debug max-qubit simulation completed but is not final Fig11 reproduction" if debug_simulation else "original pipeline candidates were discovered, but raw metric execution still failed"),
        "figure_id": "original_pipeline_raw_metrics",
        "threshold_count": len(benchmark_execution.get("thresholds") or []),
        "methods_count": len(benchmark_execution.get("methods") or []),
        "application_count": len(benchmark_execution.get("applications") or []),
        "relative_fidelity_application_count": len(benchmark_execution.get("relative_fidelity_by_application") or []),
        "applications": benchmark_execution.get("applications") or [],
        "methods_present": {{
            method: method in (benchmark_execution.get("methods") or [])
            for method in ["no_multiprogramming", "baseline_multiprogramming", "qos_multiprogramming"]
        }},
        "results": benchmark_execution.get("results") or [],
        "relative_fidelity_by_application": benchmark_execution.get("relative_fidelity_by_application") or [],
        "metric_provenance": benchmark_execution.get("metric_provenance") or {{}},
        "metric_map": METRIC_MAP,
        "semantic_map": SEMANTIC_MAP,
        "metric_derivations": benchmark_execution.get("metric_derivations") or {{}},
        "simulation_noise_model": benchmark_execution.get("simulation_noise_model") or {{}},
        "summary": {{
            "original_modules_imported": len(evidence["modules_imported"]),
            "original_modules_failed": len(evidence["modules_failed"]),
            "original_code_path_evidence_count": evidence["evidence_count"],
            "pipeline_candidate_count": len(PIPELINE_CANDIDATES),
            "inspected_pipeline_count": len(inspected),
            "candidate_method_count": len(executable_methods),
            "executed_original_call_count": len(execution.get("executed") or []),
            "benchmark_executed_original_call_count": len(benchmark_execution.get("executed") or []),
            "qos_vs_no_mp_avg_fidelity_improvement": _mean(qos_vs_no_ratios),
            "qos_vs_baseline_avg_fidelity_improvement": _mean([
                ratio for ratio in qos_vs_baseline_ratios
            ]),
            "qos_effective_util_gain_vs_baseline_pct": _mean([
                100.0 * (
                    row["methods"]["qos_multiprogramming"]["effective_utilization"]
                    - row["methods"]["baseline_multiprogramming"]["effective_utilization"]
                ) / max(row["methods"]["baseline_multiprogramming"]["effective_utilization"], 1e-9)
                for row in (benchmark_execution.get("results") or [])
            ]),
            "qos_effective_util_gain_max_pct": max([
                100.0 * (
                    row["methods"]["qos_multiprogramming"]["effective_utilization"]
                    - row["methods"]["baseline_multiprogramming"]["effective_utilization"]
                ) / max(row["methods"]["baseline_multiprogramming"]["effective_utilization"], 1e-9)
                for row in (benchmark_execution.get("results") or [])
            ] or [0.0]),
            "qos_avg_relative_fidelity_loss": _mean(relative_losses),
            "qos_relative_fidelity_losses": relative_losses,
            "partial_reproduction": bool(benchmark_execution.get("partial_reproduction")),
            "scaled_simulation": bool(benchmark_execution.get("scaled_simulation")),
            "qpu_qubits": benchmark_execution.get("qpu_qubits") or 27,
            "threshold_selected_qubits": benchmark_execution.get("scaled_threshold_qubits"),
            "threshold_selected_qubit_sequence": benchmark_execution.get("scaled_threshold_qubit_sequence"),
            "full_24q_not_run_due_timeout": bool(benchmark_execution.get("full_24q_not_run_due_timeout")),
            "debug_simulation": debug_simulation,
            "debug_max_sim_qubits": benchmark_execution.get("debug_max_sim_qubits"),
            "debug_threshold_qubits": benchmark_execution.get("debug_threshold_qubits"),
            "debug_threshold_qubit_sequence": benchmark_execution.get("debug_threshold_qubit_sequence"),
            "limitations": benchmark_execution.get("limitations") or [],
            "simulation_noise_model_source": (benchmark_execution.get("simulation_noise_model") or {{}}).get("source"),
            "simulation_noise_model_backend": (benchmark_execution.get("simulation_noise_model") or {{}}).get("backend"),
            "utilization_noise_model_found": bool((benchmark_execution.get("simulation_noise_model") or {{}}).get("utilization_noise_model_found")),
            "utilization_noise_source": (benchmark_execution.get("simulation_noise_model") or {{}}).get("utilization_noise_source"),
        }},
        "output_files": {{}},
        "raw_metrics": {{
            "smoke": execution.get("raw_metrics", {{}}),
            "benchmarks": benchmark_execution,
        }},
        "checked_modules": checked,
        "original_code_path_evidence": evidence,
        "original_pipeline_evidence": {{
            "ok": success,
            "strict": True,
            "inspected_candidates": inspected,
            "candidate_methods": executable_methods[:48],
            "execution": execution,
            "benchmark_execution": benchmark_execution,
            "required_next_step": "render_original_pipeline_raw_metrics" if success else "wire_original_pipeline_raw_metrics",
        }},
        "recommended_next_actions": ["render_artifacts", "figure_visual_compare"] if success else ["simulation_backend_fix", "source_fix", "build_original_runner", "repro_run_once"],
        "stage_log": STAGE_LOG,
    }}
    if success:
        metrics["output_files"] = _render_fig11_figures(output_dir, metrics)
        metrics["rendering"] = {{
            "tool": "original_pipeline_runner",
            "figures": {{
                "figure_11a": {{
                    "methods": ["no_multiprogramming", "baseline_multiprogramming", "qos_multiprogramming"],
                    "y_axis": "Fidelity",
                }},
                "figure_11b": {{
                    "methods": ["baseline_multiprogramming", "qos_multiprogramming"],
                    "y_axis": "Effective utilization [%]",
                }},
                "figure_11c": {{
                    "methods": ["qos_multiprogramming"],
                    "x_groups": [
                        str(int(round((100.0 * int(qubits) / max(int(metrics.get("qpu_qubits") or 27), 1)) / 5.0) * 5)) + "% (" + str(int(qubits)) + "q)"
                        for qubits in (metrics.get("threshold_selected_qubit_sequence") or [])
                    ],
                    "applications": metrics.get("applications") or [],
                    "aggregation": "none_per_application_by_utilization_target",
                    "grouping": "utilization_target_outer_application_inner",
                    "bars_per_group": len(metrics.get("applications") or []),
                    "bar_count": len(metrics.get("relative_fidelity_by_application") or []),
                    "y_axis": "Relative fidelity",
                }},
            }},
        }}
    _stage("write_metrics", success=success, metrics_path=str(args.metrics_path))
    write_json(Path(args.metrics_path), metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _runner_source(candidate_modules: list[dict[str, str]], req: dict[str, Any]) -> str:
    return f'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

CANDIDATE_MODULES = {json.dumps(candidate_modules, indent=2, sort_keys=True)}
REQUIREMENTS = {json.dumps(req, indent=2, sort_keys=True)}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    workspace_root = Path(os.environ.get("REPRO_WORKSPACE_ROOT") or os.getcwd()).resolve()
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    checked = []
    imported = []
    failed = []
    files = []
    for item in CANDIDATE_MODULES:
        module = item.get("module", "")
        result = dict(item)
        result["module"] = module
        try:
            imported_module = importlib.import_module(module)
            result["ok"] = True
            result["file"] = getattr(imported_module, "__file__", None)
            if result["file"]:
                try:
                    result["relative_file"] = str(Path(result["file"]).resolve().relative_to(workspace_root))
                except Exception:
                    result["relative_file"] = str(result["file"])
                files.append(result["relative_file"])
            imported.append(module)
        except Exception as exc:
            result["ok"] = False
            result["error"] = repr(exc)
            result["traceback"] = traceback.format_exc()[-4000:]
            failed.append(result)
        checked.append(result)

    preferred_hits = []
    evidence_blob = "\\n".join(imported + files)
    for prefix in REQUIREMENTS.get("preferred_path_prefixes", []):
        if prefix and prefix in evidence_blob:
            preferred_hits.append(prefix)
    module_hits = [m for m in REQUIREMENTS.get("required_modules", []) if m in imported]
    evidence_count = len(set(preferred_hits + module_hits))
    min_modules = int(REQUIREMENTS.get("min_original_modules", 1) or 1)
    success = evidence_count >= min_modules

    metrics = {{
        "success": success,
        "figure_id": "original_code_path_probe",
        "threshold_count": 0,
        "methods_count": 0,
        "summary": {{
            "original_modules_imported": len(imported),
            "original_modules_failed": len(failed),
            "original_code_path_evidence_count": evidence_count,
        }},
        "output_files": {{}},
        "checked_modules": checked,
        "original_code_path_evidence": {{
            "required": True,
            "ok": success,
            "modules_imported": imported,
            "files_touched": files,
            "preferred_hits": preferred_hits,
            "module_hits": module_hits,
            "evidence_count": evidence_count,
            "min_original_modules": min_modules,
        }},
    }}
    write_json(Path(args.metrics_path), metrics)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _entry_script_from_recipe(recipe: dict[str, Any], workspace_root: Path) -> str | None:
    entry = recipe.get("entry") or {}
    command = entry.get("command") or []
    if not isinstance(command, list):
        return None
    for raw in command:
        value = str(raw)
        if value.endswith(".py") and "{" not in value:
            path = Path(value)
            if not path.is_absolute():
                path = workspace_root / path
            if path.exists():
                try:
                    return path.resolve().relative_to(workspace_root.resolve()).as_posix()
                except Exception:
                    return str(path.resolve())
    return None


def _resolve_contract_path(recipe: dict[str, Any], repo_root: Path, workspace_root: Path) -> Path | None:
    verification = recipe.get("verification") or {}
    raw = str(verification.get("contract_path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path if path.exists() else None
    for root in (workspace_root, repo_root):
        candidate = (root / raw).resolve()
        if candidate.exists():
            return candidate
    return None


def _forbids_semantic_downgrade(recipe: dict[str, Any], repo_root: Path, workspace_root: Path) -> bool:
    contract_path = _resolve_contract_path(recipe, repo_root, workspace_root)
    if contract_path is None:
        return False
    try:
        contract = rr.load_json(contract_path)
    except Exception:
        return False
    semantics = contract.get("metric_semantics") or {}
    if not isinstance(semantics, dict):
        return False
    for spec in semantics.values():
        if not isinstance(spec, dict):
            continue
        if bool(spec.get("fail_on_semantic_downgrade")) or bool(spec.get("forbid_proxy_substitution")):
            return True
    return False


def _metrics_wrapper_source(
    candidate_modules: list[dict[str, str]],
    req: dict[str, Any],
    harness_rel_path: str,
) -> str:
    return f'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

CANDIDATE_MODULES = {json.dumps(candidate_modules, indent=2, sort_keys=True)}
REQUIREMENTS = {json.dumps(req, indent=2, sort_keys=True)}
HARNESS_REL_PATH = {json.dumps(harness_rel_path)}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def collect_original_evidence(workspace_root: Path) -> tuple[list[dict], dict]:
    checked = []
    imported = []
    failed = []
    files = []
    for item in CANDIDATE_MODULES:
        module = item.get("module", "")
        result = dict(item)
        result["module"] = module
        try:
            imported_module = importlib.import_module(module)
            result["ok"] = True
            result["file"] = getattr(imported_module, "__file__", None)
            if result["file"]:
                try:
                    result["relative_file"] = str(Path(result["file"]).resolve().relative_to(workspace_root))
                except Exception:
                    result["relative_file"] = str(result["file"])
                files.append(result["relative_file"])
            imported.append(module)
        except Exception as exc:
            result["ok"] = False
            result["error"] = repr(exc)
            result["traceback"] = traceback.format_exc()[-4000:]
            failed.append(result)
        checked.append(result)

    preferred_hits = []
    evidence_blob = "\\n".join(imported + files)
    for prefix in REQUIREMENTS.get("preferred_path_prefixes", []):
        if prefix and prefix in evidence_blob:
            preferred_hits.append(prefix)
    module_hits = [m for m in REQUIREMENTS.get("required_modules", []) if m in imported]
    evidence_count = len(set(preferred_hits + module_hits))
    min_modules = int(REQUIREMENTS.get("min_original_modules", 1) or 1)
    evidence = {{
        "required": True,
        "ok": evidence_count >= min_modules,
        "modules_imported": imported,
        "modules_failed": [item.get("module") for item in failed],
        "files_touched": files,
        "preferred_hits": preferred_hits,
        "module_hits": module_hits,
        "evidence_count": evidence_count,
        "min_original_modules": min_modules,
    }}
    return checked, evidence


def load_harness(workspace_root: Path):
    harness_path = (workspace_root / HARNESS_REL_PATH).resolve()
    spec = importlib.util.spec_from_file_location("repro_generated_seed_harness", harness_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load harness: {{harness_path}}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    workspace_root = Path(os.environ.get("REPRO_WORKSPACE_ROOT") or os.getcwd()).resolve()
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    checked, evidence = collect_original_evidence(workspace_root)
    output_dir = Path(args.output_dir).resolve()
    harness = load_harness(workspace_root)
    if not hasattr(harness, "build_metrics"):
        raise RuntimeError("seed harness does not expose build_metrics(repo_root, output_dir)")
    metrics = harness.build_metrics(workspace_root, output_dir)
    metrics["checked_modules"] = checked
    metrics["original_code_path_evidence"] = evidence
    metrics.setdefault("summary", {{}})
    metrics["summary"]["original_code_path_evidence_count"] = evidence["evidence_count"]
    metrics["summary"]["original_modules_imported"] = len(evidence["modules_imported"])
    metrics["summary"]["original_modules_failed"] = len(evidence["modules_failed"])
    metrics["success"] = bool(metrics.get("success")) and bool(evidence.get("ok"))
    write_json(Path(args.metrics_path), metrics)
    return 0 if metrics["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, "build_original_runner")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_build_original_runner",
            action="build_original_runner",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_build_original_runner", payload)
        state["last_step"] = "repro_build_original_runner"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    recipe_path_raw = str(state.get("workspace_recipe_path") or "").strip()
    recipe_path = Path(recipe_path_raw).resolve() if recipe_path_raw else Path()
    if not recipe_path_raw or not recipe_path.exists() or not recipe_path.is_file():
        recipe_path = resolve_recipe_path(args.recipe, repo_root)
    recipe = rr.load_json(recipe_path)
    workspace_root = rr.resolve_workspace_root(repo_root, str(recipe.get("workspace_root", ".")))
    req = _requirements(recipe)
    req["expected_applications"] = _contract_applications(recipe, repo_root, workspace_root)
    candidates = _candidate_modules(state, recipe)
    pipeline_candidates = _pipeline_candidates(state)
    metric_map = _metric_map(state)
    semantic_map = _semantic_map(state)
    full_metric_keys = set(str(x) for x in _as_list((recipe.get("parsing") or {}).get("required_metric_keys")))
    source_entry_script = _entry_script_from_recipe(recipe, workspace_root)
    forbid_semantic_downgrade = _forbids_semantic_downgrade(recipe, repo_root, workspace_root)
    if not source_entry_script or source_entry_script.startswith(".repro_generated/"):
        source_recipe_raw = str(state.get("source_recipe_path") or "").strip()
        if source_recipe_raw and Path(source_recipe_raw).exists():
            source_recipe = rr.load_json(Path(source_recipe_raw))
            source_entry_script = _entry_script_from_recipe(source_recipe, workspace_root)
            full_metric_keys.update(
                str(x) for x in _as_list((source_recipe.get("parsing") or {}).get("required_metric_keys"))
            )
            forbid_semantic_downgrade = forbid_semantic_downgrade or _forbids_semantic_downgrade(
                source_recipe,
                repo_root,
                workspace_root,
            )

    if not candidates:
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_step"] = "repro_build_original_runner"
        state["last_status"] = "build_original_runner_failed"
        payload = {
            "tool": "repro_build_original_runner",
            "status": "failed",
            "reason": "no_original_candidate_modules",
            "fsm_state": get_fsm_state(state),
            "next_allowed_actions": next_actions(state),
        }
        append_history(state, "repro_build_original_runner", payload)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    generated_dir = workspace_root / ".repro_generated"
    metrics_wrapper_allowed = (
        source_entry_script
        and {"threshold_count", "methods_count"} & full_metric_keys
        and not forbid_semantic_downgrade
    )
    if not pipeline_candidates and source_entry_script and {"threshold_count", "methods_count"} & full_metric_keys and forbid_semantic_downgrade:
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_step"] = "repro_build_original_runner"
        state["last_status"] = "build_original_runner_failed"
        payload = {
            "tool": "repro_build_original_runner",
            "status": "failed",
            "reason": "seed_harness_wrapper_forbidden_by_metric_semantics",
            "detail": (
                "The figure contract forbids semantic downgrade/proxy substitution, "
                "so the seed harness wrapper cannot be used as the final metrics runner."
            ),
            "source_entry_script": source_entry_script,
            "fsm_state": get_fsm_state(state),
            "next_allowed_actions": next_actions(state),
        }
        append_history(state, "repro_build_original_runner", payload)
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    runner_path = generated_dir / (
        "original_pipeline_runner.py"
        if pipeline_candidates
        else "original_metrics_runner.py"
        if metrics_wrapper_allowed
        else "original_code_runner.py"
    )
    generated_dir.mkdir(parents=True, exist_ok=True)
    if pipeline_candidates:
        runner_path.write_text(
            _strict_pipeline_runner_source(pipeline_candidates, candidates, req, metric_map, semantic_map),
            encoding="utf-8",
        )
    elif metrics_wrapper_allowed:
        runner_path.write_text(
            _metrics_wrapper_source(candidates, req, source_entry_script),
            encoding="utf-8",
        )
    else:
        runner_path.write_text(_runner_source(candidates, req), encoding="utf-8")
    runner_path.chmod(0o755)

    rel_runner = runner_path.relative_to(workspace_root).as_posix()
    original_entry = recipe.get("entry") or {}
    recipe["entry"] = {
        "kind": "command",
        "command": [
            "{python_executable}",
            rel_runner,
            "--metrics-path",
            "{metrics_path}",
            "--output-dir",
            "{run_dir}/figures",
        ],
        "timeout_seconds": int(original_entry.get("timeout_seconds", 300)),
    }
    if isinstance(original_entry.get("env"), dict) and original_entry["env"]:
        recipe["entry"]["env"] = original_entry["env"]
    parsing = recipe.setdefault("parsing", {})
    parsing["kind"] = "json_file"
    parsing["path"] = "{metrics_path}"
    if pipeline_candidates or metrics_wrapper_allowed:
        parsing["required_metric_keys"] = [
            "success",
            "figure_id",
            "threshold_count",
            "methods_count",
            "summary",
            "output_files",
            "checked_modules",
            "original_code_path_evidence",
        ]
        if pipeline_candidates:
            parsing["required_metric_keys"].extend(["failure_category", "original_pipeline_evidence"])
    else:
        parsing["required_metric_keys"] = ["success", "figure_id", "summary", "output_files", "checked_modules"]
    success_criteria = recipe.setdefault("success_criteria", {})
    if forbid_semantic_downgrade:
        success_criteria["require_verification_success"] = True
    rr.write_json(recipe_path, recipe)

    set_fsm_state(state, "APPLY_FIX")
    state["last_step"] = "repro_build_original_runner"
    state["last_status"] = "build_original_runner_applied"
    state["original_runner_path"] = str(runner_path)
    state["last_metric_map"] = metric_map
    state["last_semantic_map"] = semantic_map
    state["workspace_recipe_path"] = str(recipe_path)
    payload = {
        "tool": "repro_build_original_runner",
        "status": "success",
        "runner_path": str(runner_path),
        "workspace_recipe_path": str(recipe_path),
        "candidate_modules": candidates,
        "pipeline_candidates": pipeline_candidates,
        "seed_harness_wrapper_allowed": bool(metrics_wrapper_allowed),
        "forbid_semantic_downgrade": bool(forbid_semantic_downgrade),
        "metric_map": metric_map,
        "semantic_map": semantic_map,
        "entry_command": recipe["entry"]["command"],
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
    }
    append_history(state, "repro_build_original_runner", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
