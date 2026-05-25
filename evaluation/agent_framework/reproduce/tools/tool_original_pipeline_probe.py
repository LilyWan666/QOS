#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
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
    ensure_run_root,
    get_fsm_state,
    load_state,
    next_actions,
    repo_root_from_here,
    resolve_recipe_path,
    set_fsm_state,
    write_state,
)

PIPELINE_KEYWORDS = {
    "fidelity",
    "utilization",
    "multiprogram",
    "bundle",
    "circuit",
    "compose",
    "matching",
    "overlap",
    "qernel",
    "scheduler",
    "schedule",
    "backend",
    "qpu",
    "layout",
    "transpile",
    "estimate",
    "benchmark",
    "figure",
    "plot",
}
METRIC_KEYWORDS = {
    "accuracy",
    "effective",
    "fidelity",
    "latency",
    "loss",
    "relative",
    "runtime",
    "throughput",
    "utilization",
}
RISK_KEYWORDS = {
    "IBMProvider",
    "IBMQ",
    "QiskitRuntimeService",
    "qiskit_ibm_runtime",
    "provider.get_backend",
    "backend.run",
    "token",
    "QISKIT_IBM_TOKEN",
}
FRAMEWORK_PREFIX = "evaluation/agent_framework/reproduce/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument(
        "--output-root",
        default="temp/agent_framework/reproduce/tools/runs",
        help="Root directory for probe artifacts.",
    )
    return parser.parse_args()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _workspace_root(repo_root: Path, recipe: dict[str, Any]) -> Path:
    workspace = rr.resolve_workspace_root(repo_root, str(recipe.get("workspace_root", ".")))
    return workspace.resolve()


def _preferred_prefixes(recipe: dict[str, Any]) -> list[str]:
    reqs = recipe.get("reproduction_requirements") or {}
    original = reqs.get("original_code_path") or {}
    prefixes = [str(x).strip() for x in _as_list(original.get("preferred_path_prefixes")) if str(x).strip()]
    forbidden = {FRAMEWORK_PREFIX}
    return [prefix for prefix in prefixes if prefix not in forbidden and not prefix.startswith(FRAMEWORK_PREFIX)]


def _focus_terms(recipe: dict[str, Any]) -> list[str]:
    paper = recipe.get("paper") or {}
    terms = [str(term).strip().lower() for term in _as_list(paper.get("focus_terms")) if str(term).strip()]
    authoring = recipe.get("recipe_authoring") or {}
    terms.extend(
        str(term).strip().lower()
        for term in _as_list(authoring.get("required_methods"))
        if str(term).strip()
    )
    return sorted(set(terms))


def _contract_path(recipe: dict[str, Any], repo_root: Path, workspace_root: Path) -> Path | None:
    verification = recipe.get("verification") or {}
    raw_contract = str(verification.get("contract_path") or "").strip()
    if not raw_contract:
        return None
    contract_path = Path(raw_contract)
    if contract_path.is_absolute() and contract_path.exists():
        return contract_path
    for root in (workspace_root, repo_root):
        candidate = (root / raw_contract).resolve()
        if candidate.exists():
            return candidate
    return None


def _metric_terms(recipe: dict[str, Any], repo_root: Path, workspace_root: Path) -> list[str]:
    terms: set[str] = set()
    for term in _focus_terms(recipe):
        if any(keyword in term for keyword in METRIC_KEYWORDS):
            terms.add(term)

    def add_term(raw: Any) -> None:
        text = str(raw or "").strip().lower()
        if not text:
            return
        text = text.replace("[%]", "").replace("%", "").strip()
        if any(keyword in text for keyword in METRIC_KEYWORDS):
            terms.add(text)

    contract_path = _contract_path(recipe, repo_root, workspace_root)
    if contract_path:
        try:
            contract = rr.load_json(contract_path)
        except Exception:
            contract = {}
        visual = contract.get("visual_expectations") or {}
        figures = visual.get("figures") if isinstance(visual, dict) else {}
        if isinstance(figures, dict):
            for figure_id, spec in figures.items():
                if not isinstance(spec, dict):
                    continue
                add_term(figure_id)
                expected = spec.get("expected_structure") or {}
                if isinstance(expected, dict):
                    add_term(expected.get("y_axis"))
    return sorted(terms)


def _module_for_path(workspace_root: Path, path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(workspace_root)
    except ValueError:
        return None
    if rel.suffix != ".py":
        return None
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = rel.stem
    if not parts:
        return None
    if any(not part.replace("_", "a").isalnum() for part in parts):
        return None
    return ".".join(parts)


def _candidate_files(workspace_root: Path, recipe: dict[str, Any]) -> list[Path]:
    prefixes = _preferred_prefixes(recipe)
    roots: list[Path] = []
    for prefix in prefixes:
        root = workspace_root / prefix
        if root.is_file() and root.suffix == ".py":
            roots.append(root)
        elif root.is_dir():
            roots.append(root)
    if not roots:
        for child in workspace_root.iterdir():
            if child.is_dir() and not child.name.startswith(".") and child.name not in {"temp", "claw-code"}:
                roots.append(child)

    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        iterator = [root] if root.is_file() else root.rglob("*.py")
        for path in iterator:
            if any(part in {"__pycache__", ".venv", "venv", ".git"} for part in path.parts):
                continue
            try:
                rel = path.resolve().relative_to(workspace_root)
            except ValueError:
                continue
            rel_posix = rel.as_posix()
            if rel_posix.startswith(FRAMEWORK_PREFIX):
                continue
            if path not in seen:
                seen.add(path)
                files.append(path.resolve())
    return files[:250]


def _callable_signature(node: ast.AST) -> str:
    args = getattr(node, "args", None)
    if args is None:
        return "()"
    names = [arg.arg for arg in getattr(args, "args", [])]
    if getattr(args, "vararg", None):
        names.append("*" + args.vararg.arg)
    if getattr(args, "kwarg", None):
        names.append("**" + args.kwarg.arg)
    return "(" + ", ".join(names) + ")"


def _scan_file(workspace_root: Path, path: Path, focus_terms: list[str]) -> dict[str, Any] | None:
    rel = path.relative_to(workspace_root).as_posix()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"path": rel, "error": str(exc), "score": 0}

    lower = text.lower()
    score = 0
    matched_terms = sorted({term for term in focus_terms if term and (term in rel.lower() or term in lower)})
    matched_keywords = sorted({kw for kw in PIPELINE_KEYWORDS if kw in rel.lower() or kw in lower})
    score += 3 * len(matched_terms) + 2 * len(matched_keywords)
    if any(token in rel.lower() for token in ("multiprogram", "scheduler", "estimator", "backend")):
        score += 5

    result: dict[str, Any] = {
        "path": rel,
        "module": _module_for_path(workspace_root, path),
        "score": score,
        "matched_focus_terms": matched_terms,
        "matched_pipeline_keywords": matched_keywords,
        "external_risks": sorted({risk for risk in RISK_KEYWORDS if risk.lower() in lower}),
        "functions": [],
        "classes": [],
        "imports": [],
    }

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        result["syntax_error"] = str(exc)
        return result if score else None

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            body_text = ast.get_source_segment(text, node) or name
            body_lower = body_text.lower()
            local_hits = sorted({kw for kw in PIPELINE_KEYWORDS if kw in name.lower() or kw in body_lower})
            if local_hits:
                result["functions"].append({
                    "name": name,
                    "signature": _callable_signature(node),
                    "lineno": node.lineno,
                    "pipeline_keywords": local_hits,
                })
                score += len(local_hits)
        elif isinstance(node, ast.ClassDef):
            name = node.name
            methods = []
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                method_text = ast.get_source_segment(text, child) or child.name
                method_lower = method_text.lower()
                method_hits = sorted({kw for kw in PIPELINE_KEYWORDS | METRIC_KEYWORDS if kw in child.name.lower() or kw in method_lower})
                if method_hits:
                    methods.append({
                        "name": child.name,
                        "signature": _callable_signature(child),
                        "lineno": child.lineno,
                        "pipeline_keywords": method_hits,
                    })
            if any(kw in name.lower() for kw in PIPELINE_KEYWORDS) or any(
                kw in (ast.get_source_segment(text, node) or "").lower() for kw in ("fidelity", "backend", "schedule")
            ):
                result["classes"].append({"name": name, "lineno": node.lineno, "methods": methods[:24]})
                score += 2
            elif methods:
                result["classes"].append({"name": name, "lineno": node.lineno, "methods": methods[:24]})
                score += len(methods)
    result["imports"] = sorted(imports)
    result["score"] = score
    if not result["functions"] and not result["classes"] and not matched_terms and score < 6:
        return None
    return result


def _tokenize_metric(value: str) -> set[str]:
    aliases = {
        "rel": "relative",
        "util": "utilization",
        "utilisation": "utilization",
    }
    tokens = {
        aliases.get(token, token)
        for token in value.lower().replace("_", " ").replace("-", " ").split()
        if token
    }
    return {token for token in tokens if token not in {"by", "figure", "impact", "on", "of", "the", "vs", "wrt"}}


def _metric_source_candidates(metric_terms: list[str], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    metric_map: dict[str, Any] = {}
    for term in metric_terms:
        term_tokens = _tokenize_metric(term)
        if not term_tokens:
            continue
        matches: list[dict[str, Any]] = []
        for item in candidates:
            module = str(item.get("module") or "").strip()
            path = str(item.get("path") or "")
            for fn in item.get("functions") or []:
                if not isinstance(fn, dict):
                    continue
                name = str(fn.get("name") or "")
                fn_tokens = _tokenize_metric(name)
                overlap = term_tokens & fn_tokens
                if overlap:
                    matches.append({
                        "source_type": "repo_function",
                        "source": f"{module}.{name}" if module else name,
                        "path": path,
                        "lineno": fn.get("lineno"),
                        "score": len(overlap) * 10 + len(term_tokens & set(fn.get("pipeline_keywords") or [])),
                        "matched_tokens": sorted(overlap),
                    })
            for cls in item.get("classes") or []:
                if not isinstance(cls, dict):
                    continue
                class_name = str(cls.get("name") or "")
                for method in cls.get("methods") or []:
                    if not isinstance(method, dict):
                        continue
                    method_name = str(method.get("name") or "")
                    method_tokens = _tokenize_metric(method_name)
                    overlap = term_tokens & method_tokens
                    if overlap:
                        matches.append({
                            "source_type": "repo_method",
                            "source": f"{module}.{class_name}.{method_name}" if module else f"{class_name}.{method_name}",
                            "path": path,
                            "lineno": method.get("lineno"),
                            "score": len(overlap) * 10 + len(term_tokens & set(method.get("pipeline_keywords") or [])),
                            "matched_tokens": sorted(overlap),
                        })
        matches.sort(key=lambda m: (-int(m.get("score") or 0), str(m.get("source") or "")))
        best = matches[0] if matches else None
        metric_map[term] = {
            "paper_term": term,
            "selected_source": best,
            "candidates": matches[:8],
            "confidence": "high" if best and int(best.get("score") or 0) >= 20 else "medium" if best else "low",
            "required_next_action": "build_original_runner" if best else "derive_from_paper_formula",
        }
    return metric_map


def _semantic_source(
    *,
    source_type: str,
    source: str,
    path: str,
    lineno: Any,
    role: str,
    confidence: str = "high",
) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "source": source,
        "path": path,
        "lineno": lineno,
        "role": role,
        "confidence": confidence,
    }


def _semantic_map(metric_terms: list[str], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Map paper metric semantics to repo primitives, not just metric names.

    The metric map above answers "where is something named like this metric?".
    This map answers "which code primitives can generate the required data?".
    For Fig. 11(c), relative fidelity depends on selected pair/bundle execution,
    so bundle construction and pair selection are first-class semantic roles.
    """
    wants_relative_fidelity = any(
        {"relative", "fidelity"}.issubset(_tokenize_metric(term))
        for term in metric_terms
    )
    if not wants_relative_fidelity:
        return {}

    roles: dict[str, list[dict[str, Any]]] = {
        "pair_metrics": [],
        "pair_selection": [],
        "bundle_construction": [],
        "joint_circuit_construction": [],
        "fidelity_estimation": [],
    }
    for item in candidates:
        module = str(item.get("module") or "").strip()
        path = str(item.get("path") or "")
        for fn in item.get("functions") or []:
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "")
            source = f"{module}.{name}" if module else name
            lower_source = source.lower()
            lineno = fn.get("lineno")
            if name == "bundle_qernels":
                roles["bundle_construction"].append(
                    _semantic_source(
                        source_type="repo_function",
                        source=source,
                        path=path,
                        lineno=lineno,
                        role="bundle_construction",
                    )
                )
            if "fidelity" in lower_source:
                roles["fidelity_estimation"].append(
                    _semantic_source(
                        source_type="repo_function",
                        source=source,
                        path=path,
                        lineno=lineno,
                        role="fidelity_estimation",
                        confidence="medium",
                    )
                )
        for cls in item.get("classes") or []:
            if not isinstance(cls, dict):
                continue
            class_name = str(cls.get("name") or "")
            for method in cls.get("methods") or []:
                if not isinstance(method, dict):
                    continue
                method_name = str(method.get("name") or "")
                source = f"{module}.{class_name}.{method_name}" if module else f"{class_name}.{method_name}"
                lineno = method.get("lineno")
                if class_name == "Qernel" and method_name == "append_circuit":
                    roles["joint_circuit_construction"].append(
                        _semantic_source(
                            source_type="repo_method",
                            source=source,
                            path=path,
                            lineno=lineno,
                            role="joint_circuit_construction",
                        )
                    )
                if class_name == "Multiprogrammer" and method_name in {
                    "restrict_policy",
                    "re_evaluation_policy",
                    "process_qernels",
                    "run",
                }:
                    roles["pair_selection"].append(
                        _semantic_source(
                            source_type="repo_method",
                            source=source,
                            path=path,
                            lineno=lineno,
                            role="pair_selection",
                            confidence="medium" if method_name == "run" else "high",
                        )
                    )
                if class_name == "Multiprogrammer" and method_name in {
                    "spatial_utilization",
                    "effective_utilization",
                    "get_matching_score",
                }:
                    roles["pair_metrics"].append(
                        _semantic_source(
                            source_type="repo_method",
                            source=source,
                            path=path,
                            lineno=lineno,
                            role="pair_metrics",
                        )
                    )

    has_bundle = bool(roles["bundle_construction"] and roles["joint_circuit_construction"])
    has_pair_selection = bool(roles["pair_selection"])
    required_missing = [
        role
        for role, present in {
            "bundle_construction": bool(roles["bundle_construction"]),
            "joint_circuit_construction": bool(roles["joint_circuit_construction"]),
            "pair_selection": has_pair_selection,
        }.items()
        if not present
    ]
    return {
        "relative_fidelity": {
            "unit_of_execution": "pair",
            "required_execution_mode": "multiprogrammed_joint_simulation",
            "strategy": "pair_joint_simulation" if has_bundle else "unresolved",
            "roles": roles,
            "ready_for_pair_joint_runner": has_bundle and has_pair_selection,
            "required_missing_roles": required_missing,
            "required_next_action": "build_original_runner" if has_bundle else "source_fix",
        }
    }


def _summarize(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    top = candidates[:12]
    risk_files = [item for item in candidates if item.get("external_risks")]
    runnable_modules = [
        item["module"]
        for item in top
        if item.get("module") and (item.get("functions") or item.get("classes"))
    ]
    return {
        "candidate_count": len(candidates),
        "top_paths": [item["path"] for item in top],
        "runnable_module_candidates": runnable_modules[:12],
        "qpu_or_external_risk_paths": [item["path"] for item in risk_files[:12]],
        "recommended_next_actions": (
            ["simulation_backend_fix", "build_original_runner"]
            if risk_files
            else ["build_original_runner"]
        ),
    }


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, "original_pipeline_probe")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_original_pipeline_probe",
            action="original_pipeline_probe",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_original_pipeline_probe", payload)
        state["last_step"] = "repro_original_pipeline_probe"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    recipe_path_raw = str(state.get("workspace_recipe_path") or "").strip()
    recipe_path = Path(recipe_path_raw).resolve() if recipe_path_raw else Path()
    if not recipe_path_raw or not recipe_path.exists() or not recipe_path.is_file():
        recipe_path = resolve_recipe_path(args.recipe, repo_root)
    recipe = rr.load_json(recipe_path)
    workspace_root = _workspace_root(repo_root, recipe)
    focus_terms = _focus_terms(recipe)
    metric_terms = _metric_terms(recipe, repo_root, workspace_root)

    candidates = []
    for path in _candidate_files(workspace_root, recipe):
        item = _scan_file(workspace_root, path, focus_terms)
        if item is not None:
            candidates.append(item)
    candidates.sort(key=lambda item: (-int(item.get("score") or 0), str(item.get("path") or "")))

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    run_root = ensure_run_root(str(recipe.get("name") or recipe_path.stem), state, output_root)
    probe_dir = run_root / "original_pipeline_probe"
    manifest_path = probe_dir / "original_pipeline_manifest.json"
    summary = _summarize(candidates)
    metric_map = _metric_source_candidates(metric_terms, candidates)
    semantic_map = _semantic_map(metric_terms, candidates)
    manifest = {
        "success": bool(candidates),
        "tool": "repro_original_pipeline_probe",
        "workspace_root": str(workspace_root),
        "focus_terms": focus_terms,
        "metric_terms": metric_terms,
        "metric_map": metric_map,
        "semantic_map": semantic_map,
        "summary": summary,
        "candidates": candidates[:80],
    }
    _write_json(manifest_path, manifest)

    set_fsm_state(state, "CLASSIFY_FAILURE" if candidates else "TERMINAL_FAILED")
    state["last_step"] = "repro_original_pipeline_probe"
    state["last_status"] = "original_pipeline_probe_completed" if candidates else "original_pipeline_probe_failed"
    state["last_original_pipeline_probe"] = manifest
    state["last_metric_map"] = metric_map
    state["last_semantic_map"] = semantic_map
    state["original_pipeline_manifest_path"] = str(manifest_path)
    state["last_failure_classification"] = {
        "category": "original_pipeline_not_built" if candidates else "original_pipeline_not_found",
        "recoverable": bool(candidates),
        "recommended_actions": summary["recommended_next_actions"] if candidates else ["terminal_failed"],
    }
    payload = {
        "tool": "repro_original_pipeline_probe",
        "status": "success" if candidates else "failed",
        "manifest_path": str(manifest_path),
        "summary": summary,
        "metric_map": metric_map,
        "semantic_map": semantic_map,
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
    }
    append_history(state, "repro_original_pipeline_probe", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
