#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import textwrap
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


ACTION = "openevolve_target_probe"
TOOL_NAME = "repro_openevolve_target_probe"
OPENEVOLVE_REL_ROOT = Path("third_party/openevolve")


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(repo: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:
        return f"git_error:{type(exc).__name__}:{exc}"


def _parse_python(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _functions_in(path: Path) -> dict[str, dict[str, Any]]:
    tree = _parse_python(path)
    if tree is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = [arg.arg for arg in [*node.args.posonlyargs, *node.args.args]]
        if node.args.vararg:
            args.append("*" + node.args.vararg.arg)
        args.extend(arg.arg for arg in node.args.kwonlyargs)
        if node.args.kwarg:
            args.append("**" + node.args.kwarg.arg)
        out[node.name] = {
            "name": node.name,
            "kind": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
            "line": node.lineno,
            "args": args,
            "doc": ast.get_docstring(node) or "",
        }
    return out


def _signature_from_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parts = []
    defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults)
    for arg, default in zip(node.args.args, defaults):
        text = arg.arg
        if default is not None:
            try:
                text += "=" + ast.unparse(default)
            except Exception:
                text += "=..."
        parts.append(text)
    if node.args.vararg:
        parts.append("*" + node.args.vararg.arg)
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        text = arg.arg
        if default is not None:
            try:
                text += "=" + ast.unparse(default)
            except Exception:
                text += "=..."
        parts.append(text)
    if node.args.kwarg:
        parts.append("**" + node.args.kwarg.arg)
    return f"{node.name}({', '.join(parts)})"


def _class_methods(path: Path, class_name: str) -> dict[str, dict[str, Any]]:
    tree = _parse_python(path)
    if tree is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            out[child.name] = {
                "class": class_name,
                "name": child.name,
                "kind": "async_method" if isinstance(child, ast.AsyncFunctionDef) else "method",
                "line": child.lineno,
                "end_line": getattr(child, "end_lineno", None),
                "signature": _signature_from_args(child),
                "doc": ast.get_docstring(child) or "",
            }
    return out


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                segment = ast.get_source_segment(source, child) or ""
                return textwrap.dedent(segment).rstrip() + "\n"
    return ""


def _function_source(path: Path, function_name: str, class_name: str | None = None) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    if class_name:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == function_name:
                    segment = ast.get_source_segment(source, child) or ""
                    return textwrap.dedent(segment).rstrip() + "\n"
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            segment = ast.get_source_segment(source, node) or ""
            return textwrap.dedent(segment).rstrip() + "\n"
    return ""


def _call_edges(path: Path) -> dict[str, list[str]]:
    tree = _parse_python(path)
    if tree is None:
        return {}
    edges: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                calls.append(child.func.id)
        edges[node.name] = sorted(set(calls))
    return edges


def _iter_repo_python_files(repo_root: Path) -> list[Path]:
    roots = [
        repo_root / "qos",
        repo_root / "qvm",
        repo_root / "Baseline_Multiprogramming",
        repo_root / "evaluation",
    ]
    excluded_parts = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        "temp",
        "third_party",
        "openevolve_pairing",
        "agent_framework",
        "runtime_venvs",
        "figures",
        "artifacts",
    }
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel_parts = set(path.resolve().relative_to(repo_root.resolve()).parts)
            if rel_parts & excluded_parts:
                continue
            files.append(path)
    return sorted(files)


def _function_records(path: Path, repo_root: Path) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8")
    tree = _parse_python(path)
    if tree is None:
        return []
    records: list[dict[str, Any]] = []
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        class_name = None
        parent_node = parent.get(node)
        if isinstance(parent_node, ast.ClassDef):
            class_name = parent_node.name
        calls: list[str] = []
        attrs: list[str] = []
        names: list[str] = []
        constants: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            if isinstance(child, ast.Attribute):
                attrs.append(child.attr)
            elif isinstance(child, ast.Name):
                names.append(child.id)
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                constants.append(child.value)
        entry = (
            f"{_module_name(repo_root, path)}:{class_name}.{node.name}"
            if class_name
            else f"{_module_name(repo_root, path)}:{node.name}"
        )
        records.append(
            {
                "module": _module_name(repo_root, path),
                "class": class_name,
                "function": node.name,
                "entrypoint": entry,
                "signature": _signature_from_args(node),
                "source_path": _rel(repo_root, path),
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", None),
                "doc": ast.get_docstring(node) or "",
                "calls": sorted(set(calls)),
                "attrs": sorted(set(attrs)),
                "names": sorted(set(names)),
                "string_constants": constants[:20],
                "source": textwrap.dedent(ast.get_source_segment(source, node) or "").rstrip() + "\n",
            }
        )
    return records


def _module_name(repo_root: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve()).with_suffix("")
        return ".".join(rel.parts)
    except Exception:
        return path.stem


def _tokens(text: Any) -> set[str]:
    raw = str(text or "").replace("_", " ").replace("-", " ").replace(".", " ").lower()
    return {tok for tok in raw.replace("(", " ").replace(")", " ").replace(",", " ").split() if tok}


def _score_qos_target(record: dict[str, Any], inbound_callers: dict[str, list[str]]) -> tuple[int, list[str], list[str]]:
    name_tokens = _tokens(record.get("function"))
    path_tokens = _tokens(record.get("source_path"))
    class_tokens = _tokens(record.get("class"))
    sig_tokens = _tokens(record.get("signature"))
    doc_tokens = _tokens(record.get("doc"))
    call_tokens = set(record.get("calls") or [])
    body_tokens = _tokens(" ".join(record.get("names") or []) + " " + " ".join(record.get("attrs") or []))
    string_tokens = _tokens(" ".join(record.get("string_constants") or []))
    all_text_tokens = name_tokens | path_tokens | class_tokens | sig_tokens | doc_tokens | body_tokens | string_tokens

    score = 0
    evidence: list[str] = []
    penalties: list[str] = []

    if "multiprogrammer" in path_tokens or "multiprogramming" in path_tokens:
        score += 25
        evidence.append("located in multiprogramming source path")
    if "scheduler" in path_tokens:
        score += 10
        evidence.append("located in scheduler source path")
    if {"match", "matching"} & name_tokens and "score" in name_tokens:
        score += 45
        evidence.append("function name denotes matching score")
    elif {"score", "rank", "select", "pair", "pairs", "schedule"} & name_tokens:
        score += 18
        evidence.append("function name suggests scoring/ranking/selection")
    if {"q1", "q2", "backend"}.issubset(sig_tokens):
        score += 30
        evidence.append("signature accepts q1, q2, and backend")
    elif {"pairs", "qernels", "candidates"} & sig_tokens:
        score += 18
        evidence.append("signature accepts pair/candidate collection")
    if {"effective_utilization", "spatial_utilization", "fidelity", "layout", "backend"} & (call_tokens | body_tokens):
        score += 20
        evidence.append("body references utilization/fidelity/layout/backend signals")
    if {"sort", "sorted", "max"} & call_tokens:
        score += 12
        evidence.append("body ranks or selects candidates")
    if {"bundle_qernels", "check_layout_overlap", "size_overflow"} & (call_tokens | body_tokens):
        score += 15
        evidence.append("body participates in multiprogramming bundle/layout decisions")
    if {"pair", "pairs", "matching", "utilization", "fidelity", "threshold"} & (doc_tokens | string_tokens):
        score += 10
        evidence.append("docstring/comments mention pair-selection concepts")
    callers = inbound_callers.get(record.get("function") or "", [])
    if callers:
        score += min(20, 5 * len(callers))
        evidence.append(f"called by candidate selection functions: {', '.join(callers[:4])}")
    if record.get("function", "").startswith("_"):
        score -= 10
        penalties.append("private helper is less likely to be the primary evolution target")
    if "test" in path_tokens or "harness" in path_tokens:
        score -= 40
        penalties.append("test/harness code is not preferred as an evolution target")
    if record.get("function") in {"run", "__init__"}:
        score -= 15
        penalties.append("generic lifecycle function is less directly evolvable")
    if score < 0:
        score = 0
    return score, evidence, penalties


def _inbound_callers(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_name = {str(record.get("function")) for record in records}
    inbound: dict[str, list[str]] = {name: [] for name in by_name}
    for record in records:
        caller = str(record.get("entrypoint") or record.get("function"))
        for call in record.get("calls") or []:
            if call in inbound:
                inbound[call].append(caller)
    return inbound


def _qos_evolution_target_probe(repo_root: Path) -> dict[str, Any]:
    selection_target = "multiprogramming_pair_selection"
    files = _iter_repo_python_files(repo_root)
    records: list[dict[str, Any]] = []
    parse_failures: list[str] = []
    for path in files:
        try:
            records.extend(_function_records(path, repo_root))
        except Exception as exc:
            parse_failures.append(f"{_rel(repo_root, path)}: {type(exc).__name__}: {exc}")
    if not records:
        return {
            "selection_target": selection_target,
            "found": False,
            "selected": None,
            "candidates": [],
            "warnings": ["no repository Python functions discovered"],
        }

    inbound = _inbound_callers(records)
    candidates: list[dict[str, Any]] = []
    for record in records:
        score, evidence, penalties = _score_qos_target(record, inbound)
        if score <= 0 or not evidence:
            continue
        candidate = {
            key: record.get(key)
            for key in ("module", "class", "function", "entrypoint", "signature", "source_path", "line", "end_line", "source")
        }
        candidate.update(
            {
                "role": "QOS multiprogramming pair-selection target",
                "score": score,
                "evidence": evidence,
                "penalties": penalties,
                "called_by": inbound.get(record.get("function") or "", [])[:8],
            }
        )
        candidates.append(candidate)
    candidates.sort(key=lambda item: (-int(item.get("score") or 0), str(item.get("entrypoint") or "")))
    heuristic_top_candidate = candidates[0] if candidates else None
    scan_summary = {
        "files_scanned": len(files),
        "functions_scanned": len(records),
        "candidate_count": len(candidates),
        "excluded_roots": ["temp", "third_party", "evaluation/agent_framework", "evaluation/openevolve_pairing"],
        "parse_failures": parse_failures[:20],
    }
    return {
        "selection_target": selection_target,
        "found": bool(candidates),
        "selected": None,
        "selection_required": "llm_semantic_selection",
        "selection_mode": "candidate_scan_only",
        "heuristic_top_candidate": heuristic_top_candidate,
        "candidates": candidates[:25],
        "scan_summary": scan_summary,
        "source_policy": "auto_discovered_from_repository_semantics",
        "manual_seed_used": False,
        "notes": (
            "AST/static analysis only retrieves candidates and evidence. The final QOS target binding "
            "must be made by an LLM semantic-selection action and then validated deterministically."
        ),
    }


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def build_probe(repo_root: Path) -> dict[str, Any]:
    openevolve_root = (repo_root / OPENEVOLVE_REL_ROOT).resolve()
    payload: dict[str, Any] = {
        "success": False,
        "execution_mode": "openevolve_target_probe",
        "not_fig11_strict_success": True,
        "openevolve": {
            "source_root": OPENEVOLVE_REL_ROOT.as_posix(),
            "absolute_source_root": str(openevolve_root),
        },
        "target_functions": {
            "evolve": {"found": False, "candidates": []},
            "verify": {"found": False, "candidates": []},
        },
        "qos_evolution_target": {},
        "warnings": [],
    }

    if not openevolve_root.exists():
        payload["reason"] = "missing_openevolve_checkout"
        payload["warnings"].append("Run: git clone https://github.com/algorithmicsuperintelligence/openevolve.git third_party/openevolve")
        return payload

    api = openevolve_root / "openevolve" / "api.py"
    cli = openevolve_root / "openevolve" / "cli.py"
    evaluator = openevolve_root / "openevolve" / "evaluator.py"
    process_parallel = openevolve_root / "openevolve" / "process_parallel.py"
    pyproject = openevolve_root / "pyproject.toml"

    api_funcs = _functions_in(api)
    cli_funcs = _functions_in(cli)
    process_funcs = _functions_in(process_parallel)
    eval_methods = _class_methods(evaluator, "Evaluator")

    payload["openevolve"].update(
        {
            "git_commit": _git(openevolve_root, ["rev-parse", "--short", "HEAD"]),
            "git_tag": _git(openevolve_root, ["describe", "--tags", "--abbrev=0", "--always"]),
            "git_status": _git(openevolve_root, ["status", "--short", "--branch"]),
            "pyproject": _rel(repo_root, pyproject),
        }
    )

    evolve_candidates = [
        {
            "entrypoint": "openevolve.api:run_evolution",
            "file": _rel(repo_root, api),
            "line": api_funcs.get("run_evolution", {}).get("line"),
            "role": "programmatic evolution runner",
        },
        {
            "entrypoint": "openevolve.api:evolve_function",
            "file": _rel(repo_root, api),
            "line": api_funcs.get("evolve_function", {}).get("line"),
            "role": "function-level evolution API",
        },
        {
            "entrypoint": "openevolve.cli:main",
            "file": _rel(repo_root, cli),
            "line": cli_funcs.get("main", {}).get("line"),
            "role": "CLI entrypoint / openevolve-run",
        },
        {
            "entrypoint": "openevolve.process_parallel:run_evolution",
            "file": _rel(repo_root, process_parallel),
            "line": process_funcs.get("run_evolution", {}).get("line"),
            "role": "parallel evolution runner",
        },
    ]
    verify_candidates = [
        {
            "entrypoint": "openevolve.evaluator:Evaluator.evaluate_program",
            "file": _rel(repo_root, evaluator),
            "line": eval_methods.get("evaluate_program", {}).get("line"),
            "role": "OpenEvolve candidate evaluation hook",
        },
        {
            "entrypoint": "QOS-agent verify wrapper",
            "file": "evaluation/agent_framework/reproduce/figure/verify_claim.py",
            "role": "contract/provenance verification against QOS metrics",
        },
        {
            "entrypoint": "QOS-agent rank comparison wrapper",
            "file": "evaluation/agent_framework/reproduce/tools/",
            "role": "compare evolved scoring-function ranking against simulation/proxy metric provenance",
        },
    ]

    payload["target_functions"]["evolve"] = {
        "found": all(name in api_funcs for name in ("run_evolution", "evolve_function")) and "main" in cli_funcs,
        "candidates": evolve_candidates,
    }
    payload["target_functions"]["verify"] = {
        "found": "evaluate_program" in eval_methods,
        "candidates": verify_candidates,
    }
    payload["qos_evolution_target"] = _qos_evolution_target_probe(repo_root)
    payload["success"] = bool(
        payload["target_functions"]["evolve"]["found"]
        and payload["qos_evolution_target"].get("found")
        and payload["openevolve"].get("source_root") == OPENEVOLVE_REL_ROOT.as_posix()
    )
    payload["recommended_next"] = [
        "Use openevolve.api:run_evolution or openevolve.cli:main for the Evolve wrapper.",
        "Run openevolve_target_semantic_select to bind the contract target to a repository function using LLM reasoning.",
        "Implement a QOS-agent Verify wrapper that compares evolved scoring-function ranking to simulation or proxy-metric provenance.",
        "Validate proxy/evolution artifacts with qos_openevolve_proxy_search.contract.json; do not mark Fig. 11 strict reproduction success.",
    ]
    return payload


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
    probe_dir = run_root / "openevolve_target_probe"

    payload = build_probe(repo_root)
    payload["tool"] = TOOL_NAME
    payload["action"] = ACTION
    payload["fsm_state_before"] = get_fsm_state(state)
    payload["contract_path"] = (
        "evaluation/agent_framework/reproduce/examples/figure_contracts/"
        "qos_openevolve_proxy_search.contract.json"
    )
    payload["artifact_path"] = str(probe_dir / "openevolve_target_probe.json")

    _write_json(probe_dir / "openevolve_target_probe.json", payload)

    state["last_openevolve_target_probe"] = payload
    state["last_step"] = TOOL_NAME
    state["last_status"] = "openevolve_target_probe_succeeded" if payload.get("success") else "openevolve_target_probe_failed"
    set_fsm_state(state, "APPLY_FIX" if payload.get("success") else "CLASSIFY_FAILURE")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, payload)
    write_state(state_path, state)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
