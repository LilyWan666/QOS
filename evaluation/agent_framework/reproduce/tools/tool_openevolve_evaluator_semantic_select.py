#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
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


ACTION = "openevolve_evaluator_semantic_select"
TOOL_NAME = "repro_openevolve_evaluator_semantic_select"


FIG11_TERMS = [
    "Figure 11",
    "Fig. 11",
    "fidelity",
    "relative fidelity",
    "effective utilization",
    "multiprogramming",
    "multi-programming",
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


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _resolve_doc(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_doc_text(path: Path, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pdf":
        text_path = output_dir / f"{path.stem}.txt"
        proc = subprocess.run(
            ["pdftotext", "-layout", str(path), str(text_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "pdftotext failed")
        return text_path.read_text(encoding="utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def _fig11_snippets(text: str, limit: int = 14) -> list[str]:
    snippets: list[str] = []
    seen: set[str] = set()
    normalized = re.sub(r"\s+", " ", text)
    lowered_text = normalized.lower()
    anchors = [
        "figure 11",
        "fig. 11",
        "(a) impact of multi-programming on fidelity",
        "(b) effective utilization",
        "(c) relative fidelity",
        "relative fidelity w.r.t. solo",
        "effective utilization.",
    ]
    for anchor in anchors:
        start = 0
        while len(snippets) < limit:
            idx = lowered_text.find(anchor, start)
            if idx == -1:
                break
            lo = max(0, idx - 650)
            hi = min(len(normalized), idx + 950)
            snippet = normalized[lo:hi].strip()
            key = snippet[:200]
            if key not in seen:
                snippets.append(snippet[:1200])
                seen.add(key)
            start = idx + len(anchor)
    for sent in _sentences(text):
        if len(snippets) >= limit:
            break
        lowered = sent.lower()
        if any(term.lower() in lowered for term in FIG11_TERMS):
            if sent not in seen:
                snippets.append(sent[:700])
                seen.add(sent)
    return snippets


def _paper_metric_spec(recipe: dict[str, Any], repo_root: Path, artifact_dir: Path) -> dict[str, Any]:
    documents = ((recipe.get("paper") or {}).get("documents") or [])
    snippets: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for doc in documents:
        raw = str(doc.get("path") or "")
        if not raw:
            continue
        path = _resolve_doc(repo_root, raw)
        try:
            text = _read_doc_text(path, artifact_dir / "paper_text")
            lowered = text.lower()
            snippets.append(
                {
                    "label": doc.get("label") or path.stem,
                    "path": raw,
                    "resolved_path": str(path),
                    "snippets": _fig11_snippets(text),
                    "contains_fidelity": "fidelity" in lowered,
                    "contains_relative_fidelity": "relative fidelity" in lowered,
                    "contains_effective_utilization": "effective utilization" in lowered,
                }
            )
        except Exception as exc:
            errors.append({"path": raw, "error": str(exc)})

    joined = " ".join(" ".join(item.get("snippets") or []) for item in snippets).lower()
    has_fidelity = "fidelity" in joined
    has_relative = "relative fidelity" in joined
    has_eff_util = "effective utilization" in joined
    if snippets:
        has_fidelity = has_fidelity or any(bool(item.get("contains_fidelity")) for item in snippets)
        has_relative = has_relative or any(bool(item.get("contains_relative_fidelity")) for item in snippets)
        has_eff_util = has_eff_util or any(bool(item.get("contains_effective_utilization")) for item in snippets)
    return {
        "source": "paper",
        "figure_id": "fig11",
        "selection_target": "multiprogramming_pair_selection",
        "nominal_utilization_targets": ["30", "60", "88"],
        "objective_axes": [
            {
                "name": "effective_utilization",
                "role": "resource_efficiency_axis",
                "maximize": True,
                "paper_grounded": has_eff_util,
            },
            {
                "name": "fidelity_or_relative_fidelity",
                "role": "execution_quality_axis",
                "maximize": True,
                "paper_grounded": has_fidelity or has_relative,
            },
        ],
        "figures": {
            "11a": {"metric": "fidelity", "paper_grounded": has_fidelity},
            "11b": {"metric": "effective_utilization", "paper_grounded": has_eff_util},
            "11c": {"metric": "relative_fidelity_wrt_solo_execution", "paper_grounded": has_relative},
        },
        "aggregation": "selected_topk_average_pareto_rank",
        "selection_operator": "topk",
        "primary_score": "inv_avg_pareto_rank",
        "diagnostic_metrics": ["rank_agreement", "top_rank_overlap"],
        "forbid_objective_gating_by_front_overlap": True,
        "forbid_required_rank1_overlap": True,
        "aggregation_policy": "fixed_by_paper_tradeoff_no_scalar_weights",
        "forbid_weighted_sum_without_paper_weight": True,
        "forbid_single_metric_fidelity_only": True,
        "nominal_targets_are_regimes_not_objectives": True,
        "paper_snippets": snippets,
        "errors": errors,
    }


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def _iter_python_files(repo_root: Path) -> list[Path]:
    excluded = {
        ".git",
        "__pycache__",
        "temp",
        "third_party",
        "claw-code",
        "QAgent",
    }
    files: list[Path] = []
    for root, dirs, filenames in os.walk(repo_root):
        root_path = Path(root)
        dirs[:] = [dirname for dirname in dirs if dirname not in excluded]
        parts = set(root_path.relative_to(repo_root).parts)
        if parts & excluded:
            continue
        for filename in filenames:
            if filename.endswith(".py"):
                path = root_path / filename
                if path.name == "tool_openevolve_evaluator_semantic_select.py":
                    continue
                files.append(path)
    return sorted(files)


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node) or ""
    return segment[:5000]


def _candidate_score(source: str, name: str, rel_path: str) -> tuple[float, list[str], list[str]]:
    lowered = source.lower()
    name_low = name.lower()
    path_low = rel_path.lower()
    score = 0.0
    evidence: list[str] = []
    penalties: list[str] = []
    for term, weight in [
        ("effective_utilization", 4.0),
        ("relative_fidelity", 4.0),
        ("fidelity", 2.0),
        ("pareto", 3.0),
        ("rank", 2.0),
        ("topk", 2.0),
        ("top_k", 2.0),
        ("evaluate", 2.0),
    ]:
        if term in lowered or term in name_low:
            score += weight
            evidence.append(f"mentions {term}")
    if "evaluator" in name_low or "evaluator" in path_low:
        score += 3.0
        evidence.append("name/path indicates evaluator")
    if "harnesses/" in path_low or "run_qos_fig11_full.py" in path_low:
        score -= 2.0
        penalties.append("harness/runner is not an OpenEvolve evaluator")
    if "agent_framework/reproduce/tools/tool_openevolve_proxy_search.py" in path_low:
        score += 2.0
        evidence.append("current proxy-search tool generates OpenEvolve evaluator adapter")
    return score, evidence, penalties


def _scan_evaluator_candidates(repo_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in _iter_python_files(repo_root):
        rel = _rel(repo_root, path)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            name = getattr(node, "name", "")
            segment = _source_segment(source, node)
            if not segment:
                continue
            score, evidence, penalties = _candidate_score(segment, name, rel)
            if score <= 0:
                continue
            candidates.append(
                {
                    "entrypoint": f"{rel}:{name}",
                    "source_path": rel,
                    "line": getattr(node, "lineno", None),
                    "symbol": name,
                    "kind": type(node).__name__,
                    "score": score,
                    "evidence": evidence,
                    "penalties": penalties,
                    "source_excerpt": segment[:1800],
                }
            )
    return sorted(candidates, key=lambda item: (-float(item["score"]), str(item["entrypoint"])))[:20]


def _select_strategy(metric_spec: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    axes = {axis.get("name") for axis in metric_spec.get("objective_axes", [])}
    top = candidates[0] if candidates else None
    has_two_axes = "effective_utilization" in axes and "fidelity_or_relative_fidelity" in axes
    if top and top.get("score", 0) >= 10:
        return {
            "mode": "paper_metric_adapter",
            "selected_entrypoint": top.get("entrypoint"),
            "selected_source_path": top.get("source_path"),
            "reason": (
                "The repository has evaluator-related code, but the selected evaluator must be "
                "parameterized by the paper metric axes rather than used as an opaque objective."
            ),
            "requires_generated_adapter": True,
            "aggregation": "selected_topk_average_pareto_rank",
            "selection_operator": "topk",
            "primary_score": "inv_avg_pareto_rank",
            "diagnostic_metrics": ["rank_agreement", "top_rank_overlap"],
            "forbid_objective_gating_by_front_overlap": True,
            "forbid_required_rank1_overlap": True,
            "forbid_weighted_sum_without_paper_weight": True,
            "adapter_requirements": [
                "pair-level records only",
                "maximize effective_utilization",
                "maximize fidelity or relative_fidelity",
                "score selected top-k pairs by inverse average Pareto rank",
                "record rank agreement and rank-1 front overlap only as diagnostics",
            ],
        }
    return {
        "mode": "generate_paper_metric_adapter",
        "selected_entrypoint": None,
        "selected_source_path": None,
        "reason": (
            "No standalone repository evaluator satisfies the paper-derived Fig. 11 metric contract; "
            "generate an OpenEvolve evaluator adapter from the paper metric spec."
        ),
        "requires_generated_adapter": True,
        "aggregation": "selected_topk_average_pareto_rank",
        "selection_operator": "topk",
        "primary_score": "inv_avg_pareto_rank",
        "diagnostic_metrics": ["rank_agreement", "top_rank_overlap"],
        "forbid_objective_gating_by_front_overlap": True,
        "forbid_required_rank1_overlap": True,
        "forbid_weighted_sum_without_paper_weight": True,
        "adapter_requirements": [
            "pair-level records only",
            "maximize effective_utilization",
            "maximize fidelity or relative_fidelity",
            "score selected top-k pairs by inverse average Pareto rank",
            "record rank agreement and rank-1 front overlap only as diagnostics",
        ],
        "paper_metric_axes_complete": has_two_axes,
    }


def build_selection(repo_root: Path, recipe_path: Path, artifact_dir: Path) -> dict[str, Any]:
    recipe = _load_json(recipe_path)
    metric_spec = _paper_metric_spec(recipe, repo_root, artifact_dir)
    candidates = _scan_evaluator_candidates(repo_root)
    strategy = _select_strategy(metric_spec, candidates)
    grounded = all(
        bool(axis.get("paper_grounded"))
        for axis in metric_spec.get("objective_axes", [])
        if axis.get("name") in {"effective_utilization", "fidelity_or_relative_fidelity"}
    )
    return {
        "success": bool(grounded and strategy.get("requires_generated_adapter")),
        "execution_mode": ACTION,
        "selection_mode": "paper_metric_semantic_binding",
        "not_fig11_strict_success": True,
        "paper_metric_spec": metric_spec,
        "evaluator_strategy": strategy,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "warnings": [] if grounded else ["paper snippets did not ground both evaluator axes"],
    }


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
    run_root = ensure_run_root(recipe_path.stem, state, Path(args.output_root).resolve())
    artifact_dir = run_root / "openevolve_evaluator_semantic_select"
    payload = build_selection(repo_root, recipe_path, artifact_dir)
    payload["artifact_path"] = str(artifact_dir / "openevolve_evaluator_semantic_select.json")
    _write_json(Path(payload["artifact_path"]), payload)

    state["last_openevolve_evaluator_semantic_selection"] = payload
    state["last_step"] = TOOL_NAME
    state["last_status"] = "openevolve_evaluator_semantic_select_succeeded" if payload["success"] else "openevolve_evaluator_semantic_select_failed"
    set_fsm_state(state, get_fsm_state(state))
    append_history(state, TOOL_NAME, payload)
    state["next_allowed_actions"] = next_actions(state)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
