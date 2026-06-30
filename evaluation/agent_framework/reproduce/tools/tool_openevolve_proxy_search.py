#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

from figure_claim import evaluate_figure_contract  # noqa: E402
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
from physical_qpu_proxy_utils import physical_record_features  # noqa: E402
from tool_openevolve_target_probe import build_probe as build_target_probe  # noqa: E402
from tool_physical_qpu_env_probe import build_probe as build_physical_qpu_probe  # noqa: E402


ACTION = "openevolve_proxy_search"
TOOL_NAME = "repro_openevolve_proxy_search"
CONTRACT_REL_PATH = Path(
    "evaluation/agent_framework/reproduce/examples/figure_contracts/"
    "qos_openevolve_proxy_search.contract.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument(
        "--output-root",
        default="temp/agent_framework/reproduce/tools/runs",
        help="Root directory for proxy/evolution artifacts.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _pick_candidate(probe: dict[str, Any], group: str, preferred: str) -> dict[str, Any]:
    candidates = ((probe.get("target_functions") or {}).get(group) or {}).get("candidates") or []
    for candidate in candidates:
        if candidate.get("entrypoint") == preferred:
            return dict(candidate)
    for candidate in candidates:
        if candidate.get("entrypoint"):
            return dict(candidate)
    return {"entrypoint": preferred, "role": f"default {group} entrypoint"}


def _artifact_rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def _selected_qos_target(state: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    semantic = state.get("last_openevolve_target_semantic_selection")
    if isinstance(semantic, dict) and semantic.get("success") and isinstance(semantic.get("selected"), dict):
        return dict(semantic["selected"])
    qos_target = probe.get("qos_evolution_target") if isinstance(probe.get("qos_evolution_target"), dict) else {}
    selected = qos_target.get("selected") if isinstance(qos_target.get("selected"), dict) else {}
    return dict(selected) if selected else {}


def _normalize_function_indent(source: str) -> str:
    lines = textwrap.dedent(source).rstrip().splitlines()
    if len(lines) <= 1:
        return "\n".join(lines).rstrip() + "\n"

    body_indents = [
        len(line) - len(line.lstrip())
        for line in lines[1:]
        if line.strip()
    ]
    if not body_indents:
        return "\n".join(lines).rstrip() + "\n"

    min_body_indent = min(body_indents)
    if min_body_indent <= 4:
        return "\n".join(lines).rstrip() + "\n"

    normalized = [lines[0]]
    trim = min_body_indent - 4
    for line in lines[1:]:
        if line.strip() and line.startswith(" " * trim):
            normalized.append(line[trim:])
        else:
            normalized.append(line)
    return "\n".join(normalized).rstrip() + "\n"


def _compact_evolvable_function(source: str) -> str:
    """Keep target semantics but make the search/replace block easy to copy exactly."""
    normalized = _normalize_function_indent(source).rstrip()
    try:
        module = ast.parse(normalized)
    except SyntaxError:
        return normalized

    functions = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return normalized
    fn = functions[0]
    fn.name = "get_matching_score"
    fn.returns = None
    fn.decorator_list = []
    for arg in list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs):
        arg.annotation = None
    fn.args.vararg = None if fn.args.vararg is None else fn.args.vararg
    if fn.args.vararg is not None:
        fn.args.vararg.annotation = None
    if fn.args.kwarg is not None:
        fn.args.kwarg.annotation = None
    if (
        fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ):
        fn.body = fn.body[1:]
    ast.fix_missing_locations(module)
    compact = ast.unparse(fn)
    return compact.rstrip() + "\n"


class _EffectiveUtilizationNormalizer(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "effective_utilization"
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="_qos_agent_normalize_utilization", ctx=ast.Load()),
                    args=[node],
                    keywords=[],
                ),
                node,
            )
        return node


def _normalize_seed_effective_utilization_scale(source: str) -> tuple[str, bool]:
    """Normalize repo percent-scale utilization in the OpenEvolve seed only."""
    try:
        module = ast.parse(source)
    except SyntaxError:
        return source, False
    before = ast.unparse(module)
    module = _EffectiveUtilizationNormalizer().visit(module)
    ast.fix_missing_locations(module)
    after = ast.unparse(module).rstrip() + "\n"
    return after, after != before


def _proxy_depth_ratio_source() -> str:
    return '''def get_matching_score(self, q1, q2, backend, weighted=False, weights=[]):
    meta1 = q1.get_metadata()
    meta2 = q2.get_metadata()
    depth1 = float(meta1.get("depth", 0.0) or 0.0)
    depth2 = float(meta2.get("depth", 0.0) or 0.0)
    return min(depth1, depth2) / max(depth1, depth2, 1.0)'''


def _proxy_feature_spec(feature: str | None) -> dict[str, Any]:
    name = str(feature or "").strip()
    specs: dict[str, dict[str, Any]] = {
        "depth_ratio": {
            "selected_proxy_feature": "depth_ratio",
            "required_metadata": ["depth"],
            "feature_expression": "min(depth1, depth2) / max(depth1, depth2, 1.0)",
            "scale": "0_to_1_higher_is_better",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Depth similarity between the two co-scheduled circuits.",
        },
        "critical_depth_ratio": {
            "selected_proxy_feature": "critical_depth_ratio",
            "required_metadata": ["critical_depth"],
            "feature_expression": "min(critical_depth1, critical_depth2) / max(critical_depth1, critical_depth2, 1.0)",
            "scale": "0_to_1_higher_is_better",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Critical-path depth similarity between co-scheduled circuits.",
        },
        "qubit_imbalance": {
            "selected_proxy_feature": "qubit_imbalance",
            "required_metadata": ["num_qubits"],
            "feature_expression": "abs(qubits1 - qubits2) / max(qubits1 + qubits2, 1.0)",
            "scale": "0_to_1_lower_is_better",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Qubit-count imbalance between the two circuits.",
        },
        "cnot_ratio": {
            "selected_proxy_feature": "cnot_ratio",
            "required_metadata": ["num_cnot_gates"],
            "feature_expression": "min(cnot1, cnot2) / max(cnot1, cnot2, 1.0)",
            "scale": "0_to_1_higher_is_better",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "CNOT-count similarity between co-scheduled circuits.",
        },
        "nonlocal_ratio": {
            "selected_proxy_feature": "nonlocal_ratio",
            "required_metadata": ["num_nonlocal_gates"],
            "feature_expression": "min(nonlocal1, nonlocal2) / max(nonlocal1, nonlocal2, 1.0)",
            "scale": "0_to_1_higher_is_better",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Multi-qubit/nonlocal-gate similarity between co-scheduled circuits.",
        },
        "instr_ratio": {
            "selected_proxy_feature": "instr_ratio",
            "required_metadata": ["number_instructions"],
            "feature_expression": "min(instr1, instr2) / max(instr1, instr2, 1.0)",
            "scale": "0_to_1_higher_is_better",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Instruction-count similarity between co-scheduled circuits.",
        },
        "measure_ratio": {
            "selected_proxy_feature": "measure_ratio",
            "required_metadata": ["num_measurements"],
            "feature_expression": "min(measurements1, measurements2) / max(measurements1, measurements2, 1.0)",
            "scale": "0_to_1_higher_is_better",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Measurement-count similarity between co-scheduled circuits.",
        },
        "cnot_density": {
            "selected_proxy_feature": "cnot_density",
            "required_metadata": ["num_cnot_gates", "num_qubits"],
            "feature_expression": "(cnot1 + cnot2) / max(qubits1 + qubits2, 1.0)",
            "scale": "nonnegative_contextual",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Two-qubit gate density per joint qubit.",
        },
        "nonlocal_density": {
            "selected_proxy_feature": "nonlocal_density",
            "required_metadata": ["num_nonlocal_gates", "num_qubits"],
            "feature_expression": "(nonlocal1 + nonlocal2) / max(qubits1 + qubits2, 1.0)",
            "scale": "nonnegative_contextual",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Nonlocal gate density per joint qubit.",
        },
        "instr_density": {
            "selected_proxy_feature": "instr_density",
            "required_metadata": ["number_instructions", "num_qubits"],
            "feature_expression": "(instr1 + instr2) / max(qubits1 + qubits2, 1.0)",
            "scale": "nonnegative_contextual",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Instruction density per joint qubit.",
        },
        "measure_density": {
            "selected_proxy_feature": "measure_density",
            "required_metadata": ["num_measurements", "num_qubits"],
            "feature_expression": "(measurements1 + measurements2) / max(qubits1 + qubits2, 1.0)",
            "scale": "nonnegative_contextual",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Measurement density per joint qubit.",
        },
        "critical_depth_density": {
            "selected_proxy_feature": "critical_depth_density",
            "required_metadata": ["critical_depth", "num_qubits"],
            "feature_expression": "(critical_depth1 + critical_depth2) / max(qubits1 + qubits2, 1.0)",
            "scale": "nonnegative_contextual",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Critical-path depth density per joint qubit.",
        },
        "joint_qubits": {
            "selected_proxy_feature": "joint_qubits",
            "required_metadata": ["num_qubits"],
            "feature_expression": "qubits1 + qubits2",
            "scale": "nonnegative_contextual",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Total qubits used by the selected pair.",
        },
    }
    return specs.get(
        name,
        {
            "selected_proxy_feature": name or "unknown",
            "required_metadata": [],
            "feature_expression": name or "unknown",
            "scale": "unknown",
            "semantic_role": "fidelity_proxy_second_axis",
            "description": "Proxy feature selected by the semantic proxy validation tool.",
        },
    )


def _proxy_feature_specs(features: list[str] | tuple[str, ...] | None) -> list[dict[str, Any]]:
    return [_proxy_feature_spec(feature) for feature in dict.fromkeys(str(item) for item in (features or []) if str(item))]


def _semantic_factor_evidence(state: dict[str, Any]) -> dict[str, Any]:
    brainstorm = state.get("last_proxy_semantic_factor_brainstorm")
    if isinstance(brainstorm, dict) and brainstorm.get("semantic_factors"):
        return {
            key: value
            for key, value in brainstorm.items()
            if key not in {"prompt_payload", "llm_payload"}
        }

    proposal = state.get("last_proxy_metric_semantic_proposal")
    factors: list[dict[str, Any]] = []
    if isinstance(proposal, dict):
        for candidate in proposal.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            name = str(candidate.get("semantic_factor_name") or candidate.get("name") or "").strip()
            if not name:
                continue
            factors.append(
                {
                    "name": name,
                    "materialized_feature_name": candidate.get("materialized_feature_name"),
                    "paper_rationale": candidate.get("paper_rationale"),
                    "repo_rationale": candidate.get("repo_rationale"),
                    "required_metadata_keys": candidate.get("required_metadata_keys") or [],
                    "source": "proxy_metric_semantic_proposal.candidates",
                }
            )
    if factors:
        return {
            "success": True,
            "brainstorm_mode": "semantic_factor_evidence_from_metric_proposal",
            "evidence_source": "proxy_metric_semantic_proposal.candidates",
            "semantic_factors": factors,
            "upstream_brainstorm_success": bool(isinstance(brainstorm, dict) and brainstorm.get("success")),
            "manual_proxy_source_used": False,
        }

    return {
        key: value
        for key, value in (brainstorm or {}).items()
        if key not in {"prompt_payload", "llm_payload"}
    }


def _seed_proxy_scaffold_source(proxy_specs: list[dict[str, Any]] | None = None) -> str:
    selected = [str(spec.get("selected_proxy_feature")) for spec in (proxy_specs or []) if spec.get("selected_proxy_feature")]
    selected_text = ", ".join(selected) if selected else "none"
    return f'''    # QOS-Agent proxy feature scaffold.
    # Selected proxy feature(s): {selected_text}.
    # These variables are intentionally materialized for OpenEvolve; the initial
    # QOS score below is still the repository baseline unless the LLM changes it.
    meta1 = q1.get_metadata() or {{}}
    meta2 = q2.get_metadata() or {{}}
    def _qos_agent_meta(meta, key, default=0.0):
        try:
            return float(meta.get(key, default) or default)
        except Exception:
            return float(default)
    def _qos_agent_ratio(left, right):
        left = float(left or 0.0)
        right = float(right or 0.0)
        if left <= 0.0 and right <= 0.0:
            return 0.0
        return min(left, right) / max(left, right, 1.0)
    depth1 = _qos_agent_meta(meta1, "depth")
    depth2 = _qos_agent_meta(meta2, "depth")
    qubits1 = _qos_agent_meta(meta1, "num_qubits")
    qubits2 = _qos_agent_meta(meta2, "num_qubits")
    nonlocal1 = _qos_agent_meta(meta1, "num_nonlocal_gates")
    nonlocal2 = _qos_agent_meta(meta2, "num_nonlocal_gates")
    cnot1 = _qos_agent_meta(meta1, "num_cnot_gates")
    cnot2 = _qos_agent_meta(meta2, "num_cnot_gates")
    measurements1 = _qos_agent_meta(meta1, "num_measurements")
    measurements2 = _qos_agent_meta(meta2, "num_measurements")
    instr1 = _qos_agent_meta(meta1, "number_instructions")
    instr2 = _qos_agent_meta(meta2, "number_instructions")
    critical_depth1 = _qos_agent_meta(meta1, "critical_depth")
    critical_depth2 = _qos_agent_meta(meta2, "critical_depth")
    connected_components1 = _qos_agent_meta(meta1, "num_connected_components")
    connected_components2 = _qos_agent_meta(meta2, "num_connected_components")
    liveness1 = _qos_agent_meta(meta1, "liveness")
    liveness2 = _qos_agent_meta(meta2, "liveness")
    program_communication1 = _qos_agent_meta(meta1, "program_communication")
    program_communication2 = _qos_agent_meta(meta2, "program_communication")
    parallelism1 = _qos_agent_meta(meta1, "parallelism")
    parallelism2 = _qos_agent_meta(meta2, "parallelism")
    measurement1 = _qos_agent_meta(meta1, "measurement")
    measurement2 = _qos_agent_meta(meta2, "measurement")
    entanglement_ratio1 = _qos_agent_meta(meta1, "entanglement_ratio")
    entanglement_ratio2 = _qos_agent_meta(meta2, "entanglement_ratio")
    depth_ratio = _qos_agent_ratio(depth1, depth2)
    qubit_ratio = _qos_agent_ratio(qubits1, qubits2)
    nonlocal_ratio = _qos_agent_ratio(nonlocal1, nonlocal2)
    cnot_ratio = _qos_agent_ratio(cnot1, cnot2)
    measure_ratio = _qos_agent_ratio(measurements1, measurements2)
    instr_ratio = _qos_agent_ratio(instr1, instr2)
    critical_depth_ratio = _qos_agent_ratio(critical_depth1, critical_depth2)
    qubit_imbalance = abs(qubits1 - qubits2) / max(qubits1 + qubits2, 1.0)
    cnot_density = (cnot1 + cnot2) / max(qubits1 + qubits2, 1.0)
    nonlocal_density = (nonlocal1 + nonlocal2) / max(qubits1 + qubits2, 1.0)
    measure_density = (measurements1 + measurements2) / max(qubits1 + qubits2, 1.0)
    instr_density = (instr1 + instr2) / max(qubits1 + qubits2, 1.0)
    critical_depth_density = (critical_depth1 + critical_depth2) / max(qubits1 + qubits2, 1.0)
    joint_qubits = qubits1 + qubits2
'''


def _inject_seed_proxy_scaffold(source: str, proxy_specs: list[dict[str, Any]] | None = None) -> tuple[str, list[str]]:
    if not proxy_specs:
        return source, []
    lines = source.rstrip().splitlines()
    if not lines or not lines[0].lstrip().startswith("def get_matching_score"):
        return source, []
    scaffold = _seed_proxy_scaffold_source(proxy_specs).rstrip().splitlines()
    scaffold_features = [
        "depth_ratio",
        "qubit_ratio",
        "nonlocal_ratio",
        "cnot_ratio",
        "measure_ratio",
        "instr_ratio",
        "critical_depth_ratio",
        "qubit_imbalance",
        "cnot_density",
        "nonlocal_density",
        "measure_density",
        "instr_density",
        "critical_depth_density",
        "joint_qubits",
    ]
    return "\n".join([lines[0], *scaffold, *lines[1:]]).rstrip() + "\n", scaffold_features


def _build_initial_program(
    qos_target: dict[str, Any],
    seed_mode: str = "manual_qos_normalized",
    proxy_specs: list[dict[str, Any]] | None = None,
) -> str:
    seed_mode = str(seed_mode or "manual_qos_normalized").strip().lower()
    source = str(qos_target.get("source") or "").strip()
    if not source:
        raise ValueError("missing auto-discovered QOS target source; run openevolve_target_probe first")
    if seed_mode == "proxy_depth_ratio":
        source = _proxy_depth_ratio_source()
        util_scale_normalized = False
    else:
        source = _compact_evolvable_function(source).rstrip()
        if seed_mode == "manual_qos_normalized":
            source, util_scale_normalized = _normalize_seed_effective_utilization_scale(source)
        elif seed_mode == "repo_qos_raw":
            util_scale_normalized = False
        else:
            raise ValueError(
                "unsupported REPRO_OPENEVOLVE_INITIAL_SEED="
                f"{seed_mode!r}; expected repo_qos_raw, manual_qos_normalized, or proxy_depth_ratio"
            )
    scaffold_features: list[str] = []
    if seed_mode in {"repo_qos_raw", "manual_qos_normalized"}:
        source, scaffold_features = _inject_seed_proxy_scaffold(source, proxy_specs)
    source = source.rstrip()
    entrypoint = str(qos_target.get("entrypoint") or "unknown")
    source_path = str(qos_target.get("source_path") or "unknown")
    seed_notes = {
        "repo_qos_raw": "Initial seed mode: repo_qos_raw; exact repo QOS scoring, no utilization normalization.",
        "manual_qos_normalized": "Initial seed mode: manual_qos_normalized; repo QOS scoring with percent utilization normalized to [0,1].",
        "proxy_depth_ratio": "Initial seed mode: proxy_depth_ratio; pure depth-ratio proxy baseline.",
    }
    normalization_note = seed_notes.get(seed_mode, "Initial seed mode: unknown.")
    if util_scale_normalized:
        normalization_note += " The AST rewrite wrapped self.effective_utilization(...) with _qos_agent_normalize_utilization(...)."
    if scaffold_features:
        normalization_note += " The generated seed materializes proxy scaffold features without changing the initial QOS scoring logic."
    return f'''"""OpenEvolve initial program for QOS pair-selection evolution.

Generated by QOS-Agent from an auto-discovered in-repository target.
This is not a hand-written proxy seed. Source: {source_path} ({entrypoint}).
The strict Fig. 11 reproduction path still requires 8192-shot simulation provenance.
{normalization_note}
"""

from __future__ import annotations

try:
    from typing import List
except Exception:
    List = list


def _qos_agent_normalize_utilization(value):
    value = float(value or 0.0)
    return value / 100.0 if value > 1.0 else value


# EVOLVE-BLOCK-START
{source}
# EVOLVE-BLOCK-END
'''


def _format_proxy_prompt_context(
    proxy_specs: list[dict[str, Any]] | None = None,
    proxy_direction: str | None = None,
) -> str:
    if not proxy_specs:
        return (
            "## Proxy Objective Context\n\n"
            "The current run uses a configured fidelity-like/proxy metric, but no "
            "validated proxy feature spec was available. Prefer explicit metadata "
            "features and keep the score general across applications.\n"
        )
    primary = proxy_specs[0]
    direction = str(proxy_direction or "direct").strip() or "direct"
    lines = [
        "## Proxy Objective Context",
        "",
        f"Current proxy second metric: {primary.get('selected_proxy_feature')}",
        f"Proxy label transform: normalized feature with direction={direction}",
        "Pareto rank is computed over:",
        "1. effective_utilization",
        f"2. proxy_estimated_fidelity = normalized({primary.get('selected_proxy_feature')}, direction={direction})",
        "",
        "Selected proxy feature specs:",
    ]
    for spec in proxy_specs:
        lines.extend(
            [
                f"- feature: {spec.get('selected_proxy_feature')}",
                f"  expression: {spec.get('feature_expression')}",
                f"  required_metadata: {', '.join(str(item) for item in spec.get('required_metadata') or []) or 'none'}",
                f"  scale: {spec.get('scale')}",
                f"  role: {spec.get('semantic_role')}",
                f"  meaning: {spec.get('description')}",
            ]
        )
    lines.extend(
        [
            "",
            "The evaluator normalizes the selected raw feature into",
            "`proxy_estimated_fidelity` before Pareto ranking. If direction=inverse,",
            "smaller raw feature values become better proxy labels.",
            "",
            "The initial program materializes these proxy scaffold variables inside",
            "`get_matching_score`. They are available for the LLM to use, but the",
            "initial score remains the repository QOS baseline unless changed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_prompt_templates(
    template_dir: Path,
    proxy_specs: list[dict[str, Any]] | None = None,
    proxy_direction: str | None = None,
) -> None:
    proxy_context = _format_proxy_prompt_context(proxy_specs, proxy_direction=proxy_direction)
    diff_user = """# Current Program Information
- Fitness: {fitness_score}
- Metrics:
{metrics}
- Improvement focus:
{improvement_areas}

{artifacts}

PROXY_CONTEXT_PLACEHOLDER

# Program Evolution History
{evolution_history}

# Current Program
BEGIN_CURRENT_PROGRAM
{current_program}
END_CURRENT_PROGRAM

# Task
Improve only the Python function `get_matching_score(self, q1, q2, backend, weighted=False, weights=[])` to increase validation_score.
`validation_score` is the inverse average Pareto rank of the selected top-k
pairs. Pareto rank is computed from effective utilization and the proxy
metric declared in the Proxy Objective Context above. Rank agreement and rank-1 front overlap are
diagnostic only; do not optimize for mandatory overlap with the Pareto front.

Return one or more SEARCH/REPLACE blocks in exactly this format:

SEARCH_BLOCK_MARKER
the complete code between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END, including the def line
SEARCH_REPLACE_DIVIDER
the complete replacement get_matching_score function, including the same def line
REPLACE_BLOCK_MARKER

Rules:
- Do not use Markdown code fences anywhere in SEARCH or REPLACE.
- Do not write ```python or ``` anywhere in your answer.
- The SEARCH text must match the current EVOLVE block exactly, character for character.
- Replace the whole EVOLVE block in one patch. Do not patch isolated lines.
- Keep exactly this signature: get_matching_score(self, q1, q2, backend, weighted=False, weights=[]).
- Use only q1/q2 metadata, backend, and helper methods available on self.
- `self.effective_utilization(q1, q2, backend)` may return a percent-scale value;
  normalize values greater than 1 by dividing by 100 before combining with 0..1
  compatibility metrics.
- Access circuit features through `q1.get_metadata()` and `q2.get_metadata()`;
  `q1.get_metadata("depth")` style keyed access is also accepted by the evaluator;
  do not use direct attributes such as q1.selected_qubits, q2.selected_qubits,
  q1.circuit, q2.circuit, or benchmark/application names.
- Available metadata keys include: depth, num_qubits, num_clbits,
  num_nonlocal_gates, num_connected_components, number_instructions,
  num_measurements, num_cnot_gates, program_communication, liveness,
  parallelism, measurement, entanglement_ratio, critical_depth.
- Do not import external modules or use hard-coded benchmark/application names.
- The data may include a QPU feedback holdout split. Fitness is computed on
  held-out validation records only; prefer rules that generalize instead of
  memorizing utilization buckets or application names.
- Keep the same function name and signature.
- The replacement must be valid Python that parses with ast.parse.
"""
    diff_user = (
        diff_user.replace("SEARCH_BLOCK_MARKER", "<" * 7 + " SEARCH")
        .replace("SEARCH_REPLACE_DIVIDER", "=" * 7)
        .replace("REPLACE_BLOCK_MARKER", ">" * 7 + " REPLACE")
        .replace("PROXY_CONTEXT_PLACEHOLDER", proxy_context)
    )
    full_rewrite_user = """# Current Program Information
- Fitness: {fitness_score}
- Metrics:
{metrics}
- Improvement focus:
{improvement_areas}

{artifacts}

PROXY_CONTEXT_PLACEHOLDER

# Program Evolution History
{evolution_history}

# Current Program
```python
{current_program}
```

# Task
Rewrite the complete Python program to improve `validation_score`.
`validation_score` is the inverse average Pareto rank of the selected top-k
pairs. Pareto rank is computed from effective utilization and the proxy
metric declared in the Proxy Objective Context above. Rank agreement and rank-1 front overlap are
diagnostic only; do not optimize for mandatory overlap with the Pareto front.

Requirements:
- Return exactly one complete Python program in a single ```python code block.
- Preserve the module header, imports, `# EVOLVE-BLOCK-START`, and `# EVOLVE-BLOCK-END` markers.
- Keep exactly this function signature inside the evolve block:
  `def get_matching_score(self, q1, q2, backend, weighted=False, weights=[]):`
- Change the implementation inside `get_matching_score`; do not return the current program unchanged.
- The program must define `get_matching_score` at module scope and must parse with ast.parse.
- Use only q1/q2 metadata, backend, and helper methods available on self.
- `self.effective_utilization(q1, q2, backend)` may return a percent-scale value;
  normalize values greater than 1 by dividing by 100 before combining with 0..1
  compatibility metrics.
- Access circuit features through `q1.get_metadata()` and `q2.get_metadata()`;
  `q1.get_metadata("depth")` style keyed access is also accepted by the evaluator;
  do not use direct attributes such as q1.selected_qubits, q2.selected_qubits,
  q1.circuit, q2.circuit, or benchmark/application names.
- Available metadata keys include: depth, num_qubits, num_clbits,
  num_nonlocal_gates, num_connected_components, number_instructions,
  num_measurements, num_cnot_gates, program_communication, liveness,
  parallelism, measurement, entanglement_ratio, critical_depth.
- Do not import external modules or use hard-coded benchmark/application names.
- The data may include a QPU feedback holdout split. Fitness is computed on
  held-out validation records only; prefer rules that generalize instead of
  memorizing utilization buckets or application names.
"""
    full_rewrite_user = full_rewrite_user.replace("PROXY_CONTEXT_PLACEHOLDER", proxy_context)
    top_program = """### Program {program_number} (Score: {score})
BEGIN_PROGRAM
{program_snippet}
END_PROGRAM
Key features: {key_features}
"""
    _write_text(template_dir / "diff_user.txt", diff_user)
    _write_text(template_dir / "full_rewrite_user.txt", full_rewrite_user)
    _write_text(template_dir / "top_program.txt", top_program)


def _build_evaluator(training_data_path: Path) -> str:
    return f'''"""Evaluator for QOS-Agent OpenEvolve proxy-objective search.

This evaluates candidate proxy objectives against pair-level ground truth
records. The preferred proxy branch uses physical QPU records from ibm_torino
with hellinger_mean as the expensive fidelity-like label. It does not run Fig.
11 simulation and does not mark strict reproduction success.
"""

import importlib.util
import json
import math
from pathlib import Path

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:
    EvaluationResult = None


TRAINING_DATA_PATH = Path({str(training_data_path)!r})


def _safe_float(value, default=0.0):
    try:
        val = float(value)
    except Exception:
        val = default
    if math.isnan(val) or math.isinf(val):
        val = default
    return val


def _bounded01(value, default=0.0):
    return max(0.0, min(1.0, _safe_float(value, default)))


def _load_program(path):
    spec = importlib.util.spec_from_file_location("candidate_program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Backend:
    def __init__(self, num_qubits):
        self.num_qubits = max(float(num_qubits), 1.0)


class _Qernel:
    def __init__(self, features, side):
        self.features = features
        self.side = side

    def get_metadata(self, key=None):
        f = self.features
        left_qubits = float(f.get("left_qubits", 0.0))
        right_qubits = float(f.get("right_qubits", 0.0))
        qubits = left_qubits if self.side == "left" else right_qubits
        selected = max(float(f.get("selected_qubits", 1.0)), 1.0)
        suffix = "1" if self.side == "left" else "2"
        depth = float(f.get("depth_" + suffix) or max(10.0 * qubits, 1.0))
        instr = _safe_float(f.get("instr_" + suffix), depth)
        nonlocal_gates = _safe_float(f.get("nonlocal_" + suffix), 0.0)
        measurements = _safe_float(f.get("measure_" + suffix), qubits)
        cnot_gates = _safe_float(f.get("cnot_" + suffix), 0.0)
        entanglement_ratio = _bounded01(
            f.get("entanglement_ratio_" + suffix),
            nonlocal_gates / max(instr, 1.0),
        )
        measurement = _bounded01(
            f.get("measurement_" + suffix),
            measurements / max(depth, 1.0),
        )
        parallelism = _bounded01(
            f.get("parallelism_" + suffix),
            1.0 - (depth / max(instr, 1.0)),
        )
        metadata = {{
            "depth": depth,
            "num_qubits": qubits,
            "num_clbits": qubits,
            "num_nonlocal_gates": nonlocal_gates,
            "num_connected_components": float(f.get("connected_components", 1.0)),
            "number_instructions": instr,
            "num_measurements": measurements,
            "num_cnot_gates": cnot_gates,
            "program_communication": float(f.get("program_communication", 0.0)),
            "liveness": float(f.get("utilization_pressure", 0.0)),
            "parallelism": parallelism,
            "measurement": measurement,
            "entanglement_ratio": entanglement_ratio,
            "critical_depth": float(f.get("critical_depth_" + suffix, depth)),
        }}
        if key is None:
            return metadata
        return metadata.get(key, 0.0)


class _CandidateSelf:
    def effective_utilization(self, q1, q2, backend):
        return float(q1.features.get("effective_utilization_percent", 0.0))

    def entanglementComparison(self, q1, q2):
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        return _bounded01((1.0 - meta1["entanglement_ratio"]) * (1.0 - meta2["entanglement_ratio"]))

    def measurementComparison(self, q1, q2):
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        return _bounded01((1.0 - meta1["measurement"]) * (1.0 - meta2["measurement"]))

    def parallelismComparison(self, q1, q2):
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        return _bounded01((1.0 - meta1["parallelism"]) * (1.0 - meta2["parallelism"]))

    def depthComparison(self, q1, q2):
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        return _bounded01(math.exp(-0.05 * abs(float(meta1["depth"]) - float(meta2["depth"]))))

    def fidelityComparison(self, q1, q2):
        return max(0.0, min(1.0, float(q1.features.get("fidelity", 0.0))))


def _candidate_score(module, features):
    if not hasattr(module, "get_matching_score"):
        raise AttributeError("missing get_matching_score")
    q1 = _Qernel(features, "left")
    q2 = _Qernel(features, "right")
    backend = _Backend(features.get("selected_qubits", features.get("joint_qubits", 1.0)))
    return module.get_matching_score(_CandidateSelf(), q1, q2, backend, False, [])


def _stable_random_score(label):
    text = str(label or "")
    acc = 2166136261
    for ch in text:
        acc ^= ord(ch)
        acc = (acc * 16777619) & 0xFFFFFFFF
    return acc / float(0xFFFFFFFF)


def _depth_ratio_score(features):
    left = _safe_float(features.get("depth_1"), 0.0)
    right = _safe_float(features.get("depth_2"), 0.0)
    if left <= 0.0 or right <= 0.0:
        return _safe_float(features.get("depth_ratio"), 0.0)
    return min(left, right) / max(left, right, 1.0)


def _qos_basic_score(features, normalize_util):
    q1 = _Qernel(features, "left")
    q2 = _Qernel(features, "right")
    backend = _Backend(features.get("selected_qubits", features.get("joint_qubits", 1.0)))
    shim = _CandidateSelf()
    util_eff = shim.effective_utilization(q1, q2, backend)
    if normalize_util and util_eff > 1.0:
        util_eff = util_eff / 100.0
    return (
        util_eff
        + shim.entanglementComparison(q1, q2)
        + shim.measurementComparison(q1, q2)
        + shim.parallelismComparison(q1, q2)
    ) / 4.0


def _score_named_baseline(name, label, features, module=None):
    if name == "random":
        return _stable_random_score(label)
    if name == "repo_qos_raw":
        return _qos_basic_score(features, normalize_util=False)
    if name == "qos_normalized":
        return _qos_basic_score(features, normalize_util=True)
    if name == "depth_ratio_only":
        return _depth_ratio_score(features)
    if name == "current_evolved":
        if module is None:
            return -1e9
        return _candidate_score(module, features)
    raise ValueError("unknown baseline: " + str(name))


def _rank(values):
    ordered = sorted(values, key=lambda item: (-float(item[1]), str(item[0])))
    return {{label: index + 1 for index, (label, _) in enumerate(ordered)}}


def _spearman(left, right):
    labels = sorted(set(left) & set(right))
    n = len(labels)
    if n < 2:
        return 0.0
    d2 = sum((left[label] - right[label]) ** 2 for label in labels)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def _topk_overlap(sim_values, proxy_values, k=5):
    k = min(int(k), len(sim_values), len(proxy_values))
    if k <= 0:
        return 0.0
    sim_top = {{label for label, _ in sorted(sim_values, key=lambda item: (-item[1], str(item[0])))[:k]}}
    proxy_top = {{label for label, _ in sorted(proxy_values, key=lambda item: (-item[1], str(item[0])))[:k]}}
    return len(sim_top & proxy_top) / float(k)


def _dominates(left, right):
    return (
        left[0] >= right[0]
        and left[1] >= right[1]
        and (left[0] > right[0] or left[1] > right[1])
    )


def _pareto_ranks(points):
    n = len(points)
    dominates = [set() for _ in range(n)]
    dominated_count = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(points[i], points[j]):
                dominates[i].add(j)
                dominated_count[j] += 1
            elif _dominates(points[j], points[i]):
                dominates[j].add(i)
                dominated_count[i] += 1
    ranks = [0] * n
    front = [i for i in range(n) if dominated_count[i] == 0]
    rank = 1
    while front:
        next_front = []
        for i in front:
            ranks[i] = rank
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        front = next_front
        rank += 1
    return ranks


def _topk_count(n):
    if n <= 0:
        return 0
    return max(1, min(n, int(math.ceil(0.10 * float(n)))))


def _fidelity_like_label(record):
    if record.get("proxy_fidelity_label") is not None:
        return float(record.get("proxy_fidelity_label") or 0.0)
    for key in ("fidelity_like_label", "hellinger_mean", "physical_hellinger_mean", "simulation_relative_fidelity", "relative_fidelity"):
        if record.get(key) is not None:
            return float(record.get(key) or 0.0)
    return 0.0


def evaluate(program_path):
    module = _load_program(Path(program_path))
    if not hasattr(module, "get_matching_score"):
        metrics = {{
            "validation_score": 0.0,
            "combined_score": 0.0,
            "rank_agreement": 0.0,
            "topk_overlap": 0.0,
            "validity": 0.0,
            "fig11_strict_contract_satisfied": 0.0,
        }}
        if EvaluationResult is not None:
            return EvaluationResult(metrics=metrics, artifacts={{"error": "missing get_matching_score"}})
        return metrics

    data = json.loads(TRAINING_DATA_PATH.read_text(encoding="utf-8"))
    label_metric = str(data.get("label_metric") or "fidelity_like_label")
    source_kind = str(data.get("source_kind") or "unknown")
    threshold_results = []
    baseline_comparison = []
    avg_rank_values = []
    inv_avg_rank_values = []
    rank_corr_values = []
    top_rank_overlap_values = []
    valid = 0
    total = 0

    split = data.get("split") or {{}}
    train_thresholds = [str(item) for item in split.get("train_thresholds", [])]
    validation_thresholds = [str(item) for item in split.get("validation_thresholds", [])]
    eval_threshold_set = set(validation_thresholds or (data.get("thresholds") or {{}}).keys())
    train_record_count = 0
    validation_record_count = 0

    for threshold, records in sorted((data.get("thresholds") or {{}}).items()):
        if str(threshold) not in eval_threshold_set:
            train_record_count += len(records)
            continue
        validation_record_count += len(records)
        labels = []
        feature_rows = []
        pareto_points = []
        proxy_values = []
        for record in records:
            total += 1
            label = str(record.get("pair_label") or f"{{threshold}}:{{total}}")
            fidelity_like = _fidelity_like_label(record)
            eff_util = float(record.get("effective_utilization", 0.0))
            try:
                features = record.get("features") or {{}}
                proxy_score = float(_candidate_score(module, features))
                if math.isnan(proxy_score) or math.isinf(proxy_score):
                    proxy_score = -1e9
                else:
                    valid += 1
            except Exception:
                features = record.get("features") or {{}}
                proxy_score = -1e9
            labels.append(label)
            feature_rows.append(features)
            pareto_points.append((eff_util, fidelity_like))
            proxy_values.append((label, proxy_score))
        pareto_ranks = _pareto_ranks(pareto_points)
        rank_values = [(label, 1.0 / max(float(rank), 1.0)) for label, rank in zip(labels, pareto_ranks)]
        rank_corr = _spearman(_rank(rank_values), _rank(proxy_values))
        k = _topk_count(len(records))
        top_proxy_labels = [
            label for label, _score in sorted(proxy_values, key=lambda item: (-item[1], str(item[0])))[:k]
        ]
        selected_ranks = [
            pareto_ranks[labels.index(label)]
            for label in top_proxy_labels
            if label in labels
        ]
        avg_rank = sum(float(rank) for rank in selected_ranks) / len(selected_ranks) if selected_ranks else float("inf")
        inv_avg_rank = 1.0 / avg_rank if avg_rank and math.isfinite(avg_rank) else 0.0
        best_rank = min(pareto_ranks) if pareto_ranks else 1
        best_rank_labels = {{label for label, rank in zip(labels, pareto_ranks) if rank == best_rank}}
        top_rank_overlap = len(best_rank_labels & set(top_proxy_labels)) / float(min(k, len(best_rank_labels)) or 1)
        avg_rank_values.append(avg_rank)
        inv_avg_rank_values.append(inv_avg_rank)
        rank_corr_values.append(rank_corr)
        top_rank_overlap_values.append(top_rank_overlap)
        threshold_results.append({{
            "threshold": threshold,
            "split": "validation" if str(threshold) in set(validation_thresholds) else "evaluation",
            "pair_count": len(records),
            "top_k": k,
            "avg_pareto_rank": avg_rank,
            "inv_avg_pareto_rank": inv_avg_rank,
            "rank_agreement": rank_corr,
            "top_rank_overlap": top_rank_overlap,
            "pareto_rank_histogram": {{str(rank): pareto_ranks.count(rank) for rank in sorted(set(pareto_ranks))}},
        }})
        rank_by_label = {{label: rank for label, rank in zip(labels, pareto_ranks)}}
        for baseline_name in ("random", "repo_qos_raw", "qos_normalized", "depth_ratio_only", "current_evolved"):
            baseline_values = []
            for label, features in zip(labels, feature_rows):
                try:
                    score = float(_score_named_baseline(baseline_name, label, features, module))
                    if math.isnan(score) or math.isinf(score):
                        score = -1e9
                except Exception:
                    score = -1e9
                baseline_values.append((label, score))
            baseline_top_labels = [
                label
                for label, _score in sorted(baseline_values, key=lambda item: (-item[1], str(item[0])))[:k]
            ]
            baseline_selected_ranks = [rank_by_label[label] for label in baseline_top_labels if label in rank_by_label]
            baseline_avg_rank = (
                sum(float(rank) for rank in baseline_selected_ranks) / len(baseline_selected_ranks)
                if baseline_selected_ranks
                else float("inf")
            )
            baseline_best_overlap = len(best_rank_labels & set(baseline_top_labels)) / float(min(k, len(best_rank_labels)) or 1)
            baseline_comparison.append({{
                "threshold": threshold,
                "split": "validation" if str(threshold) in set(validation_thresholds) else "evaluation",
                "baseline": baseline_name,
                "top_k": k,
                "avg_pareto_rank": baseline_avg_rank,
                "inv_avg_pareto_rank": 1.0 / baseline_avg_rank if baseline_avg_rank and math.isfinite(baseline_avg_rank) else 0.0,
                "top_rank_overlap": baseline_best_overlap,
                "selected_rank_histogram": {{
                    str(rank): baseline_selected_ranks.count(rank)
                    for rank in sorted(set(baseline_selected_ranks))
                }},
                "selected_pair_labels": baseline_top_labels[: min(10, len(baseline_top_labels))],
            }})

    avg_pareto_rank = sum(avg_rank_values) / len(avg_rank_values) if avg_rank_values else float("inf")
    inv_avg_pareto_rank = sum(inv_avg_rank_values) / len(inv_avg_rank_values) if inv_avg_rank_values else 0.0
    rank_agreement = sum(rank_corr_values) / len(rank_corr_values) if rank_corr_values else 0.0
    top_rank_overlap = sum(top_rank_overlap_values) / len(top_rank_overlap_values) if top_rank_overlap_values else 0.0
    validity = valid / float(total) if total else 0.0
    rank_quality = max(0.0, min(1.0, (rank_agreement + 1.0) / 2.0))
    validation_score = inv_avg_pareto_rank * validity
    metrics = {{
        "validation_score": float(validation_score),
        "combined_score": float(validation_score),
        "avg_pareto_rank": float(avg_pareto_rank) if math.isfinite(avg_pareto_rank) else 1e9,
        "inv_avg_pareto_rank": float(inv_avg_pareto_rank),
        "rank_agreement": float(rank_agreement),
        "rank_quality": float(rank_quality),
        "top_rank_overlap": float(top_rank_overlap),
        "objective_primary_score": "inv_avg_pareto_rank",
        "objective_diagnostic_metrics": "rank_agreement,top_rank_overlap",
        "front_overlap_affects_validation_score": 0.0,
        "validity": float(validity),
        "objective_aggregation": "selected_topk_average_pareto_rank",
        "objective_selection_target": "multiprogramming_pair_selection",
        "objective_metrics": "effective_utilization," + label_metric,
        "ground_truth_source_kind": source_kind,
        "ground_truth_label_metric": label_metric,
        "evaluation_split_mode": str(split.get("mode") or "none"),
        "train_record_count": float(train_record_count),
        "validation_record_count": float(validation_record_count),
        "fig11_strict_contract_satisfied": 0.0,
    }}
    artifacts = {{
        "threshold_results": threshold_results,
        "baseline_comparison": baseline_comparison,
        "training_data_path": str(TRAINING_DATA_PATH),
        "split": split,
        "note": "Candidate objective is scored only on the held-out/evaluation split by inverse selected-top-k average Pareto rank over effective utilization and the configured fidelity-like ground-truth label. Rank agreement and rank-1 front overlap are diagnostics only.",
    }}
    if EvaluationResult is not None:
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    return metrics
'''


def _resolve_path(raw: Any, repo_root: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path if path.exists() else None


def _ground_truth_metrics_path(state: dict[str, Any], repo_root: Path) -> Path | None:
    env_path = _resolve_path(os.getenv("REPRO_OPENEVOLVE_GROUND_TRUTH_METRICS"), repo_root)
    if env_path:
        return env_path
    manifest_path = (((state.get("last_manifest") or {}).get("artifacts") or {}).get("metrics"))
    path = _resolve_path(manifest_path, repo_root)
    if path:
        return path
    return _resolve_path(state.get("last_fig11_metrics_path"), repo_root)


def _simulation_memory_path(state: dict[str, Any], repo_root: Path) -> Path | None:
    path = _resolve_path(state.get("last_simulation_memory_path"), repo_root)
    if path:
        return path
    memory = state.get("last_simulation_memory") or {}
    if isinstance(memory, dict):
        return _resolve_path(memory.get("artifact_path"), repo_root)
    return None


def _physical_records_path(state: dict[str, Any], repo_root: Path, artifact_dir: Path) -> Path | None:
    env_path = _resolve_path(os.getenv("REPRO_OPENEVOLVE_PHYSICAL_QPU_RECORDS"), repo_root)
    if env_path:
        return env_path
    path = _resolve_path(state.get("last_physical_qpu_records_path"), repo_root)
    if path:
        return path
    probe = state.get("last_physical_qpu_env_probe") or {}
    if isinstance(probe, dict):
        path = _resolve_path(probe.get("physical_records_path"), repo_root)
        if path:
            return path
    manifest_path = repo_root / "qpu_env/physical_qpu_jobs/manifest.json"
    if not manifest_path.exists():
        return None
    probe_payload = build_physical_qpu_probe(repo_root, artifact_dir / "physical_qpu_env_probe")
    return _resolve_path(probe_payload.get("physical_records_path"), repo_root)


def _layout_span(record: dict[str, Any]) -> float:
    layout = []
    for key in ("layout1", "layout2"):
        values = record.get(key) or []
        if isinstance(values, list):
            layout.extend(int(value) for value in values if isinstance(value, int))
    if not layout:
        return 0.0
    return float(max(layout) - min(layout) + 1)


def _training_record(record: dict[str, Any], selected_qubits: int | None) -> dict[str, Any]:
    joint_qubits = int(record.get("joint_qubits") or 0)
    selected = int(selected_qubits or joint_qubits or 1)
    left_qubits = float(record.get("left_qubits") or 0.0)
    right_qubits = float(record.get("right_qubits") or 0.0)
    scale_ratio = float(joint_qubits) / max(float(selected), 1.0)
    features = {
        "matching_score": float(record.get("matching_score") or record.get("selection_score") or 0.0),
        "selection_score": float(record.get("selection_score") or 0.0),
        "effective_utilization": float(record.get("effective_utilization") or 0.0),
        "effective_utilization_percent": float(record.get("effective_utilization_percent") or 0.0),
        "fidelity": float(record.get("fidelity") or 0.0),
        "solo_fidelity": float(record.get("solo_fidelity") or 0.0),
        "entanglement": float(record.get("entanglement", 0.0) or 0.0),
        "measurement": float(record.get("measurement", 0.0) or 0.0),
        "parallelism": float(record.get("parallelism", 0.0) or 0.0),
        "entanglement_ratio": float(record.get("entanglement_ratio", 0.0) or 0.0),
        "joint_qubits": float(joint_qubits),
        "selected_qubits": float(selected),
        "scale_ratio": scale_ratio,
        "source_scale_hint": scale_ratio,
        "utilization_pressure": max(0.0, scale_ratio - 1.0),
        "depth_ratio": scale_ratio,
        "left_qubits": left_qubits,
        "right_qubits": right_qubits,
        "qubit_imbalance": abs(left_qubits - right_qubits) / max(left_qubits + right_qubits, 1.0),
        "layout_span": _layout_span(record),
    }
    return {
        "pair_label": record.get("pair_label"),
        "left_application": record.get("left_application"),
        "right_application": record.get("right_application"),
        "effective_utilization": features["effective_utilization"],
        "relative_fidelity": float(record.get("relative_fidelity") or 0.0),
        "simulation_relative_fidelity": float(record.get("relative_fidelity") or 0.0),
        "features": features,
    }


def _physical_training_record(record: dict[str, Any], repo_root: Path | None = None) -> dict[str, Any] | None:
    if record.get("hellinger_mean") is None:
        return None
    features = physical_record_features(record, repo_root=repo_root)
    hellinger = float(record.get("hellinger_mean") or 0.0)
    return {
        "pair_label": record.get("pair_label"),
        "left_application": record.get("left_application"),
        "right_application": record.get("right_application"),
        "left_qubits": record.get("left_qubits"),
        "right_qubits": record.get("right_qubits"),
        "joint_qubits": record.get("joint_qubits"),
        "backend": record.get("backend"),
        "util_label": record.get("util_label"),
        "layout_mode": record.get("layout_mode"),
        "source_file": record.get("source_file"),
        "job_id": record.get("job_id"),
        "shots": record.get("shots"),
        "metric_source": record.get("metric_source") or "hellinger_mean",
        "provenance_kind": record.get("provenance_kind") or "physical_qpu_measurement",
        "effective_utilization": float(record.get("effective_utilization") or 0.0),
        "fidelity_like_label": hellinger,
        "hellinger_mean": hellinger,
        "physical_hellinger_mean": hellinger,
        "features": features,
        "feature_provenance": {
            "source": features.get("feature_source"),
            "metadata_materialized": features.get("metadata_materialized"),
            "placeholder_constant_features_forbidden": True,
        },
    }


def _parse_csv_list(raw: str, fallback: list[str]) -> list[str]:
    values = [item.strip() for item in str(raw or "").replace(";", ",").split(",") if item.strip()]
    return values or list(fallback)


def _qpu_feedback_split(thresholds: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train = _parse_csv_list(os.getenv("REPRO_QPU_FEEDBACK_TRAIN_UTILS", "util30,util60"), ["util30", "util60"])
    validate = _parse_csv_list(os.getenv("REPRO_QPU_FEEDBACK_VALIDATE_UTILS", "util88"), ["util88"])
    available = set(thresholds)
    train = [item for item in train if item in available]
    validate = [item for item in validate if item in available and item not in set(train)]
    if not validate and available:
        validate = [sorted(available, key=_threshold_key_to_float)[-1]]
        train = [item for item in sorted(available, key=_threshold_key_to_float) if item not in set(validate)]
    return {
        "mode": "holdout_by_util",
        "train_thresholds": train,
        "validation_thresholds": validate,
        "leakage_policy": "fitness_uses_validation_thresholds_only",
        "human_qpu_gate": True,
        "qpu_job_submission": "manual_only",
    }


def _rank_for_correlation(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = avg_rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    num = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_den = sum((a - left_mean) ** 2 for a in left)
    right_den = sum((b - right_mean) ** 2 for b in right)
    den = (left_den * right_den) ** 0.5
    return num / den if den else 0.0


def _spearman_values(left: list[float], right: list[float]) -> float:
    return _pearson(_rank_for_correlation(left), _rank_for_correlation(right))


def _feature_values(
    thresholds: dict[str, list[dict[str, Any]]],
    threshold_names: list[str],
    feature: str,
) -> tuple[list[float], list[float]]:
    values: list[float] = []
    labels: list[float] = []
    for threshold in threshold_names:
        for record in thresholds.get(str(threshold), []):
            features = record.get("features") if isinstance(record.get("features"), dict) else {}
            if features.get(feature) is None:
                continue
            label = record.get("hellinger_mean")
            if label is None:
                label = record.get("fidelity_like_label")
            if label is None:
                continue
            try:
                values.append(float(features.get(feature)))
                labels.append(float(label))
            except Exception:
                continue
    return values, labels


def _normalize_proxy_value(value: float, low: float, high: float, direction: str) -> float:
    if high <= low:
        scaled = 0.5
    else:
        scaled = (value - low) / (high - low)
    scaled = max(0.0, min(1.0, scaled))
    return 1.0 - scaled if direction == "inverse" else scaled


def _apply_pre_evolution_metric_proxy(
    training_payload: dict[str, Any],
    *,
    profile: dict[str, str],
    objective_resolution: dict[str, Any],
    semantic_proxy_proposal: dict[str, Any] | None = None,
    semantic_feature_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if objective_resolution.get("mode") != "proxy_substitution":
        training_payload["fitness_second_metric"] = training_payload.get("label_metric")
        return {"enabled": False, "reason": "objective_resolution_is_not_proxy_substitution"}

    thresholds = training_payload.get("thresholds") if isinstance(training_payload.get("thresholds"), dict) else {}
    split = training_payload.get("split") if isinstance(training_payload.get("split"), dict) else {}
    train_thresholds = [str(item) for item in split.get("train_thresholds", [])]
    validation_thresholds = [str(item) for item in split.get("validation_thresholds", [])]
    fallback_proxy_features = [
        "qubit_imbalance",
        "scale_ratio",
        "layout_span",
        "joint_qubits",
        "left_qubits",
        "right_qubits",
        "depth_ratio",
        "depth_density",
        "cnot_ratio",
        "cnot_density",
        "nonlocal_ratio",
        "nonlocal_density",
        "measure_ratio",
        "measure_density",
        "instr_ratio",
        "instr_density",
        "critical_depth_ratio",
        "critical_depth_density",
    ]
    proposal_candidates = []
    validation_candidates = []
    if isinstance(semantic_feature_validation, dict) and semantic_feature_validation.get("success"):
        for item in semantic_feature_validation.get("selected_feature_bundle") or []:
            if isinstance(item, dict) and item.get("feature"):
                validation_candidates.append(str(item.get("feature")))
        if not validation_candidates:
            for item in semantic_feature_validation.get("selected_feature_names") or []:
                if item:
                    validation_candidates.append(str(item))
    if isinstance(semantic_proxy_proposal, dict) and semantic_proxy_proposal.get("success"):
        for item in semantic_proxy_proposal.get("candidates") or []:
            if isinstance(item, dict) and item.get("materialized_feature_name"):
                proposal_candidates.append(str(item.get("materialized_feature_name")))
    if validation_candidates:
        configured_candidates = validation_candidates
        candidate_source = "llm_semantic_feature_validation"
    elif proposal_candidates:
        configured_candidates = proposal_candidates
        candidate_source = "llm_semantic_metric_proposal"
    else:
        configured_candidates = _parse_csv_list(
            profile.get(
                "OE_PROXY_FEATURES",
                "depth_ratio,depth_density,cnot_ratio,cnot_density,nonlocal_ratio,nonlocal_density,measure_ratio,measure_density,instr_ratio,instr_density",
            ),
            fallback_proxy_features,
        )
        candidate_source = "configured_or_legacy_fallback"
    candidates = list(dict.fromkeys(configured_candidates + ([] if proposal_candidates else fallback_proxy_features)))

    scored: list[dict[str, Any]] = []
    for feature in candidates:
        train_values, train_labels = _feature_values(thresholds, train_thresholds, feature)
        if len(train_values) < 2:
            continue
        direct = _spearman_values(train_values, train_labels)
        inverse = _spearman_values([-value for value in train_values], train_labels)
        if inverse > direct:
            direction = "inverse"
            score = inverse
        else:
            direction = "direct"
            score = direct
        all_values, _ = _feature_values(thresholds, list(thresholds.keys()), feature)
        low = min(all_values) if all_values else 0.0
        high = max(all_values) if all_values else 1.0
        validation_values, validation_labels = _feature_values(thresholds, validation_thresholds, feature)
        validation_distinct_values = len({round(float(value), 12) for value in validation_values})
        target_split_has_variance = validation_distinct_values >= 2
        validation_corr = (
            _spearman_values(
                [-value for value in validation_values] if direction == "inverse" else validation_values,
                validation_labels,
            )
            if len(validation_values) >= 2
            else None
        )
        scored.append(
            {
                "feature": feature,
                "direction": direction,
                "train_spearman": score,
                "validation_spearman": validation_corr,
                "value_min": low,
                "value_max": high,
                "train_record_count": len(train_values),
                "validation_record_count": len(validation_values),
                "validation_distinct_values": validation_distinct_values,
                "target_split_has_variance": target_split_has_variance,
            }
        )

    if not scored:
        training_payload["fitness_second_metric"] = training_payload.get("label_metric")
        return {"enabled": False, "reason": "no_proxy_feature_had_training_labels"}

    guard_enabled = os.getenv("REPRO_PROXY_GENERALIZATION_GUARD", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    try:
        min_validation_spearman = float(os.getenv("REPRO_PROXY_MIN_VALIDATION_SPEARMAN", "0.20"))
    except ValueError:
        min_validation_spearman = 0.20
    selectable = [item for item in scored if item.get("target_split_has_variance")]
    validation_selectable = [
        item
        for item in selectable
        if item.get("validation_spearman") is not None
    ]
    validation_stable = [
        item
        for item in validation_selectable
        if float(item.get("validation_spearman") or 0.0) >= min_validation_spearman
    ]
    rejected_by_generalization_guard = [
        {
            "feature": item.get("feature"),
            "train_spearman": item.get("train_spearman"),
            "validation_spearman": item.get("validation_spearman"),
            "reason": f"validation_spearman_below_{min_validation_spearman}",
        }
        for item in validation_selectable
        if float(item.get("validation_spearman") or 0.0) < min_validation_spearman
    ]
    if guard_enabled and validation_stable:
        selection_policy = "validation_first_then_train"
        selected_pool = validation_stable
        selected = sorted(
            selected_pool,
            key=lambda item: (
                -float(item.get("validation_spearman") or 0.0),
                -float(item["train_spearman"]),
                str(item["feature"]),
            ),
        )[0]
    elif guard_enabled and validation_selectable:
        selection_policy = "best_available_validation_below_threshold"
        selected_pool = validation_selectable
        selected = sorted(
            selected_pool,
            key=lambda item: (
                -float(item.get("validation_spearman") or 0.0),
                -float(item["train_spearman"]),
                str(item["feature"]),
            ),
        )[0]
    else:
        selection_policy = "train_spearman_only_no_validation_split"
        selected_pool = selectable or scored
        selected = sorted(selected_pool, key=lambda item: (-float(item["train_spearman"]), str(item["feature"])))[0]
    feature = str(selected["feature"])
    direction = str(selected["direction"])
    low = float(selected["value_min"])
    high = float(selected["value_max"])
    for records in thresholds.values():
        for record in records:
            features = record.get("features") if isinstance(record.get("features"), dict) else {}
            try:
                raw_value = float(features.get(feature))
            except Exception:
                raw_value = low
            record["proxy_fidelity_label"] = _normalize_proxy_value(raw_value, low, high, direction)
            record["proxy_metric_source"] = f"feature:{feature}:{direction}"
            record["expensive_validation_label_metric"] = training_payload.get("label_metric")

    training_payload["expensive_label_metric"] = training_payload.get("label_metric")
    training_payload["label_metric"] = "proxy_estimated_fidelity"
    training_payload["fitness_second_metric"] = "proxy_estimated_fidelity"
    training_payload["proxy_metric_validation"] = {
        "enabled": True,
        "role": "pre_evolution_expensive_metric_substitute",
        "proxy_is_evolved_function": False,
        "selected_feature": feature,
        "direction": direction,
        "selection_split": "train",
        "validation_split": "validation",
        "expensive_label_metric": training_payload.get("expensive_label_metric"),
        "selected": selected,
        "candidates": scored,
        "candidate_source": candidate_source,
        "selection_policy": selection_policy,
        "rejected_by_generalization_guard": rejected_by_generalization_guard,
        "semantic_proxy_proposal": semantic_proxy_proposal if proposal_candidates else None,
        "semantic_feature_validation": semantic_feature_validation if validation_candidates else None,
        "selection_guard": {
            "requires_target_split_feature_variance": True,
            "generalization_guard_enabled": guard_enabled,
            "min_validation_spearman": min_validation_spearman,
            "selectable_candidate_count": len(selectable),
            "validation_selectable_candidate_count": len(validation_selectable),
            "validation_stable_candidate_count": len(validation_stable),
            "fallback_used": not bool(selectable),
        },
    }
    return training_payload["proxy_metric_validation"]


def _build_training_data_from_physical_records(
    source_path: Path,
    *,
    backend_filter: str,
    layout_mode_filter: str | None = "qos",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    raw_records = payload.get("records") if isinstance(payload, dict) else payload
    thresholds: dict[str, list[dict[str, Any]]] = {}
    backend_counts: dict[str, int] = {}
    util_counts: dict[str, int] = {}
    skipped = 0
    for record in raw_records if isinstance(raw_records, list) else []:
        if not isinstance(record, dict):
            skipped += 1
            continue
        backend = str(record.get("backend") or "")
        layout_mode = str(record.get("layout_mode") or "")
        if backend_filter and backend != backend_filter:
            continue
        if layout_mode_filter and layout_mode and layout_mode != layout_mode_filter:
            continue
        row = _physical_training_record(record, repo_root=repo_root)
        if row is None:
            skipped += 1
            continue
        util_label = str(record.get("util_label") or "unknown")
        thresholds.setdefault(util_label, []).append(row)
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        util_counts[util_label] = util_counts.get(util_label, 0) + 1

    status = "pair_record_ground_truth" if thresholds else "missing_pair_records"
    split = _qpu_feedback_split(thresholds)
    train_count = sum(len(thresholds.get(key, [])) for key in split.get("train_thresholds", []))
    validation_count = sum(len(thresholds.get(key, [])) for key in split.get("validation_thresholds", []))
    return {
        "source_metrics_path": str(source_path),
        "source_kind": "physical_qpu",
        "physical_backend": backend_filter,
        "physical_layout_mode_filter": layout_mode_filter,
        "physical_metric_source": "hellinger_mean",
        "feature_materialization": {
            "source": "benchmark_qasm_metadata",
            "repo_root": str(repo_root) if repo_root else None,
            "placeholder_constant_features_forbidden": True,
        },
        "label_metric": f"{backend_filter}.hellinger_mean" if backend_filter else "hellinger_mean",
        "thresholds": thresholds,
        "split": split,
        "train_record_count": train_count,
        "validation_record_count": validation_count,
        "selected_qubits_by_threshold": {},
        "record_count": sum(len(records) for records in thresholds.values()),
        "status": status,
        "backend_counts": backend_counts,
        "util_counts": util_counts,
        "skipped_record_count": skipped,
        "no_mp_target_size_records": {},
        "scale_memory_available": False,
        "eval_shots": 8192,
        "metric_provenance_shots": 8192,
        "simulation_memory": None,
    }


def _build_training_data_from_payload(source_path: Path | None, payload: dict[str, Any], source_kind: str) -> dict[str, Any]:
    by_threshold = payload.get("selected_pair_simulation_records_by_threshold") or {}
    thresholds: dict[str, list[dict[str, Any]]] = {}
    selected_qubits_by_threshold: dict[str, int] = {}
    for threshold, threshold_payload in by_threshold.items():
        if not isinstance(threshold_payload, dict):
            continue
        selected_qubits = threshold_payload.get("selected_qubits")
        threshold_key = str(threshold)
        if selected_qubits is not None:
            try:
                selected_qubits_by_threshold[threshold_key] = int(selected_qubits)
            except Exception:
                pass
        qos_records = ((threshold_payload.get("qos") or {}).get("records") or [])
        rows = [
            _training_record(record, selected_qubits)
            for record in qos_records
            if isinstance(record, dict)
        ]
        if rows:
            thresholds[threshold_key] = rows
    no_mp_records = payload.get("no_mp_target_size_records") or {}
    if thresholds:
        status = "pair_record_ground_truth"
    elif isinstance(no_mp_records, dict) and no_mp_records:
        status = "scale_memory_only"
    else:
        status = "missing_pair_records"
    return {
        "source_metrics_path": str(source_path) if source_path else None,
        "source_kind": source_kind,
        "thresholds": thresholds,
        "selected_qubits_by_threshold": selected_qubits_by_threshold,
        "record_count": sum(len(records) for records in thresholds.values()),
        "status": status,
        "no_mp_target_size_records": no_mp_records if isinstance(no_mp_records, dict) else {},
        "scale_memory_available": status == "scale_memory_only",
        "eval_shots": payload.get("simulation_shots"),
        "metric_provenance_shots": ((payload.get("metric_provenance") or {}).get("relative_fidelity") or {}).get("shots"),
        "simulation_memory": payload.get("seed_transfer") if source_kind == "simulation_memory" else None,
    }


def _build_training_data(
    metrics_path: Path | None,
    memory_path: Path | None = None,
    physical_path: Path | None = None,
    *,
    physical_backend: str = "ibm_torino",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if physical_path is not None:
        physical = _build_training_data_from_physical_records(
            physical_path,
            backend_filter=physical_backend,
            layout_mode_filter="qos",
            repo_root=repo_root,
        )
        if physical.get("status") == "pair_record_ground_truth":
            return physical
    if memory_path is not None:
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        if isinstance(memory, dict):
            training = _build_training_data_from_payload(memory_path, memory, "simulation_memory")
            if training.get("status") in {"pair_record_ground_truth", "scale_memory_only"}:
                return training
    if metrics_path is None:
        return {
            "source_metrics_path": None,
            "source_kind": None,
            "thresholds": {},
            "record_count": 0,
            "status": "missing_ground_truth_metrics",
        }
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return _build_training_data_from_payload(metrics_path, metrics, "metrics")


def _profile_env(state: dict[str, Any]) -> dict[str, str]:
    probe = state.get("last_openevolve_param_probe") or {}
    profile = probe.get("parameter_profile") or {}
    env = profile.get("env") or {}
    return {str(key): str(value) for key, value in env.items() if value is not None}


def _profile_objective(state: dict[str, Any]) -> dict[str, Any]:
    probe = state.get("last_openevolve_param_probe") or {}
    profile = probe.get("parameter_profile") or {}
    objective = profile.get("objective") or {}
    return dict(objective) if isinstance(objective, dict) else {}


def _profile_objective_requirements(state: dict[str, Any]) -> dict[str, Any]:
    probe = state.get("last_openevolve_param_probe") or {}
    profile = probe.get("parameter_profile") or {}
    requirements = profile.get("objective_requirements") or probe.get("objective_requirements") or {}
    return dict(requirements) if isinstance(requirements, dict) else {}


def _profile_objective_resolution(state: dict[str, Any]) -> dict[str, Any]:
    probe = state.get("last_openevolve_param_probe") or {}
    profile = probe.get("parameter_profile") or {}
    resolution = profile.get("objective_resolution") or probe.get("objective_resolution") or {}
    return dict(resolution) if isinstance(resolution, dict) else {}


def _profile_settings(state: dict[str, Any]) -> dict[str, Any]:
    probe = state.get("last_openevolve_param_probe") or {}
    profile = probe.get("parameter_profile") or {}
    settings = profile.get("settings") or {}
    return dict(settings) if isinstance(settings, dict) else {}


def _profile_name(state: dict[str, Any]) -> str:
    probe = state.get("last_openevolve_param_probe") or {}
    profile = probe.get("parameter_profile") or {}
    return str(profile.get("name") or "agent_default")


def _profile_schema_path(state: dict[str, Any]) -> str | None:
    probe = state.get("last_openevolve_param_probe") or {}
    return probe.get("artifact_path")


def _param(profile: dict[str, str], name: str, fallback: str) -> str:
    return os.getenv(name, profile.get(name, fallback))


def _parse_eval_utils(raw: str) -> list[str]:
    values: list[str] = []
    for item in str(raw or "").replace(",", ":").split(":"):
        text = item.strip()
        if text:
            values.append(text)
    return values


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _openevolve_artifact_policy(state: dict[str, Any]) -> dict[str, Any]:
    forced = os.getenv("REPRO_OPENEVOLVE_INCLUDE_ARTIFACTS")
    if forced is not None:
        include = forced.strip().lower() not in {"0", "false", "no", "off"}
        return {
            "mode": "forced_by_env",
            "include_artifacts": include,
            "trigger_signals": ["REPRO_OPENEVOLVE_INCLUDE_ARTIFACTS"],
            "selected_artifact_intents": ["env_forced_all"] if include else [],
            "initial_attempt_minimal": not include,
        }
    mode = os.getenv("REPRO_OPENEVOLVE_ARTIFACT_POLICY", "minimal_then_diagnostic").strip().lower()
    if mode in {"always", "artifact_rich"}:
        return {
            "mode": mode,
            "include_artifacts": True,
            "trigger_signals": ["policy_always"],
            "selected_artifact_intents": ["evaluator_threshold_results", "baseline_comparison"],
            "initial_attempt_minimal": False,
        }
    if mode in {"never", "minimal"}:
        return {
            "mode": mode,
            "include_artifacts": False,
            "trigger_signals": [],
            "selected_artifact_intents": [],
            "initial_attempt_minimal": True,
        }

    trigger_signals: list[str] = []
    collect = state.get("last_openevolve_slurm_collect")
    if isinstance(collect, dict) and not collect.get("success"):
        trigger_signals.append("openevolve_collect_failed")
        mutation = collect.get("mutation_verification") if isinstance(collect.get("mutation_verification"), dict) else {}
        if mutation and not mutation.get("mutation_applied"):
            trigger_signals.append("no_code_mutation")
        for warning in collect.get("warnings") or []:
            text = str(warning).lower()
            if "no-op" in text or "hash matches" in text:
                trigger_signals.append("no_op_best_program")
                break
    verify = state.get("last_openevolve_proxy_verify")
    if isinstance(verify, dict) and not verify.get("success"):
        trigger_signals.append("proxy_verify_failed")
        rank = ((verify.get("verification") or {}).get("rank_comparison") or {})
        if rank.get("status"):
            trigger_signals.append(f"rank_comparison_{rank.get('status')}")
    include = bool(trigger_signals)
    return {
        "mode": "minimal_then_diagnostic",
        "include_artifacts": include,
        "trigger_signals": sorted(set(trigger_signals)),
        "selected_artifact_intents": [
            "baseline_comparison",
            "threshold_results",
            "selected_pair_rank_diagnostics",
        ]
        if include
        else [],
        "initial_attempt_minimal": not include,
        "note": (
            "First OpenEvolve attempt runs without evaluator artifacts. "
            "Diagnostic artifacts are added only after collect/verify failure signals."
        ),
    }


def _threshold_key_to_float(key: str) -> float:
    try:
        raw = float(str(key).strip().rstrip("%"))
        return raw / 100.0 if raw > 1.0 else raw
    except Exception:
        return 0.0


def _seed_transfer_payload(training_payload: dict[str, Any], profile: dict[str, str]) -> dict[str, Any]:
    transfer = _param(profile, "OE_SEED_TRANSFER", "small_to_large")
    eval_utils = _parse_eval_utils(_param(profile, "OE_EVAL_UTILS", "30:60:88"))
    present = sorted((training_payload.get("thresholds") or {}).keys(), key=_threshold_key_to_float)
    memory_seed = training_payload.get("simulation_memory") if isinstance(training_payload.get("simulation_memory"), dict) else {}
    if not present and memory_seed:
        present = sorted([str(item) for item in memory_seed.get("source_thresholds", [])], key=_threshold_key_to_float)
    present_norm = {_threshold_key_to_float(item) for item in present}
    target_thresholds = [
        item for item in eval_utils
        if not any(abs(_threshold_key_to_float(item) - seen) < 1e-9 for seen in present_norm)
    ]
    enabled = transfer == "small_to_large"
    return {
        "enabled": enabled,
        "strategy": transfer,
        "source_policy": _param(profile, "OE_SEED_SOURCE_POLICY", "completed_smaller_thresholds"),
        "target_policy": _param(profile, "OE_SEED_TARGET_POLICY", "timed_out_larger_thresholds"),
        "source_kind": training_payload.get("source_kind"),
        "label_metric": training_payload.get("label_metric"),
        "source_thresholds": present,
        "target_thresholds": target_thresholds,
        "selected_qubits_by_threshold": training_payload.get("selected_qubits_by_threshold") or {},
        "uses_simulation_grounded_sources": training_payload.get("source_kind") in {"simulation_memory", "metrics"} and bool(present),
        "uses_pair_level_ground_truth_sources": bool(present),
        "uses_pair_level_ground_truth": training_payload.get("status") == "pair_record_ground_truth",
        "strict_fig11_success": False,
        "note": "Pair-level ground-truth records seed OpenEvolve proxy search for thresholds where strict simulation is too expensive.",
    }


def _build_config(
    eval_shots: int,
    output_dir: Path,
    profile: dict[str, str],
    template_dir: Path,
    *,
    selected_proxy_feature: str | None = None,
    selected_proxy_features: list[str] | None = None,
    include_artifacts: bool = False,
) -> str:
    model = os.getenv("REPRO_OPENEVOLVE_MODEL", "Qwen2.5-14B-Instruct")
    api_base = os.getenv("REPRO_OPENEVOLVE_API_BASE", "http://localhost:8000/v1")
    api_key_ref = os.getenv("REPRO_OPENEVOLVE_API_KEY_CONFIG", "${REPRO_OPENEVOLVE_API_KEY}")
    iterations = int(os.getenv("REPRO_OPENEVOLVE_ITERATIONS", os.getenv("OE_MAX_ITERATIONS", "20")) or "20")
    population_size = int(os.getenv("REPRO_OPENEVOLVE_POPULATION_SIZE", "20") or "20")
    archive_size = int(os.getenv("REPRO_OPENEVOLVE_ARCHIVE_SIZE", "10") or "10")
    eval_utils = _param(profile, "OE_EVAL_UTILS", "30:60:88")
    multi_util_agg = _param(profile, "OE_MULTI_UTIL_AGG", "mean")
    pareto_metric = _param(profile, "OE_PARETO_SECOND_METRIC", "proxy_estimated_fidelity")
    proxy_feature = selected_proxy_feature or _param(profile, "OE_PROXY_FEATURE", "qubit_imbalance")
    proxy_features = _param(
        profile,
        "OE_PROXY_FEATURES",
        "qubit_imbalance,scale_ratio,layout_span,joint_qubits,left_qubits,right_qubits,depth_ratio,depth_density,cnot_ratio,cnot_density,nonlocal_ratio,nonlocal_density,measure_ratio,measure_density,instr_ratio,instr_density,critical_depth_ratio,critical_depth_density",
    )
    if selected_proxy_feature and selected_proxy_feature not in _parse_csv_list(proxy_features, []):
        proxy_features = proxy_features + "," + selected_proxy_feature
    if selected_proxy_features:
        proxy_features = ",".join(
            item for item in dict.fromkeys(str(item) for item in selected_proxy_features if str(item))
        )
    top_k_ratio = _param(profile, "OE_TOP_K_RATIO", "0.10")
    restrict_to_csv = _param(profile, "OE_RESTRICT_TO_PAIR_CSV", "0")
    seed_transfer = _param(profile, "OE_SEED_TRANSFER", "small_to_large")
    seed_source_policy = _param(profile, "OE_SEED_SOURCE_POLICY", "completed_smaller_thresholds")
    seed_target_policy = _param(profile, "OE_SEED_TARGET_POLICY", "timed_out_larger_thresholds")
    return f"""# Generated QOS-Agent OpenEvolve proxy-search config.
max_iterations: {iterations}
checkpoint_interval: 5
log_level: INFO
early_stopping_metric: validation_score
diff_based_evolution: false
diff_pattern: '<<<<<<< SEARCH\\n(?:```(?:python)?\\n)?(.*?)(?:\\n```)?\\n=======\\n(?:```(?:python)?\\n)?(.*?)(?:\\n```)?\\n>>>>>>> REPLACE'
database:
  population_size: {population_size}
  archive_size: {archive_size}
evaluator:
  timeout: 120
  cascade_evaluation: false
  parallel_evaluations: 1
prompt:
  template_dir: "{template_dir.as_posix()}"
  include_artifacts: {_bool_text(include_artifacts)}
  use_template_stochasticity: false
  diff_summary_max_line_len: 200
  diff_summary_max_lines: 60
llm:
  api_base: "{api_base}"
  api_key: "{api_key_ref}"
  models:
    - name: "{model}"
      weight: 1.0
  temperature: 0.2
custom:
  eval_shots: {eval_shots}
  oe_eval_utils: "{eval_utils}"
  oe_multi_util_agg: "{multi_util_agg}"
  oe_pareto_second_metric: "{pareto_metric}"
  oe_proxy_feature: "{proxy_feature}"
  oe_proxy_features: "{proxy_features}"
  oe_top_k_ratio: "{top_k_ratio}"
  oe_restrict_to_pair_csv: "{restrict_to_csv}"
  oe_seed_transfer: "{seed_transfer}"
  oe_seed_source_policy: "{seed_source_policy}"
  oe_seed_target_policy: "{seed_target_policy}"
  verification_requires_simulation: true
  output_dir: {output_dir.as_posix()}
"""


def _run_generated_baseline_comparison(evaluator_path: Path, initial_program: Path, output_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": False,
        "evaluator": str(evaluator_path),
        "program": str(initial_program),
        "artifact_path": str(output_path),
    }
    try:
        spec = importlib.util.spec_from_file_location("qos_agent_generated_proxy_evaluator", evaluator_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import generated evaluator: {evaluator_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = module.evaluate(str(initial_program))
        metrics = getattr(raw, "metrics", None)
        artifacts = getattr(raw, "artifacts", None)
        if isinstance(raw, dict):
            metrics = raw.get("metrics", metrics)
            artifacts = raw.get("artifacts", artifacts)
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        payload = {
            "success": True,
            "metrics": metrics if isinstance(metrics, dict) else {},
            "baseline_comparison": artifacts.get("baseline_comparison") or [],
            "threshold_results": artifacts.get("threshold_results") or [],
            "training_data_path": artifacts.get("training_data_path"),
            "split": artifacts.get("split"),
        }
        _write_json(output_path, payload)
        result.update(payload)
    except Exception as exc:
        result["error"] = str(exc)
        _write_json(output_path, result)
    return result


def build_proxy_payload(
    *,
    repo_root: Path,
    artifact_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    probe = state.get("last_openevolve_target_probe")
    if not isinstance(probe, dict) or not probe.get("success"):
        probe = build_target_probe(repo_root)

    evolve_target = _pick_candidate(probe, "evolve", "openevolve.api:run_evolution")
    verify_target = _pick_candidate(probe, "verify", "openevolve.evaluator:Evaluator.evaluate_program")
    openevolve = dict(probe.get("openevolve") or {})
    openevolve["entrypoint"] = evolve_target.get("entrypoint", "openevolve.api:run_evolution")

    eval_shots = int(os.getenv("REPRO_AER_SHOTS", "8192"))
    dry_run = os.getenv("REPRO_OPENEVOLVE_DRY_RUN", "1").strip().lower() not in {"0", "false", "no", "off"}
    param_probe = state.get("last_openevolve_param_probe") or {}
    training_data_schema = str(param_probe.get("training_data_schema") or "ibm_torino_physical_pair_records")
    profile = _profile_env(state)
    agent_objective = _profile_objective(state)
    objective_requirements = _profile_objective_requirements(state)
    objective_resolution = _profile_objective_resolution(state)
    agent_settings = _profile_settings(state)
    profile_name = _profile_name(state)
    profile_schema_path = _profile_schema_path(state)
    artifact_policy = _openevolve_artifact_policy(state)
    physical_backend_filter = str(
        os.getenv(
            "REPRO_OPENEVOLVE_PHYSICAL_BACKEND",
            profile.get("OE_PHYSICAL_BACKEND") or "ibm_torino",
        )
    )

    initial_program = artifact_dir / "initial_program.py"
    evaluator = artifact_dir / "evaluator.py"
    config = artifact_dir / "openevolve_config.yaml"
    template_dir = artifact_dir / "prompt_templates"
    training_data = artifact_dir / "proxy_pair_training_data.json"
    baseline_dir = artifact_dir / "baseline_programs"
    baseline_comparison_path = artifact_dir / "baseline_comparison.json"
    openevolve_output = artifact_dir / "openevolve_output"
    ground_truth_metrics = _ground_truth_metrics_path(state, repo_root)
    simulation_memory = _simulation_memory_path(state, repo_root)
    physical_records = _physical_records_path(state, repo_root, artifact_dir)
    training_payload = _build_training_data(
        ground_truth_metrics,
        simulation_memory,
        physical_records,
        physical_backend=physical_backend_filter,
        repo_root=repo_root,
    )
    pre_evolution_proxy_validation = _apply_pre_evolution_metric_proxy(
        training_payload,
        profile=profile,
        objective_resolution=objective_resolution,
        semantic_proxy_proposal=state.get("last_proxy_metric_semantic_proposal"),
        semantic_feature_validation=state.get("last_proxy_feature_semantic_validation"),
    )
    semantic_feature_validation = state.get("last_proxy_feature_semantic_validation")
    selected_proxy_features: list[str] = []
    if isinstance(semantic_feature_validation, dict) and semantic_feature_validation.get("success"):
        selected_proxy_features = [
            str(item)
            for item in (semantic_feature_validation.get("selected_feature_names") or [])
            if str(item)
        ]
    selected_proxy_feature = str(
        pre_evolution_proxy_validation.get("selected_feature")
        or objective_resolution.get("proxy_feature")
        or profile.get("OE_PROXY_FEATURE")
        or (selected_proxy_features[0] if selected_proxy_features else "")
        or "depth_ratio"
    )
    if selected_proxy_feature and selected_proxy_feature not in selected_proxy_features:
        selected_proxy_features.insert(0, selected_proxy_feature)
    selected_proxy_specs = _proxy_feature_specs(selected_proxy_features)
    if not selected_proxy_specs:
        selected_proxy_specs = [_proxy_feature_spec(selected_proxy_feature)]
    selected_proxy_direction = str(pre_evolution_proxy_validation.get("direction") or "direct")
    seed_transfer = _seed_transfer_payload(training_payload, profile)
    qos_target = _selected_qos_target(state, probe)
    semantic_selection = state.get("last_openevolve_target_semantic_selection")
    semantic_selection_valid = bool(
        isinstance(semantic_selection, dict)
        and semantic_selection.get("success")
        and isinstance(semantic_selection.get("selected"), dict)
        and qos_target.get("selection_mode") == "llm_semantic_selection"
    )
    evaluator_selection = state.get("last_openevolve_evaluator_semantic_selection")
    evaluator_selection_valid = bool(
        isinstance(evaluator_selection, dict)
        and evaluator_selection.get("success")
        and isinstance(evaluator_selection.get("paper_metric_spec"), dict)
        and isinstance(evaluator_selection.get("evaluator_strategy"), dict)
    )
    training_payload["parameter_profile"] = {
        "name": profile_name,
        "objective": agent_objective,
        "settings": agent_settings,
        "env": profile,
        "schema_artifact": profile_schema_path,
    }
    training_payload["seed_transfer"] = seed_transfer
    training_payload["qos_evolution_target"] = {
        key: value
        for key, value in qos_target.items()
        if key != "source"
        }

    if not evaluator_selection_valid:
        return {
            "success": False,
            "execution_mode": "openevolve_proxy_search",
            "not_fig11_strict_success": True,
            "reason": "missing_paper_grounded_evaluator_selection",
            "openevolve": openevolve,
            "evolution_config": {
                "eval_shots": eval_shots,
                "objective_source": "openevolve_mutates_auto_discovered_qos_target",
                "manual_seed_used": False,
                "qos_evolution_target": {
                    key: value
                    for key, value in qos_target.items()
                    if key != "source"
                },
                "target_selection": {
                    "mode": "llm_semantic_selection",
                    "success": semantic_selection_valid,
                    "selected_entrypoint": qos_target.get("entrypoint"),
                },
                "evaluator_selection": {
                    "mode": "paper_metric_semantic_binding",
                    "success": False,
                    "required": True,
                },
                "parameter_profile": profile_name,
                "training_data_schema": training_data_schema,
                "physical_backend_filter": physical_backend_filter,
                "agent_objective": agent_objective,
                "objective_requirements": objective_requirements,
                "objective_resolution": objective_resolution,
                "training_record_count": training_payload.get("record_count"),
                "training_data_status": training_payload.get("status"),
            },
            "warnings": ["missing paper-grounded evaluator binding; run openevolve_evaluator_semantic_select first"],
        }

    if not semantic_selection_valid:
        return {
            "success": False,
            "execution_mode": "openevolve_proxy_search",
            "not_fig11_strict_success": True,
            "reason": "missing_llm_semantic_target_selection",
            "openevolve": openevolve,
            "evolution_config": {
                "eval_shots": eval_shots,
                "objective_source": "openevolve_mutates_auto_discovered_qos_target",
                "manual_seed_used": False,
                "qos_evolution_target": {
                    key: value
                    for key, value in qos_target.items()
                    if key != "source"
                },
                "target_selection": {
                    "mode": "llm_semantic_selection",
                    "success": False,
                    "required": True,
                },
                "parameter_profile": profile_name,
                "training_data_schema": training_data_schema,
                "physical_backend_filter": physical_backend_filter,
                "agent_objective": agent_objective,
                "objective_requirements": objective_requirements,
                "objective_resolution": objective_resolution,
                "training_record_count": training_payload.get("record_count"),
                "training_data_status": training_payload.get("status"),
            },
            "warnings": ["missing LLM semantic target selection; run openevolve_target_semantic_select first"],
        }

    initial_seed_mode = os.getenv("REPRO_OPENEVOLVE_INITIAL_SEED", "manual_qos_normalized").strip().lower()
    _write_text(
        initial_program,
        _build_initial_program(qos_target, seed_mode=initial_seed_mode, proxy_specs=selected_proxy_specs),
    )
    baseline_program_paths = {
        "repo_qos_raw": baseline_dir / "repo_qos_raw.py",
        "manual_qos_normalized": baseline_dir / "manual_qos_normalized.py",
        "proxy_depth_ratio": baseline_dir / "proxy_depth_ratio.py",
    }
    for baseline_mode, baseline_path in baseline_program_paths.items():
        _write_text(
            baseline_path,
            _build_initial_program(qos_target, seed_mode=baseline_mode, proxy_specs=selected_proxy_specs),
        )
    _build_prompt_templates(template_dir, proxy_specs=selected_proxy_specs, proxy_direction=selected_proxy_direction)
    _write_json(training_data, training_payload)
    _write_text(evaluator, _build_evaluator(training_data))
    baseline_comparison = _run_generated_baseline_comparison(
        evaluator,
        initial_program,
        baseline_comparison_path,
    )
    _write_text(
        config,
        _build_config(
            eval_shots,
            openevolve_output,
            profile,
            template_dir,
            selected_proxy_feature=selected_proxy_feature,
            selected_proxy_features=selected_proxy_features,
            include_artifacts=bool(artifact_policy.get("include_artifacts")),
        ),
    )

    command = [
        sys.executable,
        "-m",
        "openevolve.cli",
        str(initial_program),
        str(evaluator),
        "--config",
        str(config),
        "--output",
        str(openevolve_output),
    ]

    execution_result: dict[str, Any] = {"executed": False}
    if not dry_run:
        env = dict(os.environ)
        openevolve_root = repo_root / "third_party" / "openevolve"
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(openevolve_root)
            if not current_pythonpath
            else str(openevolve_root) + os.pathsep + current_pythonpath
        )
        proc = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        execution_result = {
            "executed": True,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout.strip().splitlines()[-40:],
            "stderr_tail": proc.stderr.strip().splitlines()[-40:],
        }

    payload: dict[str, Any] = {
        "success": bool(
            probe.get("success")
            and ((probe.get("target_functions") or {}).get("evolve") or {}).get("found")
            and bool(qos_target)
            and semantic_selection_valid
            and evaluator_selection_valid
            and training_payload.get("status") == "pair_record_ground_truth"
            and int(training_payload.get("record_count") or 0) > 0
            and bool(profile)
            and (dry_run or execution_result.get("returncode") == 0)
        ),
        "execution_mode": "openevolve_proxy_search",
        "not_fig11_strict_success": True,
        "openevolve": openevolve,
        "target_functions": {
            "evolve": {
                "found": bool(((probe.get("target_functions") or {}).get("evolve") or {}).get("found")),
                "selected": evolve_target,
                "candidates": ((probe.get("target_functions") or {}).get("evolve") or {}).get("candidates") or [],
            },
            "verify": {
                "found": bool(((probe.get("target_functions") or {}).get("verify") or {}).get("found")),
                "selected": verify_target,
                "candidates": ((probe.get("target_functions") or {}).get("verify") or {}).get("candidates") or [],
            },
        },
        "evolution_config": {
            "eval_shots": eval_shots,
            "pareto_second_metric": "physical_qpu_hellinger_mean"
            if (
                training_payload.get("source_kind") == "physical_qpu"
                and training_payload.get("fitness_second_metric") != "proxy_estimated_fidelity"
            )
            else str(training_payload.get("fitness_second_metric") or "simulation_relative_fidelity"),
            "fidelity_like_label_source": (
                f"pre_evolution_proxy.{((training_payload.get('proxy_metric_validation') or {}).get('selected_feature'))}"
                if training_payload.get("fitness_second_metric") == "proxy_estimated_fidelity"
                else f"{training_payload.get('physical_backend')}.physical_qpu.{training_payload.get('physical_metric_source')}"
                if training_payload.get("source_kind") == "physical_qpu"
                else "pair_level_joint_simulation.relative_fidelity"
            ),
            "physical_backend_filter": physical_backend_filter,
            "proxy_feature": selected_proxy_feature,
            "proxy_feature_specs": selected_proxy_specs,
            "proxy_prompt_context": {
                "declares_proxy_second_axis": True,
                "primary_proxy_feature": selected_proxy_feature,
                "primary_feature_direction": selected_proxy_direction,
                "pareto_axes": [
                    "effective_utilization",
                    f"proxy_estimated_fidelity:{selected_proxy_feature}:{selected_proxy_direction}",
                ],
                "prompt_template_dir": _artifact_rel(repo_root, template_dir),
            },
            "proxy_role": "pre_evolution_expensive_metric_substitute",
            "proxy_is_evolved_function": False,
            "artifact_policy": artifact_policy,
            "pre_evolution_proxy_validation": pre_evolution_proxy_validation,
            "proxy_feature_semantic_validation": {
                key: value
                for key, value in (state.get("last_proxy_feature_semantic_validation") or {}).items()
                if key not in {"feature_scores"}
            },
            "feature_materialization": training_payload.get("feature_materialization"),
            "proxy_metric_semantic_proposal": {
                key: value
                for key, value in (state.get("last_proxy_metric_semantic_proposal") or {}).items()
                if key not in {"prompt_payload", "llm_payload"}
            },
            "proxy_semantic_factor_brainstorm": _semantic_factor_evidence(state),
            "expensive_label_metric": training_payload.get("expensive_label_metric"),
            "objective_source": "openevolve_mutates_auto_discovered_qos_target",
            "seed_objective_only": False,
            "manual_seed_used": False,
            "initial_seed": {
                "mode": initial_seed_mode,
                "available_modes": ["repo_qos_raw", "manual_qos_normalized", "proxy_depth_ratio"],
                "repo_qos_raw": "exact auto-discovered repository QOS get_matching_score; no utilization normalization",
                "manual_qos_normalized": "repository QOS get_matching_score with effective_utilization normalized to [0,1]",
                "proxy_depth_ratio": "pure depth-ratio proxy score",
                "repo_source_modified": False,
                "proxy_scaffold": {
                    "enabled": bool(selected_proxy_specs and initial_seed_mode in {"repo_qos_raw", "manual_qos_normalized"}),
                    "selected_proxy_feature": selected_proxy_feature,
                    "selected_proxy_features": selected_proxy_features,
                    "feature_specs": selected_proxy_specs,
                    "scaffold_features": [
                        "depth_ratio",
                        "qubit_ratio",
                        "nonlocal_ratio",
                        "cnot_ratio",
                        "measure_ratio",
                        "instr_ratio",
                        "critical_depth_ratio",
                        "qubit_imbalance",
                        "cnot_density",
                        "nonlocal_density",
                        "measure_density",
                        "instr_density",
                        "critical_depth_density",
                        "joint_qubits",
                    ],
                    "changes_initial_qos_score": False,
                },
            },
            "seed_effective_utilization_scale": {
                "normalization": "none"
                if initial_seed_mode == "repo_qos_raw"
                else "percent_to_unit_interval_if_value_gt_1"
                if initial_seed_mode == "manual_qos_normalized"
                else "not_applicable",
                "applied_to": "generated_initial_program_only",
                "reason": (
                    "OpenEvolve optimizes a Pareto/top-k objective that combines percent-scale "
                    "effective_utilization with 0..1 compatibility/fidelity-like signals; the "
                    "default seed normalizes utilization to keep the search space aligned with "
                    "the manual OpenEvolve reproduction."
                ),
                "repo_source_modified": False,
            },
            "objective_score_definition": {
                "primary_score": "inv_avg_pareto_rank",
                "aggregation": "selected_topk_average_pareto_rank",
                "selection_operator": "topk_by_candidate_score",
                "diagnostic_metrics": ["rank_agreement", "top_rank_overlap"],
                "front_overlap_affects_validation_score": False,
                "requires_rank1_front_overlap": False,
            },
            "qos_evolution_target": {
                key: value
                for key, value in qos_target.items()
                if key != "source"
            },
            "target_selection": {
                "mode": "llm_semantic_selection",
                "success": semantic_selection_valid,
                "selected_entrypoint": qos_target.get("entrypoint"),
                "confidence": qos_target.get("confidence"),
                "semantic_evidence": qos_target.get("semantic_evidence") or [],
                "reasoning": qos_target.get("semantic_reasoning"),
            },
            "evaluator_selection": {
                "mode": "paper_metric_semantic_binding",
                "success": evaluator_selection_valid,
                "strategy": (evaluator_selection.get("evaluator_strategy") or {}).get("mode"),
                "selected_entrypoint": (evaluator_selection.get("evaluator_strategy") or {}).get("selected_entrypoint"),
                "requires_generated_adapter": (evaluator_selection.get("evaluator_strategy") or {}).get("requires_generated_adapter"),
                "aggregation": (evaluator_selection.get("evaluator_strategy") or {}).get("aggregation"),
                "selection_operator": (evaluator_selection.get("evaluator_strategy") or {}).get("selection_operator"),
                "forbid_weighted_sum_without_paper_weight": (
                    evaluator_selection.get("evaluator_strategy") or {}
                ).get("forbid_weighted_sum_without_paper_weight"),
                "artifact_path": evaluator_selection.get("artifact_path"),
            },
            "paper_metric_basis": evaluator_selection.get("paper_metric_spec"),
            "parameter_profile": profile_name,
            "parameter_schema_artifact": profile_schema_path,
            "training_data_schema": training_data_schema,
            "baseline_programs": {
                name: _artifact_rel(repo_root, path)
                for name, path in baseline_program_paths.items()
            },
            "baseline_comparison_artifact": _artifact_rel(repo_root, baseline_comparison_path),
            "baseline_comparison": baseline_comparison,
            "training_source_kind": training_payload.get("source_kind"),
            "training_label_metric": training_payload.get("label_metric"),
            "qpu_feedback_split": training_payload.get("split"),
            "train_record_count": training_payload.get("train_record_count"),
            "validation_record_count": training_payload.get("validation_record_count"),
            "physical_records": _artifact_rel(repo_root, physical_records) if physical_records else None,
            "agent_objective": agent_objective,
            "objective_requirements": objective_requirements,
            "objective_resolution": objective_resolution,
            "agent_settings": agent_settings,
            "agent_parameter_profile_used": bool(profile),
            "manual_parameter_probe_used": False,
            "seed_transfer": seed_transfer,
            "openevolve_compat_env": {
                "OE_EVAL_UTILS": profile.get("OE_EVAL_UTILS"),
                "OE_MULTI_UTIL_AGG": profile.get("OE_MULTI_UTIL_AGG"),
                "OE_PARETO_SECOND_METRIC": profile.get("OE_PARETO_SECOND_METRIC"),
                "OE_PROXY_FEATURE": selected_proxy_feature,
                "OE_PROXY_FEATURES": ",".join(selected_proxy_features) if selected_proxy_features else profile.get("OE_PROXY_FEATURES"),
                "OE_TOP_K_RATIO": profile.get("OE_TOP_K_RATIO"),
                "OE_EVAL_SHOTS": profile.get("OE_EVAL_SHOTS"),
                "OE_RESTRICT_TO_PAIR_CSV": profile.get("OE_RESTRICT_TO_PAIR_CSV"),
                "OE_SEED_TRANSFER": profile.get("OE_SEED_TRANSFER"),
                "OE_SEED_SOURCE_POLICY": profile.get("OE_SEED_SOURCE_POLICY"),
                "OE_SEED_TARGET_POLICY": profile.get("OE_SEED_TARGET_POLICY"),
                "OE_PHYSICAL_BACKEND": profile.get("OE_PHYSICAL_BACKEND"),
            },
            "dry_run": dry_run,
            "mutation_protocol": {
                "mode": "whole_evolve_block_replace",
                "block_markers": ["# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"],
                "required_function": "get_matching_score",
                "required_signature": "get_matching_score(self, q1, q2, backend, weighted=False, weights=[])",
                "syntax_validation": "OpenEvolve/evaluator import plus planned ast.parse post-check",
                "requires_code_hash_change": True,
            },
            "config_path": _artifact_rel(repo_root, config),
            "initial_program": _artifact_rel(repo_root, initial_program),
            "evaluator": _artifact_rel(repo_root, evaluator),
            "training_data": _artifact_rel(repo_root, training_data),
            "simulation_memory": _artifact_rel(repo_root, simulation_memory) if simulation_memory else None,
            "prompt_template_dir": _artifact_rel(repo_root, template_dir),
            "training_record_count": training_payload.get("record_count"),
            "training_data_status": training_payload.get("status"),
            "output_dir": _artifact_rel(repo_root, openevolve_output),
        },
        "verification": {
            "simulation_ground_truth_reference": str(ground_truth_metrics) if ground_truth_metrics else "Fig. 11 selected-pair joint simulation with 8192 shots",
            "proxy_ground_truth_reference": _artifact_rel(repo_root, physical_records)
            if training_payload.get("source_kind") == "physical_qpu" and physical_records
            else (str(ground_truth_metrics) if ground_truth_metrics else None),
            "rank_comparison": {
                "planned": True,
                "method": "OpenEvolve evaluator scores the auto-discovered QOS pair-selection function by agreement with pair-level Pareto rank over effective utilization and the configured fidelity-like label",
                "requires_recompute": True,
                "split_policy": "fitness_uses_held_out_validation_records_only",
                "ground_truth_status": training_payload.get("status"),
                "training_record_count": training_payload.get("record_count"),
                "train_record_count": training_payload.get("train_record_count"),
                "validation_record_count": training_payload.get("validation_record_count"),
            },
            "fig11_strict_contract_satisfied": False,
        },
        "qpu_validation_plan": {
            "requires_human_approval": True,
            "automatic_qpu_submission": False,
            "current_validation_source": "held_out_existing_qpu_records",
            "backend": physical_backend_filter if training_payload.get("source_kind") == "physical_qpu" else None,
            "label_metric": training_payload.get("label_metric"),
            "suggested_next_step": "After evolution, inspect proxy ranking disagreement on the held-out split and manually choose a small number of new QPU pairs if more validation is needed.",
        },
        "run_spec": {
            "dry_run": dry_run,
            "pythonpath_prepend": "third_party/openevolve",
            "command": command,
            "execution_result": execution_result,
            "note": "Dry-run writes the OpenEvolve spec only. Set REPRO_OPENEVOLVE_DRY_RUN=0 when an LLM backend and budget are available.",
        },
        "warnings": [],
    }
    if dry_run:
        payload["warnings"].append("OpenEvolve command was not executed; this artifact is a proxy/evolution run specification.")
    if training_payload.get("status") != "pair_record_ground_truth":
        payload["warnings"].append("missing simulation pair-record ground truth; run strict Fig. 11 simulation first")
    if training_payload.get("status") == "scale_memory_only":
        payload["warnings"].append("small-scale simulation memory is available, but pair-level records are still required for rank-training")
    if not profile:
        payload["warnings"].append("missing agent OpenEvolve parameter profile; run openevolve_param_probe first")
    if not semantic_selection_valid:
        payload["warnings"].append("missing LLM semantic target selection; run openevolve_target_semantic_select first")
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
    artifact_dir = run_root / "openevolve_proxy_search"

    payload = build_proxy_payload(repo_root=repo_root, artifact_dir=artifact_dir, state=state)
    payload["tool"] = TOOL_NAME
    payload["action"] = ACTION
    payload["fsm_state_before"] = get_fsm_state(state)

    metrics_path = artifact_dir / "openevolve_proxy_search.metrics.json"
    _write_json(metrics_path, payload)

    contract_path = (repo_root / CONTRACT_REL_PATH).resolve()
    payload["contract_path"] = _artifact_rel(repo_root, contract_path)
    payload["metrics_path"] = str(metrics_path)
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        verdict = evaluate_figure_contract(contract, payload)
        verdict["contract_path"] = str(contract_path)
        verdict["metrics_path"] = str(metrics_path)
        payload["contract_verdict"] = verdict
        payload["success"] = bool(payload.get("success") and verdict.get("success"))
    else:
        payload["success"] = False
        payload["contract_verdict"] = {"success": False, "reason": f"contract not found: {contract_path}"}

    _write_json(metrics_path, payload)

    state["last_openevolve_proxy_search"] = payload
    state["last_step"] = TOOL_NAME
    state["last_status"] = "openevolve_proxy_search_succeeded" if payload.get("success") else "openevolve_proxy_search_failed"
    set_fsm_state(state, "APPLY_FIX")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, TOOL_NAME, payload)
    write_state(state_path, state)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
