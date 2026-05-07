#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-figure reproduce recipes from figure inventory manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to figures_manifest.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write generated recipes")
    parser.add_argument(
        "--paper-path",
        default="paper/qos.pdf",
        help="Paper path stored into each generated recipe paper.documents",
    )
    parser.add_argument(
        "--max-modules",
        type=int,
        default=1,
        help="Maximum number of suggested entrypoint modules included in import-smoke entry command.",
    )
    parser.add_argument(
        "--include-actions",
        default="auto_reproduce,auto_reproduce_with_verify",
        help=(
            "Comma-separated recommended_action allowlist from figure inventory. "
            "Only figures with matching recommended_action generate recipes."
        ),
    )
    parser.add_argument(
        "--attach-contract-verification",
        action="store_true",
        help=(
            "Attach figure contract verification when a contract exists. "
            "Disabled by default for import-smoke recipes."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def path_to_module(relpath: str) -> str:
    cleaned = relpath.strip().replace("\\", "/")
    if not cleaned.endswith(".py"):
        return ""
    module = cleaned[:-3].replace("/", ".")
    return module


def build_modules(figure: dict[str, Any], max_modules: int) -> list[str]:
    modules: list[str] = []
    for rel in figure.get("suggested_entrypoints", []):
        if not str(rel).startswith("qos/"):
            continue
        module = path_to_module(rel)
        if module and module not in modules:
            modules.append(module)
        if max_modules > 0 and len(modules) >= max_modules:
            break
    if not modules:
        modules.extend(["qos.backends.ibm_backends"])
    return modules


def build_recipe(
    figure: dict[str, Any],
    paper_path: str,
    max_modules: int,
    attach_contract_verification: bool,
) -> dict[str, Any]:
    figure_id = str(figure["figure_id"])
    modules = build_modules(figure, max_modules)
    contracts = (figure.get("repo_candidates") or {}).get("contracts", [])

    recipe: dict[str, Any] = {
        "name": f"qos_{figure_id}_auto_import_smoke",
        "description": f"Auto-generated import-smoke recipe for {figure_id} from figure inventory.",
        "workspace_root": ".",
        "paper": {
            "documents": [{"label": "qos_main_paper", "path": paper_path}],
            "focus_terms": [figure_id, "fidelity", "utilization", "scheduler", "error mitigation"],
        },
        "entry": {
            "kind": "command",
            "command": [
                "{python_executable}",
                "evaluation/agent_framework/reproduce/harnesses/python_import_smoke.py",
                "--metrics-path",
                "{metrics_path}",
                "--modules-json",
                json.dumps(modules),
            ],
            "timeout_seconds": 90,
        },
        "preflight": {
            "commands_exist": ["python"],
            "paths_exist": ["evaluation/agent_framework/reproduce/harnesses/python_import_smoke.py"],
            "python_imports": ["json"],
            "optional_python_imports": [],
        },
        "parsing": {
            "kind": "json_file",
            "path": "{metrics_path}",
            "required_metric_keys": ["success", "python_version", "checked_modules"],
        },
        "success_criteria": {
            "require_exit_code_zero": True,
            "require_metrics": True,
            "require_metric_success": True,
        },
        "recovery": {
            "mode": "agent_guided",
            "simulation_only": True,
            "handlers": {},
        },
        "metadata": {
            "source": "tool_generate_recipes_from_inventory.py",
            "figure_id": figure_id,
            "inventory_status": figure.get("inventory_status"),
            "suggested_entrypoints": figure.get("suggested_entrypoints", []),
            "selected_modules": modules,
            "module_selection_strategy": "prefix_order_limited",
            "max_modules": max_modules,
            "shim_policy": "disallow_generated_shims",
            "paper_ref_count": len(figure.get("paper_refs", [])),
        },
    }
    if contracts and attach_contract_verification:
        recipe["verification"] = {
            "enabled": True,
            "kind": "figure_contract",
            "contract_path": contracts[0],
            "output_path": "{run_dir}/figure_verdict.json",
        }
        recipe["success_criteria"]["require_verification_success"] = True
    return recipe


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest = load_json(manifest_path)
    figures = manifest.get("figures", [])
    include_actions = {
        item.strip() for item in str(args.include_actions).split(",") if item.strip()
    }

    generated: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for figure in figures:
        figure_id = str(figure["figure_id"])
        recommended_action = str(figure.get("recommended_action", ""))
        if include_actions and recommended_action not in include_actions:
            skipped.append(
                {
                    "figure_id": figure_id,
                    "recommended_action": recommended_action,
                    "reason": "recommended_action_not_in_include_actions",
                }
            )
            continue
        recipe = build_recipe(
            figure,
            args.paper_path,
            args.max_modules,
            args.attach_contract_verification,
        )
        out_path = output_dir / f"{recipe['name']}.json"
        write_json(out_path, recipe)
        generated.append({"figure_id": figure_id, "recipe_path": str(out_path)})

    index = {
        "manifest_path": str(manifest_path),
        "include_actions": sorted(include_actions),
        "recipe_count": len(generated),
        "recipes": generated,
        "skipped": skipped,
    }
    write_json(output_dir / "recipes.index.json", index)
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
