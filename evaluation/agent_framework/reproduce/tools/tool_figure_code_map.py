#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build figure-to-code mapping from figure inventory.")
    parser.add_argument("--manifest", required=True, help="Path to figures manifest JSON")
    parser.add_argument("--output", required=True, help="Path to figure code map JSON")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def classify_figure(fig: dict[str, Any]) -> tuple[str, bool, str]:
    refs = " ".join(str(x.get("text", "")) for x in fig.get("paper_refs", []))
    lowered = refs.lower()
    if fig.get("is_motivation_candidate") or "overview" in lowered or "foundational example" in lowered:
        return ("motivation", False, "narrative_only")
    if "challenge" in lowered or "workflow" in lowered:
        return ("method", True, "simulation_required")
    return ("result", True, "simulation_required")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    manifest = load_json(manifest_path)
    figures = manifest.get("figures", [])

    mapped: list[dict[str, Any]] = []
    for fig in figures:
        figure_id = str(fig.get("figure_id"))
        figure_type, reproduce_required, acceptance_mode = classify_figure(fig)
        repo_candidates = fig.get("repo_candidates", {})
        mapped.append(
            {
                "figure_id": figure_id,
                "figure_type": figure_type,
                "reproduce_required": reproduce_required,
                "acceptance_mode": acceptance_mode,
                "entrypoints": fig.get("suggested_entrypoints", []),
                "contracts": repo_candidates.get("contracts", []),
                "runner_candidates": repo_candidates.get("runner", []) + repo_candidates.get("python", []),
                "expected_artifacts": [f"{figure_id}.metrics.json", f"{figure_id}.summary.json"],
                "owner_tools": [
                    "ReproPreflight",
                    "ReproRunOnce",
                    "ReproConstraintResolve",
                    "ReproEnvFix",
                    "ReproSourceFix",
                    "ReproApplyFix",
                    "ReproVerifyClaim",
                    "ReproRenderArtifacts",
                ],
            }
        )

    payload = {
        "source_manifest": str(manifest_path),
        "figure_count": len(mapped),
        "required_count": sum(1 for m in mapped if m["reproduce_required"]),
        "figures": mapped,
    }
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
