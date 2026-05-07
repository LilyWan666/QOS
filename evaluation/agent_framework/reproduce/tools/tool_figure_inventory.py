#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


FIGURE_PATTERN = re.compile(r"\b(?:figure|fig\.?)\s*(\d+[a-z]?)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper+repo figure inventory manifest for full reproduction planning."
    )
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Repository root to scan for figure-related scripts/contracts.",
    )
    parser.add_argument(
        "--paper",
        required=True,
        help="Path to paper PDF/text document (absolute or relative to repo root).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output manifest JSON path.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_from_repo(repo_root: Path, raw: str) -> Path:
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (repo_root / candidate).resolve()


def read_doc_text(source_path: Path, tmp_txt: Path) -> str:
    tmp_txt.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".pdf":
        proc = subprocess.run(
            ["pdftotext", "-layout", str(source_path), str(tmp_txt)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "pdftotext failed")
        return tmp_txt.read_text(encoding="utf-8", errors="replace")
    return source_path.read_text(encoding="utf-8", errors="replace")


def normalize_fig_id(token: str) -> str:
    cleaned = token.strip().lower().replace(" ", "")
    return f"fig{cleaned}"


def collect_paper_figures(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    by_fig: dict[str, dict[str, Any]] = {}
    for idx, line in enumerate(lines, start=1):
        for match in FIGURE_PATTERN.finditer(line):
            fig_id = normalize_fig_id(match.group(1))
            rec = by_fig.setdefault(
                fig_id,
                {
                    "figure_id": fig_id,
                    "paper_refs": [],
                    "is_motivation_candidate": False,
                },
            )
            context = line.strip()
            rec["paper_refs"].append({"line": idx, "text": context[:400]})
            lowered = context.lower()
            if "motivation" in lowered or "example" in lowered:
                rec["is_motivation_candidate"] = True
    # sort refs by line, stable figure order by first mention line
    items = list(by_fig.values())
    for item in items:
        item["paper_refs"] = sorted(item["paper_refs"], key=lambda x: int(x["line"]))
    items.sort(key=lambda x: x["paper_refs"][0]["line"] if x["paper_refs"] else 10**9)
    return items


def rg_files(repo_root: Path, pattern: str) -> list[Path]:
    proc = subprocess.run(
        ["rg", "--files", str(repo_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    files = [Path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
    return [path for path in files if re.search(pattern, path.as_posix(), re.IGNORECASE)]


def rg_content(repo_root: Path, pattern: str) -> list[Path]:
    proc = subprocess.run(
        [
            "rg",
            "-l",
            "-i",
            "-g",
            "*.py",
            "-g",
            "*.sh",
            "-g",
            "*.slurm",
            "-g",
            "*.json",
            "-g",
            "*.md",
            pattern,
            str(repo_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        return []
    return [Path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]


def collect_repo_candidates(repo_root: Path, figure_id: str) -> dict[str, list[str]]:
    numeric = figure_id.replace("fig", "")
    patterns = [
        rf"(?:^|[/_])fig(?:ure)?[_-]?{re.escape(numeric)}(?:[^0-9a-z]|$)",
        rf"(?<![0-9a-z]){re.escape(figure_id)}(?![0-9a-z])",
    ]
    matched: set[Path] = set()
    for pattern in patterns:
        for path in rg_files(repo_root, pattern):
            matched.add(path)
    content_patterns = [
        rf"\bfigure\s*{re.escape(numeric)}\b",
        rf"\bfig\.?\s*{re.escape(numeric)}\b",
        rf"(?<![0-9a-z]){re.escape(figure_id)}(?![0-9a-z])",
    ]
    for pattern in content_patterns:
        for path in rg_content(repo_root, pattern):
            matched.add(path)

    grouped: dict[str, list[str]] = defaultdict(list)
    for path in sorted(matched):
        rel = str(path.resolve().relative_to(repo_root.resolve()))
        lowered = rel.lower()
        if lowered.endswith(".contract.json"):
            grouped["contracts"].append(rel)
        elif lowered.endswith(".json"):
            grouped["json"].append(rel)
        elif lowered.endswith(".py"):
            grouped["python"].append(rel)
        elif lowered.endswith(".sh") or lowered.endswith(".slurm"):
            grouped["runner"].append(rel)
        else:
            grouped["other"].append(rel)
    return dict(grouped)


def existing_relpaths(repo_root: Path, relpaths: list[str]) -> list[str]:
    out: list[str] = []
    for rel in relpaths:
        if (repo_root / rel).exists():
            out.append(rel)
    return out


def classify_figure_action(figure: dict[str, Any]) -> tuple[str, str]:
    has_runner = bool(figure.get("has_runner"))
    has_contract = bool(figure.get("has_contract"))
    is_motivation = bool(figure.get("is_motivation_candidate"))
    repo_candidates = figure.get("repo_candidates") or {}
    candidate_count = sum(
        len(repo_candidates.get(key, []))
        for key in ("python", "runner", "contracts", "json", "other")
    )

    if is_motivation and not has_runner and candidate_count <= 1:
        return (
            "skip_motivation",
            "paper context suggests motivation/example figure and no runnable repo entrypoint was found",
        )
    if has_runner:
        if has_contract:
            return (
                "auto_reproduce_with_verify",
                "runner and contract are available",
            )
        return (
            "auto_reproduce",
            "runner is available; can run and compare metrics/artifacts",
        )
    if has_contract:
        return (
            "needs_runner_binding",
            "contract exists but no runnable entrypoint is mapped",
        )
    return (
        "needs_mapping",
        "no clear runner/contract found from repo scan",
    )


def suggest_entrypoints(repo_root: Path, figure_id: str, paper_refs: list[dict[str, Any]]) -> list[str]:
    text_blob = " ".join(str(item.get("text", "")) for item in paper_refs).lower()
    suggestions: list[str] = []
    if "error mitigator" in text_blob or "§ 9.2" in text_blob or "9.2" in text_blob:
        suggestions.extend(
            [
                "qos/error_mitigator/run.py",
                "qos/error_mitigator/optimiser.py",
                "qos/error_mitigator/virtualizer.py",
            ]
        )
    if "estimator" in text_blob or "§ 9.3" in text_blob or "9.3" in text_blob:
        suggestions.extend(
            [
                "qos/estimator/estimator.py",
                "qos/scheduler/time_estimator/basic_estimator.py",
                "qos/scheduler/time_estimator/regression_estimator.py",
            ]
        )
    if "multi-programmer" in text_blob or "multi-programming" in text_blob or "§ 9.4" in text_blob or "9.4" in text_blob:
        suggestions.extend(
            [
                "qos/multiprogrammer/multiprogrammer.py",
                "evaluation/agent_framework/reproduce/harnesses/run_process_qernels_smoke.py",
                "evaluation/agent_framework/reproduce/harnesses/run_process_qernels_external.py",
            ]
        )
    if "scheduler" in text_blob or "§ 9.5" in text_blob or "9.5" in text_blob:
        suggestions.extend(
            [
                "qos/scheduler/scheduler.py",
                "qos/scheduler/multi_objective_scheduler.py",
            ]
        )
    if figure_id == "fig11":
        suggestions.extend(
            [
                "qos/multiprogrammer/multiprogrammer.py",
                "evaluation/agent_framework/reproduce/harnesses/run_process_qernels_smoke.py",
            ]
        )
    dedup: list[str] = []
    seen: set[str] = set()
    for item in suggestions:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    return existing_relpaths(repo_root, dedup)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    paper_path = resolve_from_repo(repo_root, args.paper)
    output_path = resolve_from_repo(repo_root, args.output)

    if not repo_root.exists():
        raise SystemExit(f"repo root not found: {repo_root}")
    if not paper_path.exists():
        raise SystemExit(f"paper not found: {paper_path}")

    temp_txt = output_path.parent / "_paper_inventory_source.txt"
    text = read_doc_text(paper_path, temp_txt)

    figures = collect_paper_figures(text)
    action_counts: dict[str, int] = defaultdict(int)
    for fig in figures:
        fig["repo_candidates"] = collect_repo_candidates(repo_root, fig["figure_id"])
        fig["suggested_entrypoints"] = suggest_entrypoints(
            repo_root, fig["figure_id"], fig.get("paper_refs", [])
        )
        candidates = fig["repo_candidates"]
        fig["has_contract"] = bool(candidates.get("contracts"))
        fig["has_runner"] = bool(
            candidates.get("python") or candidates.get("runner") or fig["suggested_entrypoints"]
        )
        if fig["has_runner"]:
            fig["inventory_status"] = "ready_for_recipe_authoring"
        elif fig["has_contract"]:
            fig["inventory_status"] = "contract_only_needs_runner_binding"
        else:
            fig["inventory_status"] = "missing_repo_runner"
        recommended_action, recommendation_reason = classify_figure_action(fig)
        fig["recommended_action"] = recommended_action
        fig["recommendation_reason"] = recommendation_reason
        action_counts[recommended_action] += 1

    payload = {
        "paper_path": str(paper_path),
        "repo_root": str(repo_root),
        "figure_count": len(figures),
        "summary": {
            "recommended_action_counts": dict(sorted(action_counts.items())),
        },
        "figures": figures,
        "notes": [
            "Inventory is heuristic. It maps paper figure mentions to repo candidates by filename/path similarity.",
            "Use inventory_status to prioritize recipe generation and missing-runner diagnostics.",
            "recommended_action is a planning hint for automated orchestration.",
        ],
    }
    write_json(output_path, payload)
    # keep extracted text for debugging traceability
    if temp_txt.exists():
        payload["paper_text_dump"] = str(temp_txt)
        write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
