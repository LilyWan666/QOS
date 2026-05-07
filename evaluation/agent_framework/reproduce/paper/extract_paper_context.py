#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_FOCUS_TERMS = [
    "simulation",
    "hardware",
    "real quantum",
    "benchmark",
    "fidelity",
    "utilization",
    "waiting",
    "scheduler",
    "multiprogramming",
    "error mitigation",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_doc_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root / path).resolve()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "paper"


def read_pdf_text(source_path: Path, text_output_path: Path) -> str:
    proc = subprocess.run(
        ["pdftotext", "-layout", str(source_path), str(text_output_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "pdftotext failed")
    return text_output_path.read_text(encoding="utf-8", errors="replace")


def read_text_document(source_path: Path, text_output_path: Path) -> str:
    text = source_path.read_text(encoding="utf-8", errors="replace")
    text_output_path.write_text(text, encoding="utf-8")
    return text


def extract_title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def extract_section(text: str, start_markers: list[str], end_markers: list[str]) -> str:
    lowered = text.lower()
    start_idx = -1
    for marker in start_markers:
        idx = lowered.find(marker.lower())
        if idx != -1 and (start_idx == -1 or idx < start_idx):
            start_idx = idx
    if start_idx == -1:
        return ""

    end_idx = len(text)
    for marker in end_markers:
        idx = lowered.find(marker.lower(), start_idx + 1)
        if idx != -1 and idx < end_idx:
            end_idx = idx
    section = text[start_idx:end_idx].strip()
    return re.sub(r"\n{3,}", "\n\n", section)


def sentence_split(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def matching_snippets(text: str, focus_terms: list[str], limit: int = 8) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for sentence in sentence_split(text):
        lowered = sentence.lower()
        if any(term.lower() in lowered for term in focus_terms):
            if sentence not in seen:
                seen.add(sentence)
                results.append(sentence)
            if len(results) >= limit:
                break
    return results


def build_guidance(
    recipe: dict[str, Any],
    documents: list[dict[str, Any]],
    focus_terms: list[str],
) -> list[str]:
    guidance: list[str] = []
    simulation_only = bool(recipe.get("recovery", {}).get("simulation_only"))
    all_snippets = " ".join(
        " ".join(doc.get("focus_snippets", [])) for doc in documents if doc.get("focus_snippets")
    ).lower()

    if simulation_only and ("real quantum" in all_snippets or "hardware" in all_snippets):
        guidance.append(
            "Recipe is simulation-only, so treat hardware discussion in the paper as downstream validation rather than a blocker for smoke reproduction."
        )
    if "fidelity" in all_snippets:
        guidance.append("Paper explicitly discusses fidelity, so preserve fidelity-related outputs when judging reproduction progress.")
    if "utilization" in all_snippets or "waiting" in all_snippets:
        guidance.append("Paper emphasizes system-level tradeoffs such as utilization or waiting time, which may matter in later verification even if the smoke test is narrower.")
    if "benchmark" in all_snippets:
        guidance.append("Paper references benchmark-driven evaluation, so benchmark paths and harness inputs should be treated as first-class reproduction evidence.")
    if not guidance and focus_terms:
        guidance.append(
            "Use the extracted paper snippets as grounding for experimental intent, metrics, and acceptable simplifications."
        )
    return guidance


def extract_document_context(
    repo_root: Path,
    output_dir: Path,
    document: dict[str, Any],
    focus_terms: list[str],
) -> dict[str, Any]:
    raw_path = document["path"]
    source_path = resolve_doc_path(repo_root, raw_path)
    label = document.get("label") or Path(raw_path).stem
    stem = slugify(label)
    text_output_path = output_dir / f"{stem}.txt"

    if source_path.suffix.lower() == ".pdf":
        text = read_pdf_text(source_path, text_output_path)
    else:
        text = read_text_document(source_path, text_output_path)

    abstract = extract_section(text, ["abstract"], ["1 introduction", "introduction"])
    evaluation = extract_section(
        text,
        ["9 evaluation", "evaluation", "experiments", "experimental setup"],
        ["10 ", "related work", "conclusion", "acknowledgements", "references"],
    )
    return {
        "label": label,
        "path": raw_path,
        "resolved_path": str(source_path),
        "text_path": str(text_output_path),
        "title": extract_title(text),
        "abstract_excerpt": abstract[:4000],
        "evaluation_excerpt": evaluation[:4000],
        "focus_snippets": matching_snippets(text, focus_terms, limit=10),
    }


def main() -> int:
    args = parse_args()
    recipe_path = Path(args.recipe).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    recipe = load_json(recipe_path)
    repo_root = repo_root_from_script()
    paper_cfg = recipe.get("paper", {})
    documents_cfg = paper_cfg.get("documents", [])
    focus_terms = paper_cfg.get("focus_terms", DEFAULT_FOCUS_TERMS)

    documents: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for document in documents_cfg:
        try:
            documents.append(extract_document_context(repo_root, output_dir, document, focus_terms))
        except Exception as exc:
            errors.append({"path": document.get("path", ""), "error": str(exc)})

    payload = {
        "recipe_name": recipe.get("name"),
        "focus_terms": focus_terms,
        "documents": documents,
        "document_count": len(documents),
        "errors": errors,
        "guidance": build_guidance(recipe, documents, focus_terms),
    }
    write_json(output_dir / "paper_context.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
