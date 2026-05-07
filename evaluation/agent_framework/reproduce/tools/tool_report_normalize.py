#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize batch reproduce report into required/coverage KPIs.")
    parser.add_argument("--batch-summary", required=True, help="Path to batch_summary.json")
    parser.add_argument("--figure-code-map", default="", help="Optional figure_code_map.json")
    parser.add_argument("--output", required=True, help="Path to normalized report JSON")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    batch_path = Path(args.batch_summary).resolve()
    out_path = Path(args.output).resolve()
    batch = load_json(batch_path)
    rows = batch.get("rows", [])

    code_map: dict[str, Any] = {"figures": []}
    if args.figure_code_map:
        code_map = load_json(Path(args.figure_code_map).resolve())
    req_index = {
        str(x.get("figure_id")): bool(x.get("reproduce_required", True))
        for x in code_map.get("figures", [])
    }

    normalized_rows: list[dict[str, Any]] = []
    required_total = 0
    required_passed = 0
    executed_total = len(rows)
    executed_passed = 0

    for row in rows:
        fid = str(row.get("figure_id", "unknown"))
        required = req_index.get(fid, True)
        success = str(row.get("status", "")) == "success"
        if required:
            required_total += 1
            if success:
                required_passed += 1
        if success:
            executed_passed += 1
        normalized_rows.append(
            {
                "figure_id": fid,
                "required": required,
                "status": row.get("status"),
                "returncode": row.get("returncode"),
                "run_dir": row.get("run_dir"),
                "failure_signals": row.get("failure_signals", []),
            }
        )

    payload = {
        "source_batch_summary": str(batch_path),
        "source_figure_code_map": args.figure_code_map or None,
        "kpis": {
            "required_figures_total": required_total,
            "required_figures_passed": required_passed,
            "required_pass_rate": (required_passed / required_total) if required_total else None,
            "all_executed_total": executed_total,
            "all_executed_passed": executed_passed,
            "all_executed_pass_rate": (executed_passed / executed_total) if executed_total else None,
        },
        "rows": normalized_rows,
    }
    write_json(out_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
