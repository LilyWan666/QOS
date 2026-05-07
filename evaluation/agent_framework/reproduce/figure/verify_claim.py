#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_claim import evaluate_figure_contract  # noqa: E402


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, help="Path to figure contract JSON.")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON.")
    parser.add_argument("--output", help="Path to write verdict JSON.")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the verdict JSON to stdout.",
    )
    args = parser.parse_args()

    contract_path = Path(args.contract).resolve()
    metrics_path = Path(args.metrics).resolve()

    contract = load_json(contract_path)
    metrics = load_json(metrics_path)
    verdict = evaluate_figure_contract(contract, metrics)
    verdict["contract_path"] = str(contract_path)
    verdict["metrics_path"] = str(metrics_path)

    if args.output:
        write_json(Path(args.output).resolve(), verdict)

    if args.print_json or not args.output:
        print(json.dumps(verdict, indent=2, sort_keys=True))

    return 0 if verdict.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
