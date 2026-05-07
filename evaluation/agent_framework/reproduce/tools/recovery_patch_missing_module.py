#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_path = Path(args.result_path).resolve()
    payload = {
        "success": False,
        "applied": False,
        "summary": "disabled: shim-style missing-module recovery is not allowed",
        "edits": [],
        "errors": ["shim_recovery_disabled_use_general_env_or_source_fix"],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
