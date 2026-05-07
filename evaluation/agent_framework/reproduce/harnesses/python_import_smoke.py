#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--modules-json")
    parser.add_argument("--modules-file")
    args = parser.parse_args()

    if not args.modules_json and not args.modules_file:
        raise SystemExit("one of --modules-json or --modules-file is required")

    if args.modules_json:
        modules = json.loads(args.modules_json)
    else:
        with open(args.modules_file, "r", encoding="utf-8") as f:
            modules = json.load(f)

    workspace_root = os.environ.get("REPRO_WORKSPACE_ROOT")
    if workspace_root and workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)

    checked_modules = []
    overall_success = True
    for module_name in modules:
        row = {
            "module": module_name,
            "ok": False,
        }
        try:
            importlib.import_module(module_name)
            row["ok"] = True
        except Exception as exc:
            overall_success = False
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            row["traceback"] = traceback.format_exc()
        checked_modules.append(row)

    metrics = {
        "success": overall_success,
        "checked_at": iso_now(),
        "workspace_root": workspace_root,
        "python_version": sys.version,
        "checked_modules": checked_modules,
    }

    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
