#!/usr/bin/env python3
"""
Build plotting CSV from IBM hardware sweep fidelity JSON files.

Output columns are compatible with plot_qos_vs_evolved_from_csv.py:
- name_1
- name_2
- effective_utilization
- fidelity
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _util_ratio_from_utilization_file(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    util = data.get("utilization", {})
    ratio = _safe_float(util.get("effective_utilization_ratio"))
    if ratio is not None:
        return ratio

    percent = _safe_float(util.get("effective_utilization_percent"))
    if percent is not None:
        return percent / 100.0

    return None


def _sort_key(d: Dict, path: str):
    row_idx = d.get("row_index")
    pair_idx = d.get("pair_index")
    if isinstance(row_idx, int):
        return (0, row_idx, pair_idx if isinstance(pair_idx, int) else 10**9, path)
    if isinstance(pair_idx, int):
        return (1, pair_idx, 10**9, path)
    return (2, 10**9, 10**9, path)


def build_rows(fidelity_dir: str) -> List[Dict]:
    pattern = os.path.join(fidelity_dir, "pair_*.json")
    files = sorted(glob.glob(pattern))
    parsed = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            parsed.append((path, data))
        except Exception:
            continue
    parsed.sort(key=lambda x: _sort_key(x[1], x[0]))

    rows = []
    for path, data in parsed:
        fidelity = _safe_float(data.get("hellinger_mean"))
        if fidelity is None:
            continue

        util_file = data.get("utilization_file")
        effective_utilization = _util_ratio_from_utilization_file(util_file)
        util_source = "utilization_file"

        if effective_utilization is None:
            effective_utilization = _safe_float(data.get("csv_effective_utilization"))
            util_source = "csv_effective_utilization"
            if effective_utilization is not None and effective_utilization > 1.0:
                effective_utilization = effective_utilization / 100.0

        if effective_utilization is None:
            continue

        rows.append(
            {
                "row_index": data.get("row_index"),
                "pair_index": data.get("pair_index"),
                "name_1": data.get("name_1"),
                "name_2": data.get("name_2"),
                "effective_utilization": effective_utilization,
                "fidelity": fidelity,
                "util_source": util_source,
                "fidelity_source": "hellinger_mean",
                "backend": data.get("backend"),
                "shots": data.get("shots"),
                "csv_effective_utilization": data.get("csv_effective_utilization"),
                "csv_fidelity": data.get("csv_fidelity"),
                "json_file": path,
                "utilization_file": util_file,
                "job_id": data.get("job_id"),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Build physical-metric CSV from sweep fidelity JSON files.")
    parser.add_argument("--fidelity-dir", required=True, help="Directory containing pair_*.json files.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    args = parser.parse_args()

    rows = build_rows(args.fidelity_dir)
    if not rows:
        raise SystemExit(f"[ERR] No valid rows found in: {args.fidelity_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fieldnames = [
        "row_index",
        "pair_index",
        "name_1",
        "name_2",
        "effective_utilization",
        "fidelity",
        "util_source",
        "fidelity_source",
        "backend",
        "shots",
        "csv_effective_utilization",
        "csv_fidelity",
        "job_id",
        "json_file",
        "utilization_file",
    ]

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    util_from_file = sum(1 for r in rows if r["util_source"] == "utilization_file")
    util_from_csv = sum(1 for r in rows if r["util_source"] == "csv_effective_utilization")
    print(f"[OK] Wrote {args.out_csv}")
    print(f"[INFO] Rows={len(rows)} from {args.fidelity_dir}")
    print(f"[INFO] utilization source: file={util_from_file}, csv_fallback={util_from_csv}")


if __name__ == "__main__":
    main()

