#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import Counter


def _safe_number(raw):
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _build_performance(args, csv_path):
    performance = {
        "util": _safe_number(args.util) if args.util != "" else args.util,
        "shots": _safe_number(args.shots) if args.shots != "" else args.shots,
        "workers": _safe_number(args.workers) if args.workers != "" else args.workers,
        "sample_k": _safe_number(args.sample_k) if args.sample_k != "" else args.sample_k,
        "sample_seed": _safe_number(args.sample_seed) if args.sample_seed != "" else args.sample_seed,
        "req_cpus": _safe_number(args.req_cpus) if args.req_cpus != "" else args.req_cpus,
        "req_mem_mb": _safe_number(args.req_mem_mb) if args.req_mem_mb != "" else args.req_mem_mb,
        "start_iso": args.start_iso,
        "end_iso": args.end_iso,
        "elapsed_sec": _safe_number(args.elapsed_sec),
        "user_cpu_sec": _safe_number(args.user_cpu_sec),
        "sys_cpu_sec": _safe_number(args.sys_cpu_sec),
        "cpu_pct": args.cpu_pct,
        "max_rss_kb": _safe_number(args.max_rss_kb),
        "major_page_faults": _safe_number(args.major_page_faults),
        "minor_page_faults": _safe_number(args.minor_page_faults),
        "vol_ctx_switches": _safe_number(args.vol_ctx_switches),
        "invol_ctx_switches": _safe_number(args.invol_ctx_switches),
        "exit_code": _safe_number(args.exit_code),
        "pairs_per_sec": _safe_number(args.pairs_per_sec),
        "out_csv_bytes": _safe_number(args.out_csv_bytes),
        "out_csv": args.out_csv or os.path.abspath(csv_path),
        "stdout_log": args.stdout_log,
        "stderr_log": args.stderr_log,
        "time_tsv": args.time_tsv,
    }
    return performance


def main():
    parser = argparse.ArgumentParser(
        description="Convert pair metrics CSV to per-pair JSONL and average summary JSONL."
    )
    parser.add_argument("--csv", required=True, help="Input pair metrics CSV")
    parser.add_argument("--per-pair-jsonl", required=True, help="Output per-pair JSONL path")
    parser.add_argument("--summary-jsonl", required=True, help="Output summary JSONL path")
    parser.add_argument("--util", default="", help="Target utilization label/value")
    parser.add_argument("--shots", default="", help="Target shots label/value")
    parser.add_argument("--workers", default="", help="Worker count")
    parser.add_argument("--sample-k", default="", help="Sample size if sampling mode is enabled")
    parser.add_argument("--sample-seed", default="", help="Sample seed if sampling mode is enabled")
    parser.add_argument("--req-cpus", default="", help="Requested CPUs")
    parser.add_argument("--req-mem-mb", default="", help="Requested memory in MB or scheduler units")
    parser.add_argument("--start-iso", default="", help="Start timestamp")
    parser.add_argument("--end-iso", default="", help="End timestamp")
    parser.add_argument("--elapsed-sec", default="", help="Wall time seconds")
    parser.add_argument("--user-cpu-sec", default="", help="User CPU seconds")
    parser.add_argument("--sys-cpu-sec", default="", help="System CPU seconds")
    parser.add_argument("--cpu-pct", default="", help="CPU percent as emitted by /usr/bin/time")
    parser.add_argument("--max-rss-kb", default="", help="Max RSS in KB")
    parser.add_argument("--major-page-faults", default="", help="Major page faults")
    parser.add_argument("--minor-page-faults", default="", help="Minor page faults")
    parser.add_argument("--vol-ctx-switches", default="", help="Voluntary context switches")
    parser.add_argument("--invol-ctx-switches", default="", help="Involuntary context switches")
    parser.add_argument("--exit-code", default="", help="Exit code")
    parser.add_argument("--pairs-per-sec", default="", help="Pair throughput")
    parser.add_argument("--out-csv-bytes", default="", help="Output CSV size in bytes")
    parser.add_argument("--out-csv", default="", help="Output CSV path")
    parser.add_argument("--stdout-log", default="", help="Stdout log path")
    parser.add_argument("--stderr-log", default="", help="Stderr log path")
    parser.add_argument("--time-tsv", default="", help="Raw /usr/bin/time TSV path")
    args = parser.parse_args()

    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    os.makedirs(os.path.dirname(os.path.abspath(args.per_pair_jsonl)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.summary_jsonl)), exist_ok=True)

    performance = _build_performance(args, args.csv)
    numeric_sums = {}
    numeric_counts = {}
    numeric_mins = {}
    numeric_maxs = {}
    source_counts = {}

    with open(args.per_pair_jsonl, "w", encoding="utf-8") as out:
        for row in rows:
            pair_record = dict(row)
            pair_record["pair_performance"] = {
                "pair_wall_sec": _safe_number(row.get("pair_wall_sec")),
                "pair_cpu_sec": _safe_number(row.get("pair_cpu_sec")),
                "pair_perf_source": row.get("pair_perf_source", ""),
                "worker_pid": _safe_number(row.get("worker_pid")),
            }
            pair_record["source_csv"] = os.path.abspath(args.csv)
            pair_record["run_util"] = performance["util"]
            pair_record["run_shots"] = performance["shots"]
            pair_record["run_workers"] = performance["workers"]
            pair_record["run_sample_k"] = performance["sample_k"]
            pair_record["run_sample_seed"] = performance["sample_seed"]
            pair_record["elapsed_sec"] = performance["elapsed_sec"]
            pair_record["user_cpu_sec"] = performance["user_cpu_sec"]
            pair_record["sys_cpu_sec"] = performance["sys_cpu_sec"]
            pair_record["cpu_pct"] = performance["cpu_pct"]
            pair_record["max_rss_kb"] = performance["max_rss_kb"]
            pair_record["pairs_per_sec"] = performance["pairs_per_sec"]
            pair_record["req_cpus"] = performance["req_cpus"]
            pair_record["req_mem_mb"] = performance["req_mem_mb"]
            pair_record["stdout_log"] = performance["stdout_log"]
            pair_record["stderr_log"] = performance["stderr_log"]
            pair_record["time_tsv"] = performance["time_tsv"]
            pair_record["performance"] = performance
            out.write(json.dumps(pair_record, ensure_ascii=True) + "\n")
            for key, value in row.items():
                if key.endswith("_source"):
                    source_counts.setdefault(key, Counter())[str(value)] += 1
                number = _safe_number(value)
                if number is None:
                    continue
                numeric_sums[key] = numeric_sums.get(key, 0.0) + number
                numeric_counts[key] = numeric_counts.get(key, 0) + 1
                numeric_mins[key] = number if key not in numeric_mins else min(numeric_mins[key], number)
                numeric_maxs[key] = number if key not in numeric_maxs else max(numeric_maxs[key], number)

    numeric_means = {
        key: (numeric_sums[key] / numeric_counts[key])
        for key in sorted(numeric_sums.keys())
        if numeric_counts.get(key, 0) > 0
    }

    summary = {
        "source_csv": os.path.abspath(args.csv),
        "per_pair_jsonl": os.path.abspath(args.per_pair_jsonl),
        "num_pairs": len(rows),
        "fields": fieldnames,
        "means": numeric_means,
        "mins": {key: numeric_mins[key] for key in sorted(numeric_mins.keys())},
        "maxs": {key: numeric_maxs[key] for key in sorted(numeric_maxs.keys())},
        "source_counts": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(source_counts.items())
        },
    }

    summary["performance"] = performance
    summary["util"] = performance["util"]
    summary["shots"] = performance["shots"]
    summary["workers"] = performance["workers"]
    summary["sample_k"] = performance["sample_k"]
    summary["sample_seed"] = performance["sample_seed"]
    summary["elapsed_sec"] = performance["elapsed_sec"]
    summary["user_cpu_sec"] = performance["user_cpu_sec"]
    summary["sys_cpu_sec"] = performance["sys_cpu_sec"]
    summary["max_rss_kb"] = performance["max_rss_kb"]
    summary["pairs_per_sec"] = performance["pairs_per_sec"]
    summary["req_cpus"] = performance["req_cpus"]
    summary["req_mem_mb"] = performance["req_mem_mb"]

    for key in (
        "effective_utilization",
        "fidelity",
        "shots",
        "pair_index",
        "pair_wall_sec",
        "pair_cpu_sec",
    ):
        if key in numeric_means:
            summary[f"{key}_mean"] = numeric_means[key]

    with open(args.summary_jsonl, "w", encoding="utf-8") as out:
        out.write(json.dumps(summary, ensure_ascii=True) + "\n")

    print(f"[OK] Wrote {args.per_pair_jsonl}")
    print(f"[OK] Wrote {args.summary_jsonl}")
    print(f"[INFO] num_pairs={len(rows)}")


if __name__ == "__main__":
    main()
