#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.openevolve_pairing import config
from evaluation.openevolve_pairing import evaluator_orig as evaluator


def _reset_evaluator():
    evaluator._INIT_DONE = False
    evaluator._BENCHMARKS = None
    evaluator._CANDIDATES = None
    evaluator._FEATURES = None
    evaluator._QERNEL_PAIRS = None
    evaluator._MP = None
    evaluator._PAIR_METRICS = {}
    evaluator._PAIR_RANKS = None
    evaluator._PAIR_PROXY = None
    evaluator._SIM = None


def _flatten_meta(prefix, meta):
    out = {}
    for k, v in meta.items():
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


def _pair_derived(m1, m2):
    def get(k):
        return float(m1.get(k, 0.0)), float(m2.get(k, 0.0))

    derived = {}
    for k in [
        "depth",
        "num_qubits",
        "num_nonlocal_gates",
        "num_cnot_gates",
        "num_measurements",
        "number_instructions",
    ]:
        v1, v2 = get(k)
        s = v1 + v2
        d = abs(v1 - v2)
        mx = max(v1, v2)
        mn = min(v1, v2)
        avg = s / 2.0
        ratio = (mn / mx) if mx > 0 else 1.0
        derived[f"{k}_sum"] = s
        derived[f"{k}_diff"] = d
        derived[f"{k}_max"] = mx
        derived[f"{k}_avg"] = avg
        derived[f"{k}_ratio"] = ratio
    return derived


def _extract_proxy_feature(idx, feature_name):
    f = evaluator._FEATURES[idx]
    if feature_name in f:
        return float(f.get(feature_name, 0.0))
    if feature_name == "depth_max":
        return float(max(f.get("depth_1", 0.0), f.get("depth_2", 0.0)))
    if feature_name == "cnot_max":
        return float(max(f.get("cnot_1", 0.0), f.get("cnot_2", 0.0)))
    if feature_name == "nonlocal_max":
        return float(max(f.get("nonlocal_1", 0.0), f.get("nonlocal_2", 0.0)))
    if feature_name == "measure_max":
        return float(max(f.get("measure_1", 0.0), f.get("measure_2", 0.0)))
    if feature_name == "instr_max":
        return float(max(f.get("instr_1", 0.0), f.get("instr_2", 0.0)))
    raise KeyError(f"Unsupported proxy feature: {feature_name}")


def _compute_row(idx, feature_name):
    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    circ1, circ2, name1, name2 = evaluator._CANDIDATES[idx]
    q1, q2 = evaluator._QERNEL_PAIRS[idx]
    m1 = q1.get_metadata()
    m2 = q2.get_metadata()
    raw_value = _extract_proxy_feature(idx, feature_name)
    pair_wall_sec = time.perf_counter() - t_wall0
    pair_cpu_sec = time.process_time() - t_cpu0

    row = {
        "name_1": name1,
        "name_2": name2,
        "pair_index": idx,
        "proxy_feature": feature_name,
        "proxy_raw_value": float(raw_value),
        "proxy_value": float(raw_value),
        "pair_wall_sec": float(pair_wall_sec),
        "pair_cpu_sec": float(pair_cpu_sec),
        "pair_perf_source": "measured",
        "worker_pid": int(os.getpid()),
    }
    row.update(_flatten_meta("m1_", m1))
    row.update(_flatten_meta("m2_", m2))
    row.update(_pair_derived(m1, m2))
    return row


def main():
    parser = argparse.ArgumentParser()
    util_choices = sorted(evaluator.repro.UTIL_TO_QUBITS.keys())
    parser.add_argument("--util", type=int, default=30, choices=util_choices)
    parser.add_argument("--shots", type=int, default=config.SHOTS)
    parser.add_argument("--proxy-feature", default="depth_ratio")
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--out-dir", default="pairing_metadata")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample-k", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config.TARGET_UTIL = args.util
    config.SHOTS = args.shots
    config.CANDIDATE_LIMIT = None
    evaluator.config.TARGET_UTIL = args.util
    evaluator.config.SHOTS = args.shots
    evaluator.config.CANDIDATE_LIMIT = None

    _reset_evaluator()
    evaluator._init()

    if args.debug:
        print(
            f"[Debug] TARGET_UTIL={config.TARGET_UTIL} shots={config.SHOTS} "
            f"proxy_feature={args.proxy_feature}",
            flush=True,
        )

    if args.out is None:
        suffix = ""
        if args.sample_k > 0:
            suffix = f"_sample{args.sample_k}_seed{args.sample_seed}"
        out_name = (
            f"pair_proxy_{args.proxy_feature}_util{args.util}_shots{args.shots}"
            f"{suffix}.csv"
        )
    else:
        out_name = args.out

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(__file__), out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_name if os.path.isabs(out_name) else os.path.join(out_dir, out_name)

    if args.recompute and os.path.exists(out_path):
        os.remove(out_path)

    existing = {}
    existing_rows = []
    fieldnames = None
    rows_written = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if fieldnames:
                for row in reader:
                    existing_rows.append(row)
                    n1 = row.get("name_1")
                    n2 = row.get("name_2")
                    if n1 and n2:
                        existing[tuple(sorted((n1, n2)))] = row

    total = len(evaluator._CANDIDATES)
    selected_indices = list(range(total))
    if args.sample_k > 0:
        sample_k = min(args.sample_k, total)
        rng = random.Random(args.sample_seed)
        selected_indices = sorted(rng.sample(selected_indices, sample_k))
        print(
            f"[INFO] sample mode enabled: selected {sample_k}/{total} pairs "
            f"(seed={args.sample_seed})",
            flush=True,
        )

    todo_indices = []
    for idx in selected_indices:
        _, _, name1, name2 = evaluator._CANDIDATES[idx]
        key = tuple(sorted((name1, name2)))
        if key in existing:
            continue
        todo_indices.append(idx)

    f_csv = None
    writer = None
    for idx in todo_indices:
        row = _compute_row(idx, args.proxy_feature)
        if fieldnames is None:
            fieldnames = sorted(row.keys())
            preferred = [
                "name_1",
                "name_2",
                "proxy_feature",
                "proxy_value",
                "proxy_raw_value",
                "pair_wall_sec",
                "pair_cpu_sec",
            ]
            fieldnames = preferred + [k for k in fieldnames if k not in preferred]
            f_csv = open(out_path, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
        if f_csv is None:
            f_csv = open(out_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)

        writer.writerow(row)
        rows_written.append(row)
        if args.flush_every and (len(rows_written) % args.flush_every == 0):
            f_csv.flush()
            os.fsync(f_csv.fileno())

    if f_csv is not None:
        f_csv.close()

    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
