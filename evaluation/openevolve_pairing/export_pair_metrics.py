#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:  # pragma: no cover - tqdm optional
    _HAS_TQDM = False
    def tqdm(x, **_kwargs):
        return x

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


def _init_worker(util, shots, candidate_limit):
    # Each worker has its own evaluator state.
    config.TARGET_UTIL = util
    config.SHOTS = shots
    if candidate_limit is not None and candidate_limit >= 0:
        config.CANDIDATE_LIMIT = candidate_limit
    else:
        config.CANDIDATE_LIMIT = None
    evaluator.config.TARGET_UTIL = util
    evaluator.config.SHOTS = shots
    if candidate_limit is not None and candidate_limit >= 0:
        evaluator.config.CANDIDATE_LIMIT = candidate_limit
    else:
        evaluator.config.CANDIDATE_LIMIT = None
    _reset_evaluator()
    evaluator._init()


def _compute_row(idx):
    circ1, circ2, name1, name2 = evaluator._CANDIDATES[idx]
    q1, q2 = evaluator._QERNEL_PAIRS[idx]
    m1 = q1.get_metadata()
    m2 = q2.get_metadata()
    eff, fid = evaluator._get_pair_metrics(idx)

    row = {
        "name_1": name1,
        "name_2": name2,
        "effective_utilization": float(eff),
        "fidelity": float(fid),
        "pair_index": idx,
    }
    row.update(_flatten_meta("m1_", m1))
    row.update(_flatten_meta("m2_", m2))
    row.update(_pair_derived(m1, m2))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--util", type=int, default=30, choices=[30, 45, 60, 88],
                        help="TARGET_UTIL (percent)")
    parser.add_argument("--shots", type=int, default=config.SHOTS, help="Simulation shots")
    parser.add_argument("--candidate-limit", type=int, default=-1,
                        help="Max candidates to simulate; -1 means no limit")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (0 uses SLURM_CPUS_PER_TASK)")
    parser.add_argument("--flush-every", type=int, default=10,
                        help="Flush CSV every N rows (0 disables)")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--out-dir", default="pairing_metadata")
    parser.add_argument("--out", default=None)
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute all rows and overwrite existing output file")
    parser.add_argument("--debug", action="store_true", help="Print target qubits and sample pairs")
    args = parser.parse_args()

    # Ensure both the package config and evaluator's local config module are updated.
    config.TARGET_UTIL = args.util
    config.SHOTS = args.shots
    if args.candidate_limit >= 0:
        config.CANDIDATE_LIMIT = args.candidate_limit
    else:
        config.CANDIDATE_LIMIT = None
    evaluator.config.TARGET_UTIL = args.util
    evaluator.config.SHOTS = args.shots
    if args.candidate_limit >= 0:
        evaluator.config.CANDIDATE_LIMIT = args.candidate_limit
    else:
        evaluator.config.CANDIDATE_LIMIT = None

    _reset_evaluator()
    evaluator._init()
    if args.debug:
        target_qubits = evaluator.repro.UTIL_TO_QUBITS.get(config.TARGET_UTIL)
        print(f"[Debug] TARGET_UTIL={config.TARGET_UTIL} target_qubits={target_qubits}")
        sample = evaluator._CANDIDATES[:5]
        for _c1, _c2, n1, n2 in sample:
            print(f"[Debug] pair: {n1} + {n2}")

    if args.out is None:
        out_name = f"pair_metrics_util{args.util}_shots{args.shots}.{args.format}"
    else:
        out_name = args.out
    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(__file__), out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_name if os.path.isabs(out_name) else os.path.join(out_dir, out_name)
    if args.recompute and os.path.exists(out_path):
        os.remove(out_path)

    rows = [] if args.format == "json" else None
    fieldnames = None
    f_csv = None
    writer = None
    existing = {}
    existing_rows = []
    rows_written = []
    if args.format == "csv" and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if fieldnames:
                for row in reader:
                    existing_rows.append(row)
                    n1 = row.get("name_1")
                    n2 = row.get("name_2")
                    if n1 and n2:
                        k = tuple(sorted((n1, n2)))
                        existing[k] = row

    total = len(evaluator._CANDIDATES)
    todo_indices = []
    for idx in range(total):
        _, _, name1, name2 = evaluator._CANDIDATES[idx]
        key = tuple(sorted((name1, name2)))
        if key in existing:
            continue
        todo_indices.append(idx)

    workers = args.workers
    if workers == 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if workers < 1:
        workers = 1

    def _handle_row(row):
        nonlocal fieldnames, f_csv, writer, rows
        if args.format == "json":
            rows.append(row)
            return
        if fieldnames is None:
            fieldnames = sorted(row.keys())
            preferred = ["name_1", "name_2", "effective_utilization", "fidelity"]
            fieldnames = preferred + [k for k in fieldnames if k not in preferred]
            f_csv = open(out_path, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
        if f_csv is None:
            f_csv = open(out_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)

        missing = [k for k in row.keys() if k not in fieldnames]
        if missing:
            fieldnames = fieldnames + [k for k in sorted(missing) if k not in fieldnames]
            if f_csv is not None:
                f_csv.close()
                f_csv = None
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w", newline="", encoding="utf-8") as f_tmp:
                writer_tmp = csv.DictWriter(f_tmp, fieldnames=fieldnames)
                writer_tmp.writeheader()
                for existing_row in existing_rows:
                    writer_tmp.writerow(existing_row)
                for written_row in rows_written:
                    writer_tmp.writerow(written_row)
            os.replace(tmp_path, out_path)
            f_csv = open(out_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)

        writer.writerow(row)
        rows_written.append(row)
        if args.flush_every and (len(rows_written) % args.flush_every == 0):
            f_csv.flush()
            os.fsync(f_csv.fileno())

    if workers == 1:
        iterator = tqdm(todo_indices, desc="Simulating pairs") if _HAS_TQDM else todo_indices
        for idx in iterator:
            row = _compute_row(idx)
            _handle_row(row)
            if not _HAS_TQDM:
                done = len(rows_written) if args.format != "json" else len(rows)
                if done % 10 == 0 or done == len(todo_indices):
                    print(f"  progress {done}/{len(todo_indices)}", flush=True)
    else:
        with Pool(processes=workers, initializer=_init_worker,
                  initargs=(args.util, args.shots, args.candidate_limit)) as pool:
            iterator = pool.imap_unordered(_compute_row, todo_indices, chunksize=1)
            if _HAS_TQDM:
                iterator = tqdm(iterator, total=len(todo_indices), desc="Simulating pairs")
            for row in iterator:
                _handle_row(row)
                if not _HAS_TQDM:
                    done = len(rows_written) if args.format != "json" else len(rows)
                    if done % 10 == 0 or done == len(todo_indices):
                        print(f"  progress {done}/{len(todo_indices)}", flush=True)

    if args.format == "json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    else:
        if f_csv is not None:
            f_csv.close()

    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
