#!/usr/bin/env python3
"""
Plot effective utilization vs fidelity from CSV and highlight Top-K picked by
QOS (target_orig.py) vs evolved (target.py) scoring. Labels show Pareto rank only.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.openevolve_pairing import evaluator, config


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


def _dominates(p, q):
    return (p[0] >= q[0] and p[1] >= q[1]) and (p[0] > q[0] or p[1] > q[1])


def _nondominated_ranks(points):
    n = len(points)
    dominates = [set() for _ in range(n)]
    dominated_count = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(points[i], points[j]):
                dominates[i].add(j)
                dominated_count[j] += 1
            elif _dominates(points[j], points[i]):
                dominates[j].add(i)
                dominated_count[i] += 1

    ranks = [0] * n
    front = [i for i in range(n) if dominated_count[i] == 0]
    rank = 1
    while front:
        next_front = []
        for i in front:
            ranks[i] = rank
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        front = next_front
        rank += 1
    return ranks


def _score_all(score_fn):
    scores = []
    for idx, _f in enumerate(evaluator._FEATURES):
        q1, q2 = evaluator._QERNEL_PAIRS[idx]
        try:
            s = float(score_fn(evaluator._MP, q1, q2, evaluator._SIM.backend, weighted=False, weights=[]))
        except Exception:
            s = -1e9
        scores.append(s)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV with effective_utilization and fidelity")
    parser.add_argument("--util", type=int, default=60, choices=[30, 45, 60, 88])
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--candidate-limit", type=int, default=120)
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--out-csv", default=None, help="Output CSV path with rank/flags")
    parser.add_argument("--orig-target", default=None, help="Path to original target (default: target_orig.py)")
    parser.add_argument("--new-target", default=None, help="Path to new target (default: target.py)")
    parser.add_argument("--restrict-to-csv", action="store_true",
                        help="Only score candidates that appear in CSV (Top-K chosen within CSV subset)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "effective_utilization" not in df.columns or "fidelity" not in df.columns:
        raise ValueError("CSV must contain effective_utilization and fidelity columns")

    # Compute Pareto ranks from CSV points (no simulation).
    utils = df["effective_utilization"].to_numpy(dtype=float)
    fids = df["fidelity"].to_numpy(dtype=float)
    points = list(zip(utils, fids))
    ranks = _nondominated_ranks(points)
    df["pareto_rank"] = ranks

    # Init evaluator to compute scores (no fidelity simulation).
    config.TARGET_UTIL = args.util
    config.SHOTS = args.shots
    if args.candidate_limit and args.candidate_limit > 0:
        config.CANDIDATE_LIMIT = args.candidate_limit
    else:
        config.CANDIDATE_LIMIT = None
    # evaluator imports its own `config` module; keep both in sync
    evaluator.config.TARGET_UTIL = config.TARGET_UTIL
    evaluator.config.SHOTS = config.SHOTS
    evaluator.config.CANDIDATE_LIMIT = config.CANDIDATE_LIMIT
    _reset_evaluator()
    evaluator._init()

    # Load scoring functions.
    orig_path = args.orig_target or os.path.join(os.path.dirname(__file__), "target_orig.py")
    new_path = args.new_target or os.path.join(os.path.dirname(__file__), "target.py")
    _kind_orig, score_fn_orig = evaluator._load_score_fn(orig_path)
    _kind_new, score_fn_new = evaluator._load_score_fn(new_path)

    # Map candidate name pairs -> index
    cand_map = {}
    for idx, (_c1, _c2, n1, n2) in enumerate(evaluator._CANDIDATES):
        cand_map[(n1, n2)] = idx

    if args.restrict_to_csv:
        # Only score the subset that appears in CSV.
        subset_idx = []
        for _, row in df.iterrows():
            key = (row.get("name_1"), row.get("name_2"))
            idx = cand_map.get(key, None)
            if idx is not None:
                subset_idx.append(idx)
        subset_idx = sorted(set(subset_idx))

        scores_orig = {i: None for i in subset_idx}
        scores_new = {i: None for i in subset_idx}
        for i in subset_idx:
            q1, q2 = evaluator._QERNEL_PAIRS[i]
            try:
                scores_orig[i] = float(score_fn_orig(evaluator._MP, q1, q2, evaluator._SIM.backend, weighted=False, weights=[]))
            except Exception:
                scores_orig[i] = -1e9
            try:
                scores_new[i] = float(score_fn_new(evaluator._MP, q1, q2, evaluator._SIM.backend, weighted=False, weights=[]))
            except Exception:
                scores_new[i] = -1e9

        top_orig = set(sorted(subset_idx, key=lambda i: scores_orig[i])[-args.top_k:])
        top_new = set(sorted(subset_idx, key=lambda i: scores_new[i])[-args.top_k:])
    else:
        scores_orig = _score_all(score_fn_orig)
        scores_new = _score_all(score_fn_new)
        top_orig = set(np.argsort(scores_orig)[-args.top_k:])
        top_new = set(np.argsort(scores_new)[-args.top_k:])

    # Flag CSV rows that are in Top-K sets
    in_orig = []
    in_new = []
    for _, row in df.iterrows():
        key = (row.get("name_1"), row.get("name_2"))
        idx = cand_map.get(key, None)
        in_orig.append(idx in top_orig if idx is not None else False)
        in_new.append(idx in top_new if idx is not None else False)
    df["topk_orig"] = in_orig
    df["topk_new"] = in_new

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    ax.scatter(utils, fids, s=22, c="#cccccc", alpha=0.6, label="All")

    if any(in_orig):
        sel = df["topk_orig"].to_numpy(dtype=bool)
        ax.scatter(utils[sel], fids[sel], s=44, c="#ff7f0e", marker="x", label="QOS (orig) Top-K")
        for i in np.where(sel)[0]:
            label = f"{int(df.loc[i, 'pareto_rank'])}"
            ax.text(utils[i], fids[i], label, fontsize=7, ha="left", va="bottom")

    if any(in_new):
        sel = df["topk_new"].to_numpy(dtype=bool)
        ax.scatter(
            utils[sel],
            fids[sel],
            s=40,
            facecolors="none",
            edgecolors="#1f77b4",
            marker="o",
            linewidths=1.2,
            label="Evolved (new) Top-K",
        )
        for i in np.where(sel)[0]:
            label = f"{int(df.loc[i, 'pareto_rank'])}"
            ax.text(utils[i], fids[i], label, fontsize=7, ha="left", va="bottom")

    ax.set_xlabel("Effective Utilization")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"Effective Utilization vs Fidelity (util={args.util}, Top-K={args.top_k})")
    ax.legend(loc="best")

    out_png = args.out
    if out_png is None:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_png = os.path.join(os.path.dirname(args.csv), f"{base}_qos_vs_evolved_rankonly.png")
    plt.tight_layout()
    plt.savefig(out_png)

    out_csv = args.out_csv
    if out_csv is None:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_csv = os.path.join(os.path.dirname(args.csv), f"{base}_qos_vs_evolved_rankonly.csv")
    df.to_csv(out_csv, index=False)

    print("[OK] Wrote", out_png)
    print("[OK] Wrote", out_csv)


if __name__ == "__main__":
    main()
