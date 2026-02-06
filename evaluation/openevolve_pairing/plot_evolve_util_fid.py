#!/usr/bin/env python3
"""
Plot effective utilization vs fidelity for evolved pairing score.

Outputs:
  - scatter plot with order/score labels
  - CSV of Top-K points with util/fid/score/order/rank
  - prints bottom-right point (max util, min fid) for anomaly check
"""

import argparse
import os
import sys

import numpy as np
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
    parser.add_argument("--util", type=int, default=60, choices=[30, 45, 60, 88])
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--candidate-limit", type=int, default=120)
    parser.add_argument("--target", default=None, help="Path to evolved target.py (default: openevolve_pairing/target.py)")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--out-csv", default=None, help="Output CSV path")
    parser.add_argument("--use-pareto-rank", action="store_true", help="Compute Pareto ranks")
    args = parser.parse_args()

    # Configure evaluation
    config.TARGET_UTIL = args.util
    config.SHOTS = args.shots
    config.TOP_K = args.top_k
    config.CANDIDATE_LIMIT = args.candidate_limit
    config.EVAL_MODE = "candidate"

    _reset_evaluator()
    evaluator._init()
    if args.use_pareto_rank:
        evaluator._ensure_pair_ranks()

    target_path = args.target
    if target_path is None:
        target_path = os.path.join(os.path.dirname(__file__), "target.py")
    _kind, score_fn = evaluator._load_score_fn(target_path)

    scores = _score_all(score_fn)
    top_idx = np.argsort(scores)[-config.TOP_K:]
    # Order by score descending for labeling
    top_idx = list(sorted(top_idx, key=lambda i: scores[i], reverse=True))

    pts = [evaluator._get_pair_metrics(i) for i in top_idx]
    ranks = evaluator._PAIR_RANKS if args.use_pareto_rank else None

    # Build table rows
    rows = []
    for order, idx in enumerate(top_idx, start=1):
        util, fid = evaluator._get_pair_metrics(idx)
        name1 = evaluator._CANDIDATES[idx][2]
        name2 = evaluator._CANDIDATES[idx][3]
        rank = ranks[idx] if ranks is not None else -1
        rows.append({
            "order": order,
            "name_1": name1,
            "name_2": name2,
            "effective_utilization": util,
            "fidelity": fid,
            "score": scores[idx],
            "pareto_rank": rank,
        })

    # Bottom-right point: max util, then min fid
    if rows:
        br = max(rows, key=lambda r: (r["effective_utilization"], -r["fidelity"]))
        print("[Bottom-right] order={order} score={score:.6f} util={effective_utilization:.4f} fid={fidelity:.4f} rank={pareto_rank}".format(**br))

    # Output CSV
    out_csv = args.out_csv
    if out_csv is None:
        out_csv = os.path.join(os.path.dirname(__file__), f"evolve_util_fid_top{config.TOP_K}_util{args.util}_shots{args.shots}.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        cols = ["order", "name_1", "name_2", "effective_utilization", "fidelity", "score", "pareto_rank"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write("{order},{name_1},{name_2},{effective_utilization:.6f},{fidelity:.6f},{score:.6f},{pareto_rank}\n".format(**r))

    # Plot
    utils = [p[0] for p in pts]
    fids = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    ax.scatter(utils, fids, s=36, c="#1f77b4", alpha=0.85)
    ax.set_xlabel("Effective Utilization")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"Evolved Top-{config.TOP_K} (util={args.util}, shots={args.shots})")

    # Annotate order/score
    for r in rows:
        ax.text(
            r["effective_utilization"],
            r["fidelity"],
            f"{r['order']}:{r['score']:.2f}",
            fontsize=7,
            ha="left",
            va="bottom",
        )

    out_png = args.out
    if out_png is None:
        out_png = os.path.join(os.path.dirname(__file__), f"evolve_util_fid_top{config.TOP_K}_util{args.util}_shots{args.shots}.png")
    plt.tight_layout()
    plt.savefig(out_png)

    print("[OK] Wrote", out_png)
    print("[OK] Wrote", out_csv)


if __name__ == "__main__":
    main()
