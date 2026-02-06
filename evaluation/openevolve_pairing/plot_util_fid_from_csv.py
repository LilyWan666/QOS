#!/usr/bin/env python3
"""
Plot effective utilization vs fidelity from an existing CSV.
Computes non-dominated ranks, assigns an order by score, and annotates each point.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV with effective_utilization and fidelity columns")
    parser.add_argument("--fid-weight", type=float, default=0.1, help="Weight for fidelity in score")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--out-csv", default=None, help="Output CSV path with ranks and score")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "effective_utilization" not in df.columns or "fidelity" not in df.columns:
        raise ValueError("CSV must contain effective_utilization and fidelity columns")

    utils = df["effective_utilization"].to_numpy(dtype=float)
    fids = df["fidelity"].to_numpy(dtype=float)
    points = list(zip(utils, fids))

    ranks = _nondominated_ranks(points)
    scores = (1.0 / np.array(ranks, dtype=float)) + args.fid_weight * fids
    order = np.argsort(scores)[::-1]  # descending

    out_df = df.copy()
    out_df["pareto_rank"] = ranks
    out_df["score"] = scores
    out_df["order"] = np.argsort(order) + 1

    # Bottom-right: max util, then min fid
    br_idx = max(range(len(points)), key=lambda i: (utils[i], -fids[i]))
    print(
        "[Bottom-right] index={idx} order={order} score={score:.6f} util={util:.6f} fid={fid:.6f} rank={rank}".format(
            idx=br_idx,
            order=int(out_df.loc[br_idx, "order"]),
            score=float(out_df.loc[br_idx, "score"]),
            util=float(utils[br_idx]),
            fid=float(fids[br_idx]),
            rank=int(out_df.loc[br_idx, "pareto_rank"]),
        )
    )

    # Output CSV
    out_csv = args.out_csv
    if out_csv is None:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_csv = os.path.join(os.path.dirname(args.csv), f"{base}_ranked.csv")
    out_df.to_csv(out_csv, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    ax.scatter(utils, fids, s=36, c="#1f77b4", alpha=0.85)
    ax.set_xlabel("Effective Utilization")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"Effective Utilization vs Fidelity ({os.path.basename(args.csv)})")

    for i in order:
        label = f"{int(out_df.loc[i, 'order'])}:{scores[i]:.2f}"
        ax.text(utils[i], fids[i], label, fontsize=7, ha="left", va="bottom")

    out_png = args.out
    if out_png is None:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_png = os.path.join(os.path.dirname(args.csv), f"{base}_ranked.png")
    plt.tight_layout()
    plt.savefig(out_png)

    print("[OK] Wrote", out_png)
    print("[OK] Wrote", out_csv)


if __name__ == "__main__":
    main()
