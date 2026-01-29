#!/usr/bin/env python3
"""Compare evolved pairing functions by plotting their top-K Pareto fronts."""

import argparse
import os
import random
from datetime import datetime

import numpy as np
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm optional
    def tqdm(x, **_kwargs):
        return x

from evaluation.openevolve_pairing import evaluator, config


def _pareto_frontier(points):
    # Strict non-dominated frontier for maximization.
    points = sorted(points, key=lambda p: (p[0], p[1]), reverse=True)
    frontier = []
    best_y = float("-inf")
    for x, y in points:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return sorted(frontier, key=lambda p: p[0])


def _score_all(score_fn):
    scores = []
    for idx, _f in tqdm(
        enumerate(evaluator._FEATURES),
        total=len(evaluator._FEATURES),
        desc="Scoring candidates",
    ):
        q1, q2 = evaluator._QERNEL_PAIRS[idx]
        try:
            s = float(score_fn(evaluator._MP, q1, q2, evaluator._SIM.backend, weighted=False, weights=[]))
        except Exception:
            s = -1e9
        scores.append(s)
    return scores


def _avg_metrics(selected_idx_in_top, points, top_idx, ranks):
    if not selected_idx_in_top:
        return float("nan"), float("nan"), float("nan")
    util = [points[i][0] for i in selected_idx_in_top]
    fid = [points[i][1] for i in selected_idx_in_top]
    if ranks is None:
        avg_rank = float("nan")
    else:
        avg_rank = float(np.mean([ranks[top_idx[i]] for i in selected_idx_in_top]))
    return float(np.mean(util)), float(np.mean(fid)), avg_rank


def _select_low_util_indices(points, n):
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    return order[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True, help="Path to new evolved target.py")
    parser.add_argument("--old", required=True, help="Path to original target.py")
    parser.add_argument("--out", default="pairing_compare.png", help="Output figure path")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--candidate-limit", type=int, default=config.CANDIDATE_LIMIT or 0)
    parser.add_argument("--use-pareto-rank", action="store_true", help="Compute Pareto ranks (slower)")
    parser.add_argument("--fast", action="store_true", help="Only simulate Top-K pairs")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.candidate_limit:
        config.CANDIDATE_LIMIT = args.candidate_limit
    config.TOP_K = args.top_k
    random.seed(args.seed)
    np.random.seed(args.seed)

    evaluator._init()
    if args.use_pareto_rank:
        evaluator._ensure_pair_ranks()

    _kind_new, score_fn_new = evaluator._load_score_fn(args.new)
    _kind_old, score_fn_old = evaluator._load_score_fn(args.old)

    scores_new = _score_all(score_fn_new)
    scores_old = _score_all(score_fn_old)

    top_new = np.argsort(scores_new)[-config.TOP_K:]
    top_old = np.argsort(scores_old)[-config.TOP_K:]
    union_idx = sorted(set(top_new).union(set(top_old)))

    if args.fast:
        pts_new = [evaluator._get_pair_metrics(i) for i in top_new]
        pts_old = [evaluator._get_pair_metrics(i) for i in top_old]
        pts_union = [evaluator._get_pair_metrics(i) for i in union_idx]
    else:
        pts_new = [evaluator._get_pair_metrics(i) for i in top_new]
        pts_old = [evaluator._get_pair_metrics(i) for i in top_old]
        pts_union = [evaluator._get_pair_metrics(i) for i in union_idx]
    ranks = evaluator._PAIR_RANKS if args.use_pareto_rank else None
    if ranks is not None:
        avg_rank_new = float(np.mean([ranks[i] for i in top_new])) if len(top_new) else float("inf")
        avg_rank_old = float(np.mean([ranks[i] for i in top_old])) if len(top_old) else float("inf")
    else:
        avg_rank_new = float("nan")
        avg_rank_old = float("nan")
    avg_fid_new = float(np.mean([p[1] for p in pts_new])) if pts_new else float("nan")
    avg_fid_old = float(np.mean([p[1] for p in pts_old])) if pts_old else float("nan")
    avg_proxy_new = float(np.mean([evaluator._get_pair_proxy(i) for i in top_new])) if len(top_new) else float("nan")
    avg_proxy_old = float(np.mean([evaluator._get_pair_proxy(i) for i in top_old])) if len(top_old) else float("nan")
    if args.fast:
        corr = float("nan")
    else:
        # Correlation between simulated fidelity and fast proxy over all candidates.
        all_fids = [evaluator._get_pair_metrics(i)[1] for i in range(len(evaluator._CANDIDATES))]
        all_proxy = [evaluator._get_pair_proxy(i) for i in range(len(evaluator._CANDIDATES))]
        corr = float(np.corrcoef(all_fids, all_proxy)[0, 1]) if len(all_fids) > 1 else float("nan")
    if ranks is not None:
        score_new = (1.0 / avg_rank_new) + config.FID_WEIGHT * (avg_fid_new if not np.isnan(avg_fid_new) else 0.0)
        score_old = (1.0 / avg_rank_old) + config.FID_WEIGHT * (avg_fid_old if not np.isnan(avg_fid_old) else 0.0)
        print(f"[New] avg-rank={avg_rank_new:.6f} (lower is better)", flush=True)
        print(f"[Old] avg-rank={avg_rank_old:.6f} (lower is better)", flush=True)
        print(f"[New] score={score_new:.6f}", flush=True)
        print(f"[Old] score={score_old:.6f}", flush=True)
    print(f"[New] avg-fid={avg_fid_new:.6f}", flush=True)
    print(f"[Old] avg-fid={avg_fid_old:.6f}", flush=True)
    print(f"[New] avg-proxy={avg_proxy_new:.6f}", flush=True)
    print(f"[Old] avg-proxy={avg_proxy_old:.6f}", flush=True)
    if not np.isnan(corr):
        print(f"[Proxy] fid~proxy corr={corr:.4f}", flush=True)
    if ranks is None:
        print("[Info] Pareto ranks skipped (--use-pareto-rank not set).", flush=True)

    # Sweep 1: choose Top-N from each method's Top-K.
    n_list = [4, 8, 12]
    print("[Sweep1] Top-N from each method's Top-K:", flush=True)
    for n in n_list:
        if len(pts_new) >= n:
            new_idx = _select_low_util_indices(pts_new, n)
            new_util, new_fid, new_rank = _avg_metrics(new_idx, pts_new, top_new, ranks)
        else:
            new_util = new_fid = new_rank = float("nan")
        if len(pts_old) >= n:
            old_idx = random.sample(range(len(pts_old)), n)
            old_util, old_fid, old_rank = _avg_metrics(old_idx, pts_old, top_old, ranks)
        else:
            old_util = old_fid = old_rank = float("nan")
        print(
            f"  N={n} | New(avg util={new_util:.4f}, avg fid={new_fid:.4f}, avg rank={new_rank:.3f}) "
            f"| Old(avg util={old_util:.4f}, avg fid={old_fid:.4f}, avg rank={old_rank:.3f})",
            flush=True,
        )

    # Sweep 2: avg fidelity for pairs with util >= min_util (within Top-K)
    print("[Sweep2] Avg fidelity for util >= min_util (within Top-K):", flush=True)
    min_utils = [round(0.50 + 0.02 * i, 2) for i in range(6)]
    for mu in min_utils:
        new_fids = [fid for (util, fid) in pts_new if util >= mu]
        old_fids = [fid for (util, fid) in pts_old if util >= mu]
        new_avg = float(np.mean(new_fids)) if new_fids else float("nan")
        old_avg = float(np.mean(old_fids)) if old_fids else float("nan")
        print(
            f"  min_util={mu:.2f} | New(avg fid={new_avg:.4f}, n={len(new_fids)}) "
            f"| Old(avg fid={old_avg:.4f}, n={len(old_fids)})",
            flush=True,
        )

    # Sweep 2b: policy - choose smallest util >= min_util (no fidelity knowledge)
    print("[Sweep2b] Policy: choose smallest util >= min_util (within Top-K):", flush=True)
    for mu in min_utils:
        new_candidates = [(util, fid) for (util, fid) in pts_new if util >= mu]
        old_candidates = [(util, fid) for (util, fid) in pts_old if util >= mu]
        if new_candidates:
            new_util, new_fid = min(new_candidates, key=lambda p: (p[0], -p[1]))
        else:
            new_util = new_fid = float("nan")
        if old_candidates:
            old_util, old_fid = min(old_candidates, key=lambda p: (p[0], -p[1]))
        else:
            old_util = old_fid = float("nan")
        print(
            f"  min_util={mu:.2f} | New(pick util={new_util:.4f}, fid={new_fid:.4f}) "
            f"| Old(pick util={old_util:.4f}, fid={old_fid:.4f})",
            flush=True,
        )

    # Sweep 2c: policy - average fidelity of m smallest util >= min_util
    m_list = [3, 5]
    print("[Sweep2c] Policy: avg fidelity of m smallest util >= min_util (within Top-K):", flush=True)
    for mu in min_utils:
        new_candidates = sorted([p for p in pts_new if p[0] >= mu], key=lambda p: p[0])
        old_candidates = sorted([p for p in pts_old if p[0] >= mu], key=lambda p: p[0])
        for m in m_list:
            new_take = new_candidates[:m]
            old_take = old_candidates[:m]
            new_avg = float(np.mean([fid for (_u, fid) in new_take])) if new_take else float("nan")
            old_avg = float(np.mean([fid for (_u, fid) in old_take])) if old_take else float("nan")
            print(
                f"  min_util={mu:.2f} m={m} | New(avg fid={new_avg:.4f}, n={len(new_take)}) "
                f"| Old(avg fid={old_avg:.4f}, n={len(old_take)})",
                flush=True,
            )

    # Sweep 3: util bins -> mean/max fidelity within Top-K
    print("[Sweep3] Util bins (within Top-K):", flush=True)
    bin_edges = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
    bins = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    for lo, hi in bins:
        new_fids = [fid for (util, fid) in pts_new if lo <= util < hi]
        old_fids = [fid for (util, fid) in pts_old if lo <= util < hi]
        new_mean = float(np.mean(new_fids)) if new_fids else float("nan")
        old_mean = float(np.mean(old_fids)) if old_fids else float("nan")
        new_max = float(np.max(new_fids)) if new_fids else float("nan")
        old_max = float(np.max(old_fids)) if old_fids else float("nan")
        m = min(len(new_fids), len(old_fids), 2)
        if m > 0:
            new_topm = float(np.mean(sorted(new_fids, reverse=True)[:m]))
            old_topm = float(np.mean(sorted(old_fids, reverse=True)[:m]))
        else:
            new_topm = old_topm = float("nan")
        print(
            f"  bin[{lo:.2f},{hi:.2f}) | "
            f"New(mean={new_mean:.4f}, max={new_max:.4f}, top{m}={new_topm:.4f}, n={len(new_fids)}) "
            f"| Old(mean={old_mean:.4f}, max={old_max:.4f}, top{m}={old_topm:.4f}, n={len(old_fids)})",
            flush=True,
        )
    if pts_new:
        nx = [p[0] for p in pts_new]
        ny = [p[1] for p in pts_new]
        print(f"[New] util range=({min(nx):.3f}, {max(nx):.3f}) fid range=({min(ny):.3f}, {max(ny):.3f})", flush=True)
    if pts_old:
        ox = [p[0] for p in pts_old]
        oy = [p[1] for p in pts_old]
        print(f"[Old] util range=({min(ox):.3f}, {max(ox):.3f}) fid range=({min(oy):.3f}, {max(oy):.3f})", flush=True)
    print("[New] frontier points:", _pareto_frontier(pts_new), flush=True)
    print("[Old] frontier points:", _pareto_frontier(pts_old), flush=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    if pts_new:
        nx = [p[0] for p in pts_new]
        ny = [p[1] for p in pts_new]
        ax.scatter(nx, ny, s=32, c="#1f77b4", alpha=0.9, marker="o", label="New")

    if pts_old:
        ox = [p[0] for p in pts_old]
        oy = [p[1] for p in pts_old]
        # Slightly smaller so overlaps remain visible when drawn on top.
        ax.scatter(ox, oy, s=28, c="#ff7f0e", alpha=0.9, marker="x", label="Old")

    ax.set_xlabel("Effective Utilization")
    ax.set_ylabel("Fidelity")
    if ranks is not None and not np.isnan(avg_rank_new) and not np.isnan(avg_rank_old):
        ax.set_title(f"Top-K Pareto Frontiers (New AvgRank={avg_rank_new:.3f} / Old AvgRank={avg_rank_old:.3f})")
    else:
        ax.set_title("Top-K Pareto Frontiers")
    ax.grid(True, alpha=0.2)
    ax.legend()

    out_path = args.out
    if not os.path.isabs(out_path):
        # Default relative outputs to this script's directory.
        out_path = os.path.join(os.path.dirname(__file__), out_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(out_path)
    out_path = f"{base}_{ts}{ext}"
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"[OK] Wrote {out_path}")

    # Bar plot: mean fidelity per util bin (0.50-0.60, step 0.02)
    bin_edges_bar = [round(0.50 + 0.02 * i, 2) for i in range(6)]
    bins_bar = [(bin_edges_bar[i], bin_edges_bar[i + 1]) for i in range(len(bin_edges_bar) - 1)]
    new_means = []
    old_means = []
    labels = []
    for lo, hi in bins_bar:
        new_bin = [fid for (util, fid) in pts_new if lo <= util < hi]
        old_bin = [fid for (util, fid) in pts_old if lo <= util < hi]
        new_means.append(float(np.mean(new_bin)) if new_bin else float("nan"))
        old_means.append(float(np.mean(old_bin)) if old_bin else float("nan"))
        labels.append(f"{lo:.2f}-{hi:.2f}")

    fig2, ax2 = plt.subplots(figsize=(10.0, 4.5), dpi=150)
    x = np.arange(len(labels))
    width = 0.4
    ax2.bar(x - width / 2, np.nan_to_num(new_means), width, label="New", color="#1f77b4")
    ax2.bar(x + width / 2, np.nan_to_num(old_means), width, label="Old", color="#ff7f0e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_xlabel("Utilization range")
    ax2.set_ylabel("Mean fidelity")
    ax2.set_title(f"Mean Fidelity by Utilization Bin (Top-K={config.TOP_K})")
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.legend()
    fig2.tight_layout()
    base2, ext2 = os.path.splitext(out_path)
    out_dir = os.path.dirname(out_path)
    base_name = os.path.basename(base2)
    out_bar = os.path.join(out_dir, f"bar_{base_name}{ext2}")
    fig2.savefig(out_bar)
    print(f"[OK] Wrote {out_bar}")


if __name__ == "__main__":
    main()
