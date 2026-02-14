#!/usr/bin/env python3
"""
Plot effective utilization vs fidelity from CSV and highlight Top-K picked by
QOS (target_orig.py) vs evolved (target.py) scoring. Labels show Pareto rank only.
Also outputs a binned mean-fidelity bar chart by effective_utilization.
"""

import argparse
import csv
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


def _avg_rank(top_idx, ranks):
    if not top_idx:
        return float("inf")
    if not ranks:
        return float("inf")
    selected = [ranks[i] for i in top_idx if 0 <= i < len(ranks)]
    if not selected:
        return float("inf")
    return float(np.mean(selected))


def _objective_from_avg_rank(avg_rank):
    if np.isfinite(avg_rank):
        return -float(avg_rank)
    return -100.0


def _resolve_effective_top_k(n_items, top_k, top_k_ratio):
    if n_items <= 0:
        return 0, 0.0
    ratio_raw = float(top_k_ratio or 0.0)
    ratio = ratio_raw
    # Accept either fraction (0.1) or percentage (10).
    if ratio > 1.0 and ratio <= 100.0:
        ratio = ratio / 100.0
    if ratio > 0.0:
        k = int(np.ceil(float(n_items) * ratio))
        k = max(1, min(n_items, k))
        return k, ratio
    k = int(top_k or 1)
    k = max(1, min(n_items, k))
    return k, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV with effective_utilization and fidelity")
    parser.add_argument("--util", type=int, default=60, choices=[30, 45, 60, 88])
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--top-k-ratio", type=float, default=0.0,
                        help="Top-K ratio. Accepts fraction (0.1) or percent (10 for 10%%).")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--out-csv", default=None, help="Output CSV path with rank/flags")
    parser.add_argument("--bar-out", default=None,
                        help="Output PNG path for binned mean-fidelity bar chart")
    parser.add_argument("--bar-out-csv", default=None,
                        help="Output CSV path for binned mean-fidelity stats")
    parser.add_argument("--summary-csv", default=None,
                        help="Output CSV path for objective summary (avg-rank and score)")
    parser.add_argument("--title-tag", default="",
                        help="Optional text tag appended in plot titles, e.g. PHYSICAL")
    parser.add_argument("--util-bin-width", type=float, default=0.02,
                        help="Bin width for effective_utilization in bar chart")
    parser.add_argument("--orig-target", default=None, help="Path to original target (default: target_orig.py)")
    parser.add_argument("--new-target", default=None, help="Path to new target (default: target.py)")
    parser.add_argument("--restrict-to-csv", action="store_true",
                        help="Only score candidates that appear in CSV (Top-K chosen within CSV subset)")
    args = parser.parse_args()

    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {args.csv}")
    if "effective_utilization" not in reader.fieldnames or "fidelity" not in reader.fieldnames:
        raise ValueError("CSV must contain effective_utilization and fidelity columns")

    # Compute Pareto ranks from CSV points (no simulation).
    utils = np.asarray([float(r["effective_utilization"]) for r in rows], dtype=float)
    fids = np.asarray([float(r["fidelity"]) for r in rows], dtype=float)
    points = list(zip(utils, fids))
    csv_ranks = _nondominated_ranks(points)
    for i, rank in enumerate(csv_ranks):
        rows[i]["pareto_rank"] = int(rank)

    # Init evaluator to compute scores (no fidelity simulation).
    config.TARGET_UTIL = args.util
    config.SHOTS = args.shots
    # Always use full candidate pool (no truncation).
    config.CANDIDATE_LIMIT = None
    # Always disable CSV-pruned candidate subset in plotting mode.
    config.EVAL_RESTRICT_TO_CSV = False
    # evaluator imports its own `config` module; keep both in sync
    evaluator.config.TARGET_UTIL = config.TARGET_UTIL
    evaluator.config.SHOTS = config.SHOTS
    evaluator.config.CANDIDATE_LIMIT = config.CANDIDATE_LIMIT
    evaluator.config.EVAL_RESTRICT_TO_CSV = config.EVAL_RESTRICT_TO_CSV
    _reset_evaluator()
    evaluator._init()
    evaluator._ensure_pair_ranks()

    # Load scoring functions.
    orig_path = args.orig_target or os.path.join(os.path.dirname(__file__), "target_orig.py")
    new_path = args.new_target or os.path.join(os.path.dirname(__file__), "target.py")
    _kind_orig, score_fn_orig = evaluator._load_score_fn(orig_path)
    _kind_new, score_fn_new = evaluator._load_score_fn(new_path)

    # Map candidate name pairs -> index
    cand_map = {}
    for idx, (_c1, _c2, n1, n2) in enumerate(evaluator._CANDIDATES):
        cand_map[(n1, n2)] = idx

    effective_top_k = 0
    effective_top_k_ratio = 0.0

    if args.restrict_to_csv:
        # Only score the subset that appears in CSV.
        subset_idx = []
        for row in rows:
            key = (row.get("name_1"), row.get("name_2"))
            idx = cand_map.get(key, None)
            if idx is not None:
                subset_idx.append(idx)
        subset_idx = sorted(set(subset_idx))
        effective_top_k, effective_top_k_ratio = _resolve_effective_top_k(
            len(subset_idx), args.top_k, args.top_k_ratio
        )

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

        if effective_top_k > 0:
            top_orig = set(sorted(subset_idx, key=lambda i: scores_orig[i])[-effective_top_k:])
            top_new = set(sorted(subset_idx, key=lambda i: scores_new[i])[-effective_top_k:])
        else:
            top_orig = set()
            top_new = set()
    else:
        scores_orig = _score_all(score_fn_orig)
        scores_new = _score_all(score_fn_new)
        effective_top_k, effective_top_k_ratio = _resolve_effective_top_k(
            len(scores_orig), args.top_k, args.top_k_ratio
        )
        if effective_top_k > 0:
            top_orig = set(np.argsort(scores_orig)[-effective_top_k:])
            top_new = set(np.argsort(scores_new)[-effective_top_k:])
        else:
            top_orig = set()
            top_new = set()

    top_orig_list = sorted(top_orig)
    top_new_list = sorted(top_new)
    eval_ranks = evaluator._PAIR_RANKS
    avg_rank_orig = _avg_rank(top_orig_list, eval_ranks)
    avg_rank_new = _avg_rank(top_new_list, eval_ranks)
    score_orig = _objective_from_avg_rank(avg_rank_orig)
    score_new = _objective_from_avg_rank(avg_rank_new)

    # Flag CSV rows that are in Top-K sets
    in_orig = []
    in_new = []
    for row in rows:
        key = (row.get("name_1"), row.get("name_2"))
        idx = cand_map.get(key, None)
        in_orig.append(idx in top_orig if idx is not None else False)
        in_new.append(idx in top_new if idx is not None else False)
    for i in range(len(rows)):
        rows[i]["topk_orig"] = bool(in_orig[i])
        rows[i]["topk_new"] = bool(in_new[i])

    title_tag = (args.title_tag or "").strip()
    if title_tag:
        title_tag = f" [{title_tag}]"
    top_desc = f"Top-K={effective_top_k}"
    if effective_top_k_ratio > 0.0:
        top_desc = f"{top_desc} ({effective_top_k_ratio * 100.0:.1f}%)"

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    ax.scatter(utils, fids, s=22, c="#cccccc", alpha=0.6, label="All")

    if any(in_orig):
        sel = np.asarray(in_orig, dtype=bool)
        ax.scatter(utils[sel], fids[sel], s=44, c="#ff7f0e", marker="x", label="QOS (orig) Top-K")
        for i in np.where(sel)[0]:
            label = f"{int(rows[i]['pareto_rank'])}"
            ax.text(utils[i], fids[i], label, fontsize=7, ha="left", va="bottom")

    if any(in_new):
        sel = np.asarray(in_new, dtype=bool)
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
            label = f"{int(rows[i]['pareto_rank'])}"
            ax.text(utils[i], fids[i], label, fontsize=7, ha="left", va="bottom")

    ax.set_xlabel("Effective Utilization")
    ax.set_ylabel("Fidelity")
    ax.set_title(
        f"Effective Utilization vs Fidelity (util={args.util}, {top_desc}){title_tag}"
    )
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
    out_fieldnames = list(reader.fieldnames or [])
    for extra in ("pareto_rank", "topk_orig", "topk_new"):
        if extra not in out_fieldnames:
            out_fieldnames.append(extra)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = args.summary_csv
    if summary_csv is None:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        summary_csv = os.path.join(os.path.dirname(args.csv), f"{base}_qos_vs_evolved_rankonly_avg_rank_summary.csv")
    summary_rows = [
        {
            "method": "orig",
            "selected_count": len(top_orig_list),
            "top_k": int(effective_top_k),
            "top_k_ratio": float(effective_top_k_ratio),
            "selection_scope": "csv_subset" if args.restrict_to_csv else "all_candidates",
            "avg_rank": avg_rank_orig,
            "objective_score": score_orig,
        },
        {
            "method": "new",
            "selected_count": len(top_new_list),
            "top_k": int(effective_top_k),
            "top_k_ratio": float(effective_top_k_ratio),
            "selection_scope": "csv_subset" if args.restrict_to_csv else "all_candidates",
            "avg_rank": avg_rank_new,
            "objective_score": score_new,
        },
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "selected_count",
                "top_k",
                "top_k_ratio",
                "selection_scope",
                "avg_rank",
                "objective_score",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Bar chart: mean fidelity per effective_utilization bin (for Top-K sets).
    if args.util_bin_width <= 0:
        raise ValueError("--util-bin-width must be > 0")

    min_u = float(np.min(utils))
    max_u = float(np.max(utils))
    bin_w = float(args.util_bin_width)
    start = float(np.floor(min_u / bin_w) * bin_w)
    if start > min_u:
        start -= bin_w
    end = float(np.ceil(max_u / bin_w) * bin_w)
    if end <= start:
        end = start + bin_w

    edges = np.arange(start, end + bin_w * 0.5, bin_w, dtype=float)
    if len(edges) < 2:
        edges = np.array([start, start + bin_w], dtype=float)

    sel_orig = np.asarray(in_orig, dtype=bool)
    sel_new = np.asarray(in_new, dtype=bool)

    labels = []
    orig_means = []
    new_means = []
    orig_counts = []
    new_counts = []
    bin_rows = []
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == len(edges) - 2:
            in_bin = (utils >= lo) & (utils <= hi)
        else:
            in_bin = (utils >= lo) & (utils < hi)

        orig_fids = fids[in_bin & sel_orig]
        new_fids = fids[in_bin & sel_new]
        orig_mean = float(np.mean(orig_fids)) if len(orig_fids) > 0 else float("nan")
        new_mean = float(np.mean(new_fids)) if len(new_fids) > 0 else float("nan")

        labels.append(f"{lo:.2f}-{hi:.2f}")
        orig_means.append(orig_mean)
        new_means.append(new_mean)
        orig_counts.append(int(len(orig_fids)))
        new_counts.append(int(len(new_fids)))
        bin_rows.append(
            {
                "util_lo": lo,
                "util_hi": hi,
                "orig_mean_fidelity": orig_mean,
                "orig_count": int(len(orig_fids)),
                "new_mean_fidelity": new_mean,
                "new_count": int(len(new_fids)),
            }
        )

    bar_csv = args.bar_out_csv
    if bar_csv is None:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        bar_csv = os.path.join(os.path.dirname(args.csv), f"{base}_qos_vs_evolved_rankonly_utilbin_mean_fid.csv")
    with open(bar_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "util_lo",
                "util_hi",
                "orig_mean_fidelity",
                "orig_count",
                "new_mean_fidelity",
                "new_count",
            ],
        )
        writer.writeheader()
        writer.writerows(bin_rows)

    fig2, ax2 = plt.subplots(figsize=(10.0, 4.8), dpi=150)
    x = np.arange(len(labels))
    width = 0.4
    ax2.bar(x - width / 2, np.asarray(orig_means, dtype=float), width, label="QOS (orig) Top-K", color="#ff7f0e")
    ax2.bar(
        x + width / 2,
        np.asarray(new_means, dtype=float),
        width,
        label="Evolved (new) Top-K",
        color="#1f77b4",
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.set_xlabel("Effective Utilization Bin")
    ax2.set_ylabel("Mean Fidelity")
    ax2.set_title(
        f"Mean Fidelity by Utilization Bin (bin={bin_w:.2f}, util={args.util}, {top_desc}){title_tag}"
    )
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.legend(loc="best")

    # Show sample counts for each bar pair.
    for i, (co, cn) in enumerate(zip(orig_counts, new_counts)):
        y = 0.0
        vo = orig_means[i]
        vn = new_means[i]
        if np.isfinite(vo):
            y = max(y, vo)
        if np.isfinite(vn):
            y = max(y, vn)
        ax2.text(x[i], y + 0.002, f"n={co}/{cn}", fontsize=7, ha="center", va="bottom")

    bar_png = args.bar_out
    if bar_png is None:
        png_base = os.path.splitext(os.path.basename(out_png))[0]
        bar_png = os.path.join(os.path.dirname(out_png), f"{png_base}_utilbin_mean_fid.png")
    plt.tight_layout()
    plt.savefig(bar_png)

    print("[OK] Wrote", out_png)
    print("[OK] Wrote", out_csv)
    print("[OK] Wrote", summary_csv)
    print("[OK] Wrote", bar_png)
    print("[OK] Wrote", bar_csv)
    print(f"[Objective] orig avg-rank={avg_rank_orig:.6f} score={score_orig:.6f}")
    print(f"[Objective] new  avg-rank={avg_rank_new:.6f} score={score_new:.6f}")


if __name__ == "__main__":
    main()
