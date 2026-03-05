#!/usr/bin/env python3
"""
Plot Physical Fidelity by Train-Util and Overall Score (same style as
proxy_depth_ratio_trainutil_on_physical_summary.png). Reads agg CSV and optional
extra rows for u67/u81; writes proxy_depth_ratio_trainutil_on_physical_summary.png.
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(
    SCRIPT_DIR,
    "openevolve_output",
    "_proxy_depthratio_singleutil_physical_306088_8196_20260220",
)
AGG_CSV = os.path.join(OUT_DIR, "proxy_depth_ratio_singleutil_physical_agg.csv")
OUT_PNG = os.path.join(OUT_DIR, "proxy_depth_ratio_trainutil_on_physical_summary.png")

EVAL_SHOTS = 8196


def _weighted_mean_from_utilbin_csv(path: str, mean_key: str, count_key: str) -> float:
    total = 0.0
    cnt = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                m = float(row.get(mean_key, "nan"))
            except Exception:
                m = float("nan")
            try:
                c = int(float(row.get(count_key, 0) or 0))
            except Exception:
                c = 0
            if c > 0 and np.isfinite(m):
                total += m * c
                cnt += c
    return float(total / cnt) if cnt > 0 else float("nan")


def _eval_util_mean_new_fid(phys_eval_dir: str, eval_util: int, ckpt: str) -> float:
    # Example:
    #   .../phys_eval_u67/ckpt10/pair_metrics_util60_shots8196_qos_vs_ckpt10_physical_rankonly_utilbin_mean_fid.csv
    path = os.path.join(
        phys_eval_dir,
        ckpt,
        f"pair_metrics_util{eval_util}_shots{EVAL_SHOTS}_qos_vs_{ckpt}_physical_rankonly_utilbin_mean_fid.csv",
    )
    if not os.path.isfile(path):
        return float("nan")
    return _weighted_mean_from_utilbin_csv(path, "new_mean_fidelity", "new_count")


def compute_trainutil_metrics_from_phys_eval(phys_eval_dir: str) -> tuple[float, float, float]:
    """Return (top10_avg, top20_avg, overall_avg) over eval utils 30/60/88."""
    eval_utils = [30, 60, 88]
    t10 = [_eval_util_mean_new_fid(phys_eval_dir, u, "ckpt10") for u in eval_utils]
    t20 = [_eval_util_mean_new_fid(phys_eval_dir, u, "ckpt20") for u in eval_utils]
    top10_avg = float(np.mean([v for v in t10 if np.isfinite(v)])) if any(np.isfinite(v) for v in t10) else float("nan")
    top20_avg = float(np.mean([v for v in t20 if np.isfinite(v)])) if any(np.isfinite(v) for v in t20) else float("nan")
    overall = float((top10_avg + top20_avg) / 2.0) if np.isfinite(top10_avg) and np.isfinite(top20_avg) else float("nan")
    return top10_avg, top20_avg, overall


def load_agg_and_extras():
    rows = []
    with open(AGG_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            row["train_util"] = int(row["train_util"])
            rows.append(row)

    # Derive baseline directly from agg rows to avoid stale hardcoded values.
    if rows:
        base_top10 = float(np.mean([float(r["top10_base_avg_eval306088"]) for r in rows]))
        base_top20 = float(np.mean([float(r["top20_base_avg_eval306088"]) for r in rows]))
        base_overall = float(np.mean([float(r["overall_base"]) for r in rows]))
        physical_shots = int(float(rows[0].get("physical_shots", EVAL_SHOTS) or EVAL_SHOTS))
        eval_shots = int(float(rows[0].get("eval_shots", EVAL_SHOTS) or EVAL_SHOTS))
    else:
        base_top10 = float("nan")
        base_top20 = float("nan")
        base_overall = float("nan")
        physical_shots = EVAL_SHOTS
        eval_shots = EVAL_SHOTS

    u67_dir = os.path.join(OUT_DIR, "phys_eval_u67")
    u81_dir = os.path.join(OUT_DIR, "phys_eval_u81")
    present_utils = {r["train_util"] for r in rows}
    if 67 not in present_utils and os.path.isdir(u67_dir):
        u67_top10, u67_top20, u67_overall = compute_trainutil_metrics_from_phys_eval(u67_dir)
        rows.append(
            {
                "train_util": 67,
                "top10_new_avg_eval306088": u67_top10,
                "top20_new_avg_eval306088": u67_top20,
                "top10_base_avg_eval306088": base_top10,
                "top20_base_avg_eval306088": base_top20,
                "overall_new": u67_overall,
                "overall_base": base_overall,
                "delta_overall": u67_overall - base_overall if np.isfinite(u67_overall) else float("nan"),
                "physical_shots": physical_shots,
                "eval_shots": eval_shots,
            }
        )
    if 81 not in present_utils and os.path.isdir(u81_dir):
        u81_top10, u81_top20, u81_overall = compute_trainutil_metrics_from_phys_eval(u81_dir)
        rows.append(
            {
                "train_util": 81,
                "top10_new_avg_eval306088": u81_top10,
                "top20_new_avg_eval306088": u81_top20,
                "top10_base_avg_eval306088": base_top10,
                "top20_base_avg_eval306088": base_top20,
                "overall_new": u81_overall,
                "overall_base": base_overall,
                "delta_overall": u81_overall - base_overall if np.isfinite(u81_overall) else float("nan"),
                "physical_shots": physical_shots,
                "eval_shots": eval_shots,
            }
        )

    # Sort by train_util
    rows.sort(key=lambda r: r["train_util"])
    return rows, base_top10, base_top20, base_overall, physical_shots, eval_shots


def main():
    rows, base_top10, base_top20, base_overall, physical_shots, eval_shots = load_agg_and_extras()
    train_utils = [r["train_util"] for r in rows]
    labels = [f"u{u}" for u in train_utils]
    top10 = np.array([float(r["top10_new_avg_eval306088"]) for r in rows])
    top20 = np.array([float(r["top20_new_avg_eval306088"]) if r.get("top20_new_avg_eval306088") else np.nan for r in rows])
    overall = np.array([float(r["overall_new"]) for r in rows])
    # Mask nan for plotting (matplotlib bars with nan may show as 0)
    top10_safe = np.where(np.isfinite(top10), top10, 0.0)
    top20_safe = np.where(np.isfinite(top20), top20, 0.0)
    overall_safe = np.where(np.isfinite(overall), overall, 0.0)
    has_u81 = len(overall) > 0 and np.isnan(overall[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    x = np.arange(len(labels))
    width = 0.35

    # Left: Physical Fidelity by Train-Util (Top10% blue, Top20% orange)
    bars1 = ax1.bar(x - width / 2, top10_safe, width, label="Top10%", color="#1f77b4")
    bars2 = ax1.bar(x + width / 2, top20_safe, width, label="Top20%", color="#ff7f0e")
    ax1.axhline(base_top10, color="#1f77b4", linestyle="--", linewidth=1, alpha=0.8, label=f"Baseline Top10% ({base_top10:.3f})")
    ax1.axhline(base_top20, color="#ff7f0e", linestyle="--", linewidth=1, alpha=0.8, label=f"Baseline Top20% ({base_top20:.3f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Physical Mean Fidelity")
    ax1.set_xlabel("Train-Util Method")
    ax1.set_title("Physical Fidelity by Train-Util Method")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(0.0, 0.9)
    ax1.grid(True, axis="y", alpha=0.2)
    # Mark u81 as N/A if nan
    if has_u81:
        idx81 = next((i for i, u in enumerate(train_utils) if u == 81), None)
        if idx81 is not None:
            ax1.text(idx81, 0.05, "N/A", ha="center", va="bottom", fontsize=9)

    # Right: Overall Score (green)
    bars3 = ax2.bar(x, overall_safe, width=0.6, color="#2ca02c", label="Mean(Top10%, Top20%)")
    ax2.axhline(base_overall, color="gray", linestyle="--", linewidth=1, alpha=0.8, label=f"Baseline overall ({base_overall:.3f})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Mean(Top10%, Top20%)")
    ax2.set_xlabel("Train-Util Method")
    ax2.set_title("Overall Score (Higher Better)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0.0, 0.9)
    ax2.grid(True, axis="y", alpha=0.2)
    for i, (v, v_safe) in enumerate(zip(overall, overall_safe)):
        if np.isfinite(v):
            ax2.text(i, v_safe + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        else:
            ax2.text(i, 0.02, "N/A", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"Depth-ratio evolved per train util, evaluated on Physical util30/60/88 (shots={physical_shots}; eval shots={eval_shots})",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    plt.close()
    print("[OK] Wrote", OUT_PNG)


if __name__ == "__main__":
    main()
