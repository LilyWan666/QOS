#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_topx_ablation_physical import compute_run_metrics


ROOT = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing")
OPENEVOLVE_OUT = ROOT / "openevolve_output"
FIG_BASE = ROOT / "figures" / "ablation_artifact_physical"


RUN_GROUPS: Dict[str, List[Path]] = {
    "full": [
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_143101_flash_top01_combined50_50_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_221737_flash_top01_combined50_50_r2_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_004743_flash_top01_combined50_50_r3_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
    ],
    "none": [
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_192340_flash_top01_combined50_50_noart_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_005950_flash_top01_combined50_50_noart_r2_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_005950_flash_top01_combined50_50_noart_r3_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
    ],
    "rankdist": [
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_010250_flash_top01_combined50_50_rankdist_only_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_020229_flash_top01_combined50_50_rankdist_only_r2_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_020229_flash_top01_combined50_50_rankdist_only_r3_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
    ],
    "toppairs": [
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_010250_flash_top01_combined50_50_toppairs_only_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_020229_flash_top01_combined50_50_toppairs_only_r2_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_020229_flash_top01_combined50_50_toppairs_only_r3_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
    ],
    "topcols": [
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_010250_flash_top01_combined50_50_topcols_only_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_020229_flash_top01_combined50_50_topcols_only_r2_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_020229_flash_top01_combined50_50_topcols_only_r3_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
    ],
}

LABELS = {
    "full": "111",
    "none": "000",
    "rankdist": "100",
    "toppairs": "010",
    "topcols": "001",
}

SETTING_ORDER = ["none", "topcols", "toppairs", "rankdist", "full"]

FULL_LABELS = {
    "full": "Full Artifacts",
    "none": "No Artifacts",
    "rankdist": "rank_distribution_csv only",
    "toppairs": "top_pairs_metrics_csv only",
    "topcols": "top_rank_pairs_all_columns_csv only",
}


def aggregate_group_metrics(run_dirs: List[Path], machine: str) -> Dict[str, float]:
    avg_fidelity = []
    avg_norm_util = []
    for run_dir in run_dirs:
        metrics = compute_run_metrics(run_dir, machine)
        avg_fidelity.append((metrics[0.10]["mean_fidelity"] + metrics[0.20]["mean_fidelity"]) / 2.0)
        avg_norm_util.append(
            ((metrics[0.10]["mean_normalized_eff_util"] + metrics[0.20]["mean_normalized_eff_util"]) / 2.0) * 100.0
        )
    return {
        "avg_fidelity_mean": float(np.mean(avg_fidelity)),
        "avg_fidelity_var": float(np.var(avg_fidelity)),
        "avg_norm_util_mean_pct": float(np.mean(avg_norm_util)),
        "avg_norm_util_var_pct": float(np.var(avg_norm_util)),
    }


def make_plot(rows: List[Dict[str, float]], out_dir: Path) -> None:
    labels = [LABELS[row["setting"]] for row in rows]
    x = np.arange(len(labels))
    fidelity_mean = [row["avg_fidelity_mean"] for row in rows]
    fidelity_std = [math.sqrt(row["avg_fidelity_var"]) for row in rows]
    norm_mean = [row["avg_norm_util_mean_pct"] for row in rows]
    norm_std = [math.sqrt(row["avg_norm_util_var_pct"]) for row in rows]

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }
    )

    fig, ax1 = plt.subplots(figsize=(7.3, 4.05), facecolor="white")
    ax1.set_facecolor("white")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)

    width = 0.36
    ax1.bar(
        x - width / 2,
        fidelity_mean,
        width=width,
        yerr=fidelity_std,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
    )
    ax1.set_ylabel("Fidelity")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    fid_lower = min(m - s for m, s in zip(fidelity_mean, fidelity_std))
    fid_upper = max(m + s for m, s in zip(fidelity_mean, fidelity_std))
    fid_pad = max(0.005, (fid_upper - fid_lower) * 0.08)
    ax1.set_ylim(fid_lower - fid_pad, fid_upper + fid_pad)

    ax2 = ax1.twinx()
    ax2.bar(
        x + width / 2,
        norm_mean,
        width=width,
        yerr=norm_std,
        color="#F58518",
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
    )
    ax2.set_ylabel("Norm. Util.")
    norm_lower = min(m - s for m, s in zip(norm_mean, norm_std))
    norm_upper = max(m + s for m, s in zip(norm_mean, norm_std))
    norm_pad = max(0.2, (norm_upper - norm_lower) * 0.08)
    ax2.set_ylim(norm_lower - norm_pad, norm_upper + norm_pad)

    legend_handles = [
        Patch(facecolor="#4C78A8", edgecolor="black", label="Fidelity"),
        Patch(facecolor="#F58518", edgecolor="black", label="Norm. Util."),
    ]
    ax1.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_dir / "figure_artifact_ablation_dual_axis.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "figure_artifact_ablation_dual_axis.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", choices=["torino", "marrakesh"], default="torino")
    args = parser.parse_args()

    out_dir = FIG_BASE / f"{args.machine}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for setting in SETTING_ORDER:
        run_dirs = RUN_GROUPS[setting]
        rows.append({"setting": setting, **aggregate_group_metrics(run_dirs, args.machine)})

    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "avg_fidelity_mean",
                "avg_fidelity_var",
                "avg_norm_util_mean_pct",
                "avg_norm_util_var_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "label_mapping.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "short_label", "full_label"])
        for setting in SETTING_ORDER:
            writer.writerow([setting, LABELS[setting], FULL_LABELS[setting]])

    make_plot(rows, out_dir)
    print(f"[ok] wrote {out_dir / 'summary.csv'}")
    print(f"[ok] wrote {out_dir / 'label_mapping.csv'}")
    print(f"[ok] wrote {out_dir / 'figure_artifact_ablation_dual_axis.png'}")
    print(f"[ok] wrote {out_dir / 'figure_artifact_ablation_dual_axis.pdf'}")


if __name__ == "__main__":
    main()
