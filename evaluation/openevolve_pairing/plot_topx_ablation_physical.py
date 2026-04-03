#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.openevolve_pairing import config, evaluator  # noqa: E402
from evaluation.openevolve_pairing.build_physical_pair_csv import build_rows  # noqa: E402


ROOT = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing")
OPENEVOLVE_OUT = ROOT / "openevolve_output"
IBM_JOBS = Path("/work/nvme/betu/lily/QOS/ibm_quantum/jobs")
FIG_BASE = ROOT / "figures" / "ablation_topx_physical"

EVAL_UTILS = [30, 60, 88]
PHYSICAL_TOP_RATIOS = [0.10, 0.20]
TARGET_QUBITS = {30: 8, 60: 16, 88: 24}
BACKEND_QUBITS = {"torino": 133, "marrakesh": 156}

RUN_GROUPS: Dict[str, List[Path]] = {
    "top1": [
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_143101_flash_top01_combined50_50_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_221737_flash_top01_combined50_50_r2_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_004743_flash_top01_combined50_50_r3_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
    ],
    "top5": [
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_190239_flash_top05_combined50_50_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter90",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_221736_flash_top05_combined50_50_r2_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_004742_flash_top05_combined50_50_r3_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
    ],
    "top10": [
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_190238_flash_top10_combined50_50_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter90",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_221736_flash_top10_combined50_50_r2_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_004742_flash_top10_combined50_50_r3_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
    ],
    "top15": [
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_190238_flash_top15_combined50_50_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter90",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_221737_flash_top15_combined50_50_r2_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_004741_flash_top15_combined50_50_r3_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
    ],
    "top20": [
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_193441_flash_top20_combined50_50_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_221737_flash_top20_combined50_50_r2_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
        OPENEVOLVE_OUT / "GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_004742_flash_top20_combined50_50_r3_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100",
    ],
}


def reset_eval() -> None:
    evaluator._INIT_DONE = False
    evaluator._BENCHMARKS = None
    evaluator._CANDIDATES = None
    evaluator._FEATURES = None
    evaluator._QERNEL_PAIRS = None
    evaluator._MP = None
    evaluator._PAIR_METRICS = {}
    evaluator._PAIR_METADATA = {}
    evaluator._PAIR_METADATA_COLUMNS = []
    evaluator._PAIR_RANKS = None
    evaluator._PAIR_PROXY = None
    evaluator._SIM = None


def init_for_util(util: int) -> None:
    config.TARGET_UTIL = util
    config.SHOTS = 1000
    config.CANDIDATE_LIMIT = None
    config.EVAL_RESTRICT_TO_CSV = False
    evaluator.config.TARGET_UTIL = util
    evaluator.config.SHOTS = 1000
    evaluator.config.CANDIDATE_LIMIT = None
    evaluator.config.EVAL_RESTRICT_TO_CSV = False
    reset_eval()
    evaluator._init()


def scores_for_target(target_path: Path) -> np.ndarray:
    _, score_fn = evaluator._load_score_fn(str(target_path))
    scores = []
    for idx in range(len(evaluator._CANDIDATES)):
        q1, q2 = evaluator._QERNEL_PAIRS[idx]
        try:
            score = float(score_fn(evaluator._MP, q1, q2, evaluator._SIM.backend, weighted=False, weights=[]))
        except Exception:
            score = -1e9
        scores.append(score)
    return np.asarray(scores, dtype=float)


def select_name_pairs(scores: np.ndarray, name_map: Dict[Tuple[str, str], int], top_ratio: float) -> set[Tuple[str, str]]:
    k = max(1, int(math.ceil(len(scores) * top_ratio)))
    top_idx = set(np.argsort(scores)[-k:].tolist())
    return {(n1, n2) for (n1, n2), idx in name_map.items() if idx in top_idx}


def physical_map_for(machine: str, util: int) -> Dict[Tuple[str, str], Tuple[float, float]]:
    fidelity_dir = IBM_JOBS / f"util{util}_ibm_{machine}_shots8192" / "fidelity"
    rows = build_rows(str(fidelity_dir))
    return {
        (str(r["name_1"]), str(r["name_2"])): (float(r["effective_utilization"]), float(r["fidelity"]))
        for r in rows
    }


def mean_on_physical(selected_pairs: set[Tuple[str, str]], physical_map: Dict[Tuple[str, str], Tuple[float, float]]) -> Tuple[float, float]:
    effs = []
    fids = []
    for key in selected_pairs:
        if key in physical_map:
            eff, fid = physical_map[key]
            effs.append(eff)
            fids.append(fid)
    return float(np.mean(effs)), float(np.mean(fids))


def compute_run_metrics(run_dir: Path, machine: str) -> Dict[float, Dict[str, float]]:
    target_path = run_dir / "best" / "best_program.py"
    backend_qubits = BACKEND_QUBITS[machine]
    out: Dict[float, Dict[str, float]] = {}
    for phys_top_ratio in PHYSICAL_TOP_RATIOS:
        rels = []
        effs = []
        fids = []
        for util in EVAL_UTILS:
            init_for_util(util)
            name_map = {(n1, n2): idx for idx, (_c1, _c2, n1, n2) in enumerate(evaluator._CANDIDATES)}
            scores = scores_for_target(target_path)
            selected = select_name_pairs(scores, name_map, phys_top_ratio)
            eff, fid = mean_on_physical(selected, physical_map_for(machine, util))
            rel = eff / (TARGET_QUBITS[util] / backend_qubits)
            rels.append(rel)
            effs.append(eff)
            fids.append(fid)
        out[phys_top_ratio] = {
            "mean_normalized_eff_util": float(np.mean(rels)),
            "mean_effective_utilization": float(np.mean(effs)),
            "mean_fidelity": float(np.mean(fids)),
        }
    return out


def aggregate_group_metrics(run_dirs: List[Path], machine: str) -> Dict[str, float]:
    by_ratio = {
        ratio: {"mean_normalized_eff_util": [], "mean_effective_utilization": [], "mean_fidelity": []}
        for ratio in PHYSICAL_TOP_RATIOS
    }
    for run_dir in run_dirs:
        run_metrics = compute_run_metrics(run_dir, machine)
        for ratio in PHYSICAL_TOP_RATIOS:
            for key, value in run_metrics[ratio].items():
                by_ratio[ratio][key].append(value)
    out = {}
    for ratio in PHYSICAL_TOP_RATIOS:
        prefix = "top10" if abs(ratio - 0.10) < 1e-12 else "top20"
        out[f"{prefix}_fidelity_mean"] = float(np.mean(by_ratio[ratio]["mean_fidelity"]))
        out[f"{prefix}_fidelity_var"] = float(np.var(by_ratio[ratio]["mean_fidelity"]))
        out[f"{prefix}_normalized_eff_util_mean"] = float(np.mean(by_ratio[ratio]["mean_normalized_eff_util"]))
        out[f"{prefix}_normalized_eff_util_var"] = float(np.var(by_ratio[ratio]["mean_normalized_eff_util"]))
        out[f"{prefix}_effective_utilization_mean"] = float(np.mean(by_ratio[ratio]["mean_effective_utilization"]))
        out[f"{prefix}_effective_utilization_var"] = float(np.var(by_ratio[ratio]["mean_effective_utilization"]))

    avg_fids = []
    avg_norm_utils = []
    avg_eff_utils = []
    for run_idx in range(len(run_dirs)):
        avg_fids.append(
            float(
                np.mean(
                    [
                        by_ratio[0.10]["mean_fidelity"][run_idx],
                        by_ratio[0.20]["mean_fidelity"][run_idx],
                    ]
                )
            )
        )
        avg_norm_utils.append(
            float(
                np.mean(
                    [
                        by_ratio[0.10]["mean_normalized_eff_util"][run_idx],
                        by_ratio[0.20]["mean_normalized_eff_util"][run_idx],
                    ]
                )
            )
        )
        avg_eff_utils.append(
            float(
                np.mean(
                    [
                        by_ratio[0.10]["mean_effective_utilization"][run_idx],
                        by_ratio[0.20]["mean_effective_utilization"][run_idx],
                    ]
                )
            )
        )

    out["avg_fidelity_mean"] = float(np.mean(avg_fids))
    out["avg_fidelity_var"] = float(np.var(avg_fids))
    out["avg_normalized_eff_util_mean"] = float(np.mean(avg_norm_utils))
    out["avg_normalized_eff_util_var"] = float(np.var(avg_norm_utils))
    out["avg_effective_utilization_mean"] = float(np.mean(avg_eff_utils))
    out["avg_effective_utilization_var"] = float(np.var(avg_eff_utils))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", choices=sorted(BACKEND_QUBITS), default="torino")
    parser.add_argument("--fid-ymin", type=float, default=None)
    parser.add_argument("--fid-ymax", type=float, default=None)
    parser.add_argument("--norm-ymin", type=float, default=None)
    parser.add_argument("--norm-ymax", type=float, default=None)
    args = parser.parse_args()

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = FIG_BASE / f"{args.machine}_run_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    labels = []
    avg_fidelity_mean = []
    avg_fidelity_std = []
    avg_norm_util_mean = []
    avg_norm_util_std = []

    for label, run_dirs in RUN_GROUPS.items():
        metrics = aggregate_group_metrics(run_dirs, args.machine)
        row = {"setting": label, **metrics}
        rows.append(row)
        labels.append(label.replace("top", "Top ") + "%")
        avg_fidelity_mean.append(metrics["avg_fidelity_mean"])
        avg_fidelity_std.append(float(math.sqrt(metrics["avg_fidelity_var"])))
        avg_norm_util_mean.append(metrics["avg_normalized_eff_util_mean"] * 100.0)
        avg_norm_util_std.append(float(math.sqrt(metrics["avg_normalized_eff_util_var"])) * 100.0)

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "top10_fidelity_mean",
                "top10_fidelity_var",
                "top20_fidelity_mean",
                "top20_fidelity_var",
                "avg_fidelity_mean",
                "avg_fidelity_var",
                "top10_normalized_eff_util_mean",
                "top10_normalized_eff_util_var",
                "top20_normalized_eff_util_mean",
                "top20_normalized_eff_util_var",
                "avg_normalized_eff_util_mean",
                "avg_normalized_eff_util_var",
                "top10_effective_utilization_mean",
                "top10_effective_utilization_var",
                "top20_effective_utilization_mean",
                "top20_effective_utilization_var",
                "avg_effective_utilization_mean",
                "avg_effective_utilization_var",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(labels))
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
        }
    )
    fig, ax1 = plt.subplots(figsize=(10.5, 5.8), facecolor="white")
    ax1.set_facecolor("white")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    width = 0.36
    ax1.bar(
        x - width / 2,
        avg_fidelity_mean,
        width=width,
        yerr=avg_fidelity_std,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        label="Avg Fidelity",
    )
    ax1.set_ylabel("Avg Fidelity")
    ax1.set_xlabel("Top Percentage Used During Evolution")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    fid_lower = min(m - s for m, s in zip(avg_fidelity_mean, avg_fidelity_std))
    fid_upper = max(m + s for m, s in zip(avg_fidelity_mean, avg_fidelity_std))
    fid_pad = max(0.005, (fid_upper - fid_lower) * 0.08)
    fid_ymin = args.fid_ymin if args.fid_ymin is not None else fid_lower - fid_pad
    fid_ymax = args.fid_ymax if args.fid_ymax is not None else fid_upper + fid_pad
    ax1.set_ylim(fid_ymin, fid_ymax)

    ax2 = ax1.twinx()
    ax2.bar(
        x + width / 2,
        avg_norm_util_mean,
        width=width,
        yerr=avg_norm_util_std,
        color="#F58518",
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        label="Avg Normalized Eff. Util.",
    )
    ax2.set_ylabel("Avg Normalized Eff. Util. (%)")
    norm_lower = min(m - s for m, s in zip(avg_norm_util_mean, avg_norm_util_std))
    norm_upper = max(m + s for m, s in zip(avg_norm_util_mean, avg_norm_util_std))
    norm_pad = max(0.2, (norm_upper - norm_lower) * 0.08)
    norm_ymin = args.norm_ymin if args.norm_ymin is not None else norm_lower - norm_pad
    norm_ymax = args.norm_ymax if args.norm_ymax is not None else norm_upper + norm_pad
    ax2.set_ylim(norm_ymin, norm_ymax)

    legend_handles = [
        Patch(facecolor="#4C78A8", edgecolor="black", label="Avg Fidelity"),
        Patch(facecolor="#F58518", edgecolor="black", label="Avg Normalized Eff. Util."),
    ]
    ax1.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=2, frameon=False)

    fig.tight_layout()
    png_path = out_dir / "figure_topx_ablation_dual_axis.png"
    pdf_path = out_dir / "figure_topx_ablation_dual_axis.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {png_path}")
    print(f"[ok] wrote {pdf_path}")


if __name__ == "__main__":
    main()
