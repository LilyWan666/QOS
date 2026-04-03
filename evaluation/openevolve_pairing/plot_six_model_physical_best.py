#!/usr/bin/env python3
import csv
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.openevolve_pairing import config, evaluator  # noqa: E402
from evaluation.openevolve_pairing.build_physical_pair_csv import build_rows  # noqa: E402


ROOT = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing")
OPENEVOLVE_OUT = ROOT / "openevolve_output"
IBM_JOBS = Path("/work/nvme/betu/lily/QOS/ibm_quantum/jobs")
FIG_BASE = ROOT / "figures" / "six_model_physical_best"

EVAL_UTILS = [30, 60, 88]
TOP_RATIOS = [0.10, 0.20]
TARGET_QUBITS = {30: 8, 60: 16, 88: 24}
BACKEND_QUBITS = 133


@dataclass(frozen=True)
class MethodSpec:
    label: str
    glob_pattern: str
    include_in_bar: bool


METHOD_SPECS: List[MethodSpec] = [
    MethodSpec(
        label="Gemini 3 Pro",
        glob_pattern="GeminiMulti_gemini-3-pro-preview*_iter100_*/*run01_*",
        include_in_bar=True,
    ),
    MethodSpec(
        label="Gemini 3 Flash",
        glob_pattern="GeminiMulti_gemini-3-flash-preview*_iter100_*/*run01_*",
        include_in_bar=True,
    ),
    MethodSpec(
        label="GPT-5 mini",
        glob_pattern="OpenAI_gpt-5-mini_iter100_*",
        include_in_bar=True,
    ),
    MethodSpec(
        label="GPT-5.3 Codex",
        glob_pattern="OpenAI_gpt-5.3-codex_iter100_*",
        include_in_bar=True,
    ),
    MethodSpec(
        label="Claude Sonnet 4.6",
        glob_pattern="OpenAI_claude-sonnet-4-6_iter100_*",
        include_in_bar=True,
    ),
    MethodSpec(
        label="Claude Opus 4.6",
        glob_pattern="OpenAI_claude-opus-4-6_iter100_*",
        include_in_bar=True,
    ),
]

TARGET_LABEL = "target.py (orig)"
TARGET_PATH = ROOT / "target.py"


def pick_latest_run(glob_pattern: str) -> Path:
    matches = sorted(
        OPENEVOLVE_OUT.glob(glob_pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No run matches: {glob_pattern}")
    run_dir = matches[0]
    best_path = run_dir / "best" / "best_program.py"
    if not best_path.is_file():
        raise FileNotFoundError(f"Missing best program: {best_path}")
    return run_dir


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


def scores_for_target(target_path: Path) -> np.ndarray:
    _, score_fn = evaluator._load_score_fn(str(target_path))
    scores = []
    for idx in range(len(evaluator._CANDIDATES)):
        q1, q2 = evaluator._QERNEL_PAIRS[idx]
        try:
            s = float(score_fn(evaluator._MP, q1, q2, evaluator._SIM.backend, weighted=False, weights=[]))
        except Exception:
            s = -1e9
        scores.append(s)
    return np.asarray(scores, dtype=float)


def select_name_pairs(scores: np.ndarray, name_map: Dict[Tuple[str, str], int], top_ratio: float):
    k = max(1, int(math.ceil(len(scores) * top_ratio)))
    top_idx = set(np.argsort(scores)[-k:].tolist())
    out = set()
    for (n1, n2), idx in name_map.items():
        if idx in top_idx:
            out.add((n1, n2))
    return out, k


def mean_on_physical(selected_pairs, physical_map):
    fids = []
    utils = []
    miss = 0
    for key in selected_pairs:
        if key not in physical_map:
            miss += 1
            continue
        eff, fid = physical_map[key]
        utils.append(eff)
        fids.append(fid)
    if not fids:
        return float("nan"), float("nan"), 0, miss
    return float(np.mean(fids)), float(np.mean(utils)), len(fids), miss


def ratio_label(top_ratio: float) -> str:
    if abs(top_ratio - 0.10) < 1e-12:
        return "Top10%"
    if abs(top_ratio - 0.20) < 1e-12:
        return "Top20%"
    return f"Top{top_ratio * 100:.1f}%"


def main() -> None:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = FIG_BASE / f"run_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = []
    for spec in METHOD_SPECS:
        run_dir = pick_latest_run(spec.glob_pattern)
        methods.append(
            {
                "label": spec.label,
                "include_in_bar": spec.include_in_bar,
                "run_dir": str(run_dir),
                "target_path": run_dir / "best" / "best_program.py",
            }
        )
    methods.append(
        {
            "label": TARGET_LABEL,
            "include_in_bar": False,
            "run_dir": "N/A",
            "target_path": TARGET_PATH,
        }
    )

    method_lookup = {m["label"]: m for m in methods}
    if not TARGET_PATH.is_file():
        raise FileNotFoundError(f"Missing target.py: {TARGET_PATH}")

    detail_rows = []

    for util in EVAL_UTILS:
        fidelity_dir = IBM_JOBS / f"util{util}_ibm_torino_shots8192" / "fidelity"
        if not fidelity_dir.is_dir():
            raise FileNotFoundError(f"Missing physical fidelity dir: {fidelity_dir}")

        phys_rows = build_rows(str(fidelity_dir))
        if not phys_rows:
            raise RuntimeError(f"No physical rows parsed for util={util} from {fidelity_dir}")

        phys_csv = out_dir / f"pair_metrics_util{util}_shots8192_from_json.csv"
        with phys_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(phys_rows[0].keys()))
            writer.writeheader()
            writer.writerows(phys_rows)

        physical_map = {}
        for r in phys_rows:
            key = (str(r["name_1"]), str(r["name_2"]))
            physical_map[key] = (float(r["effective_utilization"]), float(r["fidelity"]))

        config.TARGET_UTIL = util
        config.SHOTS = 1000
        config.CANDIDATE_LIMIT = None
        config.EVAL_RESTRICT_TO_CSV = False
        evaluator.config.TARGET_UTIL = config.TARGET_UTIL
        evaluator.config.SHOTS = config.SHOTS
        evaluator.config.CANDIDATE_LIMIT = config.CANDIDATE_LIMIT
        evaluator.config.EVAL_RESTRICT_TO_CSV = config.EVAL_RESTRICT_TO_CSV
        reset_eval()
        evaluator._init()

        name_map = {}
        for idx, (_c1, _c2, n1, n2) in enumerate(evaluator._CANDIDATES):
            name_map[(n1, n2)] = idx

        score_cache = {}
        for m in methods:
            score_cache[m["label"]] = scores_for_target(m["target_path"])

        for top_ratio in TOP_RATIOS:
            for m in methods:
                selected, k = select_name_pairs(score_cache[m["label"]], name_map, top_ratio)
                mean_fid, mean_eff, hit_count, miss_count = mean_on_physical(selected, physical_map)
                detail_rows.append(
                    {
                        "method": m["label"],
                        "include_in_bar": int(bool(m["include_in_bar"])),
                        "run_dir": m["run_dir"],
                        "target_path": str(m["target_path"]),
                        "eval_util": util,
                        "physical_fidelity_dir": str(fidelity_dir),
                        "top_ratio": top_ratio,
                        "top_label": ratio_label(top_ratio),
                        "selected_k": k,
                        "matched_count": hit_count,
                        "missing_count": miss_count,
                        "mean_fidelity": mean_fid,
                        "mean_effective_utilization": mean_eff,
                        "mean_normalized_effective_utilization": (
                            mean_eff / (TARGET_QUBITS[util] / BACKEND_QUBITS)
                            if np.isfinite(mean_eff)
                            else float("nan")
                        ),
                    }
                )

    detail_csv = out_dir / "detail_metrics.csv"
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "include_in_bar",
                "run_dir",
                "target_path",
                "eval_util",
                "physical_fidelity_dir",
                "top_ratio",
                "top_label",
                "selected_k",
                "matched_count",
                "missing_count",
                "mean_fidelity",
                "mean_effective_utilization",
                "mean_normalized_effective_utilization",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    agg_rows = []
    for m in methods:
        for top_ratio in TOP_RATIOS:
            sub = [r for r in detail_rows if r["method"] == m["label"] and abs(r["top_ratio"] - top_ratio) < 1e-12]
            fvals = [float(r["mean_fidelity"]) for r in sub if np.isfinite(float(r["mean_fidelity"]))]
            uvals = [float(r["mean_effective_utilization"]) for r in sub if np.isfinite(float(r["mean_effective_utilization"]))]
            nuvals = [
                float(r["mean_normalized_effective_utilization"])
                for r in sub
                if np.isfinite(float(r["mean_normalized_effective_utilization"]))
            ]
            agg_rows.append(
                {
                    "method": m["label"],
                    "include_in_bar": int(bool(m["include_in_bar"])),
                    "run_dir": m["run_dir"],
                    "target_path": str(m["target_path"]),
                    "top_ratio": top_ratio,
                    "top_label": ratio_label(top_ratio),
                    "avg_mean_fidelity_eval306088": float(np.mean(fvals)) if fvals else float("nan"),
                    "avg_mean_effective_utilization_eval306088": float(np.mean(uvals)) if uvals else float("nan"),
                    "avg_mean_normalized_effective_utilization_eval306088": float(np.mean(nuvals)) if nuvals else float("nan"),
                }
            )

    agg_csv = out_dir / "aggregate_metrics.csv"
    with agg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "include_in_bar",
                "run_dir",
                "target_path",
                "top_ratio",
                "top_label",
                "avg_mean_fidelity_eval306088",
                "avg_mean_effective_utilization_eval306088",
                "avg_mean_normalized_effective_utilization_eval306088",
            ],
        )
        writer.writeheader()
        writer.writerows(agg_rows)

    bar_methods = [m["label"] for m in methods if m["include_in_bar"]]
    colors = {
        "Gemini 3 Pro": "#4C78A8",
        "Gemini 3 Flash": "#F58518",
        "GPT-5 mini": "#54A24B",
        "GPT-5.3 Codex": "#E45756",
        "Claude Sonnet 4.6": "#B279A2",
        "Claude Opus 4.6": "#72B7B2",
        TARGET_LABEL: "#000000",
    }

    x = np.arange(len(bar_methods))
    width = 0.38

    def _series(metric_key: str, top_ratio: float) -> np.ndarray:
        vals = []
        for label in bar_methods:
            row = next(
                r for r in agg_rows if r["method"] == label and abs(r["top_ratio"] - top_ratio) < 1e-12
            )
            vals.append(float(row[metric_key]))
        return np.asarray(vals, dtype=float)

    fid_top10 = _series("avg_mean_fidelity_eval306088", 0.10)
    fid_top20 = _series("avg_mean_fidelity_eval306088", 0.20)
    util_top10 = _series("avg_mean_normalized_effective_utilization_eval306088", 0.10) * 100.0
    util_top20 = _series("avg_mean_normalized_effective_utilization_eval306088", 0.20) * 100.0
    fid_base_top10 = float(
        next(
            r for r in agg_rows if r["method"] == TARGET_LABEL and abs(r["top_ratio"] - 0.10) < 1e-12
        )["avg_mean_fidelity_eval306088"]
    )
    fid_base_top20 = float(
        next(
            r for r in agg_rows if r["method"] == TARGET_LABEL and abs(r["top_ratio"] - 0.20) < 1e-12
        )["avg_mean_fidelity_eval306088"]
    )
    util_base_top10 = 100.0 * float(
        next(
            r for r in agg_rows if r["method"] == TARGET_LABEL and abs(r["top_ratio"] - 0.10) < 1e-12
        )["avg_mean_normalized_effective_utilization_eval306088"]
    )
    util_base_top20 = 100.0 * float(
        next(
            r for r in agg_rows if r["method"] == TARGET_LABEL and abs(r["top_ratio"] - 0.20) < 1e-12
        )["avg_mean_normalized_effective_utilization_eval306088"]
    )

    fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=220)
    b1 = ax1.bar(x - width / 2, fid_top10, width=width, label="Top10%", color="#4C78A8", edgecolor="black")
    b2 = ax1.bar(x + width / 2, fid_top20, width=width, label="Top20%", color="#F58518", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_methods, rotation=16, ha="right")
    ax1.set_ylabel("Mean Fidelity")
    ax1.set_title("Best Model Fidelity (avg over eval util 30/60/88)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.axhline(
        fid_base_top10,
        color="#4C78A8",
        ls="--",
        lw=1.2,
        alpha=0.85,
        label=f"Baseline Top10% ({fid_base_top10:.3f})",
    )
    ax1.axhline(
        fid_base_top20,
        color="#F58518",
        ls="--",
        lw=1.2,
        alpha=0.85,
        label=f"Baseline Top20% ({fid_base_top20:.3f})",
    )
    ax1.legend(loc="best")
    ax1.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax1.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    fig1.tight_layout()
    fig1.savefig(out_dir / "figure_fidelity_top10_top20.png", dpi=220)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 6), dpi=220)
    c1 = ax2.bar(x - width / 2, util_top10, width=width, label="Top10%", color="#4C78A8", edgecolor="black")
    c2 = ax2.bar(x + width / 2, util_top20, width=width, label="Top20%", color="#F58518", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_methods, rotation=16, ha="right")
    ax2.set_ylabel("Mean Normalized Eff. Util. (%)")
    ax2.set_title("Best Model Normalized Effective Utilization (avg over eval util 30/60/88)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.axhline(
        util_base_top10,
        color="#4C78A8",
        ls="--",
        lw=1.2,
        alpha=0.85,
        label=f"Baseline Top10% ({util_base_top10:.3f})",
    )
    ax2.axhline(
        util_base_top20,
        color="#F58518",
        ls="--",
        lw=1.2,
        alpha=0.85,
        label=f"Baseline Top20% ({util_base_top20:.3f})",
    )
    ax2.legend(loc="best")
    ax2.bar_label(c1, fmt="%.3f", padding=2, fontsize=8)
    ax2.bar_label(c2, fmt="%.3f", padding=2, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_dir / "figure_effutil_top10_top20.png", dpi=220)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10.5, 7.2), dpi=240)
    short_name = {
        "Gemini 3 Pro": "Gemini Pro",
        "Gemini 3 Flash": "Gemini Flash",
        "GPT-5 mini": "GPT-5 mini",
        "GPT-5.3 Codex": "GPT-5.3 Codex",
        "Claude Sonnet 4.6": "Sonnet 4.6",
        "Claude Opus 4.6": "Opus 4.6",
        "QOS+QOS": "QOS+QOS",
        "MP+QOS": "MP+QOS",
    }

    model_legend_handles = []
    for m in methods:
        m_label = m["label"]
        row10 = next(r for r in agg_rows if r["method"] == m_label and abs(r["top_ratio"] - 0.10) < 1e-12)
        row20 = next(r for r in agg_rows if r["method"] == m_label and abs(r["top_ratio"] - 0.20) < 1e-12)
        x10 = float(row10["avg_mean_normalized_effective_utilization_eval306088"]) * 100.0
        y10 = float(row10["avg_mean_fidelity_eval306088"])
        x20 = float(row20["avg_mean_normalized_effective_utilization_eval306088"]) * 100.0
        y20 = float(row20["avg_mean_fidelity_eval306088"])

        color = colors.get(m_label, "#777777")
        is_target = m_label == TARGET_LABEL
        z = 6 if is_target else 4

        ax3.scatter(
            [x10],
            [y10],
            s=95 if not is_target else 160,
            marker="o",
            color=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=z + 1,
        )
        ax3.scatter(
            [x20],
            [y20],
            s=95 if not is_target else 160,
            marker="^",
            color=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=z + 1,
        )
        model_legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
                linestyle="None",
                markersize=8.5 if not is_target else 10.0,
                label=short_name.get(m_label, m_label),
            )
        )

    ax3.set_xlabel("Mean Normalized Eff. Util. (%)")
    ax3.set_ylabel("Mean Fidelity")
    ax3.set_title("Fidelity vs Normalized Effective Utilization (Top10/Top20, with target.py)")
    ax3.grid(alpha=0.22, linestyle=":")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="#333333", linestyle="None", label="Top10%"),
        plt.Line2D([0], [0], marker="^", color="#333333", linestyle="None", label="Top20%"),
    ]
    marker_legend = ax3.legend(handles=legend_handles, loc="lower right", title="Selection")
    ax3.add_artist(marker_legend)
    ax3.legend(
        handles=model_legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
        title="Method",
        fontsize=8.4,
        title_fontsize=9.0,
    )
    fig3.tight_layout()
    fig3.savefig(out_dir / "figure_scatter_effutil_vs_fidelity_with_target.png", dpi=240)
    plt.close(fig3)

    method_csv = out_dir / "method_run_mapping.csv"
    with method_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "include_in_bar", "run_dir", "target_path"])
        writer.writeheader()
        for m in methods:
            writer.writerow(
                {
                    "method": m["label"],
                    "include_in_bar": int(bool(m["include_in_bar"])),
                    "run_dir": m["run_dir"],
                    "target_path": str(m["target_path"]),
                }
            )

    print(f"[OK] Output: {out_dir}")
    print("[OK] Figures:")
    print(f"  - {out_dir / 'figure_fidelity_top10_top20.png'}")
    print(f"  - {out_dir / 'figure_effutil_top10_top20.png'}")
    print(f"  - {out_dir / 'figure_scatter_effutil_vs_fidelity_with_target.png'}")
    print("[OK] CSV:")
    print(f"  - {method_csv}")
    print(f"  - {detail_csv}")
    print(f"  - {agg_csv}")


if __name__ == "__main__":
    main()
