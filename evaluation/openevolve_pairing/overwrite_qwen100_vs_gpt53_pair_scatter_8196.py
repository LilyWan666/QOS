#!/usr/bin/env python3
import csv
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.openevolve_pairing import config, evaluator


ROOT = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing")
OUT = ROOT / "openevolve_output" / "_qwen100_vs_gpt53codex100_pair_scatter_physical8196_20260303"
PAIR = ROOT / "pairing_metadata"

TARGET_ORIG = ROOT / "target_orig.py"
TARGET_QWEN = (
    ROOT
    / "openevolve_output/Qwen2.5-14B-Instruct_iter500_20260213_130246_u306088_s1000_top10_avgrank_iter500/checkpoints/checkpoint_100/best_program.py"
)
TARGET_GPT = (
    ROOT
    / "openevolve_output/OpenAI_gpt-5.3-codex_iter100_20260303_005505_proxy_depthratio_avg306088_gpt53codex_iter100_20260303_login/best/best_program.py"
)

EVAL_UTILS = [30, 60, 88]
TOPS = [0.10, 0.20]


def _reset_eval():
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


def _load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append(
            {
                "name_1": r["name_1"],
                "name_2": r["name_2"],
                "effective_utilization": float(r["effective_utilization"]),
                "fidelity": float(r["fidelity"]),
            }
        )
    return out


def _scores_for_target(util: int, target_path: Path):
    config.TARGET_UTIL = util
    config.SHOTS = 1000
    config.CANDIDATE_LIMIT = None
    config.EVAL_RESTRICT_TO_CSV = False
    evaluator.config.TARGET_UTIL = config.TARGET_UTIL
    evaluator.config.SHOTS = config.SHOTS
    evaluator.config.CANDIDATE_LIMIT = config.CANDIDATE_LIMIT
    evaluator.config.EVAL_RESTRICT_TO_CSV = config.EVAL_RESTRICT_TO_CSV
    _reset_eval()
    evaluator._init()
    _kind, score_fn = evaluator._load_score_fn(str(target_path))

    scores = []
    name_map = {}
    for idx, (_c1, _c2, n1, n2) in enumerate(evaluator._CANDIDATES):
        q1, q2 = evaluator._QERNEL_PAIRS[idx]
        try:
            s = float(score_fn(evaluator._MP, q1, q2, evaluator._SIM.backend, weighted=False, weights=[]))
        except Exception:
            s = -1e9
        scores.append(s)
        name_map[(n1, n2)] = idx
    return np.asarray(scores, dtype=float), name_map


def _select_name_pairs(scores, name_map, top_ratio):
    n = len(scores)
    k = max(1, int(math.ceil(n * top_ratio)))
    top_idx = set(np.argsort(scores)[-k:].tolist())
    out = set()
    for (n1, n2), idx in name_map.items():
        if idx in top_idx:
            out.add((n1, n2))
    return out, k


def _save_points_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name_1",
                "name_2",
                "effective_utilization",
                "fidelity",
                "in_orig",
                "in_qwen",
                "in_gpt",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["name_1"],
                    r["name_2"],
                    f'{r["effective_utilization"]:.12g}',
                    f'{r["fidelity"]:.12g}',
                    int(r["in_orig"]),
                    int(r["in_qwen"]),
                    int(r["in_gpt"]),
                ]
            )


def _plot_pair_scatter(out_png: Path, out_pdf: Path, rows, util: int, top_label: str):
    x = np.array([r["effective_utilization"] for r in rows], dtype=float)
    y = np.array([r["fidelity"] for r in rows], dtype=float)
    in_orig = np.array([r["in_orig"] for r in rows], dtype=bool)
    in_qwen = np.array([r["in_qwen"] for r in rows], dtype=bool)
    in_gpt = np.array([r["in_gpt"] for r in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, s=20, c="#bfbfbf", alpha=0.5, label=f"All pairs (n={len(rows)})", zorder=1)
    ax.scatter(
        x[in_orig], y[in_orig], s=80, c="#ff8c00", edgecolors="white", linewidths=0.7,
        label=f"QOS orig {top_label} (n={int(in_orig.sum())})", zorder=2
    )
    ax.scatter(
        x[in_qwen], y[in_qwen], s=260, facecolors="none", edgecolors="#1f77b4", linewidths=2.0,
        label=f"Qwen100 {top_label} (n={int(in_qwen.sum())})", zorder=3
    )
    ax.scatter(
        x[in_gpt], y[in_gpt], s=130, marker="X", c="#ff1493", edgecolors="#222222", linewidths=0.8,
        label=f"GPT53codex100 {top_label} (n={int(in_gpt.sum())})", zorder=4
    )

    oq = int(np.logical_and(in_orig, in_qwen).sum())
    og = int(np.logical_and(in_orig, in_gpt).sum())
    qg = int(np.logical_and(in_qwen, in_gpt).sum())
    ax.text(
        0.01, 0.02, f"Orig∩Qwen={oq} | Orig∩GPT={og} | Qwen∩GPT={qg}",
        transform=ax.transAxes, fontsize=10, color="#555555"
    )

    ax.set_title(f"Pair-level Scatter (util={util}, {top_label}, physical shots=8196)")
    ax.set_xlabel("Effective Utilization")
    ax.set_ylabel("Fidelity")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)


def _method_case_means(rows, mask):
    arr_f = np.array([r["fidelity"] for r in rows], dtype=float)
    arr_u = np.array([r["effective_utilization"] for r in rows], dtype=float)
    idx = np.array(mask, dtype=bool)
    if idx.sum() == 0:
        return np.nan, np.nan
    return float(arr_f[idx].mean()), float(arr_u[idx].mean())


def _plot_overall_figs(records):
    methods = ["orig", "qwen100", "gpt53codex100"]
    labels = ["QOS orig", "Qwen100", "GPT53codex100"]
    colors = ["#ff8c00", "#1f77b4", "#ff1493"]

    fid_mean = {}
    util_mean = {}
    for m in methods:
        fid_mean[m] = float(np.mean([r[f"{m}_fid"] for r in records]))
        util_mean[m] = float(np.mean([r[f"{m}_util"] for r in records]))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ax = axs[0]
    for i, m in enumerate(methods):
        ax.scatter(util_mean[m], fid_mean[m], s=140, c=colors[i], label=labels[i], zorder=3)
    ax.annotate("", xy=(util_mean["qwen100"], fid_mean["qwen100"]), xytext=(util_mean["orig"], fid_mean["orig"]),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.8))
    ax.annotate("", xy=(util_mean["gpt53codex100"], fid_mean["gpt53codex100"]), xytext=(util_mean["orig"], fid_mean["orig"]),
                arrowprops=dict(arrowstyle="->", color="#ff1493", lw=1.8))
    ax.set_xlabel("Overall Mean Effective Utilization")
    ax.set_ylabel("Overall Mean Fidelity")
    ax.set_title("Overall Mean: Fidelity vs Effective Utilization")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axs[1]
    x = np.arange(2)
    width = 0.22
    base = np.array([fid_mean["orig"], util_mean["orig"]], dtype=float)
    q = np.array([fid_mean["qwen100"], util_mean["qwen100"]], dtype=float)
    g = np.array([fid_mean["gpt53codex100"], util_mean["gpt53codex100"]], dtype=float)
    q_gain = (q - base) / np.maximum(np.abs(base), 1e-12) * 100.0
    g_gain = (g - base) / np.maximum(np.abs(base), 1e-12) * 100.0
    ax.axhline(0.0, color="#666666", lw=1.0)
    ax.bar(x - width / 2, q_gain, width=width, color="#1f77b4", label="Qwen100 vs orig")
    ax.bar(x + width / 2, g_gain, width=width, color="#ff1493", label="GPT53codex100 vs orig")
    ax.set_xticks(x, ["Fidelity", "Effective Utilization"])
    ax.set_ylabel("Relative Gain (%)")
    ax.set_title("Relative Gain vs QOS orig")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.suptitle("Qwen100 vs GPT53codex100 (physical util30/60/88, Top10/Top20 avg)")
    fig.tight_layout()
    fig.savefig(OUT / "qwen100_vs_gpt53codex100_overall_and_gain_combined.png", dpi=180)
    fig.savefig(OUT / "qwen100_vs_gpt53codex100_overall_and_gain_combined.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(labels))
    w = 0.35
    fvals = [fid_mean[m] for m in methods]
    uvals = [util_mean[m] for m in methods]
    ax.bar(x - w / 2, fvals, width=w, color="#4c78a8", label="Mean Fidelity")
    ax.bar(x + w / 2, uvals, width=w, color="#72b7b2", label="Mean Effective Utilization")
    ax.set_xticks(x, labels)
    ax.set_title("Overall Averages (util30/60/88, Top10/Top20 avg)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "qwen100_vs_gpt53codex100_overall_bar.png", dpi=180)
    fig.savefig(OUT / "qwen100_vs_gpt53codex100_overall_bar.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_records = []

    for util in EVAL_UTILS:
        rows = _load_rows(PAIR / f"pair_metrics_util{util}_shots8196.csv")
        scores_orig, name_map = _scores_for_target(util, TARGET_ORIG)
        scores_qwen, _ = _scores_for_target(util, TARGET_QWEN)
        scores_gpt, _ = _scores_for_target(util, TARGET_GPT)

        for top_ratio in TOPS:
            top_label = "Top10%" if abs(top_ratio - 0.10) < 1e-12 else "Top20%"
            top_orig, _ = _select_name_pairs(scores_orig, name_map, top_ratio)
            top_qwen, _ = _select_name_pairs(scores_qwen, name_map, top_ratio)
            top_gpt, _ = _select_name_pairs(scores_gpt, name_map, top_ratio)

            out_rows = []
            for r in rows:
                key = (r["name_1"], r["name_2"])
                out_rows.append(
                    {
                        **r,
                        "in_orig": key in top_orig,
                        "in_qwen": key in top_qwen,
                        "in_gpt": key in top_gpt,
                    }
                )

            t = 10 if top_ratio == 0.10 else 20
            csv_path = OUT / f"u{util}_t{t}_pair_scatter_points.csv"
            png_path = OUT / f"u{util}_t{t}_pair_scatter_qwen100_vs_gpt53codex100.png"
            pdf_path = OUT / f"u{util}_t{t}_pair_scatter_qwen100_vs_gpt53codex100.pdf"
            _save_points_csv(csv_path, out_rows)
            _plot_pair_scatter(png_path, pdf_path, out_rows, util, top_label)

            of, ou = _method_case_means(out_rows, [r["in_orig"] for r in out_rows])
            qf, qu = _method_case_means(out_rows, [r["in_qwen"] for r in out_rows])
            gf, gu = _method_case_means(out_rows, [r["in_gpt"] for r in out_rows])
            all_records.append(
                {
                    "util": util,
                    "top": int(t),
                    "orig_fid": of,
                    "orig_util": ou,
                    "qwen100_fid": qf,
                    "qwen100_util": qu,
                    "gpt53codex100_fid": gf,
                    "gpt53codex100_util": gu,
                }
            )

    _plot_overall_figs(all_records)
    print("[DONE] overwritten all plots/csv in", OUT, flush=True)
    for util in EVAL_UTILS:
        n = len(_load_rows(PAIR / f"pair_metrics_util{util}_shots8196.csv"))
        print(f"[INFO] util{util} rows={n}", flush=True)


if __name__ == "__main__":
    main()
