#!/usr/bin/env python3
"""Plot depth-ratio vs fidelity on physical backends."""

import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ROOT = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing")
JOBS = Path("/work/nvme/betu/lily/QOS/ibm_quantum/jobs")
PAIRING_METADATA = ROOT / "pairing_metadata"
OUT_DIR = ROOT / "figures" / "depth_ratio_fidelity_backend_summary"
TARGET_ORIG = ROOT / "target_orig.py"
UTILS = [30, 60, 88]
BACKEND_DIRS = {
    "torino": {
        30: JOBS / "util30_ibm_torino_shots8192" / "fidelity",
        60: JOBS / "util60_ibm_torino_shots8192" / "fidelity",
        88: JOBS / "util88_ibm_torino_shots8192" / "fidelity",
    },
    "marrakesh": {
        30: JOBS / "util30_ibm_marrakesh_shots8192" / "fidelity",
        60: JOBS / "util60_ibm_marrakesh_shots8192" / "fidelity",
        88: JOBS / "util88_ibm_marrakesh_shots8192" / "fidelity",
    },
}
UTIL_STYLE = {
    30: {"color": "#4C78A8", "marker": "o", "label": "util 30%"},
    60: {"color": "#F58518", "marker": "s", "label": "util 60%"},
    88: {"color": "#54A24B", "marker": "D", "label": "util 88%"},
}


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _depth_ratio_from_utilization_file(path: Path):
    data = _load_json(path)
    circuits = data.get("circuits", [])
    if len(circuits) != 2:
        raise ValueError(f"Expected 2 circuits in {path}, got {len(circuits)}")
    depth_1 = int(circuits[0]["depth"])
    depth_2 = int(circuits[1]["depth"])
    denom = max(depth_1, depth_2, 1)
    return min(depth_1, depth_2) / denom


def _load_backend_rows(backend: str):
    rows = []
    for util in UTILS:
        fidelity_dir = BACKEND_DIRS[backend][util]
        for json_path in sorted(fidelity_dir.glob("pair_*.json")):
            data = _load_json(json_path)
            fidelity = _safe_float(data.get("hellinger_mean"))
            util_file = data.get("utilization_file")
            name_1 = str(data.get("name_1", "")).strip()
            name_2 = str(data.get("name_2", "")).strip()
            if fidelity is None or not util_file or not name_1 or not name_2:
                continue
            depth_ratio = _depth_ratio_from_utilization_file(Path(util_file))
            rows.append(
                {
                    "util": util,
                    "backend": backend,
                    "name_1": name_1,
                    "name_2": name_2,
                    "depth_ratio": depth_ratio,
                    "fidelity": fidelity,
                }
            )
    return rows


def _build_average_rows(torino_rows, marrakesh_rows):
    torino_map = {
        (r["util"], r["name_1"], r["name_2"]): r
        for r in torino_rows
    }
    marrakesh_map = {
        (r["util"], r["name_1"], r["name_2"]): r
        for r in marrakesh_rows
    }
    keys = sorted(set(torino_map) & set(marrakesh_map))
    rows = []
    for key in keys:
        t_row = torino_map[key]
        m_row = marrakesh_map[key]
        rows.append(
            {
                "util": key[0],
                "backend": "average",
                "name_1": key[1],
                "name_2": key[2],
                "depth_ratio": float(np.mean([t_row["depth_ratio"], m_row["depth_ratio"]])),
                "fidelity": float(np.mean([t_row["fidelity"], m_row["fidelity"]])),
            }
        )
    return rows


def _load_sim_rows():
    rows = []
    for util in UTILS:
        csv_path = PAIRING_METADATA / f"pair_metrics_util{util}_shots1000.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                depth_ratio = _safe_float(row.get("depth_ratio"))
                fidelity = _safe_float(row.get("fidelity"))
                name_1 = str(row.get("name_1", "")).strip()
                name_2 = str(row.get("name_2", "")).strip()
                if depth_ratio is None or fidelity is None or not name_1 or not name_2:
                    continue
                rows.append(
                    {
                        "util": util,
                        "backend": "simulation_1000",
                        "name_1": name_1,
                        "name_2": name_2,
                        "depth_ratio": depth_ratio,
                        "fidelity": fidelity,
                    }
                )
    return rows


def _pair_key(util, name_1, name_2):
    return (int(util),) + tuple(sorted((str(name_1).strip(), str(name_2).strip())))


def _reset_evaluator_orig(evaluator_orig):
    evaluator_orig._INIT_DONE = False
    evaluator_orig._BENCHMARKS = None
    evaluator_orig._CANDIDATES = None
    evaluator_orig._FEATURES = None
    evaluator_orig._QERNEL_PAIRS = None
    evaluator_orig._MP = None
    evaluator_orig._PAIR_METRICS = {}
    evaluator_orig._PAIR_RANKS = None
    evaluator_orig._PAIR_PROXY = None
    evaluator_orig._SIM = None


def _select_top_name_pairs_qos_orig(top_ratio=0.10):
    from evaluation.openevolve_pairing import config as pair_config
    from evaluation.openevolve_pairing import evaluator_orig

    selected = set()
    for util in UTILS:
        pair_config.TARGET_UTIL = int(util)
        pair_config.SHOTS = 1000
        pair_config.CANDIDATE_LIMIT = None
        evaluator_orig.config.TARGET_UTIL = pair_config.TARGET_UTIL
        evaluator_orig.config.SHOTS = pair_config.SHOTS
        evaluator_orig.config.CANDIDATE_LIMIT = pair_config.CANDIDATE_LIMIT
        _reset_evaluator_orig(evaluator_orig)
        evaluator_orig._init()
        _, score_fn = evaluator_orig._load_score_fn(str(TARGET_ORIG))
        scores = []
        for idx in range(len(evaluator_orig._CANDIDATES)):
            q1, q2 = evaluator_orig._QERNEL_PAIRS[idx]
            try:
                score = float(
                    score_fn(
                        evaluator_orig._MP,
                        q1,
                        q2,
                        evaluator_orig._SIM.backend,
                        weighted=False,
                        weights=[],
                    )
                )
            except Exception:
                score = -1e9
            scores.append(score)
        scores = np.asarray(scores, dtype=float)
        k = max(1, int(math.ceil(len(scores) * float(top_ratio))))
        top_idx = set(np.argsort(scores)[-k:].tolist())
        for idx in top_idx:
            _, _, name_1, name_2 = evaluator_orig._CANDIDATES[idx]
            selected.add(_pair_key(util, name_1, name_2))
    return selected


def _filter_rows_by_selected(rows, selected_keys):
    return [
        row
        for row in rows
        if _pair_key(row["util"], row["name_1"], row["name_2"]) in selected_keys
    ]


def _pearson_r(xs, ys):
    if len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _add_fit_line(ax, xs, ys):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, linestyle="--", linewidth=1.4, color="black", alpha=0.85)


def _plot_panel(ax, rows, title):
    xs = [r["depth_ratio"] for r in rows]
    ys = [r["fidelity"] for r in rows]
    pearson_r = _pearson_r(xs, ys)

    for util in UTILS:
        util_rows = [r for r in rows if r["util"] == util]
        if not util_rows:
            continue
        style = UTIL_STYLE[util]
        ax.scatter(
            [r["depth_ratio"] for r in util_rows],
            [r["fidelity"] for r in util_rows],
            s=28,
            alpha=0.8,
            color=style["color"],
            marker=style["marker"],
            edgecolors="black",
            linewidths=0.35,
            label=style["label"],
        )

    _add_fit_line(ax, xs, ys)

    ax.set_title(f"{title}\nPearson r = {pearson_r:.3f}")
    ax.set_xlabel("Depth Ratio")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_xlim(-0.02, 1.02)

    finite_y = [y for y in ys if math.isfinite(y)]
    if finite_y:
        y_min = min(finite_y)
        y_max = max(finite_y)
        pad = max(0.02, 0.08 * (y_max - y_min if y_max > y_min else 0.1))
        ax.set_ylim(max(-0.02, y_min - pad), min(1.02, y_max + pad))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    torino_rows = _load_backend_rows("torino")
    marrakesh_rows = _load_backend_rows("marrakesh")
    average_rows = _build_average_rows(torino_rows, marrakesh_rows)
    sim_rows = _load_sim_rows()
    top10_selected = _select_top_name_pairs_qos_orig(0.10)
    torino_top10_rows = _filter_rows_by_selected(torino_rows, top10_selected)
    marrakesh_top10_rows = _filter_rows_by_selected(marrakesh_rows, top10_selected)
    average_top10_rows = _build_average_rows(torino_top10_rows, marrakesh_top10_rows)
    sim_top10_rows = _filter_rows_by_selected(sim_rows, top10_selected)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, dpi=220)
    _plot_panel(axes[0], torino_rows, "Torino")
    _plot_panel(axes[1], marrakesh_rows, "Marrakesh")
    _plot_panel(axes[2], average_rows, "Average")
    axes[0].set_ylabel("Fidelity")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "depth_ratio_fidelity_backend_summary.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "depth_ratio_fidelity_backend_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    for stem, rows, title in (
        ("depth_ratio_fidelity_torino", torino_rows, "Torino"),
        ("depth_ratio_fidelity_marrakesh", marrakesh_rows, "Marrakesh"),
        ("depth_ratio_fidelity_average", average_rows, "Average"),
        ("depth_ratio_fidelity_sim1000", sim_rows, "Simulation (1000 shots)"),
        ("depth_ratio_fidelity_top10_torino", torino_top10_rows, "Torino Top10%"),
        ("depth_ratio_fidelity_top10_marrakesh", marrakesh_top10_rows, "Marrakesh Top10%"),
        ("depth_ratio_fidelity_top10_average", average_top10_rows, "Average Top10%"),
        ("depth_ratio_fidelity_top10_sim1000", sim_top10_rows, "Simulation Top10% (1000 shots)"),
    ):
        fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.6), dpi=220)
        _plot_panel(ax, rows, title)
        ax.set_ylabel("Fidelity")
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8), sharey=True, dpi=220)
    _plot_panel(axes[0], torino_top10_rows, "Torino Top10%")
    _plot_panel(axes[1], marrakesh_top10_rows, "Marrakesh Top10%")
    _plot_panel(axes[2], average_top10_rows, "Average Top10%")
    _plot_panel(axes[3], sim_top10_rows, "Simulation Top10%")
    axes[0].set_ylabel("Fidelity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "depth_ratio_fidelity_top10_summary.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "depth_ratio_fidelity_top10_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    for name, rows in (
        ("torino", torino_rows),
        ("marrakesh", marrakesh_rows),
        ("average", average_rows),
        ("simulation_1000", sim_rows),
        ("top10_torino", torino_top10_rows),
        ("top10_marrakesh", marrakesh_top10_rows),
        ("top10_average", average_top10_rows),
        ("top10_simulation_1000", sim_top10_rows),
    ):
        r = _pearson_r([row["depth_ratio"] for row in rows], [row["fidelity"] for row in rows])
        print(f"{name}: n={len(rows)} Pearson r={r:.6f}")


if __name__ == "__main__":
    main()
