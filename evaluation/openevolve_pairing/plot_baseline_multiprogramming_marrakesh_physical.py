#!/usr/bin/env python3
"""Evaluate baseline multiprogramming ranking against physical-machine results."""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.openevolve_pairing import config, evaluator  # noqa: E402
from evaluation.openevolve_pairing.build_physical_pair_csv import build_rows  # noqa: E402
from qos.error_mitigator.analyser import SupermarqFeaturesAnalysisPass  # noqa: E402
from qos.types.types import Qernel  # noqa: E402


ROOT = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing")
TARGET_PATH = ROOT / "target_baseline_multiprogramming.py"
PAIRING_SHOTS = 1000
PHYSICAL_SHOTS = 8192
UTILS = [30, 60, 88]
TOP_RATIOS = [0.10, 0.20]
PAIRING_MODE_TO_FIG_DIR = {
    "score_topk": {
        "marrakesh": ROOT / "figures" / "baseline_multiprogramming_marrakesh_physical",
        "torino": ROOT / "figures" / "baseline_multiprogramming_torino_physical",
    },
    "process_qernels": {
        "marrakesh": ROOT / "figures" / "baseline_process_qernels_marrakesh_physical",
        "torino": ROOT / "figures" / "baseline_process_qernels_torino_physical",
    },
    "random": {
        "marrakesh": ROOT / "figures" / "baseline_random_marrakesh_physical",
        "torino": ROOT / "figures" / "baseline_random_torino_physical",
    },
}
MACHINE_TO_PHYSICAL_DIRS = {
    "marrakesh": {
        30: Path("/work/nvme/betu/lily/QOS/ibm_quantum/jobs/util30_ibm_marrakesh_shots8192/fidelity"),
        60: Path("/work/nvme/betu/lily/QOS/ibm_quantum/jobs/util60_ibm_marrakesh_shots8192/fidelity"),
        88: Path("/work/nvme/betu/lily/QOS/ibm_quantum/jobs/util88_ibm_marrakesh_shots8192/fidelity"),
    },
    "torino": {
        30: Path("/work/nvme/betu/lily/QOS/ibm_quantum/jobs/util30_ibm_torino_shots8192/fidelity"),
        60: Path("/work/nvme/betu/lily/QOS/ibm_quantum/jobs/util60_ibm_torino_shots8192/fidelity"),
        88: Path("/work/nvme/betu/lily/QOS/ibm_quantum/jobs/util88_ibm_torino_shots8192/fidelity"),
    },
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--machine",
        choices=sorted(MACHINE_TO_PHYSICAL_DIRS.keys()),
        default="marrakesh",
        help="Physical-machine family to evaluate.",
    )
    parser.add_argument(
        "--pairing-mode",
        choices=("score_topk", "process_qernels", "random"),
        default="score_topk",
        help="Pairing selection mode. process_qernels matches reproduce_fig11 default behavior.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used for random pairing mode.")
    return parser.parse_args()


def _reset_evaluator() -> None:
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


def _configure_evaluator(util: int) -> None:
    config.TARGET_UTIL = int(util)
    config.SHOTS = int(PAIRING_SHOTS)
    config.CANDIDATE_LIMIT = None
    config.EVAL_RESTRICT_TO_CSV = False
    evaluator.config.TARGET_UTIL = config.TARGET_UTIL
    evaluator.config.SHOTS = config.SHOTS
    evaluator.config.CANDIDATE_LIMIT = config.CANDIDATE_LIMIT
    evaluator.config.EVAL_RESTRICT_TO_CSV = config.EVAL_RESTRICT_TO_CSV
    _reset_evaluator()
    evaluator._init()
    evaluator._ensure_pair_ranks()


def _ratio_label(top_ratio: float) -> str:
    return f"Top{int(round(top_ratio * 100.0))}%"


def _selected_k(n_items: int, top_ratio: float) -> int:
    if n_items <= 0:
        return 0
    return max(1, min(n_items, int(math.ceil(float(n_items) * float(top_ratio)))))


def _score_all(score_fn) -> np.ndarray:
    scores = []
    for idx in range(len(evaluator._CANDIDATES)):
        q1, q2 = evaluator._QERNEL_PAIRS[idx]
        try:
            score = float(
                score_fn(
                    evaluator._MP,
                    q1,
                    q2,
                    evaluator._SIM.backend,
                    weighted=False,
                    weights=[],
                )
            )
        except Exception:
            score = -1e9
        scores.append(score)
    return np.asarray(scores, dtype=float)


def _stable_desc_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="mergesort")


def _random_order(n_items: int, seed: int) -> np.ndarray:
    rng = random.Random(int(seed))
    order = list(range(int(n_items)))
    rng.shuffle(order)
    return np.asarray(order, dtype=int)


def _random_sample_indices(n_items: int, sample_size: int, seed: int) -> set[int]:
    rng = random.Random(int(seed))
    indices = list(range(int(n_items)))
    if sample_size >= n_items:
        return set(indices)
    return set(rng.sample(indices, int(sample_size)))


def _ranked_pairs_via_process_qernels():
    analyser = SupermarqFeaturesAnalysisPass()
    qernel_pool = {}
    backend = evaluator._SIM.backend
    target_qubits = int(evaluator.repro.UTIL_TO_QUBITS[config.TARGET_UTIL])

    for name, sizes in evaluator._BENCHMARKS.items():
        for nq, circ in sizes.items():
            if int(nq) > target_qubits:
                continue
            q = Qernel(circ)
            analyser.run(q)
            qernel_pool[q] = {
                "circuit": circ,
                "name": f"{name}-{nq}",
                "layout": list(range(int(nq))),
            }

    def pair_filter(q1, q2, _l1, _l2, _backend):
        return (q1.get_circuit().num_qubits + q2.get_circuit().num_qubits) == target_qubits

    qernel_dict = {
        q: [(meta["layout"], backend, 1.0)]
        for q, meta in qernel_pool.items()
    }
    ranked = evaluator._MP.process_qernels(
        qernel_dict,
        threshold=0.0,
        dry_run=True,
        pair_filter=pair_filter,
        return_ranked=True,
    )
    seen = set()
    rows = []
    for q1, q2, layout1, layout2, score, spatial_util, _backend in ranked:
        key = tuple(sorted((id(q1), id(q2))))
        if key in seen:
            continue
        seen.add(key)
        meta1 = qernel_pool.get(q1)
        meta2 = qernel_pool.get(q2)
        if not meta1 or not meta2:
            continue
        rows.append(
            {
                "name_1": meta1["name"],
                "name_2": meta2["name"],
                "score": float(score),
                "spatial_utilization": float(spatial_util),
                "layout_1": " ".join(str(x) for x in layout1),
                "layout_2": " ".join(str(x) for x in layout2),
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_name(v: object) -> str:
    return str(v or "").strip()


def _load_physical_rows(
    util: int,
    out_dir: Path,
    machine: str,
    physical_dirs: Dict[int, Path],
) -> Tuple[List[Dict], Dict[Tuple[str, str], Dict]]:
    fidelity_dir = physical_dirs[int(util)]
    if not fidelity_dir.is_dir():
        raise FileNotFoundError(f"Missing physical fidelity dir: {fidelity_dir}")
    rows = build_rows(str(fidelity_dir))
    if not rows:
        raise RuntimeError(f"No physical rows parsed from {fidelity_dir}")

    out_csv = out_dir / f"pair_metrics_util{util}_shots{PHYSICAL_SHOTS}_{machine}_from_json.csv"
    _write_csv(out_csv, rows, list(rows[0].keys()))

    ordered_map: Dict[Tuple[str, str], Dict] = {}
    for row in rows:
        key = (_normalize_name(row.get("name_1")), _normalize_name(row.get("name_2")))
        if key not in ordered_map:
            ordered_map[key] = row
    return rows, ordered_map


def _match_physical_pair(name_1: str, name_2: str, ordered_map: Dict[Tuple[str, str], Dict]):
    key = (_normalize_name(name_1), _normalize_name(name_2))
    if key in ordered_map:
        return ordered_map[key], "ordered"
    swapped = (key[1], key[0])
    if swapped in ordered_map:
        return ordered_map[swapped], "swapped"
    return None, "missing"


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.8,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )


def _save_figure(fig: plt.Figure, path_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bar(rows: List[Dict], value_key: str, ylabel: str, out_base: Path) -> None:
    _set_plot_style()
    order = [30, 60, 88, "Mean"]
    labels = ["util30", "util60", "util88", "Mean"]
    top_order = ["Top10%", "Top20%"]
    colors = {"Top10%": "#4c78a8", "Top20%": "#f58518"}

    values = {top: [] for top in top_order}
    for bucket in order:
        for top in top_order:
            match = next((r for r in rows if r["util_bucket"] == bucket and r["top_label"] == top), None)
            values[top].append(float(match[value_key]) if match and match[value_key] != "" else np.nan)

    x = np.arange(len(order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for idx, top in enumerate(top_order):
        offset = (idx - 0.5) * width
        bars = ax.bar(x + offset, values[top], width=width, label=top, color=colors[top], edgecolor="#222222", linewidth=0.8)
        for bar, val in zip(bars, values[top]):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    _save_figure(fig, out_base)


def _plot_scatter(rows: List[Dict], out_base: Path) -> None:
    _set_plot_style()
    marker_map = {"Top10%": "o", "Top20%": "s"}
    color_map = {30: "#4c78a8", 60: "#54a24b", 88: "#e45756", "Mean": "#222222"}
    fig, ax = plt.subplots(figsize=(6.8, 5.3))

    for row in rows:
        x = float(row["mean_effective_utilization"])
        y = float(row["mean_fidelity"])
        bucket = row["util_bucket"]
        top_label = row["top_label"]
        label = f"u{bucket}" if bucket != "Mean" else "Mean"
        ax.scatter(
            x,
            y,
            s=90 if bucket == "Mean" else 70,
            marker=marker_map[top_label],
            c=color_map[bucket],
            edgecolors="#222222",
            linewidths=0.8,
            alpha=0.95,
        )
        ax.text(x, y, f" {label}-{top_label}", fontsize=11, va="center", ha="left")

    ax.set_xlabel("Mean Effective Utilization")
    ax.set_ylabel("Mean Fidelity")
    ax.grid(True)
    _save_figure(fig, out_base)


def _aggregate_rows(summary_rows: List[Dict]) -> List[Dict]:
    agg_rows = []
    for top_ratio in TOP_RATIOS:
        top_label = _ratio_label(top_ratio)
        sub = [r for r in summary_rows if abs(float(r["top_ratio"]) - top_ratio) < 1e-12]
        fvals = [float(r["mean_fidelity"]) for r in sub if r["mean_fidelity"] != ""]
        uvals = [float(r["mean_effective_utilization"]) for r in sub if r["mean_effective_utilization"] != ""]
        matched = [int(r["matched_count"]) for r in sub]
        missing = [int(r["missing_count"]) for r in sub]
        selected = [int(r["selected_count"]) for r in sub]
        ordered = [int(r["ordered_match_count"]) for r in sub]
        swapped = [int(r["swapped_match_count"]) for r in sub]
        agg_rows.append(
            {
                "util_bucket": "Mean",
                "top_ratio": top_ratio,
                "top_label": top_label,
                "selected_count_mean": float(np.mean(selected)) if selected else float("nan"),
                "selected_count_sum": int(sum(selected)),
                "matched_count_mean": float(np.mean(matched)) if matched else float("nan"),
                "matched_count_sum": int(sum(matched)),
                "missing_count_mean": float(np.mean(missing)) if missing else float("nan"),
                "missing_count_sum": int(sum(missing)),
                "ordered_match_count_sum": int(sum(ordered)),
                "swapped_match_count_sum": int(sum(swapped)),
                "mean_fidelity": float(np.mean(fvals)) if fvals else float("nan"),
                "mean_effective_utilization": float(np.mean(uvals)) if uvals else float("nan"),
                "join_coverage_mean": float(np.mean([m / s for m, s in zip(matched, selected) if s > 0])) if selected else float("nan"),
            }
        )
    return agg_rows


def main() -> None:
    args = _parse_args()
    machine = str(args.machine)
    pairing_mode = str(args.pairing_mode)
    seed = int(args.seed)
    fig_base = PAIRING_MODE_TO_FIG_DIR[pairing_mode][machine]
    physical_dirs = MACHINE_TO_PHYSICAL_DIRS[machine]

    if not TARGET_PATH.is_file():
        raise FileNotFoundError(f"Missing baseline target wrapper: {TARGET_PATH}")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = fig_base / f"run_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "machine": machine,
        "pairing_mode": pairing_mode,
        "seed": seed,
        "target_path": str(TARGET_PATH),
        "ranking_shots": PAIRING_SHOTS,
        "physical_shots": PHYSICAL_SHOTS,
        "utils": UTILS,
        "top_ratios": TOP_RATIOS,
        "physical_dirs": {str(k): str(v) for k, v in physical_dirs.items()},
        "join_policy": "ordered_first_then_swapped",
    }
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, sort_keys=True)

    _kind, score_fn = evaluator._load_score_fn(str(TARGET_PATH))

    summary_rows: List[Dict] = []
    join_rows: List[Dict] = []
    join_overview_rows: List[Dict] = []

    for util in UTILS:
        phys_rows, ordered_physical = _load_physical_rows(util, out_dir, machine, physical_dirs)

        _configure_evaluator(util)
        per_util_join_counter: Counter = Counter()
        util_join_rows: List[Dict] = []
        if pairing_mode == "process_qernels":
            ranked_rows = _ranked_pairs_via_process_qernels()
            ranked_count = len(ranked_rows)
            top_sets = {}
            for top_ratio in TOP_RATIOS:
                k = _selected_k(ranked_count, top_ratio)
                top_sets[top_ratio] = set(range(k))
            for idx, pair_row in enumerate(ranked_rows):
                name_1 = pair_row["name_1"]
                name_2 = pair_row["name_2"]
                physical_row, match_mode = _match_physical_pair(name_1, name_2, ordered_physical)
                matched = physical_row is not None
                if matched:
                    per_util_join_counter[match_mode] += 1
                else:
                    per_util_join_counter["missing"] += 1

                physical_name_1 = _normalize_name(physical_row.get("name_1")) if matched else ""
                physical_name_2 = _normalize_name(physical_row.get("name_2")) if matched else ""

                row = {
                    "eval_util": int(util),
                    "name_1": _normalize_name(name_1),
                    "name_2": _normalize_name(name_2),
                    "score": float(pair_row["score"]),
                    "score_rank": int(idx + 1),
                    "pareto_rank": "",
                    "selected_top10": int(idx in top_sets[0.10]),
                    "selected_top20": int(idx in top_sets[0.20]),
                    "matched_physical": int(matched),
                    "physical_match_mode": match_mode,
                    "physical_name_1": physical_name_1,
                    "physical_name_2": physical_name_2,
                    "effective_utilization": float(physical_row["effective_utilization"]) if matched else "",
                    "fidelity": float(physical_row["fidelity"]) if matched else "",
                    "physical_backend": physical_row.get("backend", "") if matched else "",
                    "physical_job_id": physical_row.get("job_id", "") if matched else "",
                    "physical_json_file": physical_row.get("json_file", "") if matched else "",
                    "spatial_utilization": float(pair_row["spatial_utilization"]),
                    "layout_1": pair_row["layout_1"],
                    "layout_2": pair_row["layout_2"],
                }
                util_join_rows.append(row)
                join_rows.append(row)
        elif pairing_mode == "random":
            ranked_count = len(evaluator._CANDIDATES)
            order = _random_order(ranked_count, seed + int(util))
            score_rank = np.empty(ranked_count, dtype=int)
            score_rank[order] = np.arange(1, ranked_count + 1, dtype=int)
            top_sets = {}
            for top_ratio in TOP_RATIOS:
                k = _selected_k(ranked_count, top_ratio)
                top_sets[top_ratio] = _random_sample_indices(
                    ranked_count,
                    k,
                    seed + int(util) * 1000 + int(round(top_ratio * 100.0)),
                )

            for idx, (_circ1, _circ2, name_1, name_2) in enumerate(evaluator._CANDIDATES):
                physical_row, match_mode = _match_physical_pair(name_1, name_2, ordered_physical)
                matched = physical_row is not None
                if matched:
                    per_util_join_counter[match_mode] += 1
                else:
                    per_util_join_counter["missing"] += 1

                physical_name_1 = _normalize_name(physical_row.get("name_1")) if matched else ""
                physical_name_2 = _normalize_name(physical_row.get("name_2")) if matched else ""

                row = {
                    "eval_util": int(util),
                    "name_1": _normalize_name(name_1),
                    "name_2": _normalize_name(name_2),
                    "score": float(ranked_count - int(score_rank[idx]) + 1),
                    "score_rank": int(score_rank[idx]),
                    "pareto_rank": int(evaluator._PAIR_RANKS[idx]),
                    "selected_top10": int(idx in top_sets[0.10]),
                    "selected_top20": int(idx in top_sets[0.20]),
                    "matched_physical": int(matched),
                    "physical_match_mode": match_mode,
                    "physical_name_1": physical_name_1,
                    "physical_name_2": physical_name_2,
                    "effective_utilization": float(physical_row["effective_utilization"]) if matched else "",
                    "fidelity": float(physical_row["fidelity"]) if matched else "",
                    "physical_backend": physical_row.get("backend", "") if matched else "",
                    "physical_job_id": physical_row.get("job_id", "") if matched else "",
                    "physical_json_file": physical_row.get("json_file", "") if matched else "",
                }
                util_join_rows.append(row)
                join_rows.append(row)
        else:
            scores = _score_all(score_fn)
            order = _stable_desc_order(scores)
            score_rank = np.empty(len(scores), dtype=int)
            score_rank[order] = np.arange(1, len(scores) + 1, dtype=int)
            top_sets = {}
            for top_ratio in TOP_RATIOS:
                k = _selected_k(len(scores), top_ratio)
                top_sets[top_ratio] = set(order[:k].tolist())

            for idx, (_circ1, _circ2, name_1, name_2) in enumerate(evaluator._CANDIDATES):
                physical_row, match_mode = _match_physical_pair(name_1, name_2, ordered_physical)
                matched = physical_row is not None
                if matched:
                    per_util_join_counter[match_mode] += 1
                else:
                    per_util_join_counter["missing"] += 1

                physical_name_1 = _normalize_name(physical_row.get("name_1")) if matched else ""
                physical_name_2 = _normalize_name(physical_row.get("name_2")) if matched else ""

                row = {
                    "eval_util": int(util),
                    "name_1": _normalize_name(name_1),
                    "name_2": _normalize_name(name_2),
                    "score": float(scores[idx]),
                    "score_rank": int(score_rank[idx]),
                    "pareto_rank": int(evaluator._PAIR_RANKS[idx]),
                    "selected_top10": int(idx in top_sets[0.10]),
                    "selected_top20": int(idx in top_sets[0.20]),
                    "matched_physical": int(matched),
                    "physical_match_mode": match_mode,
                    "physical_name_1": physical_name_1,
                    "physical_name_2": physical_name_2,
                    "effective_utilization": float(physical_row["effective_utilization"]) if matched else "",
                    "fidelity": float(physical_row["fidelity"]) if matched else "",
                    "physical_backend": physical_row.get("backend", "") if matched else "",
                    "physical_job_id": physical_row.get("job_id", "") if matched else "",
                    "physical_json_file": physical_row.get("json_file", "") if matched else "",
                }
                util_join_rows.append(row)
                join_rows.append(row)

        detail_csv = out_dir / f"baseline_mp_util{util}_joined_physical.csv"
        _write_csv(detail_csv, util_join_rows, list(util_join_rows[0].keys()))

        ranked_pair_count = len(util_join_rows)
        matched_total = per_util_join_counter["ordered"] + per_util_join_counter["swapped"]
        join_overview_rows.append(
            {
                "eval_util": int(util),
                "ranked_pair_count": ranked_pair_count,
                "physical_row_count": len(phys_rows),
                "matched_count": matched_total,
                "ordered_match_count": per_util_join_counter["ordered"],
                "swapped_match_count": per_util_join_counter["swapped"],
                "missing_count": per_util_join_counter["missing"],
                "join_coverage": float(matched_total) / float(ranked_pair_count) if ranked_pair_count else float("nan"),
                "joined_csv": str(detail_csv),
            }
        )

        util_summary_rows: List[Dict] = []
        for top_ratio in TOP_RATIOS:
            top_label = _ratio_label(top_ratio)
            selected_rows = [r for r in util_join_rows if int(r[f"selected_top{int(round(top_ratio * 100.0))}"]) == 1]
            matched_rows = [r for r in selected_rows if int(r["matched_physical"]) == 1]
            ordered_selected = sum(1 for r in matched_rows if r["physical_match_mode"] == "ordered")
            swapped_selected = sum(1 for r in matched_rows if r["physical_match_mode"] == "swapped")
            mean_fid = float(np.mean([float(r["fidelity"]) for r in matched_rows])) if matched_rows else float("nan")
            mean_eff = (
                float(np.mean([float(r["effective_utilization"]) for r in matched_rows])) if matched_rows else float("nan")
            )
            summary_row = {
                "eval_util": int(util),
                "top_ratio": float(top_ratio),
                "top_label": top_label,
                "selected_count": len(selected_rows),
                "matched_count": len(matched_rows),
                "missing_count": len(selected_rows) - len(matched_rows),
                "ordered_match_count": ordered_selected,
                "swapped_match_count": swapped_selected,
                "ranked_pair_count": ranked_pair_count,
                "physical_row_count": len(phys_rows),
                "mean_fidelity": mean_fid,
                "mean_effective_utilization": mean_eff,
                "join_coverage": float(len(matched_rows)) / float(len(selected_rows)) if selected_rows else float("nan"),
                "joined_csv": str(detail_csv),
            }
            util_summary_rows.append(summary_row)
            summary_rows.append(summary_row)

        util_summary_csv = out_dir / f"baseline_mp_util{util}_summary.csv"
        _write_csv(util_summary_csv, util_summary_rows, list(util_summary_rows[0].keys()))

    join_overview_csv = out_dir / "join_coverage_by_util.csv"
    _write_csv(join_overview_csv, join_overview_rows, list(join_overview_rows[0].keys()))

    summary_csv = out_dir / "summary_by_util.csv"
    _write_csv(summary_csv, summary_rows, list(summary_rows[0].keys()))

    aggregate_rows = _aggregate_rows(summary_rows)
    aggregate_csv = out_dir / "aggregate_summary.csv"
    _write_csv(aggregate_csv, aggregate_rows, list(aggregate_rows[0].keys()))

    plot_rows = []
    for row in summary_rows:
        plot_rows.append(
            {
                "util_bucket": int(row["eval_util"]),
                "top_label": row["top_label"],
                "mean_fidelity": row["mean_fidelity"],
                "mean_effective_utilization": row["mean_effective_utilization"],
            }
        )
    plot_rows.extend(aggregate_rows)

    _plot_grouped_bar(
        plot_rows,
        value_key="mean_fidelity",
        ylabel="Mean Physical Fidelity",
        out_base=out_dir / "figure_fidelity_top10_top20",
    )
    _plot_grouped_bar(
        plot_rows,
        value_key="mean_effective_utilization",
        ylabel="Mean Physical Effective Utilization",
        out_base=out_dir / "figure_effutil_top10_top20",
    )
    _plot_scatter(plot_rows, out_base=out_dir / "figure_scatter_effutil_vs_fidelity")

    print(f"[OK] Output dir: {out_dir}")
    print(f"[OK] Summary by util: {summary_csv}")
    print(f"[OK] Aggregate summary: {aggregate_csv}")
    print(f"[OK] Join coverage: {join_overview_csv}")


if __name__ == "__main__":
    main()
