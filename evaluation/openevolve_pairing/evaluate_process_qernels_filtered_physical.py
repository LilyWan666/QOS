#!/usr/bin/env python3
"""Evaluate the static process_qernels filter set against physical-machine results."""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
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
from qos.error_mitigator.analyser import SupermarqFeaturesAnalysisPass  # noqa: E402
from qos.types.types import Qernel  # noqa: E402


ROOT = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing")
PAIRING_SHOTS = 1000
PHYSICAL_SHOTS = 8192
UTILS = [30, 60, 88]
TOP_RATIOS = [0.10, 0.20]
MACHINE_TO_FIG_DIR = {
    "marrakesh": ROOT / "figures" / "process_qernels_filtered_marrakesh_physical",
    "torino": ROOT / "figures" / "process_qernels_filtered_torino_physical",
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
    parser.add_argument("--machine", choices=sorted(MACHINE_TO_PHYSICAL_DIRS.keys()), default="torino")
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args()


def _reset_evaluator():
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


def _configure_evaluator(util: int):
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


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ratio_label(top_ratio: float) -> str:
    return f"Top{int(round(top_ratio * 100.0))}%"


def _selected_k(total_count: int, top_ratio: float) -> int:
    if total_count <= 0:
        return 0
    return max(1, int(math.ceil(float(total_count) * float(top_ratio))))


def _normalize_name(v: object) -> str:
    return str(v or "").strip()


def _load_physical_rows(util: int, out_dir: Path, machine: str, physical_dirs: Dict[int, Path]):
    fidelity_dir = physical_dirs[int(util)]
    rows = build_rows(str(fidelity_dir))
    out_csv = out_dir / f"pair_metrics_util{util}_shots{PHYSICAL_SHOTS}_{machine}_from_json.csv"
    _write_csv(out_csv, rows, list(rows[0].keys()))
    ordered_map = {}
    for row in rows:
        key = (_normalize_name(row.get("name_1")), _normalize_name(row.get("name_2")))
        if key not in ordered_map:
            ordered_map[key] = row
    return rows, ordered_map


def _match_physical_pair(name_1: str, name_2: str, ordered_map):
    key = (_normalize_name(name_1), _normalize_name(name_2))
    if key in ordered_map:
        return ordered_map[key], "ordered"
    swapped = (key[1], key[0])
    if swapped in ordered_map:
        return ordered_map[swapped], "swapped"
    return None, "missing"


def _process_qernels_filtered_rows(threshold: float):
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
        threshold=float(threshold),
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
    rows.sort(key=lambda r: (-float(r["score"]), -float(r["spatial_utilization"]), r["name_1"], r["name_2"]))
    return rows


def _set_plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.8,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )


def _save_figure(fig: plt.Figure, path_base: Path):
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_metric(rows: List[Dict], value_key: str, ylabel: str, out_base: Path):
    _set_plot_style()
    order = [30, 60, 88, "Mean"]
    labels = ["util30", "util60", "util88", "Mean"]
    top_order = [_ratio_label(r) for r in TOP_RATIOS]
    colors = {"Top10%": "#4c78a8", "Top20%": "#f58518"}
    x = np.arange(len(order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for idx, top_label in enumerate(top_order):
        vals = []
        for bucket in order:
            row = next(r for r in rows if r["util_bucket"] == bucket and r["top_label"] == top_label)
            vals.append(float(row[value_key]))
        offsets = x + (idx - (len(top_order) - 1) / 2.0) * width
        bars = ax.bar(
            offsets,
            vals,
            width=width,
            color=colors[top_label],
            edgecolor="#222222",
            linewidth=0.8,
            label=top_label,
        )
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=12)
    _save_figure(fig, out_base)


def _aggregate_rows(summary_rows: List[Dict]) -> List[Dict]:
    aggregate_rows = []
    for top_ratio in TOP_RATIOS:
        top_label = _ratio_label(top_ratio)
        rows = [r for r in summary_rows if float(r["top_ratio"]) == float(top_ratio)]
        aggregate_rows.append(
            {
                "util_bucket": "Mean",
                "eval_util": "Mean",
                "top_ratio": float(top_ratio),
                "top_label": top_label,
                "selected_count": float(np.mean([float(r["selected_count"]) for r in rows])),
                "matched_count": float(np.mean([float(r["matched_count"]) for r in rows])),
                "missing_count": float(np.mean([float(r["missing_count"]) for r in rows])),
                "ordered_match_count": int(sum(int(r["ordered_match_count"]) for r in rows)),
                "swapped_match_count": int(sum(int(r["swapped_match_count"]) for r in rows)),
                "ranked_pair_count": float(np.mean([float(r["ranked_pair_count"]) for r in rows])),
                "physical_row_count": float(np.mean([float(r["physical_row_count"]) for r in rows])),
                "mean_fidelity": float(np.mean([float(r["mean_fidelity"]) for r in rows])),
                "mean_effective_utilization": float(np.mean([float(r["mean_effective_utilization"]) for r in rows])),
                "join_coverage": float(np.mean([float(r["join_coverage"]) for r in rows])),
            }
        )
    return aggregate_rows


def main():
    args = _parse_args()
    machine = str(args.machine)
    threshold = float(args.threshold)
    out_root = MACHINE_TO_FIG_DIR[machine]
    physical_dirs = MACHINE_TO_PHYSICAL_DIRS[machine]

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"run_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "machine": machine,
                "pairing_logic": "multiprogrammer.process_qernels static filter",
                "threshold": threshold,
                "ranking_shots": PAIRING_SHOTS,
                "physical_shots": PHYSICAL_SHOTS,
                "utils": UTILS,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    join_rows = []
    summary_rows = []
    join_overview_rows = []

    for util in UTILS:
        phys_rows, ordered_physical = _load_physical_rows(util, out_dir, machine, physical_dirs)
        _configure_evaluator(util)
        selected_rows = _process_qernels_filtered_rows(threshold=threshold)

        per_util_join_counter: Counter = Counter()
        util_join_rows: List[Dict] = []
        for idx, pair_row in enumerate(selected_rows, start=1):
            name_1 = pair_row["name_1"]
            name_2 = pair_row["name_2"]
            physical_row, match_mode = _match_physical_pair(name_1, name_2, ordered_physical)
            matched = physical_row is not None
            if matched:
                per_util_join_counter[match_mode] += 1
            else:
                per_util_join_counter["missing"] += 1
            row = {
                "eval_util": int(util),
                "name_1": name_1,
                "name_2": name_2,
                "selected_rank": int(idx),
                "score": float(pair_row["score"]),
                "spatial_utilization": float(pair_row["spatial_utilization"]),
                "layout_1": pair_row["layout_1"],
                "layout_2": pair_row["layout_2"],
                "selected_top10": 0,
                "selected_top20": 0,
                "matched_physical": int(matched),
                "physical_match_mode": match_mode,
                "effective_utilization": float(physical_row["effective_utilization"]) if matched else "",
                "fidelity": float(physical_row["fidelity"]) if matched else "",
                "physical_backend": physical_row.get("backend", "") if matched else "",
                "physical_job_id": physical_row.get("job_id", "") if matched else "",
                "physical_json_file": physical_row.get("json_file", "") if matched else "",
            }
            util_join_rows.append(row)
            join_rows.append(row)

        detail_csv = out_dir / f"process_qernels_util{util}_joined_physical.csv"
        _write_csv(detail_csv, util_join_rows, list(util_join_rows[0].keys()) if util_join_rows else [])

        ranked_pair_count = len(util_join_rows)
        top_sets = {}
        for top_ratio in TOP_RATIOS:
            k = _selected_k(ranked_pair_count, top_ratio)
            top_sets[top_ratio] = set(range(1, k + 1))
        for row in util_join_rows:
            row["selected_top10"] = int(int(row["selected_rank"]) in top_sets[0.10])
            row["selected_top20"] = int(int(row["selected_rank"]) in top_sets[0.20])

        _write_csv(detail_csv, util_join_rows, list(util_join_rows[0].keys()) if util_join_rows else [])

        matched_total = per_util_join_counter["ordered"] + per_util_join_counter["swapped"]
        join_overview_rows.append(
            {
                "eval_util": int(util),
                "selected_pair_count": ranked_pair_count,
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
            util_summary_rows.append(
                {
                    "util_bucket": int(util),
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
                    "mean_fidelity": float(np.mean([float(r["fidelity"]) for r in matched_rows])) if matched_rows else float("nan"),
                    "mean_effective_utilization": float(np.mean([float(r["effective_utilization"]) for r in matched_rows])) if matched_rows else float("nan"),
                    "join_coverage": float(len(matched_rows)) / float(len(selected_rows)) if selected_rows else float("nan"),
                }
            )
        util_summary_csv = out_dir / f"process_qernels_util{util}_summary.csv"
        _write_csv(util_summary_csv, util_summary_rows, list(util_summary_rows[0].keys()))
        summary_rows.extend(util_summary_rows)

    join_overview_csv = out_dir / "join_coverage_by_util.csv"
    _write_csv(join_overview_csv, join_overview_rows, list(join_overview_rows[0].keys()))

    aggregate_rows = _aggregate_rows(summary_rows)
    all_summary_rows = summary_rows + aggregate_rows
    summary_csv = out_dir / "summary_by_util.csv"
    _write_csv(summary_csv, all_summary_rows, list(all_summary_rows[0].keys()))

    _plot_metric(all_summary_rows, "mean_fidelity", "Mean Physical Fidelity", out_dir / "figure_fidelity_top10_top20")
    _plot_metric(
        all_summary_rows,
        "mean_effective_utilization",
        "Mean Physical Effective Utilization",
        out_dir / "figure_effutil_top10_top20",
    )

    print(f"[OK] Output dir: {out_dir}")
    print(f"[OK] Summary by util: {summary_csv}")
    print(f"[OK] Join coverage: {join_overview_csv}")


if __name__ == "__main__":
    main()
